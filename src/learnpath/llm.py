"""LLM rerank. The LLM picks 5 of the 20 shortlisted courses with a reason each,
and every returned ID is checked against the shortlist (anything else is
replaced from the hybrid ranking).

Answers go into a JSONL file as they come in, so a stopped run can continue
without paying for the same calls twice.
"""

import json
import os
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from . import config
from .retrieval import STYLE_TEXT

PROMPTS = {}

PROMPTS["v1"] = """You recommend online courses to one student on a learning platform.
You get the student's profile, their recent courses, and a list of candidate courses the platform already shortlisted. Pick exactly 5 courses from that list.

Rules:
- Only use course_id values from the candidate list. Never invent a course.
- Match the student's level. Do not jump more than one level above what they have finished.
- Respect their weekly time and how they like to learn.
- Prefer a coherent path over five near-identical courses.
- List the 5 courses in the order the student should take them.
- Give one reason per course, at most 25 words, about this student. Do not restate the course description.

Answer with JSON only, in this shape:
{"recommendations": [{"course_id": "C0000", "reason": "..."}]}"""

# v2: v1 ignored the platform's ranking and lost recall (see notebook 06),
# so the model now sees the rank and how many students took each course
PROMPTS["v2"] = """You recommend online courses to one student on a learning platform.
You get the student's profile, their recent courses, and a list of candidate courses the platform already shortlisted. Pick exactly 5 courses from that list.

The candidates are ranked. #1 is the platform's best guess, based on what similar students took and how close the course is to this student's interests. "taken by" is how many students on the platform took the course.

Rules:
- Only use course_id values from the candidate list. Never invent a course.
- Start from the top of the ranking. Only skip a high-ranked course if it clearly does not fit this student (wrong level, outside their interests, too long for their time), and only pick a low-ranked one if it clearly fits better.
- Match the student's level. Do not jump more than one level above what they have finished.
- Respect their weekly time and how they like to learn.
- Prefer a coherent path over five near-identical courses.
- List the 5 courses in the order the student should take them.
- Give one reason per course, at most 25 words, about this student. Do not restate the course description.

Answer with JSON only, in this shape:
{"recommendations": [{"course_id": "C0000", "reason": "..."}]}"""


def _short(text, n_words):
    w = str(text).split()
    return " ".join(w[:n_words]) + ("..." if len(w) > n_words else "")


def build_messages(student, history, candidates, n_history=5, version=config.PROMPT_VERSION):
    # candidates = course rows plus enrollments, completion_rate and median_hours from the logs
    lines = [
        "STUDENT",
        f"- Stated interests: {student['stated_interests']}",
        f"- Goal: {student['goal']}",
        f"- Self-reported level: {student['self_reported_level']}",
        f"- Learns best through: {STYLE_TEXT.get(student['learning_style'], student['learning_style'])}",
        f"- Time available: about {student['weekly_hours']:g} hours per week",
        "",
        "RECENT COURSES (oldest first)",
    ]
    if history is None or len(history) == 0:
        lines.append("- None yet. This student just signed up.")
    else:
        for r in history.sort_values("enrolled_at").tail(n_history).itertuples():
            status = "finished" if r.completed else f"stopped at {int(r.progress * 100)}%"
            lines.append(f"- {r.name} ({r.level if isinstance(r.level, str) else 'level unknown'}), {status}")
    lines += ["", "CANDIDATES"]
    for rank, r in enumerate(candidates.itertuples(), 1):
        level = r.level if isinstance(r.level, str) else "level unknown"
        fmt = "guided project" if r.format == "guided_project" else "course"
        comp = f"{r.completion_rate:.0%} finish" if r.completion_rate == r.completion_rate else "new, no data"
        hrs = f"~{r.median_hours:.0f}h" if r.median_hours == r.median_hours else "time unknown"
        skills = ", ".join(r.skills_str.split("; ")[:5])
        stats = f"rating {r.rating:.1f}, {comp}"
        if version == "v1":
            head = f"{r.course_id} | {r.name}"
        else:
            head = f"{r.course_id} | #{rank} | {r.name}"
            stats += f", taken by {int(r.enrollments)} students"
        lines.append(f"{head} | {level} | {r.subdomain.replace('-', ' ')} | {fmt}, {hrs} | "
                     f"{stats} | skills: {skills} | {_short(r.description, 30)}")
    return [{"role": "system", "content": PROMPTS[version]},
            {"role": "user", "content": "\n".join(lines)}]


def parse_and_validate(text, candidate_ids, fallback_order, k=config.TOP_K):
    # final list + counters (invented, repeated, filled) for the evaluation
    out = {"recs": [], "reasons": {}, "n_invalid": 0, "n_duplicate": 0,
           "n_filled": 0, "parse_error": False}
    try:
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        items = json.loads(cleaned).get("recommendations", [])
    except Exception:
        items, out["parse_error"] = [], True
    allowed = set(candidate_ids)
    for it in items:
        cid = str(it.get("course_id", "")).strip() if isinstance(it, dict) else ""
        if cid not in allowed:
            out["n_invalid"] += 1
            continue
        if cid in out["recs"]:
            out["n_duplicate"] += 1
            continue
        if len(out["recs"]) < k:
            out["recs"].append(cid)
            out["reasons"][cid] = str(it.get("reason", "")).strip()
    for cid in fallback_order:
        if len(out["recs"]) >= k:
            break
        if cid not in out["recs"]:
            out["recs"].append(cid)
            out["n_filled"] += 1
    return out


def _uses_fixed_temperature(model):
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4")) or "reasoner" in m


def chat(client, model, messages, retries=3):
    kwargs = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    if not _uses_fixed_temperature(model):
        kwargs["temperature"] = 0
    if model.startswith("deepseek") and not config.DEEPSEEK_THINKING:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    last = None
    for attempt in range(retries):
        try:
            t = time.time()
            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None)
            return {"text": resp.choices[0].message.content or "",
                    "latency_s": round(time.time() - t, 2),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "error": None}
        except Exception as e:  # network, rate limit, wrong model name
            last = repr(e)
            time.sleep(2 ** attempt)
    return {"text": "", "latency_s": None, "prompt_tokens": None,
            "completion_tokens": None, "error": last}


def cost_usd(model, prompt_tokens, completion_tokens):
    p_in, p_out = config.PRICES.get(model, (float("nan"), float("nan")))
    if prompt_tokens is None or completion_tokens is None:
        return float("nan")
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1e6


def run_batch(items, client, model, out_path, k=config.TOP_K, log_every=25):
    # skips students that already have an answer in out_path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["student_id"])
    todo = [it for it in items if it["student_id"] not in done]
    print(f"{model}: {len(done)} already saved, {len(todo)} to run")
    with out_path.open("a") as f:
        for n, it in enumerate(todo, 1):
            r = chat(client, model, it["messages"])
            v = parse_and_validate(r["text"], it["candidate_ids"], it["fallback"], k)
            rec = {"student_id": it["student_id"], "model": model,
                   "prompt_version": it.get("prompt_version", config.PROMPT_VERSION),
                   "candidate_ids": it["candidate_ids"], **r, **v,
                   "cost_usd": cost_usd(model, r["prompt_tokens"], r["completion_tokens"])}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % log_every == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done")
    return load_results(out_path)


def load_results(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])


def get_secret(name):
    # Colab Secrets first, then an environment variable (laptop). Never put keys in code.
    try:
        from google.colab import userdata  # type: ignore
    except ImportError:
        userdata = None
    if userdata is not None:
        try:
            val = userdata.get(name)
            if val:
                return val
        except Exception as e:
            kind = type(e).__name__
            if kind == "NotebookAccessError":
                raise RuntimeError(f"{name} is in Colab Secrets but this notebook can't read it. "
                                   "Key icon on the left > switch on 'Notebook access' next to it.") from None
            if kind != "SecretNotFoundError":
                raise
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"No secret called {name}. Key icon on the left > Add new secret, "
                           "name it exactly like that and switch on 'Notebook access'.")
    return val


def make_client(provider):
    from openai import OpenAI
    if provider == "openai":
        return OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    if provider == "deepseek":
        return OpenAI(api_key=get_secret("DEEPSEEK_API_KEY"), base_url=config.DEEPSEEK_BASE_URL)
    raise ValueError(provider)


class MockClient:
    """Fake client for testing the pipeline offline. Sometimes returns a fake ID
    or a repeat so the checks get used. Its numbers mean nothing."""

    def __init__(self, seed=0, p_bad=0.15):
        self.rng = random.Random(seed)
        self.p_bad = p_bad
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, model, messages, **kw):
        user = messages[-1]["content"]
        cands = re.findall(r"^(C\d{4}) \|", user, flags=re.M)
        picks = cands[:8]
        self.rng.shuffle(picks)
        picks = picks[:5]
        if self.rng.random() < self.p_bad:
            picks[-1] = "C9999"
        if self.rng.random() < self.p_bad:
            picks[-2] = picks[0]
        body = {"recommendations": [{"course_id": c, "reason": "Mock reason."} for c in picks]}
        usage = SimpleNamespace(prompt_tokens=len(user) // 4 + 250, completion_tokens=120)
        msg = SimpleNamespace(content=json.dumps(body))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)
