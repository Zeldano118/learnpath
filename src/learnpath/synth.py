"""Synthetic students and their learning history.

There is no student dataset, so this simulates one with simple rules
(assumptions A1 to A8 in notebook 02). true_interests and true_level only
drive the simulation; no model gets to see them.
"""

import numpy as np
import pandas as pd

from . import config

AGE_BANDS = (["18-24", "25-34", "35-44", "45+"], [0.35, 0.35, 0.18, 0.12])
EDUCATION = (["High school", "Bachelor's", "Master's or higher"], [0.28, 0.52, 0.20])
REGIONS = (["Southeast Asia", "South Asia", "Middle East & North Africa", "Europe",
            "North America", "Sub-Saharan Africa", "Latin America"],
           [0.22, 0.18, 0.12, 0.16, 0.14, 0.09, 0.09])
STYLES = (["visual", "reading", "hands_on"], [0.40, 0.25, 0.35])
GOALS = (["career switch", "upskill for current job", "academic study", "personal interest"],
         [0.30, 0.35, 0.15, 0.20])
# chance of true level 1/2/3 by education
LEVEL_BY_EDU = {"High school": [0.70, 0.25, 0.05],
                "Bachelor's": [0.45, 0.40, 0.15],
                "Master's or higher": [0.25, 0.45, 0.30]}
LEVEL_NAMES = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}

P_INTEREST_TOPIC = 0.80   # A1: picks from own interests
P_TRACK = 0.40            # A4: students who follow a popular path
P_STALE_INTEREST = 0.25   # A7: stated interest is out of date


def _topic_label(sub):
    return sub.replace("-", " ")


def build_world(courses, rng):
    """Hidden course traits: popularity, hours, and popular paths."""
    c = courses.copy()
    # A3: long-tail popularity, a bit higher for well rated courses
    base = rng.lognormal(mean=0.0, sigma=1.0, size=len(c))
    c["pop"] = np.clip(base * (1 + 1.5 * (c["rating"] - 4.5)), 0.05, None)
    # hours to finish are not in the dataset, guided projects are short
    hours_by_level = c["level_num"].map({1: 10, 2: 15, 3: 20}).fillna(12)
    c["est_hours"] = np.where(c["format"] == "guided_project",
                              rng.uniform(1, 2.5, len(c)),
                              hours_by_level * rng.lognormal(0, 0.35, len(c)))
    c["est_hours"] = c["est_hours"].round(1)

    # sort ties by name so the random draws are the same on every pandas version
    topics = c["subdomain"].value_counts().sort_index().sort_values(ascending=False, kind="stable")
    topic_weight = (topics ** 0.7) / (topics ** 0.7).sum()

    # A4: 3 popular paths per topic, each 2 beginner + 2 intermediate + 1 advanced course.
    # CF can pick these up, text similarity can't.
    tracks = {}
    for sub, n in topics.items():
        if n < 12:
            continue
        pool = c[c["subdomain"] == sub]
        for t in range(3):
            path = []
            for lvl, k in ((1, 2), (2, 2), (3, 1)):
                cand = pool[(pool["level_num"] == lvl) & (~pool["course_id"].isin(path))]
                if len(cand) == 0:
                    continue
                p = cand["pop"] / cand["pop"].sum()
                take = min(k, len(cand))
                path += list(rng.choice(cand["course_id"], size=take, replace=False, p=p))
            if len(path) >= 3:
                tracks.setdefault(sub, []).append(path)
    return c, topic_weight, tracks


def _pick(options, rng):
    names, probs = options
    return rng.choice(names, p=probs)


def make_student(sid, topic_weight, rng, all_topics):
    edu = _pick(EDUCATION, rng)
    true_level = int(rng.choice([1, 2, 3], p=LEVEL_BY_EDU[edu]))
    n_int = 1 if rng.random() < 0.6 else 2
    first = rng.choice(topic_weight.index, p=topic_weight.values)
    interests = [first]
    if n_int == 2:
        rest = topic_weight.drop(first)
        interests.append(rng.choice(rest.index, p=(rest / rest.sum()).values))
    stated = list(interests)
    if rng.random() < P_STALE_INTEREST:
        stated[-1] = rng.choice(all_topics)
    # A8: self-reported level is right 70% of the time, else off by one
    self_level = true_level if rng.random() < 0.7 else int(np.clip(true_level + rng.choice([-1, 1]), 1, 3))
    return {
        "student_id": sid,
        "age_band": _pick(AGE_BANDS, rng),
        "education_level": edu,
        "region": _pick(REGIONS, rng),
        "learning_style": _pick(STYLES, rng),
        "weekly_hours": float(np.clip(np.round(rng.lognormal(np.log(5), 0.5) * 2) / 2, 1, 20)),
        "goal": _pick(GOALS, rng),
        "self_reported_level": LEVEL_NAMES[self_level],
        "stated_interests": "; ".join(_topic_label(t) for t in dict.fromkeys(stated)),
        "true_interests": "; ".join(interests),
        "true_level": true_level,
    }


def _completion_prob(st, course, cur_level):
    # A5 and A6: harder than real level, little time, wrong format -> less likely to finish
    gap = max(0.0, (course["level_num"] if course["level_num"] == course["level_num"] else 2) - st["true_level"])
    fmt_match = (st["learning_style"] == "hands_on" and course["format"] == "guided_project") or \
                (st["learning_style"] == "reading" and course["format"] == "course")
    z = (0.4
         + 0.9 * (st["weekly_hours"] - 5) / 5
         - 1.3 * gap
         + 0.5 * fmt_match
         + 0.6 * (course["rating"] - 4.5)
         + 0.9 * (course["format"] == "guided_project"))
    return 1 / (1 + np.exp(-z))


def simulate_history(st, n_steps, world, tracks, topic_weight, rng, start):
    by_topic = world.groupby("subdomain")
    interests = st["true_interests"].split("; ")
    cur_level = max(1, st["true_level"] - (1 if rng.random() < 0.5 else 0))
    track = None
    if rng.random() < P_TRACK:
        options = [p for t in interests for p in tracks.get(t, [])]
        if options:
            track = list(options[rng.integers(len(options))])
    taken, rows = set(), []
    date = start
    for _ in range(n_steps):
        course = None
        if track:
            nxt = [cid for cid in track if cid not in taken]
            if nxt and rng.random() < 0.6:
                course = world.loc[nxt[0]]
        if course is None:
            if rng.random() < P_INTEREST_TOPIC:
                w = [0.65, 0.35] if len(interests) == 2 else [1.0]
                topic = rng.choice(interests, p=w)
            else:
                topic = rng.choice(topic_weight.index, p=topic_weight.values)
            pool = by_topic.get_group(topic)
            pool = pool[~pool.index.isin(taken)]
            lvl = pool["level_num"].fillna(cur_level)
            pool = pool[lvl <= cur_level + 1]   # A2: stay near the current level
            if len(pool) == 0:
                continue
            lvl = pool["level_num"].fillna(cur_level)
            w = pool["pop"] * np.exp(-1.2 * (lvl - cur_level).abs())
            if st["learning_style"] == "hands_on":
                w = w * np.where(pool["format"] == "guided_project", 1.6, 1.0)
            elif st["learning_style"] == "reading":
                w = w * np.where(pool["format"] == "course", 1.2, 1.0)
            course = pool.loc[rng.choice(pool.index, p=(w / w.sum()).values)]
        cid = course.name
        taken.add(cid)
        p_done = _completion_prob(st, course, cur_level)
        completed = rng.random() < p_done
        progress = 1.0 if completed else float(np.round(rng.beta(2, 3) * 0.95, 2))
        hours = float(np.round(course["est_hours"] * progress * rng.lognormal(0, 0.25), 1))
        rating = np.nan
        if progress >= 0.5 and rng.random() < 0.6:
            rating = int(np.clip(round(course["rating"] + rng.normal(0, 0.5) - (0 if completed else 0.8)), 1, 5))
        rows.append({"student_id": st["student_id"], "course_id": cid,
                     "enrolled_at": date.strftime("%Y-%m-%d"),
                     "time_spent_hours": hours, "progress": progress,
                     "completed": int(completed), "rating": rating})
        lvl = course["level_num"]
        # A2: finishing a course at your level sometimes moves you up
        if completed and lvl == lvl and lvl >= cur_level and cur_level < 3 and rng.random() < 0.35:
            cur_level += 1
        date = date + pd.Timedelta(days=int(3 + rng.exponential(12)))
    return rows


def generate(courses, n_students=config.N_STUDENTS, n_cold=config.N_COLD_START, seed=config.SEED):
    rng = np.random.default_rng(seed)
    world, topic_weight, tracks = build_world(courses, rng)
    world = world.set_index("course_id")
    all_topics = list(topic_weight.index)
    t0, t1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-30")

    students, interactions = [], []
    for i in range(1, n_students + 1):
        st = make_student(f"S{i:04d}", topic_weight, rng, all_topics)
        n_steps = int(np.clip(config.MIN_HISTORY + rng.poisson(5), config.MIN_HISTORY, config.MAX_HISTORY))
        start = t0 + pd.Timedelta(days=int(rng.integers(0, (t1 - t0).days)))
        rows = simulate_history(st, n_steps, world, tracks, topic_weight, rng, start)
        if len(rows) < config.MIN_HISTORY:
            continue
        students.append(st)
        interactions += rows

    cold, cold_truth = [], []
    for i in range(1, n_cold + 1):
        st = make_student(f"N{i:04d}", topic_weight, rng, all_topics)
        start = pd.Timestamp("2025-07-01") + pd.Timedelta(days=int(rng.integers(0, 60)))
        rows = simulate_history(st, 3, world, tracks, topic_weight, rng, start)
        st["split"] = "val" if i <= n_cold // 2 else "test"
        cold.append(st)
        cold_truth += [{"student_id": st["student_id"], "course_id": r["course_id"]} for r in rows]

    students = pd.DataFrame(students)
    inter = pd.DataFrame(interactions)
    inter = assign_splits(inter)
    world_out = world.reset_index()[["course_id", "pop", "est_hours"]]
    return students, inter, pd.DataFrame(cold), pd.DataFrame(cold_truth), world_out, tracks


def assign_splits(inter):
    # leave-last-out: last 2 courses = test, the one before = val, rest = train
    inter = inter.sort_values(["student_id", "enrolled_at"]).reset_index(drop=True)
    pos_from_end = inter.groupby("student_id").cumcount(ascending=False)
    inter["split"] = np.select([pos_from_end < 2, pos_from_end == 2], ["test", "val"], "train")
    return inter
