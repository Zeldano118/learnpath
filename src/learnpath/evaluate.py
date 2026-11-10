"""Offline metrics. Accuracy: did the hidden next courses make the top 5?
Quality: right level, no skipped prerequisites, how much of the catalog gets shown."""

import numpy as np
import pandas as pd

from . import config


def recall_at_k(recs, relevant, k=config.TOP_K):
    if not relevant:
        return np.nan
    return len(set(recs[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recs, relevant, k=config.TOP_K):
    rel = set(relevant)
    dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(recs[:k]) if c in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else np.nan


def hit_at_k(recs, relevant, k=config.TOP_K):
    return float(bool(set(recs[:k]) & set(relevant)))


def evaluate_lists(rec_lists, truth, students, courses, checker=None, history=None,
                   k=config.TOP_K):
    # rec_lists and truth are both {student_id: [course_id, ...]}
    level_of = dict(zip(courses["course_id"], courses["level_num"]))
    topic_of = dict(zip(courses["course_id"], courses["subdomain"]))
    true_level = dict(zip(students["student_id"], students["true_level"]))
    interests = dict(zip(students["student_id"], students["true_interests"].str.split("; ")))
    rows = []
    for sid, recs in rec_lists.items():
        recs = list(recs)[:k]
        on_interest = np.mean([topic_of.get(c) in interests[sid] for c in recs]) if recs else np.nan
        lv = [level_of.get(c) for c in recs]
        lv = [x for x in lv if x == x and x is not None]
        too_hard = np.mean([x > true_level[sid] + 1 for x in lv]) if lv else np.nan
        viol = np.nan
        if checker is not None and history is not None:
            h = history.get(sid)
            taken = set(h["course_id"]) if h is not None else set()
            viol = np.mean([checker.violates(c, taken, true_level[sid]) for c in recs])
        rows.append({
            "student_id": sid,
            "recall@5": recall_at_k(recs, truth.get(sid, []), k),
            "ndcg@5": ndcg_at_k(recs, truth.get(sid, []), k),
            "hit@5": hit_at_k(recs, truth.get(sid, []), k),
            "on_interest": on_interest,
            "too_hard": too_hard,
            "prereq_violation": viol,
        })
    per = pd.DataFrame(rows)
    n_unique = len({c for recs in rec_lists.values() for c in list(recs)[:k]})
    summary = {
        "students": len(per),
        "recall@5": per["recall@5"].mean(),
        "recall@5_ci95": 1.96 * per["recall@5"].std() / np.sqrt(len(per)),
        "ndcg@5": per["ndcg@5"].mean(),
        "hit@5": per["hit@5"].mean(),
        "on_interest_share": per["on_interest"].mean(),
        "too_hard_share": per["too_hard"].mean(),
        "prereq_violation_share": per["prereq_violation"].mean(),
        "catalog_coverage": n_unique / len(courses),
    }
    return summary, per
