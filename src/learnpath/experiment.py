"""Helpers shared by the notebooks so they all use the same splits and samples."""

import json

import numpy as np
import pandas as pd

from . import catalog, config, graph, retrieval
from .hybrid import Recommender, tune_alpha, tune_cold_weight

PARAMS_FILE = config.RESULTS / "params.json"


def load_data():
    courses = catalog.load_courses()
    students = pd.read_csv(config.STUDENTS)
    inter = pd.read_csv(config.INTERACTIONS)
    cold = pd.read_csv(config.COLD_START)
    cold_truth = pd.read_csv(config.COLD_START_TRUTH)
    return courses, students, inter, cold, cold_truth


def split_context(inter, split):
    # "val": learn from train, check on val (tuning)
    # "test": learn from train + val, check on test (final numbers)
    if split == "val":
        fit, target = inter[inter["split"] == "train"], inter[inter["split"] == "val"]
    else:
        fit, target = inter[inter["split"] != "test"], inter[inter["split"] == "test"]
    hist = {k: g for k, g in fit.groupby("student_id")}
    truth = {k: g["course_id"].tolist() for k, g in target.groupby("student_id")}
    truth_done = {k: g.loc[g["completed"] == 1, "course_id"].tolist() for k, g in target.groupby("student_id")}
    return fit, hist, truth, truth_done


def cold_context(cold, cold_truth, split):
    ids = set(cold.loc[cold["split"] == split, "student_id"])
    students = cold[cold["student_id"].isin(ids)]
    truth = {k: g["course_id"].tolist() for k, g in cold_truth[cold_truth["student_id"].isin(ids)].groupby("student_id")}
    return students, truth


def course_stats(courses, fit_inter):
    # things the platform would know from its own logs
    g = fit_inter.groupby("course_id")
    stats = pd.DataFrame({
        "enrollments": g.size(),
        "completion_rate": g["completed"].mean(),
        "median_hours": fit_inter[fit_inter["completed"] == 1].groupby("course_id")["time_spent_hours"].median(),
    })
    return courses.merge(stats, left_on="course_id", right_index=True, how="left") \
                  .fillna({"enrollments": 0})


def load_params():
    return json.loads(PARAMS_FILE.read_text()) if PARAMS_FILE.exists() else {}


def save_params(params):
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_FILE.write_text(json.dumps(params, indent=2))


def tune_space(courses, students, inter, cold, cold_truth, space, prereq_graph, n_tune=800):
    # tune alpha and the new-student popularity weight for one space, val data only
    fit, hist, truth, _ = split_context(inter, "val")
    rec = Recommender(courses, space, prereq_graph, fit)
    sample = students.sample(n=min(n_tune, len(students)), random_state=config.SEED).to_dict("records")
    alpha, a_tab = tune_alpha(rec, sample, hist, truth)
    cold_val, cold_val_truth = cold_context(cold, cold_truth, "val")
    beta, b_tab = tune_cold_weight(rec, cold_val.to_dict("records"), cold_val_truth)
    best_val = float(a_tab["val_recall@5"].max())
    return {"alpha": alpha, "cold_pop_weight": beta, "val_recall@5": best_val}, a_tab, b_tab


def llm_eval_students(students, n=config.LLM_EVAL_STUDENTS):
    return students.sample(n=n, random_state=config.SEED)["student_id"].tolist()


def best_space(params, spaces):
    ranked = sorted((p["val_recall@5"], name) for name, p in params.items() if name in spaces)
    return ranked[-1][1] if ranked else "tfidf"


def build_llm_items(rec, courses_with_stats, students_df, hist, k_pool=config.RERANK_POOL,
                    version=config.PROMPT_VERSION):
    # shortlist + prompt for every student
    from .llm import build_messages
    info = courses_with_stats.set_index("course_id")
    items = []
    for st in students_df.to_dict("records"):
        h = hist.get(st["student_id"])
        cand = rec.recommend(st, h, k=k_pool, return_frame=True)
        cand = cand[["course_id"]].join(info, on="course_id")
        hh = None
        if h is not None:
            hh = h.merge(courses_with_stats[["course_id", "name", "level"]], on="course_id", how="left")
        items.append({
            "student_id": st["student_id"],
            "messages": build_messages(st, hh, cand, version=version),
            "candidate_ids": cand["course_id"].tolist(),
            "fallback": cand["course_id"].tolist(),
            "prompt_version": version,
        })
    return items
