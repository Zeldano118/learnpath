"""The hybrid recommender: CF + content, merged by rank, then the graph check.

CF and content scores are on different scales, so they are merged with
reciprocal rank fusion instead of adding raw scores. alpha = weight on CF.
"""

import numpy as np
import pandas as pd

from . import config
from .cf import ItemKNN
from .graph import PrereqChecker

LEVEL_FROM_NAME = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
RRF_K = 60


def ranks(scores):
    # rank 1 = best, ties keep catalog order
    order = np.argsort(-scores, kind="stable")
    r = np.empty_like(order)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def estimate_level(student, history, level_of):
    # what the platform can see: average level of recent finished courses,
    # or the self-reported level if there are none
    if history is not None and len(history):
        done = history[history["completed"] == 1].sort_values("enrolled_at").tail(5)
        lv = [level_of.get(c) for c in done["course_id"]]
        lv = [x for x in lv if x == x and x is not None]
        if lv:
            return int(round(float(np.mean(lv))))
    return LEVEL_FROM_NAME.get(student["self_reported_level"], 1)


class Recommender:
    def __init__(self, courses, space, prereq_graph, train_interactions,
                 alpha=0.5, cold_pop_weight=0.5, use_graph=True,
                 level_penalty=0.7, prereq_penalty=0.9):
        self.courses = courses
        self.ids = courses["course_id"].to_numpy()
        self.space = space
        self.level_of = dict(zip(courses["course_id"], courses["level_num"]))
        self.levels = courses["level_num"].to_numpy()
        self.cf = ItemKNN(self.ids).fit(train_interactions)
        self.checker = PrereqChecker(prereq_graph, courses) if prereq_graph is not None else None
        self.alpha = alpha
        self.cold_pop_weight = cold_pop_weight
        self._pop_rank = ranks(self.cf.popularity)
        self.use_graph = use_graph and self.checker is not None
        self.level_penalty = level_penalty
        self.prereq_penalty = prereq_penalty
        # embedding files might store courses in a different order
        self._space_order = np.array([space.idx[c] for c in self.ids]) if space is not None else None

    def _content(self, student, history):
        q = self.space.query_vector(student, history)
        return self.space.scores(q)[self._space_order]

    def score(self, student, history, mode="hybrid", alpha=None, cold_pop_weight=None):
        a = self.alpha if alpha is None else alpha
        b = self.cold_pop_weight if cold_pop_weight is None else cold_pop_weight
        if mode == "popular":
            s = self.cf.popularity.copy()
        elif mode == "cf":
            s = self.cf.scores(history)
        elif mode == "content":
            s = self._content(student, history)
        else:
            cf = self.cf.scores(history)
            content = self._content(student, history)
            if np.any(cf):
                s = a / (RRF_K + ranks(cf)) + (1 - a) / (RRF_K + ranks(content))
            else:
                # new student: no CF signal, so mix content with popularity instead
                s = b / (RRF_K + self._pop_rank) + (1 - b) / (RRF_K + ranks(content))
        return np.asarray(s, dtype=float)

    def recommend(self, student, history, k=config.TOP_K, mode="hybrid", alpha=None,
                  cold_pop_weight=None, use_graph=None, return_frame=False):
        s = self.score(student, history, mode, alpha, cold_pop_weight)
        s = s - s.min() + 1e-9   # keep scores positive so the penalties shrink them
        taken = set(history["course_id"]) if history is not None and len(history) else set()
        seen = np.isin(self.ids, list(taken))
        s[seen] = -np.inf
        graph_on = self.use_graph if use_graph is None else use_graph
        lvl = estimate_level(student, history, self.level_of)
        if graph_on and mode in ("hybrid", "content"):
            top = np.argsort(-s)[: max(k * 10, 200)]
            for i in top:
                L = self.levels[i]
                if L == L and L > lvl + 1:
                    s[i] *= self.level_penalty
                if self.checker.violates(self.ids[i], taken, lvl):
                    s[i] *= self.prereq_penalty
        top = np.argsort(-s, kind="stable")[:k]
        if not return_frame:
            return list(self.ids[top])
        out = self.courses.iloc[top].copy()
        out["score"] = s[top]
        out["est_student_level"] = lvl
        return out


def _grid_search(rec, students, history_by_student, truth_by_student, param, grid, k):
    from .evaluate import recall_at_k
    rows = []
    for v in grid:
        r = [recall_at_k(rec.recommend(st, history_by_student.get(st["student_id"]), k=k,
                                       use_graph=False, **{param: v}),
                         truth_by_student[st["student_id"]], k)
             for st in students]
        rows.append({param: v, "val_recall@5": float(np.nanmean(r))})
    res = pd.DataFrame(rows)
    best = float(res.sort_values("val_recall@5", ascending=False, kind="stable").iloc[0][param])
    return best, res


def tune_alpha(rec, students, history_by_student, truth_by_student,
               grid=(0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0), k=config.TOP_K):
    # weight on CF for students with history (val split)
    return _grid_search(rec, students, history_by_student, truth_by_student, "alpha", grid, k)


def tune_cold_weight(rec, cold_students, truth_by_student,
                     grid=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0), k=config.TOP_K):
    # weight on popularity for new students (only the 100 val new students)
    return _grid_search(rec, cold_students, {}, truth_by_student, "cold_pop_weight", grid, k)
