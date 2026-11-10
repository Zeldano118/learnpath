"""Item-based collaborative filtering (two courses are similar if the same
students take both) and course popularity."""

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


class ItemKNN:
    def __init__(self, course_ids):
        self.course_ids = list(course_ids)
        self.idx = {c: i for i, c in enumerate(self.course_ids)}

    def fit(self, interactions):
        users = interactions["student_id"].astype("category")
        self.user_idx = {u: i for i, u in enumerate(users.cat.categories)}
        rows = users.cat.codes.to_numpy()
        cols = interactions["course_id"].map(self.idx).to_numpy()
        # a finished course counts more than one dropped early
        vals = (0.5 + interactions["progress"]).to_numpy()
        X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(self.user_idx), len(self.course_ids)))
        Xn = normalize(X, axis=0)   # cosine between course columns
        S = (Xn.T @ Xn).tocsr()
        S.setdiag(0)
        S.eliminate_zeros()
        self.S = S
        self.popularity = np.asarray((X > 0).sum(axis=0)).ravel().astype(float)
        return self

    def scores(self, history):
        if history is None or len(history) == 0:
            return np.zeros(len(self.course_ids))
        cols = history["course_id"].map(self.idx).to_numpy()
        w = (0.5 + history["progress"]).to_numpy()
        v = sparse.csr_matrix((w, (np.zeros_like(cols), cols)), shape=(1, len(self.course_ids)))
        return np.asarray((v @ self.S).todense()).ravel()
