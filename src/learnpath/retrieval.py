"""Content retrieval. Three "spaces" with the same interface: TF-IDF (built
here), Sentence-BERT and OpenAI embeddings (made in notebook 03, saved as .npy)."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config

STYLE_TEXT = {"visual": "video lessons", "reading": "reading", "hands_on": "hands-on practice"}


def profile_text(student):
    # what the student filled in at sign-up, as one sentence
    return (f"Interested in {student['stated_interests']}. "
            f"Goal: {student['goal']}. Level: {student['self_reported_level']}. "
            f"Prefers {STYLE_TEXT.get(student['learning_style'], student['learning_style'])}.")


class Space:
    """Course vectors (and profile vectors if we have them), L2-normalized."""

    def __init__(self, name, course_ids, course_matrix, profile_ids=None, profile_matrix=None,
                 encoder=None):
        self.name = name
        self.course_ids = list(course_ids)
        self.idx = {c: i for i, c in enumerate(self.course_ids)}
        self.C = course_matrix
        self.profile_idx = {p: i for i, p in enumerate(profile_ids or [])}
        self.P = profile_matrix
        self.encoder = encoder  # only TF-IDF can encode new text without an API

    def profile_vector(self, student):
        sid = student["student_id"]
        if sid in self.profile_idx:
            return self.P[self.profile_idx[sid]]
        if self.encoder is not None:
            return self.encoder(profile_text(student))
        return None

    def query_vector(self, student, history, profile_weight=0.3, decay=0.85):
        # recent courses weighted by progress, plus 30% profile.
        # no history (new student) -> profile only
        prof = self.profile_vector(student)
        if history is None or len(history) == 0:
            return prof
        h = history.sort_values("enrolled_at")
        rows = [self.idx[c] for c in h["course_id"]]
        n = len(rows)
        w = (0.5 + h["progress"].to_numpy()) * decay ** np.arange(n - 1, -1, -1)
        if sparse.issparse(self.C):
            v = sparse.csr_matrix(w) @ self.C[rows]
            v = np.asarray(v.todense()).ravel()
        else:
            v = w @ self.C[rows]
        v = v / (np.linalg.norm(v) + 1e-12)
        if prof is not None:
            p = np.asarray(prof.todense()).ravel() if sparse.issparse(prof) else np.asarray(prof).ravel()
            v = (1 - profile_weight) * v + profile_weight * p
        return v / (np.linalg.norm(v) + 1e-12)

    def scores(self, q):
        if q is None:
            return np.zeros(len(self.course_ids))
        if sparse.issparse(q):
            q = np.asarray(q.todense()).ravel()
        return np.asarray(self.C @ q).ravel()


def tfidf_space(courses):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2,
                          sublinear_tf=True, max_features=60000)
    C = vec.fit_transform(courses["doc"])
    enc = lambda text: normalize(vec.transform([text]))[0]
    return Space("tfidf", courses["course_id"], C, encoder=enc)


def save_embeddings(name, course_ids, course_matrix, profile_ids, profile_matrix):
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(config.DATA_PROCESSED / f"emb_{name}_courses.npy", course_matrix.astype(np.float32))
    np.save(config.DATA_PROCESSED / f"emb_{name}_profiles.npy", profile_matrix.astype(np.float32))
    pd.DataFrame({"course_id": course_ids}).to_csv(config.DATA_PROCESSED / f"emb_{name}_course_ids.csv", index=False)
    pd.DataFrame({"student_id": profile_ids}).to_csv(config.DATA_PROCESSED / f"emb_{name}_profile_ids.csv", index=False)


def load_space(name):
    # None if notebook 03 hasn't made it yet
    f = config.DATA_PROCESSED / f"emb_{name}_courses.npy"
    if not f.exists():
        return None
    C = normalize(np.load(f))
    P = normalize(np.load(config.DATA_PROCESSED / f"emb_{name}_profiles.npy"))
    cids = pd.read_csv(config.DATA_PROCESSED / f"emb_{name}_course_ids.csv")["course_id"]
    pids = pd.read_csv(config.DATA_PROCESSED / f"emb_{name}_profile_ids.csv")["student_id"].tolist()
    return Space(name, cids, C, pids, P)


def available_spaces(courses):
    spaces = {"tfidf": tfidf_space(courses)}
    for name in ("sbert", "openai"):
        s = load_space(name)
        if s is not None:
            spaces[name] = s
    return spaces
