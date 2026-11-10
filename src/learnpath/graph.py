"""Skill graph (skills that show up on the same courses) and the inferred
prerequisite graph between courses. The dataset has no real prerequisites,
so the second one is only a rough guide."""

from collections import Counter
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from . import config


def skill_document_frequency(courses):
    return Counter(s for skills in courses["skills"] for s in set(skills))


def generic_skills(courses, max_share=0.05):
    # skills on >5% of courses ("project", "analysis") link everything, leave them out
    df = skill_document_frequency(courses)
    n = len(courses)
    return {s for s, c in df.items() if c / n > max_share}


def build_skill_graph(courses, min_courses=3, min_cooccur=2):
    df = skill_document_frequency(courses)
    generic = generic_skills(courses)
    keep = {s for s, c in df.items() if c >= min_courses and s not in generic}
    pairs = Counter()
    for skills in courses["skills"]:
        ss = sorted(set(skills) & keep)
        pairs.update(combinations(ss, 2))
    G = nx.Graph()
    for (a, b), w in pairs.items():
        if w >= min_cooccur:
            G.add_edge(a, b, weight=w)
    for s in G.nodes:
        G.nodes[s]["courses"] = df[s]
    return G


def skill_communities(G, resolution=1.0, seed=config.SEED):
    comms = nx.community.louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return sorted(comms, key=len, reverse=True)


def build_prereq_graph(courses, min_jaccard=0.15, max_parents=3):
    generic = generic_skills(courses)
    G = nx.DiGraph()
    G.add_nodes_from(courses["course_id"])
    sets = {r.course_id: set(r.skills) - generic for r in courses.itertuples()}
    known = courses.dropna(subset=["level_num"])
    for _, grp in known.groupby("subdomain"):
        ids = grp["course_id"].tolist()
        lv = dict(zip(grp["course_id"], grp["level_num"]))
        for b in ids:
            cands = []
            for a in ids:
                if lv[a] >= lv[b] or not sets[a] or not sets[b]:
                    continue
                j = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
                if j >= min_jaccard:
                    cands.append((j, a))
            for j, a in sorted(cands, reverse=True)[:max_parents]:
                G.add_edge(a, b, weight=round(j, 3))
    return G


class PrereqChecker:
    def __init__(self, G, courses):
        self.G = G
        self.level = dict(zip(courses["course_id"], courses["level_num"]))

    def violates(self, course_id, taken, student_level):
        # True if the course is above the student's level and they took none of its parents
        parents = list(self.G.predecessors(course_id)) if course_id in self.G else []
        if not parents:
            return False
        lvl = self.level.get(course_id)
        if lvl != lvl or lvl is None or lvl <= student_level:
            return False
        return not any(p in taken for p in parents)

    def order(self, course_ids):
        # study order: by level, prerequisites first
        sub = self.G.subgraph(course_ids)
        try:
            topo = {c: i for i, c in enumerate(nx.topological_sort(sub))}
        except nx.NetworkXUnfeasible:
            topo = {c: 0 for c in course_ids}
        def lvl(c):
            v = self.level.get(c)
            return 2 if v is None or v != v else v
        return sorted(course_ids, key=lambda c: (lvl(c), topo.get(c, 0)))
