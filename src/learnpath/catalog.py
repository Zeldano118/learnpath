"""Load and clean the Coursera catalog. Every rule adds a line to a small
report (rule, rows affected, note) so notebook 01 can show what changed."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import config

LEVEL_MAP = {
    "Beginner": "Beginner",
    "Intermediate": "Intermediate",
    # Coursera's middle level for language courses
    "Conversant": "Intermediate",
    "Advanced": "Advanced",
    "Not Calibrated": None,
}
LEVEL_NUM = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}

# The Skills field ends with two Coursera category slugs,
# e.g. "... Machine Learning data-science machine-learning"
DOMAINS = {
    "business", "computer-science", "data-science", "life-sciences",
    "physical-science-and-engineering", "social-sciences", "arts-and-humanities",
    "information-technology", "language-learning", "personal-development",
    "math-and-logic",
}
# these two subdomains have only one course each
SUBDOMAIN_MERGE = {"astronomy": "physics-and-astronomy", "rationality": "philosophy"}

GUIDED_PROJECT_PROVIDER = "Coursera Project Network"


def load_raw(path=config.RAW_CATALOG):
    df = pd.read_csv(path)
    return df.rename(columns={
        "Course Name": "name",
        "University": "provider",
        "Difficulty Level": "level_raw",
        "Course Rating": "rating_raw",
        "Course URL": "url",
        "Course Description": "description",
        "Skills": "skills_raw",
    })


# accented letters got lost in the CSV export, fixed by hand
PROVIDER_FIX = {
    "�cole Polytechnique": "École Polytechnique",
    "�cole Polytechnique F�d�rale de Lausanne": "École Polytechnique Fédérale de Lausanne",
    "�cole des Ponts ParisTech": "École des Ponts ParisTech",
    "Universit� Bocconi": "Università Bocconi",
    "Universitat Aut�noma de Barcelona": "Universitat Autònoma de Barcelona",
    "Ludwig-Maximilians-Universit�t M�nchen (LMU)": "Ludwig-Maximilians-Universität München (LMU)",
    "�cole normale sup�rieure": "École normale supérieure",
    "CentraleSup�lec": "CentraleSupélec",
    "Universidad Nacional Aut�noma de M�xico": "Universidad Nacional Autónoma de México",
    "Technische Universit�t M�nchen (TUM)": "Technische Universität München (TUM)",
    "Funda��o Instituto de Administra��o": "Fundação Instituto de Administração",
    "Institut Mines-T�l�com": "Institut Mines-Télécom",
    "(ISC)�": "(ISC)²",
    "Pontificia Universidad Cat�lica de Chile": "Pontificia Universidad Católica de Chile",
}


def _fix_text(s):
    # The export turned curly quotes, dashes and accents into U+FFFD.
    # "You?ll" gets its apostrophe back, " ? " was a dash, the rest is dropped.
    s = str(s)
    s = re.sub(r"(?<=[A-Za-z])�(?=(?:s|ll|re|ve|t|d|m)\b)", "'", s)
    s = s.replace(" � ", " - ")
    s = s.replace("�", "")
    return re.sub(r"\s+", " ", s).strip()


def _split_skills(raw):
    # skills are split by two spaces, the last chunk also has the 2 slugs
    parts = [p.strip() for p in str(raw).split("  ") if p.strip()]
    domain = subdomain = None
    if parts:
        tokens = parts[-1].split()
        if len(tokens) >= 3 and tokens[-2] in DOMAINS and tokens[-1].islower():
            domain, subdomain = tokens[-2], tokens[-1]
            parts[-1] = " ".join(tokens[:-2])
    skills = []
    for p in parts:
        p = p.lower().strip()
        if p and p not in skills:
            skills.append(p)
    subdomain = SUBDOMAIN_MERGE.get(subdomain, subdomain)
    return skills, domain, subdomain


def clean_catalog(df):
    """Returns (clean_df, report)."""
    report = []
    df = df.copy()
    n0 = len(df)

    # 1. same course page listed twice
    df = df.drop_duplicates(subset="url", keep="first")
    report.append(("Drop duplicate course URLs", n0 - len(df), "kept the first row"))

    # 2. broken text
    n_fixed = int(df["provider"].isin(PROVIDER_FIX).sum())
    df["provider"] = df["provider"].replace(PROVIDER_FIX)
    report.append(("Restore accented provider names", n_fixed, f"{len(PROVIDER_FIX)} universities"))
    n_text = int((df["name"].str.contains("�") | df["description"].str.contains("�")).sum())
    report.append(("Repair broken characters in names/descriptions", n_text,
                   "apostrophes restored, dashes kept, unrecoverable characters dropped"))
    for col in ("name", "description", "provider"):
        df[col] = df[col].map(_fix_text)

    # 3. descriptions that are empty or mostly "????"
    words = df["description"].str.split().str.len()
    q_ratio = df["description"].str.count(r"\?") / df["description"].str.len().clip(lower=1)
    bad = (words < 3) | (q_ratio > 0.3)
    report.append(("Drop unusable descriptions", int(bad.sum()),
                   "fewer than 3 words or mostly '?' characters"))
    df = df[~bad].copy()
    df["short_description"] = df["description"].str.split().str.len() < 15
    report.append(("Flag short descriptions (<15 words)", int(df["short_description"].sum()),
                   "kept; skills carry most of the signal for these"))

    # 4. level
    df["level"] = df["level_raw"].map(LEVEL_MAP)
    report.append(("Map 'Conversant' to Intermediate", int((df["level_raw"] == "Conversant").sum()),
                   "Coursera uses it for language courses"))
    report.append(("Level 'Not Calibrated' set to unknown", int(df["level"].isna().sum()),
                   "kept in catalog, skipped by level filters"))
    df["level_num"] = df["level"].map(LEVEL_NUM)

    # 5. rating
    df["rating"] = pd.to_numeric(df["rating_raw"], errors="coerce")
    df["rating_imputed"] = df["rating"].isna()
    medians = df.groupby("level", dropna=False)["rating"].median()
    overall = df["rating"].median()
    fill = df["level"].map(medians).fillna(overall)
    df["rating"] = df["rating"].fillna(fill)
    report.append(("Impute 'Not Calibrated' ratings", int(df["rating_imputed"].sum()),
                   "median rating of the same level, flagged in rating_imputed"))

    # 6. skills and categories
    parsed = df["skills_raw"].map(_split_skills)
    df["skills"] = parsed.map(lambda t: t[0])
    df["domain"] = parsed.map(lambda t: t[1])
    df["subdomain"] = parsed.map(lambda t: t[2])
    missing_cat = df["subdomain"].isna()
    if missing_cat.any():
        df = _fill_categories(df, missing_cat)
    report.append(("Infer missing category from nearest courses", int(missing_cat.sum()),
                   "5-NN on TF-IDF of name + skills"))

    # 7. format
    df["format"] = np.where(df["provider"] == GUIDED_PROJECT_PROVIDER, "guided_project", "course")

    df = df.reset_index(drop=True)
    df.insert(0, "course_id", [f"C{i:04d}" for i in range(1, len(df) + 1)])
    df["skills_str"] = df["skills"].map("; ".join)
    df["doc"] = df.apply(course_document, axis=1)

    cols = ["course_id", "name", "provider", "level", "level_num", "rating", "rating_imputed",
            "domain", "subdomain", "format", "skills_str", "description", "short_description",
            "url", "doc"]
    return df[cols], report


def _fill_categories(df, missing):
    text = (df["name"] + " " + df["skills"].map(" ".join)).values
    vec = TfidfVectorizer(stop_words="english", min_df=2).fit(text)
    X = vec.transform(text)
    known = np.where(~missing.values)[0]
    unknown = np.where(missing.values)[0]
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(X[known])
    _, idx = nn.kneighbors(X[unknown])
    for row, neigh in zip(unknown, idx):
        votes = df.iloc[known[neigh]]["subdomain"].value_counts()
        sub = sorted(votes[votes == votes.max()].index)[0]   # alphabetical on ties, same on every pandas version
        dom = df.iloc[known[neigh]].loc[lambda d: d["subdomain"] == sub, "domain"].iloc[0]
        df.iat[row, df.columns.get_loc("subdomain")] = sub
        df.iat[row, df.columns.get_loc("domain")] = dom
    return df


def course_document(row, max_desc_words=180):
    # text used for retrieval and embeddings, name and skills first
    desc = " ".join(str(row["description"]).split()[:max_desc_words])
    skills = row["skills_str"] if "skills_str" in row else "; ".join(row["skills"])
    level = row["level"] if isinstance(row["level"], str) else "Unknown level"
    return (f"{row['name']}. Level: {level}. Topic: {row['subdomain'].replace('-', ' ')}. "
            f"Skills: {skills}. {desc}")


def load_courses(path=config.COURSES):
    df = pd.read_csv(path)
    df["skills"] = df["skills_str"].fillna("").map(lambda s: [x for x in s.split("; ") if x])
    return df
