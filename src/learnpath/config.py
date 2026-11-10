"""Settings and paths for the whole project.

Paths are relative to the project folder, so the code runs the same on a laptop
and from Google Drive in Colab. LEARNPATH_DIR overrides the folder.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("LEARNPATH_DIR", Path(__file__).resolve().parents[2]))

DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
DATA_SYNTHETIC = PROJECT_DIR / "data" / "synthetic"
RESULTS = PROJECT_DIR / "results"
FIGURES = RESULTS / "figures"
LLM_DIR = RESULTS / "llm"

RAW_CATALOG = DATA_RAW / "Coursera.csv"
COURSES = DATA_PROCESSED / "courses.csv"
STUDENTS = DATA_SYNTHETIC / "students.csv"
INTERACTIONS = DATA_SYNTHETIC / "interactions.csv"
COLD_START = DATA_SYNTHETIC / "cold_start_students.csv"
COLD_START_TRUTH = DATA_SYNTHETIC / "cold_start_truth.csv"

SEED = 42

# synthetic data
N_STUDENTS = 2000
N_COLD_START = 200     # 100 for tuning, 100 for testing
MIN_HISTORY = 4
MAX_HISTORY = 15

# recommendation
TOP_K = 5              # courses shown to the student
N_CANDIDATES = 50
RERANK_POOL = 20       # courses the LLM gets to choose from
LLM_EVAL_STUDENTS = 300

# models (check names against your own account, notebook 05 does a test call first)
SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_EMBED_DIM = 512
OPENAI_CHAT_MODEL = "gpt-4o-mini"
DEEPSEEK_CHAT_MODEL = "deepseek-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# deepseek-flash "thinks" by default, which made each call ~10x more expensive
# and ~5x slower in the first trial. Off keeps it comparable to gpt-4o-mini.
DEEPSEEK_THINKING = False

# USD per 1M tokens (input, output), only used to report cost
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "deepseek-flash": (0.30, 1.20),
}

PROMPT_VERSION = "v2"
# Goes into the LLM result file names. Change it to start a fresh run
# instead of resuming from saved answers.
LLM_RUN = "run3"
# runs compared in notebook 06 (run1 was the trial with DeepSeek thinking on)
LLM_RUNS = {"run2": "v1", "run3": "v2"}


def ensure_dirs():
    for p in (DATA_PROCESSED, DATA_SYNTHETIC, FIGURES, LLM_DIR):
        p.mkdir(parents=True, exist_ok=True)
