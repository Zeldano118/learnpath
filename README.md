# LearnPath

A course recommender that combines collaborative filtering, content retrieval, an inferred prerequisite graph and an LLM. I built it to test one question: can an LLM make course recommendations more personal than collaborative filtering alone?

The course catalog is real (3,422 Coursera courses after cleaning). The students are simulated, because there was no student data to work with: 2,000 students with a learning history and 200 new students with none.

## How it works

1. **Retrieve.** Item-based CF and a content model (TF-IDF) each rank courses for the student. The two rankings are merged with reciprocal rank fusion. New students get content mixed with popularity instead of CF.
2. **Filter.** The prerequisite graph pushes down courses more than one level above the student, or whose prerequisites they haven't taken.
3. **Augment.** The top 20 courses go into a prompt with the student's profile and recent courses.
4. **Generate.** The LLM picks 5, puts them in study order and writes a short reason for each.
5. **Check.** Every returned course ID must be in the shortlist. Anything else is replaced from the ranking.

## Results (test split)

Students with a history (2,000):

| System | Recall@5 | Picks above level |
|---|---|---|
| Most popular | 0.007 | 0.0% |
| Content only (TF-IDF) | 0.046 | 9.0% |
| CF only | 0.197 | 10.8% |
| Hybrid, no graph | 0.197 | 9.8% |
| Hybrid + graph | 0.168 | 2.5% |

New students (100): the hybrid reaches 0.103, against 0.017 for popular courses.

LLM rerank on 300 students with a history and 100 new students:

| | Recall@5, history | Recall@5, new | Interest fit, new |
|---|---|---|---|
| Shortlist top 5 (no LLM) | 0.163 | 0.103 | 40% |
| DeepSeek, prompt v2 | 0.108 | 0.107 | 65% |
| GPT-4o-mini, prompt v2 | 0.132* | 0.093* | 55% |

\* GPT-4o-mini returned 238 IDs from outside the shortlist with prompt v2. The check replaced them, and counting only its own picks its recall is 0.100 and 0.070.

In short: the LLM didn't beat the shortlist for students with a history, but for new students DeepSeek matched it while picking courses that fit their interests much better. Full tables are in `results/`.

## Repo layout

```
notebooks/   01 to 06, run in order (03 and 05 need API keys)
src/         the learnpath package the notebooks import
data/        simulated students (the Coursera data isn't included, see data/README.md)
results/     metric tables and the raw LLM answers (results/llm)
```

## Running it

1. Download `Coursera.csv` from Kaggle into `data/raw/` (see `data/README.md`).
2. `pip install -r requirements.txt`
3. Run the notebooks in order. They work in Colab (keys go in Colab Secrets as `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`) or locally with the keys as environment variables.

The whole run cost me about $1.05 in API calls.

## Limitations

- The students are simulated, so the numbers compare the parts under my assumptions, not on real learners.
- Stated interests are topic names, which favors TF-IDF over embeddings.
- Prerequisites are inferred from level and skill overlap, and Coursera's skill tags are noisy.
- I wrote prompt v2 after seeing the v1 results. I ran it once without further tuning, but its numbers are still a bit optimistic.

I used an AI assistant to help write the code. The approach, the model choices and the runs are mine.
