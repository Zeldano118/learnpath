# Data

## raw/ (not in the repo)

`Coursera.csv` from the Kaggle dataset "Coursera Courses Dataset 2021" by Khushee Kapoor: 3,522 rows, 7 columns (course name, university, difficulty level, rating, URL, description, skills). Download it from Kaggle and place it at `data/raw/Coursera.csv`. It is left out of the repo because the data was scraped from Coursera and its license is not stated.

## processed/ (not in the repo)

- `courses.csv`: the cleaned catalog (3,422 courses), made by notebook 01. It comes from the same Kaggle file, so it's left out too.
- `emb_sbert_*`, `emb_openai_*`: course and student vectors, made by notebook 03.

## synthetic/

Made by notebook 02 with seed 42. None of it describes real people.

- `students.csv`: 2,000 students. `true_interests` and `true_level` are hidden from all models; they only drive the simulation.
- `interactions.csv`: about 18,000 enrollments with time spent, progress, completion, optional rating, and the train/val/test split.
- `cold_start_students.csv`, `cold_start_truth.csv`: 200 new students with no history (100 for tuning, 100 for testing) and the first courses they went on to take.
