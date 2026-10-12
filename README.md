# CampusHelp AI — Reproducible Research Prototype

This is a **new, reproducible synthetic-data implementation** of a 5-category student helpdesk router. The CSV is generated from 50 deterministic sentence patterns per category and is **not** the user's earlier 250-message checkpoint dataset. No real students or university services were tested. Do not claim otherwise.

## Run

```bash
python -m pip install -r requirements.txt
python build_project.py
python evaluate.py
python app.py
```

Open the localhost URL shown by Gradio (usually http://127.0.0.1:7860). For hosted demos, place the files in a public Hugging Face Space and set the startup file to `app.py`. Do not upload private student data. No verified public demo URL exists yet. During a separate qualitative test, the model incorrectly suggested Financial Aid for "My wifi is not working" (probability 0.323); its low-confidence review flag triggered at the default 0.60 threshold.

## Reproduce

- `build_project.py` deterministically generates 250 labeled messages, 50 per category (`random.Random(42)`).
- `evaluate.py` uses `train_test_split(test_size=0.2, stratify=y, random_state=42)` and identical 50-message test records for three methods.
- TF-IDF features are fitted **only on training data** via pipelines.
- Metrics and errors are saved to `results.json`, predictions to `heldout_predictions.csv`.
- `model.joblib` is a locally trained logistic regression; no pretrained language model is used.
- `app.py` serves the trained model with campus-office suggestions and a user-adjustable human-review threshold. Scores are classifier probabilities, **not calibrated reliability estimates**.

## Major limitations

The data are synthetic and generated from a small number of repeated patterns, so held-out examples closely resemble training examples. Real student communication, mixed requests, ambiguous labels, changing office names, and out-of-scope questions may perform much worse. The office descriptions are generic and are not verified NC A&T directory links. No human-subject study or real-person testing was performed for this package; verify whether a future user study requires IRB review.

## Project deliverables

- `app.py`, `router.py`, `model.joblib`: prototype
- `synthetic_student_messages.csv`, `build_project.py`: data + generator
- `evaluate.py`, `results.json`, `heldout_predictions.csv`: reproducible experiment and weak cases
- `paper.tex`, `references.bib`, `final_submission.pdf`: paper source and PDF
- `DEMO_SCRIPT.md`: walkthrough and paper review plan

Paper venue: ACMSE 2027 Poster Track, 2 pages including references, double-anonymous review, Nov 14, 2026 deadline. IMPORTANT: The ACMSE website requires its own modified `acmart` bundle, which could not be downloaded here. Before conference submission, import this `paper.tex` into the **official ACMSE 2027 template**, replacing the preamble if necessary, then recheck page count and venue constraints. Do not upload the course supplemental link page to a double-anonymous venue.


## Independent stress-test evaluation (completed)
After generating/training the original 250-question project, run `python add_challenge.py`. This writes `challenge_student_messages.csv` (60 independently authored, hypothetical prompts), `challenge_predictions.csv`, and `challenge_results.json`. The script does **not** refit the demo Logistic Regression model; it reconstructs Naive Bayes with the original fixed training split for comparison. The challenge set was never used for training or selection.

Challenge accuracy / macro-F1: keyword 0.600 / 0.6084; Naive Bayes 0.600 / 0.5841; Logistic Regression 0.5833 / 0.5621. The manually assigned primary categories have not been adjudicated by a second annotator; several ambiguous prompts might have multiple sensible destinations. Avoid claiming this challenge set represents the real student population.

The original training and test set reuse 10 topic strings across five templates, making the held-out score optimistic. The independent challenge exposes generalization limitations, not deployment readiness.
