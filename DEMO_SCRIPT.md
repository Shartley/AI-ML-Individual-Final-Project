# Recorded demo and paper review (4–6 minutes)

1. **Intro (30s)**: State the research question: can classic supervised classifiers route synthetic student questions to five office categories? Show the dataset and limitations.
2. **Training (45s)**: Show `python build_project.py` then `python evaluate.py`, `results.json`, shared 80/20 split, train-only TF-IDF.
3. **Model comparison (45s)**: Explain keyword baseline, Naive Bayes, Logistic Regression and numerical accuracy/macro-F1 from `results.json`.
4. **Live demo (90s)**: Run `python app.py`; test one academic, one financial-aid, and one housing question. Show changing review threshold and the office recommendation.
5. **Weak case (45s)**: Open `heldout_predictions.csv` and `results.json`; show an actual misclassification, **if one exists**, or use an ambiguous mixed-intent example explicitly labeled a qualitative challenge (not a held-out model error).
6. **Paper review (60s)**: Scroll the 2-page venue-formatted paper and identify Introduction, Related Work, Methodology, Results, Discussion, References and supplemental links page.
7. **Limitations (30s)**: Explain that synthetic near-template duplication inflates performance, predictions are not official directory referrals, and the study included no real student evaluations.

Record your screen and upload to a shareable provider; verify in an incognito window. The demo recording is not yet created.


### New stress-test evidence to show during recording
Run `python add_challenge.py`, open `challenge_results.json` or `challenge_predictions.csv`, and show the three challenge accuracies: 60.0%, 60.0%, and 58.3%. Explain that these questions were hypothetical and independently authored, not taken from students. In the interface test “Where do I appeal a grade that seems wrong?” and discuss any wrong prediction without promising that the same behavior will occur for another software version. Point out that the 100% synthetic holdout scores did not generalize.
