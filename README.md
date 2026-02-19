# 🫀 Heart Disease Prediction
### Stacking Ensemble · Recursive Feature Elimination · Target Encoding

> Kaggle Playground Series S6E2 — binary classification predicting heart disease presence/absence, evaluated on ROC-AUC.

---

## Pipeline

```
Raw Data → Target Encoding (TE) → Optuna Tuning → RFECV → Stacking Ensemble → Submission
```

| Step | Technique | Detail |
|------|-----------|--------|
| Encoding | Target Encoding | 5-fold cross-fitting to prevent leakage |
| Tuning | Optuna (TPE, 50 trials) | XGBoost hyperparameter search, ROC-AUC objective |
| Feature Selection | RFECV | Tuned XGBoost estimator, optimal subset via CV |
| Modelling | Stacking Ensemble | LR + Random Forest + XGBoost → Logistic meta-learner |
| Evaluation | Stratified 5-Fold CV | ROC-AUC |

---

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook Heart_Disease_Stacking_RFE_TE.ipynb
```

> **Note:** Place `train.csv` and `test.csv` in the working directory before running. Download via `kaggle competitions download -c playground-series-s6e2`.

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
category_encoders
optuna
matplotlib
```

---

## Output

Running the notebook end-to-end produces `submission_stacking_roc.csv` — a file with predicted heart disease probabilities ready for Kaggle submission.

*AI Generated README.md, reviewed before upload*
