"""Logistic Regression baseline (no imbalance handling), 5-fold stratified CV.

Run in VS Code: open this file and press the Run button, or in the terminal:
    python baseline_logreg.py
The "# %%" markers also let you run it cell by cell in the Interactive Window.
Only train.csv is used. test.csv is never loaded.
"""
# %% Imports and settings
import argparse
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET = "SeriousDlqin2yrs"
SCORING = {"PR-AUC": "average_precision", "ROC-AUC": "roc_auc", "Recall": "recall"}

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="data/train.csv")
ap.add_argument("--out", default="modeling/baseline_cv.csv")
args, _ = ap.parse_known_args()  # parse_known_args keeps notebook/Interactive runs working

# %% Load data and sanity checks
df = pd.read_csv(args.train)
y = df.pop(TARGET)
X = df

print("=" * 52)
print(" DATA CHECK")
print("=" * 52)
print(f" Rows                : {X.shape[0]:,}")
print(f" Features            : {X.shape[1]}")
print(f" Default rate        : {y.mean():.2%}")
print(f" Missing values      : {int(X.isna().sum().sum())}")

# %% Cross-validation
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=SEED)),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)

# %% Per-fold and summary tables
folds = pd.DataFrame({m: res[f"test_{m}"] for m in SCORING})
folds.index = [f"Fold {i}" for i in range(1, 6)]

summary = pd.DataFrame({"Mean": folds.mean(), "Std": folds.std()}).round(4)

print()
print("=" * 52)
print(" LOGISTIC REGRESSION BASELINE (5-fold stratified CV)")
print("=" * 52)
print("\n Per-fold scores")
print(folds.round(4).to_string())
print("\n Summary")
print(summary.to_string())
print(f"\n Random-guess PR-AUC (default rate): {y.mean():.4f}")
print(" Recall is measured at the default 0.5 threshold.")

# %% Save results
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
out = summary.reset_index().rename(columns={"index": "Metric"})
out.to_csv(args.out, index=False)
print(f"\n Saved: {args.out}")
