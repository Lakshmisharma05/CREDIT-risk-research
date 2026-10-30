"""12 experiments: 3 models x 4 imbalance techniques, 5-fold stratified CV.

Usage:
    python run_experiments.py                       # all 12
    python run_experiments.py --models xgboost      # one model (split across PCs)
    python run_experiments.py --train data/train.csv --out modeling/cv_results.csv

Needs: pandas numpy scikit-learn xgboost lightgbm imbalanced-learn
Only train.csv is used. test.csv is never loaded here.
Results are saved after every experiment and finished ones are skipped on rerun.
"""
import argparse, os, re, time
import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
TARGET = "SeriousDlqin2yrs"
TECHNIQUES = ["none", "class_weight", "smote", "undersample"]
MODELS = ["logreg", "xgboost", "lightgbm"]
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "recall": "recall"}


def make_model(name, technique, spw):
    cw = technique == "class_weight"
    if name == "logreg":
        return LogisticRegression(max_iter=1000, random_state=SEED,
                                  class_weight="balanced" if cw else None)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             scale_pos_weight=spw if cw else 1.0,
                             tree_method="hist", eval_metric="aucpr",
                             n_jobs=1, random_state=SEED)
    if name == "lightgbm":
        return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              class_weight="balanced" if cw else None,
                              n_jobs=1, random_state=SEED, verbose=-1)
    raise ValueError(name)


def make_pipeline(name, technique, spw):
    steps = []
    if name == "logreg":
        steps.append(("scale", StandardScaler()))
    if technique == "smote":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif technique == "undersample":
        steps.append(("sampler", RandomUnderSampler(random_state=SEED)))
    steps.append(("model", make_model(name, technique, spw)))
    return Pipeline(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--out", default="modeling/cv_results.csv")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = ap.parse_args()

    df = pd.read_csv(args.train)
    y = df.pop(TARGET)
    # LightGBM rejects some characters (e.g. hyphens) in feature names
    X = df.rename(columns=lambda c: re.sub(r"[^0-9a-zA-Z_]", "_", c))
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"X={X.shape}, default rate={y.mean():.4f}, scale_pos_weight={spw:.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = pd.read_csv(args.out) if os.path.exists(args.out) else pd.DataFrame()
    rows = done.to_dict("records")
    finished = {(r["model"], r["technique"]) for r in rows}

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name in args.models:
        for tech in TECHNIQUES:
            if (name, tech) in finished:
                print(f"skip {name}/{tech} (already done)")
                continue
            t0 = time.time()
            res = cross_validate(make_pipeline(name, tech, spw), X, y, cv=cv,
                                 scoring=SCORING, n_jobs=-1)
            row = {"model": name, "technique": tech}
            for k in SCORING:
                row[f"{k}_mean"] = round(res[f"test_{k}"].mean(), 4)
                row[f"{k}_std"] = round(res[f"test_{k}"].std(), 4)
            row["seconds"] = round(time.time() - t0)
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(row)

    print(pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
