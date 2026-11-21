"""
Task 1: Run all 12 ML experiments -> cv_results.csv

12 experiments = 3 models x 4 imbalance techniques
  Models    : Logistic Regression, XGBoost, LightGBM
  Imbalance : none, class_weight, SMOTE, RandomUnderSampler

Resampling sirf training folds ke andar hoti hai (imblearn Pipeline), isliye CV mein leakage nahi hoti.
Test set (20%) bilkul touch nahi hota; wo final evaluation ke liye alag save hota hai.

Run:
    pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn openpyxl xlrd
    python run_12_experiments.py --data "default of credit card clients.xls"
"""
import argparse, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
SEED, N_FOLDS = 42, 5


def load_prepare(path):
    if path.endswith((".xls", ".xlsx")):
        df = pd.read_excel(path, header=1)
    else:
        df = pd.read_csv(path)
        if "LIMIT_BAL" not in df.columns:
            df = pd.read_csv(path, header=1)
    df.columns = [c.strip().replace(".", "_").upper() for c in df.columns]
    df = df.rename(columns={"DEFAULT_PAYMENT_NEXT_MONTH": "TARGET", "PAY_1": "PAY_0"})
    if "TARGET" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "TARGET"})
    df = df.drop(columns=["ID"], errors="ignore").drop_duplicates()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    pay = [c for c in df.columns if c.startswith("PAY_") and not c.startswith("PAY_AMT")]
    bill = [f"BILL_AMT{i}" for i in range(1, 7)]
    amt = [f"PAY_AMT{i}" for i in range(1, 7)]
    df["AVG_DELAY"] = df[pay].mean(axis=1)
    df["MAX_DELAY"] = df[pay].max(axis=1)
    df["UTILIZATION"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["PAY_RATIO"] = df[amt].sum(axis=1) / (df[bill].sum(axis=1).abs() + 1)
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_model(model_name, technique, spw):
    """Model + imbalance technique ka pipeline banata hai."""
    cw = technique == "class_weight"
    if model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced" if cw else None)
    elif model_name == "XGBoost":
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                colsample_bytree=0.8, eval_metric="auc", random_state=SEED,
                                n_jobs=-1, scale_pos_weight=spw if cw else 1.0)
    else:
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
                                 colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1,
                                 scale_pos_weight=spw if cw else 1.0)
    steps = []
    if technique == "SMOTE":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif technique == "undersampling":
        steps.append(("sampler", RandomUnderSampler(random_state=SEED)))
    if model_name == "LogisticRegression":
        steps.insert(0, ("scaler", StandardScaler()))   # scaling sampler se pehle
    steps.append(("clf", clf))
    return Pipeline(steps)


def main(path):
    df = load_prepare(path)
    X, y = df.drop(columns="TARGET"), df["TARGET"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    os.makedirs("data_split", exist_ok=True)
    X_tr.assign(TARGET=y_tr).to_csv("data_split/train.csv", index=False)
    X_te.assign(TARGET=y_te).to_csv("data_split/test.csv", index=False)
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()

    models = ["LogisticRegression", "XGBoost", "LightGBM"]
    techniques = ["none", "class_weight", "SMOTE", "undersampling"]
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
               "recall": "recall", "precision": "precision", "balanced_acc": "balanced_accuracy"}
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)

    rows, n = [], 0
    for m in models:
        for t in techniques:
            n += 1
            t0 = time.time()
            s = cross_validate(make_model(m, t, spw), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
            row = {"exp_id": n, "model": m, "imbalance_technique": t}
            for k in scoring:
                row[f"{k}_mean"] = s[f"test_{k}"].mean()
                row[f"{k}_std"] = s[f"test_{k}"].std()
            row["time_sec"] = time.time() - t0
            rows.append(row)
            print(f"[{n:2d}/12] {m:19s} + {t:13s} AUC={row['roc_auc_mean']:.4f} "
                  f"PR-AUC={row['pr_auc_mean']:.4f} F1={row['f1_mean']:.4f} Recall={row['recall_mean']:.4f}")

    res = pd.DataFrame(rows).round(4)
    res.to_csv("cv_results.csv", index=False)
    best = res.sort_values(["pr_auc_mean", "roc_auc_mean"], ascending=False).iloc[0]
    print("\nSaved: cv_results.csv (12 rows)")
    print(f"Best (PR-AUC): {best['model']} + {best['imbalance_technique']} "
          f"| PR-AUC={best['pr_auc_mean']} AUC={best['roc_auc_mean']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    main(ap.parse_args().data)
