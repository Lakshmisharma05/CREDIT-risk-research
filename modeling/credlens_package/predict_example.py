"""
Saved models se prediction.

Chalane ka tarika (is file ko results/ folder mein rakho, jahan models/ aur features.json hain):
    python predict_example.py                      # data_split/test.csv pe demo + metrics
    python predict_example.py --raw naya_data.csv  # naye RAW CSV (original columns) pe predict
Output: predictions.csv (probability + har variant ka default flag)
"""
import argparse, json
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score

MODEL = "HistGradientBoosting"
TECHS = ["none", "class_weight", "SMOTE", "undersampling"]


def prepare_raw(df):
    """Raw CSV -> training jaisi features (cleaning training ke jaisi hi honi chahiye)."""
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]).copy()
    late = [c for c in df.columns if c.startswith("NumberOfTime")]
    for c in late:
        df.loc[df[c] >= 90, c] = np.nan            # 96/98 special codes -> missing
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10)
    df["age"] = df["age"].clip(18, 100)
    df["TOTAL_PAST_DUE"] = df[late].fillna(0).sum(axis=1)
    df["INCOME_MISSING"] = df["MonthlyIncome"].isna().astype(int)
    return df


def main(raw_path=None):
    feats = json.load(open("features.json"))
    thr = json.load(open("models/thresholds.json"))
    if raw_path:
        df = prepare_raw(pd.read_csv(raw_path))
    else:
        df = pd.read_csv("data_split/test.csv")           # already prepared
    X = df[feats]
    y = df["TARGET"] if "TARGET" in df else (df["SeriousDlqin2yrs"] if "SeriousDlqin2yrs" in df else None)

    out = pd.DataFrame(index=df.index)
    for t in TECHS:
        pipe = joblib.load(f"models/{MODEL}_{t}.joblib")
        p = pipe.predict_proba(X)[:, 1]
        out[f"prob_{t}"] = p.round(4)
        out[f"default_{t}"] = (p >= thr[t]).astype(int)
        line = f"{t:14s} threshold={thr[t]:.2f} flagged={int((p >= thr[t]).sum())}/{len(X)}"
        if y is not None and y.nunique() > 1:
            pred = (p >= thr[t]).astype(int)
            line += (f" | AUC={roc_auc_score(y, p):.4f} PR-AUC={average_precision_score(y, p):.4f}"
                     f" Recall={recall_score(y, pred):.4f} Precision={precision_score(y, pred, zero_division=0):.4f}")
        print(line)
    out.to_csv("predictions.csv", index=False)
    print("\nSaved predictions.csv\n", out.head())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="raw CSV path (original columns)")
    main(ap.parse_args().raw)
