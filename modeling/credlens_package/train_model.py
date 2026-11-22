"""
CREDLENS training code: final model + calibrator + model_config.json banata hai.

    python train_model.py --data cs-training.csv --params best_params.json
Output: models/credlens_model.joblib, models/credlens_calibrator.joblib, models/model_config.json

Design: HistGradientBoosting (class_weight=balanced) + median imputer. Class weight se probabilities
inflate hoti hain, isliye train out-of-fold predictions pe Platt (sigmoid) calibration lagayi jati hai
(test set calibration/threshold ke liye use nahi hota). Threshold = calibrated OOF probability pe F1-max.
"""
import argparse, datetime, json, os, platform
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, brier_score_loss, balanced_accuracy_score)
from credlens_predictor import prepare_raw, to_logit, FEATURES

SEED, TARGET = 42, "SeriousDlqin2yrs"


def build(params):
    d = dict(max_iter=250, learning_rate=0.05, max_depth=4); d.update(params)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **d))])


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "recall": recall_score(y, pred),
            "f1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "brier": brier_score_loss(y, p), "threshold": thr}


def main(data, params_path, out="models"):
    os.makedirs(out, exist_ok=True)
    params = json.load(open(params_path))["best_params"]
    df = pd.read_csv(data)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]).drop_duplicates()
    y = df[TARGET]; X = prepare_raw(df.drop(columns=TARGET))[FEATURES]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = build(params).fit(Xtr, ytr)
    oof = cross_val_predict(build(params), Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            method="predict_proba")[:, 1]
    iso = LogisticRegression(C=1e6, max_iter=1000).fit(to_logit(oof), ytr)   # Platt: ranking (AUC/PR-AUC) same rehta hai
    ts = np.linspace(0.02, 0.95, 94)
    thr = float(ts[np.argmax([f1_score(ytr, iso.predict_proba(to_logit(oof))[:, 1] >= t) for t in ts])])

    raw = model.predict_proba(Xte)[:, 1]; cal = iso.predict_proba(to_logit(raw))[:, 1]
    m_cal, m_raw = scores(yte, cal, thr), scores(yte, raw, 0.77)
    print("calibrated test:", {k: round(v, 4) for k, v in m_cal.items()})

    joblib.dump(model, f"{out}/credlens_model.joblib"); joblib.dump(iso, f"{out}/credlens_calibrator.joblib")
    cfg = {"model_version": "1.0.0", "created": datetime.date.today().isoformat(),
           "algorithm": "HistGradientBoostingClassifier (class_weight=balanced) + SimpleImputer(median) + Platt (sigmoid) calibration",
           "hyperparameters": {**{"max_iter": 250, "learning_rate": 0.05, "max_depth": 4}, **params},
           "tuning": "30-trial random search, 3-fold CV PR-AUC", "random_seed": SEED,
           "split": {"test_size": 0.2, "stratified": True, "n_train": len(Xtr), "n_test": len(Xte),
                     "train_default_rate": round(float(ytr.mean()), 4), "test_default_rate": round(float(yte.mean()), 4)},
           "feature_order": FEATURES, "target": TARGET, "positive_class": "1 = serious delinquency within 2 years",
           "decision_threshold": thr, "threshold_rule": "F1-max on calibrated train out-of-fold probabilities",
           "risk_bands": {"medium_min": round(thr / 2, 4), "high_min": round(thr, 4),
                          "rule": "Low < thr/2 <= Medium < thr <= High"},
           "test_metrics_calibrated": {k: round(v, 4) for k, v in m_cal.items()},
           "test_metrics_uncalibrated_raw_score_thr0.77": {k: round(v, 4) for k, v in m_raw.items()},
           "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                           "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}}
    json.dump(cfg, open(f"{out}/model_config.json", "w"), indent=2)
    print("saved to", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--params", default="best_params.json")
    a = ap.parse_args(); main(a.data, a.params)
