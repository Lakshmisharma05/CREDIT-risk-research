"""
CREDLENS AI: preprocessing + prediction module (desktop integration ke liye)

Usage (Python):
    from credlens_predictor import CredLensPredictor
    p = CredLensPredictor()                       # models/ folder isi file ke saath hona chahiye
    p.predict_one({...10 raw fields...})          # -> dict (class + probability)
    p.predict([{...}, {...}])                     # -> list[dict]
    p.predict_df(df)                              # -> DataFrame

CLI:
    python credlens_predictor.py --json sample_input.json
    python credlens_predictor.py --csv raw_data.csv --out predictions.csv

Ye module hi poora preprocessing karta hai (cleaning + feature engineering + column order).
Training mein bhi yahi prepare_raw() use hota hai, isliye train/serve mismatch nahi hota.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))

RAW_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse", "DebtRatio",
    "MonthlyIncome", "NumberOfOpenCreditLinesAndLoans", "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfDependents"]
DERIVED_COLUMNS = ["TOTAL_PAST_DUE", "INCOME_MISSING"]
FEATURES = RAW_COLUMNS + DERIVED_COLUMNS            # model ko isi order mein chahiye
LATE_COLUMNS = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
                "NumberOfTime60-89DaysPastDueNotWorse"]


def prepare_raw(df):
    """Raw columns -> cleaned + engineered features (training aur prediction dono mein same)."""
    df = df.copy()
    for c in RAW_COLUMNS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in LATE_COLUMNS:
        df.loc[df[c] >= 90, c] = np.nan                       # 96/98 = special codes -> missing
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=10)
    df["age"] = df["age"].clip(18, 100)
    df["TOTAL_PAST_DUE"] = df[LATE_COLUMNS].fillna(0).sum(axis=1)
    df["INCOME_MISSING"] = df["MonthlyIncome"].isna().astype(int)
    return df                                                  # median imputation model-pipeline ke andar hai


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def validate(df):
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


class CredLensPredictor:
    def __init__(self, model_dir=None):
        d = model_dir or os.path.join(HERE, "models")
        self.cfg = json.load(open(os.path.join(d, "model_config.json")))
        self.model = joblib.load(os.path.join(d, "credlens_model.joblib"))
        self.calibrator = joblib.load(os.path.join(d, "credlens_calibrator.joblib"))
        self.threshold = self.cfg["decision_threshold"]
        self.bands = self.cfg["risk_bands"]
        self.version = self.cfg["model_version"]

    def _band(self, p):
        if p >= self.bands["high_min"]: return "High"
        if p >= self.bands["medium_min"]: return "Medium"
        return "Low"

    def predict_df(self, df):
        validate(df)
        X = prepare_raw(df)[FEATURES]
        raw = self.model.predict_proba(X)[:, 1]
        prob = self.calibrator.predict_proba(to_logit(raw))[:, 1]      # Platt calibration
        out = pd.DataFrame({
            "class": (prob >= self.threshold).astype(int),
            "label": np.where(prob >= self.threshold, "Likely Default", "Unlikely Default"),
            "probability": np.round(prob, 4),
            "risk_band": [self._band(p) for p in prob],
            "threshold": round(self.threshold, 4),
            "model_version": self.version}, index=df.index)
        return out

    def predict(self, records):
        return self.predict_df(pd.DataFrame(records)).to_dict(orient="records")

    def predict_one(self, record):
        return self.predict([record])[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="single record JSON file")
    ap.add_argument("--csv", help="raw CSV file")
    ap.add_argument("--out", default="predictions.csv")
    a = ap.parse_args()
    p = CredLensPredictor()
    if a.json:
        print(json.dumps(p.predict_one(json.load(open(a.json))), indent=2))
    elif a.csv:
        df = pd.read_csv(a.csv)
        res = p.predict_df(df); res.to_csv(a.out, index=False)
        print(f"Saved {a.out} ({len(res)} rows)"); print(res.head())
    else:
        ap.error("--json ya --csv do")
