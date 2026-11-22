import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (brier_score_loss, roc_auc_score,
                             average_precision_score, confusion_matrix)

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ---------- 1. Data ----------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
for df in (train, test):
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

TARGET = "SeriousDlqin2yrs"
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET].values

spw = (y_train == 0).sum() / (y_train == 1).sum()

def make_model():
    return xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, eval_metric="aucpr", random_state=42)

# ---------- 2. Before calibration ----------
base = make_model()
base.fit(X_train, y_train)
p_before = base.predict_proba(X_test)[:, 1]

# ---------- 3. After calibration ----------
calib = CalibratedClassifierCV(estimator=make_model(), method="isotonic", cv=5)
calib.fit(X_train, y_train)
p_after = calib.predict_proba(X_test)[:, 1]

print("Brier before:", round(brier_score_loss(y_test, p_before), 4))
print("Brier after :", round(brier_score_loss(y_test, p_after), 4))

# Calibration plot
plt.figure(figsize=(6, 6))
for p, name in [(p_before, "Before"), (p_after, "After (isotonic)")]:
    frac, mean_pred = calibration_curve(y_test, p, n_bins=10)
    plt.plot(mean_pred, frac, marker="o", label=name)
plt.plot([0, 1], [0, 1], "k--", label="Perfect")
plt.xlabel("Predicted probability")
plt.ylabel("Actual default rate")
plt.title("Calibration curve")
plt.legend()
plt.savefig("outputs/figures/calibration.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------- 4. Bootstrap 95% CI ----------
rng = np.random.default_rng(42)
aucs, praucs = [], []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    if y_test[idx].sum() == 0:
        continue
    aucs.append(roc_auc_score(y_test[idx], p_after[idx]))
    praucs.append(average_precision_score(y_test[idx], p_after[idx]))

auc_ci = np.percentile(aucs, [2.5, 97.5]).round(4)
pr_ci = np.percentile(praucs, [2.5, 97.5]).round(4)
print("ROC-AUC 95% CI:", auc_ci)
print("PR-AUC  95% CI:", pr_ci)

# ---------- 5. Business threshold ----------
COST_FN, COST_FP = 5, 1   # FN (default miss) ko FP se 5 guna mehenga maana
best = None
for t in np.arange(0.05, 0.95, 0.01):
    pred = (p_after >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cost = COST_FN * fn + COST_FP * fp
    if best is None or cost < best[1]:
        best = (round(t, 2), cost, tp, fp, fn, tn)
print("Best threshold:", best[0], "| Cost:", best[1])
pd.DataFrame([best], columns=["threshold", "cost", "TP", "FP", "FN", "TN"]).to_csv(
    "outputs/tables/best_threshold.csv", index=False)

# ---------- 6. Age-wise fairness (SEX nahi hai, age group use karo) ----------
explainer = shap.TreeExplainer(base)
sv = explainer.shap_values(X_test)
shap_df = pd.DataFrame(sv, columns=X_test.columns)

groups = pd.cut(X_test["age"], bins=[0, 35, 55, 200],
                labels=["<35", "35-55", "55+"])
rows = []
for g in ["<35", "35-55", "55+"]:
    m = (groups == g).values
    if m.sum() == 0:
        continue
    yt, pp = y_test[m], p_after[m]
    pred = (pp >= best[0]).astype(int)
    rows.append({
        "age_group": g,
        "n": int(m.sum()),
        "actual_default_rate": round(yt.mean(), 4),
        "mean_pred_prob": round(pp.mean(), 4),
        "recall_at_best_thr": round(((pred == 1) & (yt == 1)).sum() / max(yt.sum(), 1), 4),
        "mean_abs_shap_MonthlyIncome": round(np.abs(shap_df.loc[m, "MonthlyIncome"]).mean(), 4),
        "mean_abs_shap_RevolvingUtil": round(np.abs(shap_df.loc[m, "RevolvingUtilizationOfUnsecuredLines"]).mean(), 4),
    })
fair = pd.DataFrame(rows)
fair.to_csv("outputs/tables/fairness_age.csv", index=False)
print(fair)

# ---------- 7. Final summary table ----------
summary = pd.DataFrame([{
    "model": "XGBoost (calibrated, isotonic)",
    "ROC_AUC": round(roc_auc_score(y_test, p_after), 4),
    "ROC_AUC_CI95": f"{auc_ci[0]} - {auc_ci[1]}",
    "PR_AUC": round(average_precision_score(y_test, p_after), 4),
    "PR_AUC_CI95": f"{pr_ci[0]} - {pr_ci[1]}",
    "Brier_before": round(brier_score_loss(y_test, p_before), 4),
    "Brier_after": round(brier_score_loss(y_test, p_after), 4),
    "best_threshold": best[0],
}])
summary.to_csv("outputs/tables/final_summary.csv", index=False)
print(summary.T)
print("\nSab files outputs/ folder me ban gayi hain.")