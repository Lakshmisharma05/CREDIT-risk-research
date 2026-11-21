"""
Tasks 4-9 (Task 1 ke baad chalta hai; data_split/ aur cv_results.csv chahiye)

  4. Optuna hyperparameter tuning (30-50 trials)   -> best_params.json, optuna_trials.csv
  5. Retrain selected model on full training data  -> models/final_model.joblib
  6. Train all four imbalance variants (tuned)     -> 4 fitted pipelines
  7. Save all four with Joblib                     -> models/<model>_<technique>.joblib
  8. Update report with final results/interpretation -> Final_Results_Report.pdf
  9. Handover package for Member 3                 -> HANDOVER_Member3.md, features.json,
                                                      predict_example.py, final_test_results.csv

Run:  python tasks_4_to_9.py --trials 40
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, balanced_accuracy_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import lightgbm as lgb
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 42
TECHS = ["none", "class_weight", "SMOTE", "undersampling"]
TL = {"none": "None", "class_weight": "Class weight", "SMOTE": "SMOTE", "undersampling": "Undersampling"}


# ------------------------------------------------------------ model factory
def build(model, tech, spw, params):
    cw = tech == "class_weight"
    if model == "LogisticRegression":
        clf = LogisticRegression(max_iter=3000, class_weight="balanced" if cw else None, **params)
    elif model == "XGBoost":
        clf = xgb.XGBClassifier(eval_metric="auc", random_state=SEED, n_jobs=-1,
                                scale_pos_weight=spw if cw else 1.0, **params)
    else:
        clf = lgb.LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1, subsample_freq=1,
                                 scale_pos_weight=spw if cw else 1.0, **params)
    steps = []
    if model == "LogisticRegression":
        steps.append(("scaler", StandardScaler()))
    if tech == "SMOTE":
        steps.append(("sampler", SMOTE(random_state=SEED)))
    elif tech == "undersampling":
        steps.append(("sampler", RandomUnderSampler(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def suggest(trial, model):
    if model == "LogisticRegression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    p = {"n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
         "subsample": trial.suggest_float("subsample", 0.6, 1.0),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True)}
    if model == "XGBoost":
        p.update(max_depth=trial.suggest_int("max_depth", 3, 8),
                 min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                 gamma=trial.suggest_float("gamma", 0, 1))
    else:
        p.update(num_leaves=trial.suggest_int("num_leaves", 15, 127),
                 min_child_samples=trial.suggest_int("min_child_samples", 10, 100))
    return p


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, pred), "recall": recall_score(y, pred),
            "precision": precision_score(y, pred), "balanced_acc": balanced_accuracy_score(y, pred),
            "ks": float((tpr - fpr).max()), "brier": brier_score_loss(y, p), "threshold": thr}


def best_thr(y, p):
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[np.argmax([f1_score(y, p >= t) for t in ts])])


# --------------------------------------------------------------------- main
def main(trials, cv_csv="cv_results.csv"):
    os.makedirs("models", exist_ok=True)
    os.makedirs("report_assets", exist_ok=True)
    tr, te = pd.read_csv("data_split/train.csv"), pd.read_csv("data_split/test.csv")
    X_tr, y_tr = tr.drop(columns="TARGET"), tr["TARGET"]
    X_te, y_te = te.drop(columns="TARGET"), te["TARGET"]
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    # model selection from Task 1/3 results (best PR-AUC)
    cvr = pd.read_csv(cv_csv)
    sel = cvr.loc[cvr["pr_auc_mean"].idxmax()]
    model, tech0 = sel["model"], sel["imbalance_technique"]
    print(f"Selected from CV: {model} + {tech0} (PR-AUC {sel['pr_auc_mean']:.4f})")

    # ---- Task 4: Optuna
    def objective(trial):
        return cross_val_score(build(model, tech0, spw, suggest(trial, model)), X_tr, y_tr,
                               cv=cv, scoring="average_precision", n_jobs=1).mean()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_params
    json.dump({"model": model, "tuned_on_technique": tech0, "n_trials": trials,
               "best_cv_pr_auc": study.best_value, "best_params": best},
              open("best_params.json", "w"), indent=2)
    tdf = study.trials_dataframe(); tdf.to_csv("optuna_trials.csv", index=False)
    plt.figure(figsize=(6.5, 3.2))
    plt.plot(tdf["number"], tdf["value"], "o", alpha=.5, label="trial")
    plt.plot(tdf["number"], tdf["value"].cummax(), "r-", label="best so far")
    plt.xlabel("Trial"); plt.ylabel("CV PR-AUC"); plt.legend(); plt.title("Optuna optimisation history")
    plt.tight_layout(); plt.savefig("report_assets/optuna_history.png", dpi=200); plt.close()
    print("Best params:", best, "| CV PR-AUC:", round(study.best_value, 4))

    # ---- Task 5: retrain selected model on full training data
    final = build(model, tech0, spw, best).fit(X_tr, y_tr)
    joblib.dump(final, "models/final_model.joblib")

    # ---- Task 6 + 7: four imbalance variants, tuned params, joblib
    rows, curves, thresholds = [], {}, {}
    for t in TECHS:
        pipe = build(model, t, spw, best)
        oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thr = best_thr(y_tr, oof)                       # threshold train OOF se (test leakage nahi)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_te)[:, 1]
        joblib.dump(pipe, f"models/{model}_{t}.joblib")
        thresholds[t] = thr
        rows.append({"model": model, "imbalance_technique": t, **metrics(y_te, p, thr)})
        curves[t] = p
        print(f"{model} + {t}: test PR-AUC {rows[-1]['pr_auc']:.4f} recall {rows[-1]['recall']:.4f}")
    json.dump(thresholds, open("models/thresholds.json", "w"), indent=2)
    json.dump(list(X_tr.columns), open("features.json", "w"), indent=2)
    res = pd.DataFrame(rows)
    res.round(4).to_csv("final_test_results.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for t, p in curves.items():
        f, tp, _ = roc_curve(y_te, p); ax[0].plot(f, tp, label=TL[t])
        pr, rc, _ = precision_recall_curve(y_te, p); ax[1].plot(rc, pr, label=TL[t])
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_title("ROC (test)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR")
    ax[1].set_title("Precision-Recall (test)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("report_assets/test_curves.png", dpi=200); plt.close()
    res.set_index("imbalance_technique")[["roc_auc", "pr_auc", "f1", "recall", "precision"]].plot(
        kind="bar", figsize=(7, 3.4), rot=0, title="Test metrics by imbalance variant")
    plt.tight_layout(); plt.savefig("report_assets/test_bars.png", dpi=200); plt.close()

    # ---- Task 8: Final report
    make_pdf(cvr, sel, study, best, res, y_te, model, tech0, len(X_tr), len(X_te))
    # ---- Task 9: handover
    handover(model, tech0, best, study, res, thresholds)
    print("\nDone: Final_Results_Report.pdf, HANDOVER_Member3.md, models/, final_test_results.csv")


# ------------------------------------------------------------------- report
def _tbl(data, widths, font=8):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, -1), font),
                           ("GRID", (0, 0), (-1, -1), .4, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")])]))
    return t


def make_pdf(cvr, sel, study, best, res, y_te, model, tech0, ntr, nte):
    ss = getSampleStyleSheet()
    B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.5, leading=14, spaceAfter=6)
    doc = SimpleDocTemplate("Final_Results_Report.pdf", pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm, title="Final Results Report")
    W = A4[0] - 3.6*cm
    st = [Paragraph("Final Results Report: Tuning, Final Models and Interpretation", ss["Title"]), Spacer(1, 6)]

    st += [Paragraph("1. Model selection from the 12 CV experiments", ss["Heading1"]),
           Paragraph(f"Sabse high PR-AUC wala experiment <b>{model} + {TL[tech0]}</b> tha "
                     f"(5-fold CV PR-AUC {sel['pr_auc_mean']:.4f}, ROC-AUC {sel['roc_auc_mean']:.4f}, "
                     f"Recall {sel['recall_mean']:.4f}). Isi model family ko tuning ke liye chuna gaya.", B)]
    top = cvr.sort_values("pr_auc_mean", ascending=False).head(5)
    rows = [["Rank", "Model", "Technique", "PR-AUC", "ROC-AUC", "Recall"]]
    for i, (_, r) in enumerate(top.iterrows(), 1):
        rows.append([i, r["model"], TL.get(r["imbalance_technique"], r["imbalance_technique"]),
                     f"{r['pr_auc_mean']:.4f}", f"{r['roc_auc_mean']:.4f}", f"{r['recall_mean']:.4f}"])
    st += [_tbl(rows, [1.2*cm, 4*cm, 3.2*cm, 2.8*cm, 2.8*cm, 2.8*cm]), Spacer(1, 6)]

    gain = study.best_value - sel["pr_auc_mean"]
    st += [Paragraph("2. Hyperparameter tuning (Optuna)", ss["Heading1"]),
           Paragraph(f"{len(study.trials)} trials (TPE sampler, seed {SEED}), objective = 5-fold CV PR-AUC on the "
                     f"training set. Best CV PR-AUC <b>{study.best_value:.4f}</b> versus {sel['pr_auc_mean']:.4f} "
                     f"untuned (change {gain:+.4f}).", B)]
    rows = [["Parameter", "Best value"]] + [[k, f"{v:.5g}" if isinstance(v, float) else str(v)] for k, v in best.items()]
    st += [_tbl(rows, [6*cm, 5*cm]), Spacer(1, 6), Image("report_assets/optuna_history.png", W, W * 3.2 / 6.5), PageBreak()]

    st += [Paragraph("3. Final model and the four imbalance variants", ss["Heading1"]),
           Paragraph(f"Selected model ko poore training data ({ntr} rows) pe retrain kiya gaya (models/final_model.joblib). "
                     f"Phir wahi tuned parameters ke saath chaaron imbalance variants train kiye gaye aur hold-out "
                     f"test set ({nte} rows, {y_te.mean():.1%} defaulters) pe evaluate kiye gaye. Classification "
                     f"threshold training out-of-fold predictions se F1 maximise karke chuna gaya, test set se nahi.", B)]
    rows = [["Variant", "ROC-AUC", "PR-AUC", "F1", "Recall", "Precision", "BalAcc", "KS", "Thr"]]
    for _, r in res.iterrows():
        rows.append([TL[r["imbalance_technique"]]] + [f"{r[k]:.4f}" for k in
                    ["roc_auc", "pr_auc", "f1", "recall", "precision", "balanced_acc", "ks"]] + [f"{r['threshold']:.2f}"])
    st += [_tbl(rows, [3*cm] + [(W - 3*cm) / 8] * 8, 7.5), Spacer(1, 8),
           Image("report_assets/test_curves.png", W, W * 3.8 / 9), Spacer(1, 6),
           Image("report_assets/test_bars.png", W * .85, W * .85 * 3.4 / 7), PageBreak()]

    b_pr, b_rec, b_f1 = (res.loc[res[k].idxmax()] for k in ["pr_auc", "recall", "f1"])
    ref = res[res["imbalance_technique"] == tech0].iloc[0]
    st += [Paragraph("4. Interpretation", ss["Heading1"]),
           Paragraph(f"<b>Ranking quality:</b> test PR-AUC sabse zyada <b>{TL[b_pr['imbalance_technique']]}</b> variant ka hai "
                     f"({b_pr['pr_auc']:.4f}). Tuning config ({TL[tech0]}) ka test PR-AUC {ref['pr_auc']:.4f} hai, jabki "
                     f"CV mein {study.best_value:.4f} tha; ye farak generalisation gap dikhata hai.", B),
           Paragraph(f"<b>Defaulter detection:</b> sabse high recall {TL[b_rec['imbalance_technique']]} ka hai "
                     f"({b_rec['recall']:.4f}, precision {b_rec['precision']:.4f}). Best F1: "
                     f"{TL[b_f1['imbalance_technique']]} ({b_f1['f1']:.4f}).", B),
           Paragraph("<b>Trade-off:</b> weighting/resampling variants recall badhate hain lekin precision girate hain. "
                     "Isliye final variant business cost (missed default vs false alarm) ke hisaab se chuna jaye. "
                     "Resampling variants ke Brier scores probabilities ko miscalibrated dikha sakte hain; "
                     "probability ke roop mein use karne se pehle calibration karni chahiye.", B),
           Paragraph("<b>Limitations:</b> ek hi train/test split; tuning ek technique (" + TL[tech0] + ") pe hui aur "
                     "baaki variants pe wahi parameters reuse hue; SHAP/fairness analysis is report ka hissa nahi hai "
                     "(Member 3 ka kaam).", B)]
    doc.build(st)


def handover(model, tech0, best, study, res, thr):
    tbl = "| Variant | ROC-AUC | PR-AUC | F1 | Recall | Precision | Threshold |\n|---|---|---|---|---|---|---|\n"
    for _, r in res.iterrows():
        tbl += (f"| {r['imbalance_technique']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1']:.4f} | "
                f"{r['recall']:.4f} | {r['precision']:.4f} | {r['threshold']:.2f} |\n")
    md = f"""# Handover to Member 3: Models and Methodology

## Files
- `models/final_model.joblib`: selected model ({model} + {tech0}), full train data pe fitted
- `models/{model}_<technique>.joblib`: four variants (none, class_weight, SMOTE, undersampling)
- `models/thresholds.json`: har variant ka decision threshold (train OOF F1 se)
- `features.json`: input columns ka exact order
- `data_split/train.csv`, `data_split/test.csv`: wahi split jo modelling mein use hua
- `cv_results.csv` (12 experiments), `best_params.json`, `optuna_trials.csv`, `final_test_results.csv`
- `Detailed_Modeling_Report.pdf`, `Final_Results_Report.pdf`

## Methodology
1. Data: UCI Default of Credit Card Clients; EDUCATION/MARRIAGE invalid codes merge; 5 engineered features.
2. Split: 80/20 stratified (seed {SEED}); test set sirf final evaluation ke liye.
3. 12 experiments: 3 models x 4 imbalance techniques, 5-fold stratified CV; resampling sirf training folds mein (no leakage).
4. Selection: best CV PR-AUC -> {model} + {tech0}.
5. Optuna ({len(study.trials)} trials, TPE) on CV PR-AUC; best CV PR-AUC {study.best_value:.4f}.
6. Best params: `{json.dumps(best)}`
7. Four variants same tuned params ke saath; threshold train out-of-fold F1 se.

## Final test results
{tbl}
## Member 3 ke liye (SHAP / metrics)
- Tree models (XGBoost/LightGBM) ke liye `shap.TreeExplainer(pipe.named_steps["clf"])` use karo; data ko pipeline ke
  scaler/sampler se nahi guzarna (sampler sirf training mein active hota hai). Logistic Regression ke liye scaled data chahiye.
- Probability reporting se pehle calibration check karo (resampled variants miscalibrated ho sakte hain).
- `python predict_example.py` loading ka example hai.
"""
    open("HANDOVER_Member3.md", "w", encoding="utf-8").write(md)
    ex = f'''import json, joblib, pandas as pd
feats = json.load(open("features.json")); thr = json.load(open("models/thresholds.json"))
test = pd.read_csv("data_split/test.csv")
X, y = test[feats], test["TARGET"]
for t in ["none", "class_weight", "SMOTE", "undersampling"]:
    pipe = joblib.load(f"models/{model}_{{t}}.joblib")
    p = pipe.predict_proba(X)[:, 1]
    print(t, "predicted defaulters:", int((p >= thr[t]).sum()), "of", len(X))
'''
    open("predict_example.py", "w").write(ex)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40, help="Optuna trials (30-50)")
    main(ap.parse_args().trials)
