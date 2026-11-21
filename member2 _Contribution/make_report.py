"""
cv_results.csv  ->  Detailed_Modeling_Report.pdf  (tables + charts + auto interpretation)

Run (run_12_experiments.py chalane ke baad, usi folder mein):
    pip install reportlab matplotlib pandas
    python make_report.py --csv cv_results.csv --out Detailed_Modeling_Report.pdf
Sabhi numbers seedha CSV se aate hain; kuch hardcoded nahi hai.
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak)

METRICS = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1", "F1"), ("recall", "Recall"),
           ("precision", "Precision"), ("balanced_acc", "Bal. Acc")]
TECH_LABEL = {"none": "None", "class_weight": "Class weight", "SMOTE": "SMOTE",
              "undersampling": "Undersampling"}


def charts(df, d):
    paths = []
    piv = df.pivot(index="model", columns="imbalance_technique", values="pr_auc_mean")
    ax = piv.plot(kind="bar", figsize=(7, 3.6), rot=0)
    ax.set_ylabel("PR-AUC (5-fold CV mean)"); ax.set_title("PR-AUC by model and imbalance technique")
    ax.legend(title="Technique", fontsize=7); plt.tight_layout()
    p = f"{d}/c1.png"; plt.savefig(p, dpi=200); plt.close(); paths.append(p)

    plt.figure(figsize=(6.2, 4.4))
    for m, g in df.groupby("model"):
        plt.scatter(g["recall_mean"], g["precision_mean"], s=70, label=m)
        for _, r in g.iterrows():
            plt.annotate(TECH_LABEL.get(r["imbalance_technique"], r["imbalance_technique"]),
                         (r["recall_mean"], r["precision_mean"]), fontsize=6, xytext=(3, 3),
                         textcoords="offset points")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall trade-off (12 experiments)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    p = f"{d}/c2.png"; plt.savefig(p, dpi=200); plt.close(); paths.append(p)

    piv = df.pivot(index="model", columns="imbalance_technique", values="roc_auc_mean")
    plt.figure(figsize=(6.2, 2.8)); plt.imshow(piv.values, cmap="Blues", aspect="auto")
    plt.xticks(range(piv.shape[1]), piv.columns, fontsize=7); plt.yticks(range(piv.shape[0]), piv.index, fontsize=7)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            plt.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    plt.title("ROC-AUC heatmap"); plt.tight_layout()
    p = f"{d}/c3.png"; plt.savefig(p, dpi=200); plt.close(); paths.append(p)
    return paths


def tbl(data, widths=None, font=7):
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), font), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), .4, colors.grey), ("ALIGN", (3, 1), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def main(csv, out):
    df = pd.read_csv(csv)
    os.makedirs("report_assets", exist_ok=True)
    c1, c2, c3 = charts(df, "report_assets")
    ss = getSampleStyleSheet()
    H1, H2 = ss["Heading1"], ss["Heading2"]
    B = ParagraphStyle("B", parent=ss["Normal"], fontSize=9.5, leading=14, spaceAfter=6)
    S = ParagraphStyle("S", parent=B, fontSize=8, textColor=colors.grey)
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8*cm, rightMargin=1.8*cm,
                            topMargin=1.8*cm, bottomMargin=1.8*cm, title="Modeling Report")
    W = A4[0] - 3.6*cm
    st = []

    # ---- derived facts
    best_pr = df.loc[df["pr_auc_mean"].idxmax()]
    best_auc = df.loc[df["roc_auc_mean"].idxmax()]
    best_rec = df.loc[df["recall_mean"].idxmax()]
    best_f1 = df.loc[df["f1_mean"].idxmax()]
    by_tech = df.groupby("imbalance_technique")[[k + "_mean" for k, _ in METRICS]].mean()
    by_model = df.groupby("model")[[k + "_mean" for k, _ in METRICS]].mean()
    lab = lambda r: f"{r['model']} + {TECH_LABEL.get(r['imbalance_technique'], r['imbalance_technique'])}"

    # ---- Title
    st += [Paragraph("Modeling Report: Credit Default Prediction", ss["Title"]),
           Paragraph("Comparison of 12 ML experiments (3 models x 4 imbalance techniques) on the UCI "
                     "Default of Credit Card Clients dataset", ss["Heading3"]), Spacer(1, 8)]

    st += [Paragraph("1. Objective", H1),
           Paragraph("Is report ka maqsad ye tay karna hai ki credit-card default prediction ke liye kaunsa model aur "
                     "kaunsi class-imbalance technique sabse behtar hai. Target variable 'default payment next month' "
                     "hai, aur dataset imbalanced hai (lagbhag 22% defaulters).", B)]

    st += [Paragraph("2. Experimental setup", H1)]
    setup = [["Item", "Detail"],
             ["Dataset", "UCI Default of Credit Card Clients (30,000 clients, 23 features + 5 engineered)"],
             ["Engineered features", "AVG_DELAY, MAX_DELAY, UTILIZATION, PAY_RATIO, BILL_TREND"],
             ["Split", "80% train / 20% hold-out test (stratified, seed 42). Test set CV mein use nahi hua"],
             ["Validation", "5-fold stratified cross-validation on the training set"],
             ["Models", "Logistic Regression, XGBoost, LightGBM (default-style, untuned)"],
             ["Imbalance techniques", "None, Class weight, SMOTE, Random undersampling"],
             ["Leakage control", "SMOTE/undersampling sirf training folds ke andar (imblearn Pipeline)"],
             ["Metrics", "ROC-AUC, PR-AUC, F1, Recall, Precision, Balanced Accuracy (mean and std over folds)"]]
    t = tbl([[Paragraph(str(c), ParagraphStyle("c", fontSize=8, leading=10)) for c in r] for r in setup],
            [4*cm, W - 4*cm], 8)
    st += [t, Spacer(1, 6),
           Paragraph("Note: PR-AUC ko primary metric rakha gaya hai kyunki imbalanced data mein ROC-AUC optimistic "
                     "ho sakta hai. Recall defaulters pakadne ki ability batata hai, jo credit risk mein mehngi "
                     "galti (missed default) se bachata hai.", B), PageBreak()]

    # ---- Results tables
    st += [Paragraph("3. Complete results table", H1),
           Paragraph("Table 1: 5-fold CV mean for all 12 experiments (bold = best in column).", B)]
    hdr = ["#", "Model", "Technique"] + [n for _, n in METRICS]
    rows = [hdr]
    bestcols = {k: df[k + "_mean"].max() for k, _ in METRICS}
    for _, r in df.iterrows():
        rows.append([int(r["exp_id"]), r["model"], TECH_LABEL.get(r["imbalance_technique"], r["imbalance_technique"])] +
                    [f"{r[k + '_mean']:.4f}" for k, _ in METRICS])
    t = tbl(rows, [0.8*cm, 3.2*cm, 2.6*cm] + [(W - 6.6*cm) / 6] * 6)
    ts = []
    for ci, (k, _) in enumerate(METRICS):
        for ri in range(1, len(rows)):
            if abs(float(rows[ri][3 + ci]) - bestcols[k]) < 1e-9:
                ts += [("FONTNAME", (3 + ci, ri), (3 + ci, ri), "Helvetica-Bold"),
                       ("BACKGROUND", (3 + ci, ri), (3 + ci, ri), colors.HexColor("#CDEBC5"))]
    t.setStyle(TableStyle(ts)); st += [t, Spacer(1, 10)]

    st += [Paragraph("Table 2: Fold-to-fold standard deviation (stability).", B)]
    rows = [hdr]
    for _, r in df.iterrows():
        rows.append([int(r["exp_id"]), r["model"], TECH_LABEL.get(r["imbalance_technique"], r["imbalance_technique"])] +
                    [f"{r[k + '_std']:.4f}" for k, _ in METRICS])
    st += [tbl(rows, [0.8*cm, 3.2*cm, 2.6*cm] + [(W - 6.6*cm) / 6] * 6), PageBreak()]

    # ---- Comparison
    st += [Paragraph("4. Comparison of the 12 experiments", H1),
           Image(c1, W, W * 3.6 / 7), Spacer(1, 6),
           Image(c2, W * .85, W * .85 * 4.4 / 6.2), Spacer(1, 6),
           Image(c3, W * .85, W * .85 * 2.8 / 6.2), PageBreak()]

    st += [Paragraph("4.1 Average effect of each imbalance technique", H2)]
    rows = [["Technique"] + [n for _, n in METRICS]]
    for tname, r in by_tech.iterrows():
        rows.append([TECH_LABEL.get(tname, tname)] + [f"{r[k + '_mean']:.4f}" for k, _ in METRICS])
    st += [tbl(rows, [3.2*cm] + [(W - 3.2*cm) / 6] * 6, 8), Spacer(1, 8),
           Paragraph("4.2 Average performance of each model", H2)]
    rows = [["Model"] + [n for _, n in METRICS]]
    for mname, r in by_model.iterrows():
        rows.append([mname] + [f"{r[k + '_mean']:.4f}" for k, _ in METRICS])
    st += [tbl(rows, [3.2*cm] + [(W - 3.2*cm) / 6] * 6, 8), Spacer(1, 10)]

    # ---- Interpretation (auto-generated from data)
    tech_rec = by_tech["recall_mean"].idxmax(); tech_prec = by_tech["precision_mean"].idxmax()
    tech_pr = by_tech["pr_auc_mean"].idxmax()
    model_pr = by_model["pr_auc_mean"].idxmax()
    base = df[df["imbalance_technique"] == "none"]["recall_mean"].mean()
    st += [Paragraph("5. Interpretation", H1),
           Paragraph(f"<b>Best ranking quality (PR-AUC):</b> {lab(best_pr)} (PR-AUC {best_pr['pr_auc_mean']:.4f}, "
                     f"ROC-AUC {best_pr['roc_auc_mean']:.4f}). Highest ROC-AUC: {lab(best_auc)} "
                     f"({best_auc['roc_auc_mean']:.4f}).", B),
           Paragraph(f"<b>Best defaulter detection (Recall):</b> {lab(best_rec)} with recall "
                     f"{best_rec['recall_mean']:.4f} (precision {best_rec['precision_mean']:.4f}). "
                     f"Best F1 balance: {lab(best_f1)} (F1 {best_f1['f1_mean']:.4f}).", B),
           Paragraph(f"<b>Model family:</b> averaged over all techniques, {model_pr} gives the best PR-AUC "
                     f"({by_model.loc[model_pr, 'pr_auc_mean']:.4f}).", B),
           Paragraph(f"<b>Imbalance techniques:</b> on average '{TECH_LABEL.get(tech_rec, tech_rec)}' gives the "
                     f"highest recall ({by_tech.loc[tech_rec, 'recall_mean']:.4f}, versus {base:.4f} with no handling), "
                     f"while '{TECH_LABEL.get(tech_prec, tech_prec)}' keeps precision highest "
                     f"({by_tech.loc[tech_prec, 'precision_mean']:.4f}). Best PR-AUC on average: "
                     f"'{TECH_LABEL.get(tech_pr, tech_pr)}'. Resampling/weighting recall badhata hai lekin "
                     "precision ghatati hai, yani ye ek threshold/cost trade-off hai, free improvement nahi.", B),
           Paragraph("<b>Stability:</b> Table 2 ke std values dekho. Agar do experiments ke mean metrics ka farak "
                     "std se chhota hai, to unhe statistically alag nahi maanna chahiye.", B),
           Paragraph("6. Recommendation and next steps", H1),
           Paragraph(f"Hyperparameter tuning (Optuna, 30-50 trials) ke liye <b>{lab(best_pr)}</b> ko primary "
                     f"candidate rakho (PR-AUC ke basis pe). Agar business ko missed defaults ki cost zyada "
                     f"lagti hai, to {lab(best_rec)} ko bhi tuning shortlist mein rakho. Tuning ke baad final model "
                     "ko full training data pe retrain karke hold-out test set pe ek hi baar evaluate karo.", B),
           Paragraph("Limitations: models untuned hain; sirf ek random split/seed use hua; threshold 0.5 pe F1, "
                     "Recall aur Precision nikle hain (cost-based threshold alag se choose karna chahiye).", S)]
    doc.build(st)
    print("Saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="cv_results.csv")
    ap.add_argument("--out", default="Detailed_Modeling_Report.pdf")
    a = ap.parse_args()
    main(a.csv, a.out)
