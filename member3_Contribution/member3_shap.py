
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "SeriousDlqin2yrs"

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# Balanced sample weights
weights = compute_sample_weight("balanced", y_train)

# Hist Gradient Boosting Model
model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_leaf_nodes=15,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights)

# Predictions
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Evaluation
print("MODEL: HistGradientBoostingClassifier")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("PR-AUC:", average_precision_score(y_test, probabilities))

# SHAP explanation using permutation explainer
# HistGradientBoosting is explained through model predictions
background = X_train.sample(
    n=min(100, len(X_train)), random_state=42
)

X_explain = X_test.sample(
    n=min(500, len(X_test)), random_state=42
)

def predict_positive(data):
    data = pd.DataFrame(data, columns=X_train.columns)
    return model.predict_proba(data)[:, 1]

explainer = shap.Explainer(
    predict_positive,
    background,
    algorithm="permutation"
)

shap_values = explainer(
    X_explain,
    max_evals=2 * X_train.shape[1] + 1
)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_explain, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

# SHAP Feature Importance
shap.summary_plot(
    shap_values, X_explain, plot_type="bar", show=False
)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", bbox_inches="tight")
plt.close()

# Waterfall Plot
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("shap_waterfall.png", bbox_inches="tight")
plt.close()

# Top 3 Feature Dependence Plots
importance = np.abs(shap_values.values).mean(axis=0)
top_features = X_explain.columns[
    np.argsort(importance)[-3:]
]

for feature in top_features:
    shap.dependence_plot(
        feature,
        shap_values.values,
        X_explain,
        show=False
    )
    plt.tight_layout()
    plt.savefig(
        f"dependence_{feature}.png",
        bbox_inches="tight"
    )
    plt.close()

print("\nAll SHAP plots saved successfully.")
print("Explained samples:", X_explain.shape)
print("Current folder:", os.getcwd())
