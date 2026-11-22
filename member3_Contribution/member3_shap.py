import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight

# Load the team's datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "SeriousDlqin2yrs"

# Separate features and target
X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

# Train Gradient Boosting with balanced sample weights
weights = compute_sample_weight("balanced", y_train)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=weights)

# Predict probabilities and classes
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("ROC-AUC:", roc_auc_score(y_test, probabilities))
print("PR-AUC (Average Precision):",
      average_precision_score(y_test, probabilities))

# SHAP explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

# Feature importance bar plot
shap.summary_plot(
    shap_values, X_test, plot_type="bar", show=False
)
plt.tight_layout()
plt.savefig("shap_feature_importance.png", bbox_inches="tight")
plt.close()

print("\nSHAP plots saved successfully.")

# Waterfall Plot
if isinstance(shap_values, list):
    shap_values_2d = shap_values[1]
else:
    shap_values_2d = shap_values

base_value = float(np.array(explainer.expected_value).flatten()[-1])

explanation = shap.Explanation(
    values=shap_values_2d[0],
    base_values=base_value,
    data=X_test.iloc[0].values,
    feature_names=X_test.columns.tolist()
)

explanation = shap.Explanation(
    values=shap_values_2d[0],
    base_values=base_value,
    data=X_test.iloc[0].values,
    feature_names=X_test.columns.tolist()
)

shap.plots.waterfall(explanation, show=False)
plt.savefig("shap_waterfall.png", bbox_inches="tight")
plt.close()

# Dependence Plots - Top 3 Features
importance = np.abs(shap_values_2d).mean(axis=0)
top_features = X_test.columns[np.argsort(importance)[-3:]]

for feature in top_features:
    shap.dependence_plot(
        feature,
        shap_values_2d,
        X_test,
        show=False
    )
    plt.savefig(f"dependence_{feature}.png", bbox_inches="tight")
    plt.close()

print("Waterfall and Dependence plots saved successfully.")


print("SHAP VALUES SHAPE:", np.array(shap_values).shape)
print("X_TEST SHAPE:", X_test.shape)
print("Current folder:", os.getcwd())
