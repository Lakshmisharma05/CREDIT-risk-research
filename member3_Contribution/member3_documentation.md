
# Member 3 – SHAP Explainability and Model Analysis

## 1. Project Overview
This contribution focuses on model evaluation, explainability
and comparison for the Credit Risk Assessment project.

## 2. Model Used
GradientBoostingClassifier was used for SHAP analysis.

## 3. Model Evaluation
- Accuracy: 0.79
- ROC-AUC: 0.8684
- PR-AUC: 0.4034
- Recall (Class 1): 0.79
- Precision (Class 1): 0.21

## 4. SHAP Analysis
The following visualizations were generated:
- SHAP Summary Plot
- SHAP Feature Importance Plot
- SHAP Waterfall Plot
- Three SHAP Dependence Plots

## 5. Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Gradient Boosting | 0.8684 | 0.4034 |
| XGBoost (Calibrated) | 0.8675 | 0.4028 |

## 6. Conclusion
Both models achieved similar ROC-AUC and PR-AUC scores.
Gradient Boosting showed slightly higher values on these
two metrics in the available results.
SHAP was used to interpret feature contributions and
understand model predictions.
