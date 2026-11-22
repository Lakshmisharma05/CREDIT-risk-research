# Member 3 – SHAP Explainability and Model Analysis

## 1. Project Overview
This contribution focuses on model evaluation, explainability, and comparison for the Credit Risk Assessment project.

## 2. Model Used
The **HistGradientBoostingClassifier** from Scikit-learn was used for credit risk prediction and SHAP-based explainability.

Balanced sample weights were applied during model training to address class imbalance.

## 3. Model Evaluation

The model was evaluated on the test dataset containing 29,994 records.

- Accuracy: 0.80
- ROC-AUC: 0.8694
- PR-AUC: 0.4046
- Recall (Class 1): 0.79
- Precision (Class 1): 0.22
- F1-score (Class 1): 0.34

## 4. SHAP Analysis

Permutation-based SHAP explanations were generated to understand the contribution of individual features to model predictions.

The following visualizations were generated:

- SHAP Summary Plot
- SHAP Feature Importance Plot
- SHAP Waterfall Plot
- Three SHAP Dependence Plots

SHAP analysis was performed on a sample of 500 test records.

## 5. Model Comparison

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| HistGradientBoostingClassifier | 0.8694 | 0.4046 |
| XGBoost (Calibrated) | 0.8675 | 0.4028 |

## 6. Conclusion

HistGradientBoostingClassifier achieved a ROC-AUC of 0.8694 and PR-AUC of 0.4046.

It showed slightly higher values than the calibrated XGBoost model in the available comparison results.

SHAP was used to interpret feature contributions and improve the understanding of model predictions.

The model achieved a recall of 0.79 for the high-risk class, helping identify a substantial proportion of potential defaulters. However, its precision of 0.22 indicates that some predicted high-risk cases were not actual defaulters.