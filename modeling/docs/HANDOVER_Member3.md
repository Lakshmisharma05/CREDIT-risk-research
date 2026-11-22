# Handover to Member 3: Models and Methodology

## Files
- `models/final_model.joblib`: HistGradientBoosting + class_weight, full train data pe fitted (preprocessing pipeline ke andar)
- `models/HistGradientBoosting_<technique>.joblib`: four variants (none, class_weight, SMOTE, undersampling)
- `models/thresholds.json`: har variant ka threshold (train OOF F1)
- `features.json`: input columns ka exact order; `data_split/train.csv`, `data_split/test.csv`
- `cv_results.csv`, `best_params.json`, `tuning_trials.csv`, `final_test_results.csv`
- `Detailed_Modeling_Report.pdf`, `Final_Results_Report.pdf`

## Methodology
1. Data: uploaded CSV (SeriousDlqin2yrs target, 6.7% defaulters). Codes >=90 in past-due columns missing mane gaye; utilization clip 10; age clip 18-100; median imputation (train folds ke andar); TOTAL_PAST_DUE, INCOME_MISSING features.
2. Split 80/20 stratified (seed 42); test sirf final evaluation ke liye.
3. 12 experiments: 3 models x 4 imbalance techniques, 5-fold CV; SMOTE/undersampling sirf training folds mein.
4. Selection: best CV PR-AUC -> HistGradientBoosting + class_weight. Tuning: 30-trial random search.
5. Best params: `{"learning_rate": 0.16002621824387603, "max_depth": 4, "max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 2.512779099948726, "max_iter": 250}`

## Final test results
| Variant | ROC-AUC | PR-AUC | F1 | Recall | Precision | Threshold |
|---|---|---|---|---|---|---|
| none | 0.8628 | 0.3761 | 0.4085 | 0.4314 | 0.3879 | 0.24 |
| class_weight | 0.8620 | 0.3827 | 0.4214 | 0.4813 | 0.3748 | 0.77 |
| SMOTE | 0.8463 | 0.3589 | 0.3834 | 0.4389 | 0.3404 | 0.25 |
| undersampling | 0.8506 | 0.3326 | 0.3831 | 0.4190 | 0.3529 | 0.85 |

## Member 3 ke liye
- SHAP: `shap.TreeExplainer(pipe.named_steps["clf"])` ko `pipe.named_steps["pre"].transform(X)` ke output pe chalao (HistGradientBoosting ke liye shap version check karo; slow ho to KernelExplainer/sample use karo).
- Resampled variants ki probabilities miscalibrated ho sakti hain; calibration check karo.
- Note: libraries sklearn-based hain; paper mein XGBoost/LightGBM chahiye to `run_all.py` apne laptop pe chalao.
