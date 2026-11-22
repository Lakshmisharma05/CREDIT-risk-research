# CREDLENS AI — Data Documentation and Project Methodology

**Project:** Explainable Credit Risk Assessment Using SHAP-Augmented Gradient Boosting on Imbalanced Financial Data  
**Application:** CREDLENS AI  
**Contributor:** Riya — Data, Preprocessing and Application  
**EDA Notebook:** `member1_data.ipynb`

---

## 1. Project Overview

CREDLENS AI is an explainable credit risk assessment platform. It analyzes borrower information, estimates the probability of serious delinquency within two years, assigns risk categories and provides feature-level explanations.

The project combines:
- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Imbalanced classification experiments
- Gradient boosting and probability calibration
- SHAP-based model explanations
- A desktop application for risk analysis and simulation

The model provides decision-support estimates. It does not guarantee future borrower behaviour or replace professional lending decisions.

## 2. Dataset Source and Description

The project uses the **Give Me Some Credit** dataset from the Kaggle competition.

- Main source file: `cs-training.csv`
- Original dataset size: 150,000 borrower records
- Input variables: 10 raw features
- Target variable: `SeriousDlqin2yrs`
- Task: Binary classification of serious delinquency within two years

The original source file is not stored in the repository. Processed datasets and project files are maintained separately.

## 3. Target Variable

`SeriousDlqin2yrs` indicates whether a borrower experienced serious delinquency:

| Value | Meaning |
|---|---|
| 0 | No serious delinquency within two years |
| 1 | Serious delinquency — 90 or more days past due within two years |

The positive class (`1`) is the minority class.

## 4. Feature Dictionary

The dataset contains the following 10 raw input features:

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Proportion of available unsecured credit being used |
| `age` | Borrower's age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Number of times 30–59 days past due |
| `DebtRatio` | Debt-to-income ratio recorded in the dataset |
| `MonthlyIncome` | Borrower's monthly income |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Number of times 90 or more days late |
| `NumberRealEstateLoansOrLines` | Number of real-estate loans or lines |
| `NumberOfTime60-89DaysPastDueNotWorse` | Number of times 60–89 days past due |
| `NumberOfDependents` | Number of dependents |

The final prediction pipeline also creates two derived features:
- `TOTAL_PAST_DUE`: Sum of the three past-due count features after special-code handling.
- `INCOME_MISSING`: Binary indicator showing whether monthly income is missing.

The final model therefore uses 12 features.

## 5. Class Imbalance

The original dataset has the following target distribution:

| Class | Records | Percentage |
|---|---:|---:|
| 0 — No serious delinquency | 139,974 | 93.32% |
| 1 — Serious delinquency | 10,026 | 6.68% |
| **Total** | **150,000** | **100%** |

The minority-to-majority distribution is approximately 1:14.

A model that predicts only the majority class could achieve high accuracy while failing to identify defaulters. Therefore, accuracy alone is not sufficient.

The project considers metrics such as:
- Precision and recall
- F1-score
- ROC-AUC and PR-AUC
- Balanced accuracy
- KS statistic
- Brier score for probability quality

The class distribution is illustrated in `figures/class_distribution.png`.

## 6. Missing Values and Data Quality

Initial EDA identified missing values in two columns:

| Feature | Missing records | Percentage |
|---|---:|---:|
| `MonthlyIncome` | 29,731 | 19.82% |
| `NumberOfDependents` | 3,924 | 2.62% |

No missing values were reported in the other raw columns during this initial check. No exact duplicate rows were found in that check.

The missing-value distribution is shown in `figures/missing_values.png`.

## 7. Initial EDA: Outliers and Observations

Initial EDA highlighted several unusual values:

- `age` ranged from 0 to 109.
- Past-due count columns contained special values such as 96 and 98.
- `RevolvingUtilizationOfUnsecuredLines` had an extreme maximum of 50,708.
- `DebtRatio` had a highly skewed distribution, with a median around 0.37 and a much larger maximum.
- `MonthlyIncome` also contained extreme values.

The initial outlier analysis is illustrated in `figures/outliers_before_after.png`.

These observations informed the data-cleaning and model-preprocessing decisions described below.

## 8. EDA Cleaning and Processed Dataset

The initial EDA workflow used a stratified 80/20 train-test split with `random_state=42`.

The documented EDA cleaning steps included:

1. **Missing values:** Median imputation based on training data.
2. **Invalid ages:** Records with age below 18 or above 100 were removed in this EDA workflow. Fourteen records were removed in total.
3. **Past-due counts:** Values were capped at 20 in the EDA-cleaned datasets. The 96/98 values were investigated separately because their meaning was not explicitly established by the dataset documentation.
4. **Extreme values:** Utilization, debt ratio and monthly income were capped using training-set 99th-percentile values, then applied to the test set.
5. **Scaling:** The saved EDA datasets were not scaled. Scaling, where required, belongs inside the relevant modeling pipeline.

The EDA documentation records 119,992 training rows and 29,994 test rows after cleaning. These counts belong to the EDA preprocessing workflow and should not be confused with the separate final model split.

## 9. Final Model Preprocessing and Feature Engineering

The final modeling and prediction pipeline uses the shared preprocessing function `prepare_raw()` in:

`modeling/credlens_package/credlens_predictor.py`

This function is used during both training and prediction to keep preprocessing consistent.

Its documented operations are:

1. Convert available raw input columns to numeric values.
2. Treat values greater than or equal to 90 in the three past-due columns as missing special codes.
3. Clip `RevolvingUtilizationOfUnsecuredLines` at an upper limit of 10.
4. Clip `age` to the range 18–100.
5. Create `TOTAL_PAST_DUE` by summing the three past-due columns after missing-value handling.
6. Create `INCOME_MISSING` to indicate missing monthly income.
7. Apply median imputation inside the model pipeline.

**Important distinction:** The EDA workflow's documented capping of past-due counts at 20 is not the same as the final model pipeline's handling of values greater than or equal to 90 as missing. These describe different processing stages and should not be treated as one identical operation.

The final model feature order is stored in `modeling/credlens_package/models/model_config.json`.

## 10. Model Development Methodology

The modeling experiments compared three classifier families across four imbalance-handling approaches.

| Classifier | Imbalance approaches |
|---|---|
| Logistic Regression | None, class weighting, SMOTE, undersampling |
| Random Forest | None, class weighting, SMOTE, undersampling |
| HistGradientBoosting | None, class weighting, SMOTE, undersampling |

This produced 12 experiment combinations.

The modeling workflow used:
- Stratified 80/20 train-test split with seed 42.
- Five-fold cross-validation for model comparisons.
- SMOTE and undersampling within training folds.
- PR-AUC as the primary model-selection metric.
- A 30-trial random search for hyperparameter tuning.

The selected model was **HistGradientBoosting with class weighting**.

### Selected hyperparameters

| Parameter | Value |
|---|---:|
| `max_iter` | 250 |
| `learning_rate` | 0.160026 |
| `max_depth` | 4 |
| `max_leaf_nodes` | 15 |
| `min_samples_leaf` | 100 |
| `l2_regularization` | 2.512779 |

The model uses median imputation as part of its preprocessing pipeline.

## 11. Final Model Evaluation

### 11.1 Comparison of imbalance approaches

The following results are the reported final test results for the four HistGradientBoosting variants. These are the uncalibrated variant-comparison results.

| Variant | ROC-AUC | PR-AUC | F1 | Recall | Precision | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| None | 0.8628 | 0.3761 | 0.4085 | 0.4314 | 0.3879 | 0.24 |
| Class weighting | 0.8620 | 0.3827 | 0.4214 | 0.4813 | 0.3748 | 0.77 |
| SMOTE | 0.8463 | 0.3589 | 0.3834 | 0.4389 | 0.3404 | 0.25 |
| Undersampling | 0.8506 | 0.3326 | 0.3831 | 0.4190 | 0.3529 | 0.85 |

Class weighting achieved the highest PR-AUC among these four reported variants.

### 11.2 Calibrated deployed model

The deployed model configuration identifies version `1.0.0` and uses Platt (sigmoid) calibration.

The calibrated holdout test metrics recorded in `model_config.json` are:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.8620 |
| PR-AUC | 0.3827 |
| Recall | 0.4888 |
| F1-score | 0.4261 |
| Precision | 0.3776 |
| Balanced accuracy | 0.7155 |
| Brier score | 0.0503 |
| Decision threshold | 0.21 |

The model configuration records 23,953 training rows and 5,989 test rows for this modeling split, with default rates of approximately 6.69% and 6.70%, respectively.

**Threshold clarification:** The class-weight comparison table reports an uncalibrated threshold of 0.77. The deployed calibrated model uses a threshold of 0.21. These thresholds belong to different probability scales and must not be interchanged.

## 12. Probability Calibration and Risk Categories

The training workflow generates out-of-fold training probabilities and fits a Logistic Regression calibration model using their logit-transformed values. This is Platt (sigmoid) calibration.

The decision threshold is selected using the F1-max rule on calibrated training out-of-fold probabilities. The holdout test set is reserved for final evaluation.

The deployed model uses the following risk bands:

| Risk band | Calibrated probability |
|---|---|
| Low | Below 0.105 |
| Medium | 0.105 to below 0.21 |
| High | 0.21 or above |

The decision threshold is 0.21. The risk-band boundaries are stored in `model_config.json`.

## 13. Explainability and SHAP

CREDLENS AI includes an Explainability Lab that presents SHAP-based feature contributions for a borrower prediction.

The application explains contributions to the calibrated default probability:
- Positive SHAP contributions increase the model's predicted default probability.
- Negative SHAP contributions decrease the model's predicted default probability.

SHAP explanations describe how model features contribute to an individual prediction. They do not establish causation and should not be interpreted as proof of why a borrower will or will not default.

## 14. CREDLENS AI Application Modules

The desktop application contains seven modules:

| Module | Purpose |
|---|---|
| Dashboard | Project overview and model status |
| Data Intelligence | Dataset overview, data quality, missing values and feature analysis |
| Risk Analysis | Historical default distribution and model-predicted risk categories |
| Explainability Lab | SHAP-based explanations of calibrated default probability |
| Simulation Lab | Interactive borrower risk simulation using model predictions |
| Fairness & Reliability | Calibrated model metrics and exploratory age-group comparisons |
| Report Studio | Project summary and overview of the analysis sections |

The Simulation Lab accepts monthly income, debt ratio and credit utilization as user inputs. The remaining seven raw features are filled using dataset medians, as documented in the application guide.

The Fairness & Reliability module includes exploratory age-group comparisons of actual defaults and predicted risk. These comparisons do not constitute a formal fairness certification.

Report Studio summarizes the data, risk analysis, explainability, simulation, fairness and reliability sections.

## 15. Limitations and Responsible Interpretation

- The dataset represents historical borrower information and may not reflect every current lending environment.
- Predictive performance does not guarantee future outcomes.
- Class imbalance makes evaluation using accuracy alone unsuitable.
- Special values in past-due columns are treated as missing by the final predictor; their original meaning is not assumed.
- SHAP values indicate model contributions, not causal relationships.
- Age-group comparisons are exploratory and do not certify that the model is fair.
- Simulation results depend on user inputs and median-filled values for the other features.
- Model predictions are decision-support estimates, not guaranteed outcomes or automated lending decisions.

## 16. Repository Structure and Reproducibility

Relevant project files include:

```text
data/
    train.csv
    test.csv

figures/
    class_distribution.png
    missing_values.png
    outliers_before_after.png

member1_data.ipynb
DATA_DOCUMENTATION.md
USER_GUIDE.md

modeling/
    credlens_package/
        credlens_predictor.py
        train_model.py
        models/
            credlens_model.joblib
            credlens_calibrator.joblib
            model_config.json
    results/
        cv_results.csv
        final_test_results.csv
        customer_predictions.csv
    docs/
        HANDOVER_Member3.md
    reports/
        Detailed_Modeling_Report.pdf
        Final_Results_Report.pdf
        CREDLENS_Complete_Project_Report.pdf

app/
    main.py

The EDA notebook documents the initial data exploration and preprocessing workflow. The modeling package contains the final prediction preprocessing, model, calibrator and configuration. The application uses these project components to present borrower risk analysis and explanations.

The model configuration records random_seed = 42, the feature order, model version, decision threshold and evaluation metrics.

Disclaimer: CREDLENS AI is an academic credit-risk analysis project. Its outputs are estimates for educational and decision-support purposes only, not financial advice or guaranteed predictions.
