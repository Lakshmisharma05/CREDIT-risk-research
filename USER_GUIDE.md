# CREDLENS AI – User Guide

## 1. Introduction

CREDLENS AI is a desktop-based credit risk analysis application developed for academic and research purposes.

It uses a trained machine learning model and SHAP-based explainability to explore borrower default risk.

## 2. Application Modules

### 2.1 Dashboard
Provides an overview of borrower records, risk distribution, and model-related information.

### 2.2 Data Intelligence
Allows users to explore:
- Dataset overview
- Feature information
- Missing values
- Feature distributions
- Correlation analysis
- Outlier analysis

### 2.3 Risk Analysis
Displays:
- Historical actual default distribution
- Model-predicted Low, Medium, and High risk categories

Historical defaults and model predictions are presented separately.

### 2.4 Explainability Lab

This module provides SHAP-based explanations of model predictions.

Steps:
1. Open Explainability Lab.
2. Click GENERATE SHAP EXPLANATION.
3. Wait for the explanation to generate.
4. Review the prediction label, risk category, and default probability.
5. Examine the SHAP feature contribution chart.

Chart interpretation:
- Red bars indicate features increasing predicted default probability.
- Blue bars indicate features decreasing predicted default probability.
- Bar length represents the magnitude of the feature contribution.

Note: The current implementation generates an explanation for a sample borrower from the dataset.

### 2.5 Simulation Lab

This module allows users to simulate borrower risk using three input values.

Required inputs:
- Monthly Income
- Debt Ratio
- Credit Utilization

Steps:
1. Open Simulation Lab.
2. Enter monthly income.
3. Enter debt ratio (example: 0.35).
4. Enter credit utilization (example: 0.40).
5. Click RUN RISK SIMULATION.
6. Review the prediction results.

The application displays:
- Prediction label
- Risk category
- Default probability
- Model version

Important: The remaining seven borrower features are populated using dataset median values. Therefore, the simulation is an illustrative estimate, not a complete borrower assessment.

### 2.6 Fairness & Reliability

This module displays calibrated model evaluation metrics and an exploratory age-group comparison.

Users can examine:
- Model performance metrics
- Decision threshold
- Actual default rates by age group
- Predicted risk distribution by age group

Age-group comparisons are descriptive and do not independently establish or certify model fairness.

### 2.7 Report Studio

Provides an overview of the project's analysis, risk assessment, explainability, simulation, fairness, and reliability components.

## 3. Running the Application

1. Clone the project repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using requirements.txt.
4. Place the required cs-training.csv dataset in the repository root.
5. Ensure the trained model files are available.
6. Run the application using:

```bash
python app/main.py
Refer to README.md for complete installation instructions.
4. Required Files
Dataset:

- cs-training.csv

Model files:
- modeling/credlens_package/models/credlens_model.joblib
- modeling/credlens_package/models/credlens_calibrator.joblib
- modeling/credlens_package/models/model_config.json
5. Troubleshooting
Dataset not found

- Ensure cs-training.csv is placed in the repository root.

Model loading error

- Check that all three model files are present in the models directory.

Missing Python packages
- Activate the project virtual environment.
- Run pip install -r requirements.txt.
Invalid simulation input
- Enter numeric values.
- Ensure monthly income is greater than zero.
- Ensure debt ratio and credit utilization are non-negative.
6. Disclaimer
CREDLENS AI is an academic research project. Predictions are estimates and are not guaranteed outcomes, professional financial advice, or actual lending decisions.
