
# CREDLENS AI
### Explainable Credit Intelligence & Risk Simulation Software

CREDLENS AI is a desktop-based credit risk analysis application developed using Python, PySide6, Machine Learning, and SHAP explainability.

The application helps explore historical credit data, estimate borrower default risk, understand model predictions, and examine model reliability.

## Features

### 1. Dashboard
- Overview of borrower records and risk distribution.
- Key dataset and model indicators.

### 2. Data Intelligence
- Dataset overview and feature analysis.
- Missing-value analysis.
- Feature distributions, correlation, and outlier analysis.

### 3. Risk Analysis
- Historical actual default distribution.
- Model-predicted Low, Medium, and High risk categories.

### 4. Explainability Lab
- SHAP-based feature contribution analysis.
- Explains factors influencing calibrated default probability.

### 5. Simulation Lab
- Interactive borrower risk simulation.
- Estimates default probability and risk category.
- Remaining borrower features use dataset median values.

### 6. Fairness & Reliability
- Calibrated holdout test performance metrics.
- Exploratory age-group comparison.
- Model reliability information.

### 7. Report Studio
- Overview of project analysis, model, explainability, and evaluation.

## Tech Stack

- Python
- PySide6
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- Joblib
- SHAP

## Project Setup

### 1. Clone the repository

```bash
git clone https://github.com/Lakshmisharma05/CREDIT-risk-research.git
cd CREDIT-risk-research
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv-app
source .venv-app/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the required dataset

Place `cs-training.csv` in the repository root directory.

The dataset is excluded from Git through `.gitignore` and must be shared separately with team members.

### 5. Run the application

```bash
python app/main.py
```

## Model Files

The trained model files are located in:

`modeling/credlens_package/models/`

- `credlens_model.joblib` – Trained machine learning model.
- `credlens_calibrator.joblib` – Probability calibration component.
- `model_config.json` – Model configuration, threshold, risk bands, and evaluation metrics.

## Project Structure

```text
CREDIT-risk-research/
│
├── app/
│   └── main.py
│
├── modeling/
│   ├── credlens_package/
│   │   └── models/
│   └── results/
│
├── requirements.txt
├── README.md
└── cs-training.csv
```

## Important Notes

- The dataset is not included in the repository.
- The application requires the trained model files.
- Prediction cache files may be generated or updated during application use.
- Age-group comparisons are exploratory and do not independently certify model fairness.
- Model predictions are estimates and should not be treated as guaranteed outcomes.

## Disclaimer

This project is developed for academic and research purposes only. It is not professional financial advice and must not be used as the sole basis for actual lending decisions.
