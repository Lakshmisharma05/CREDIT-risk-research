# CREDLENS AI
### Explainable Credit Intelligence & Risk Simulation Software

CREDLENS AI is a desktop-based credit risk analysis application built using Python, PySide6, Machine Learning, and SHAP explainability.

## Features
- Interactive credit risk dashboard
- Dataset intelligence and risk analysis
- Borrower risk prediction using a trained model
- SHAP-based explainability
- Credit risk simulation
- Fairness and reliability analysis
- Project reports and results

## Tech Stack
- Python
- PySide6
- Pandas and NumPy
- Matplotlib
- Scikit-learn
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

### 4. Run the application
```bash
python app/main.py
```

## Required Files
The application requires:
- `cs-training.csv` in the repository root
- Trained model files in `modeling/credlens_package/models/`
- Prediction cache in `modeling/results/customer_predictions.csv`

The dataset is not included in this repository because it is excluded through `.gitignore`. Ensure the required files are shared with the team before running the app.

## Team Testing
Team members can clone the repository, install dependencies, run the application, and report bugs or suggest improvements.

## Disclaimer
This project is developed for academic and research purposes. Predictions are not professional financial advice or actual lending decisions.