# FraudShield AI

FraudShield AI is a machine learning web application for detecting suspicious financial transactions. It uses a LightGBM classification model with fraud-focused feature engineering and provides an interactive Streamlit dashboard for analytics, model performance review, and real-time transaction scoring.

## Features

- Interactive fraud analytics dashboard
- Real-time single-transaction fraud prediction
- LightGBM model training pipeline
- Feature engineering for transaction amount, balances, account behavior, and payment type
- Model metrics, confusion matrix, ROC curve, and feature importance views
- CSV report generation for scored transactions

## Tech Stack

- Python
- Streamlit
- LightGBM
- scikit-learn
- pandas
- NumPy
- Plotly
- imbalanced-learn

## Project Structure

```text
.
├── pipeline.py                  # End-to-end data, training, and app pipeline
├── requirements.txt             # Python dependencies
├── src/
│   ├── app.py                   # Streamlit application
│   ├── train_model.py           # Model training logic
│   ├── generate_sample_data.py  # Synthetic dataset generator
│   ├── fraud_core.py            # Core fraud logic
│   ├── performance.py           # Performance tracking helpers
│   └── utils.py                 # Shared utilities and chart helpers
├── docs/                        # Project documentation
└── Reports/                     # Academic report templates/assets
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The App

```bash
streamlit run src/app.py
```

## Run The Full Pipeline

```bash
python pipeline.py
```

Useful options:

```bash
python pipeline.py --data-rows 50000
python pipeline.py --skip-data-gen
python pipeline.py --no-app
```

## Data And Model Files

Large generated files are intentionally ignored by Git:

- `data/*.csv`
- `models/*.pkl`
- `outputs/`
- local virtual environments

This keeps the GitHub repository small and resume-friendly. Regenerate the dataset and model artifacts locally using `pipeline.py`.

## Documentation

Project documentation is available in the `docs/` folder, including the high-level design, low-level design, synopsis, and final project report.
