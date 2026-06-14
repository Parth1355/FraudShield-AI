# High Level Design

## Title
FraudShield AI: High Level Design Document

## Student Details
- Name: `[Your Name]`
- Roll No.: `[Your Roll Number]`
- Guide Name: `[Guide Name]`

## 1. Introduction
This document presents the high level design of FraudShield AI, a financial fraud detection web application. The system is designed to process historical transaction data, train a machine learning model, and provide real-time fraud predictions along with analytics dashboards.

## 2. System Overview
FraudShield AI consists of three major layers:
- Data and model layer
- Application logic layer
- Presentation layer

The solution uses a dataset of financial transactions, generates engineered features, trains a LightGBM classifier, and exposes the results through a Streamlit application.

## 3. Major Modules
### 3.1 Data Preparation Module
- Loads transaction data from CSV.
- Handles missing values and duplicate records.
- Removes administrative fields such as `isFlaggedFraud` when needed.
- Generates engineered features such as transaction frequency and balance discrepancy.

### 3.2 Model Training Module
- Splits data into training and testing datasets.
- Preprocesses numeric and categorical features.
- Trains a LightGBM classifier.
- Stores trained artifacts:
  - `model.pkl`
  - `preprocessor.pkl`
  - `metrics.pkl`

### 3.3 Analytics Module
- Displays fraud distribution and transaction insights.
- Shows class imbalance comparison, category fraud rate, hourly trend, and correlation heatmap.
- Enables report export in CSV format.

### 3.4 Prediction Module
- Accepts transaction inputs from the user.
- Builds a single-row feature set consistent with the training pipeline.
- Generates fraud probability and prediction label.
- Displays risk level and probability visualization.

### 3.5 Model Insights Module
- Displays feature importance.
- Shows confusion matrix and ROC curve.
- Presents classification report and derived metrics.

## 4. Architectural Design
### 4.1 Architecture Style
The system follows a modular, layered architecture:
- Presentation Layer: Streamlit interface
- Business Logic Layer: utility functions and prediction logic
- Data/Model Layer: dataset files, preprocessing pipeline, model artifacts

### 4.2 High Level Components
- User Interface
- Data Loader
- Feature Engineering Engine
- Preprocessing Pipeline
- Model Trainer
- Prediction Engine
- Visualization Engine
- Report Exporter

## 5. System Flow
1. User or developer provides the financial transaction dataset.
2. Training script preprocesses the dataset and creates engineered features.
3. LightGBM model is trained and model artifacts are stored.
4. Streamlit app loads dataset and trained artifacts.
5. User explores analytics, enters new transaction details, or reviews model diagnostics.
6. System returns fraud probability, label, charts, and downloadable reports.

## 6. Input and Output
### Inputs
- Historical transaction CSV dataset
- User-entered transaction details for prediction
- Threshold value for decision boundary

### Outputs
- Fraud prediction label
- Fraud probability and risk level
- Analytics dashboard charts
- Model performance metrics
- Fraud transaction CSV report

## 7. Technology Stack
- Python
- Streamlit
- LightGBM
- scikit-learn
- pandas
- numpy
- Plotly
- imbalanced-learn

## 8. High Level Data Flow
1. CSV data is loaded.
2. Data cleaning and feature engineering are performed.
3. Features are transformed by preprocessing pipeline.
4. Model is trained and metrics are saved.
5. Streamlit app reads artifacts and supports analytics and prediction.

## 9. Security and Limitations
### Security Considerations
- Input is processed locally within the application.
- No direct integration with external payment systems is included.
- The current system is intended for academic and demo use.

### Limitations
- Uses static dataset rather than live transaction feed.
- Does not include user authentication or access control.
- No automated model retraining pipeline is implemented.

## 10. Conclusion
The high level design organizes the fraud detection project into independent modules for data preparation, training, analytics, prediction, and reporting. This makes the system easy to understand, maintain, and extend.
