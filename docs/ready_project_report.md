# Savitribai Phule Pune University

## Final Project Report
### On
## FraudShield AI: Financial Fraud Detection Dashboard with ML-Based Transaction Risk Prediction

**Team members:**  
`[Student Name]` (`[Roll No.]`)

**Guided by:** `[Guide Name]`  
**Industry Mentor:** `[If Applicable]`

## Table of Contents
- Executive Summary
- 1. Background
- 1.1 Aim
- 1.2 Technologies
- 1.3 Hardware Architecture
- 1.4 Software Architecture
- 2. System
- 2.1 Requirements
- 2.1.1 Functional requirements
- 2.1.2 User requirements
- 2.1.3 Environmental requirements
- 2.2 Design and Architecture
- 2.3 Implementation
- 2.4 Testing
- 2.5 Graphical User Interface (GUI) Layout
- 2.6 Customer testing
- 2.7 Evaluation
- 3. Snapshots of the Project
- 4. Conclusions
- 5. Further development or research
- 6. References
- 7. Appendix

## Executive Summary

The FraudShield AI system is a web-based application designed to analyze financial transaction data and provide intelligent fraud prediction using machine learning techniques. The project addresses a critical real-world problem where fraudulent transactions are very rare compared to legitimate ones, but the impact of missed fraud can be severe.

This system integrates data analytics, visualization, and predictive modelling into a single dashboard. Users can explore fraud trends, inspect transaction patterns, and predict fraud risk for a manually entered transaction based on transaction type, amount, and sender/receiver balance values. In addition to model scoring, the application applies hard consistency checks so that obviously impossible balance transitions are flagged directly.

The system is developed using Python, Streamlit, Scikit-learn, LightGBM, Pandas, NumPy, and Plotly. It represents a complete workflow from data loading to preprocessing, feature engineering, model training, calibrated prediction, deployment, analytics, and reporting.

## 1. Background

Financial fraud detection has become increasingly important because of the rapid growth of digital payment systems. Traditional rule-based systems are easy to implement but often fail to identify new fraud behaviour. Purely manual verification is also too slow for large transaction volumes.

This project addresses these gaps by building a smart fraud analytics system that combines:
- analytical insights on transaction data
- predictive modelling for fraud detection
- interactive dashboards for users
- model diagnostics for interpretability

### 1.1 Aim
To develop a system that:
- analyzes financial transaction dataset
- predicts fraud probability for a new transaction
- provides an interactive dashboard for insights
- supports fraud-focused review and reporting

### 1.2 Technologies
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- LightGBM
- Plotly
- Pickle

### 1.3 Hardware Architecture
- Processor: Intel i3 or above
- RAM: Minimum 4 GB, recommended 8 GB
- Storage: At least 10 GB free space

### 1.4 Software Architecture
- Operating System: Windows / Linux / macOS
- Python 3.x environment
- Required libraries installed
- Local file-based dataset and model artifact storage

## 2. System

### 2.1 Requirements

#### 2.1.1 Functional requirements
- Load transaction dataset
- Display analytics dashboard
- Accept user transaction inputs
- Predict fraud probability
- Apply rule-based transaction validation
- Display model metrics and charts
- Export fraud transaction report

#### 2.1.2 User requirements
- Simple and interactive UI
- Fast response time
- Understandable fraud decision output
- Easy navigation across pages

#### 2.1.3 Environmental requirements
- Python environment installed
- Required libraries available
- Compatible system with local execution support

### 2.2 Design and Architecture
Components:
- UI Module: Streamlit interface
- Data Module: CSV loading and preparation
- Feature Engineering Module: fraud-sensitive derived features
- Model Module: LightGBM with calibrated probabilities
- Rule Engine: hard balance-consistency checks
- Visualization Module: charts and diagnostics
- Performance Module: timing and deferred heavy rendering

### 2.3 Implementation
The implementation is divided across a few core files:
- `fraud_core.py` handles shared cleaning and feature engineering
- `train_model.py` handles training, preprocessing, calibration, and artifact generation
- `utils.py` handles cached loaders, prediction, charting, and rule logic
- `app.py` handles page rendering and interaction
- `performance.py` tracks major operation timings

The app currently contains four pages:
- Home
- Analytics Dashboard
- Fraud Prediction
- Model Insights

The model uses numeric and categorical transaction fields, adds engineered features such as hour-of-day, balance ratios, discrepancy signals, and log amount, then predicts fraud probability through a calibrated LightGBM classifier.

### 2.4 Testing

#### 2.4.1 Test Plan Objectives
- Ensure system reliability
- Validate prediction correctness
- Verify UI behaviour

#### 2.4.2 Data Entry
- Valid inputs through dropdowns and numeric widgets
- Numeric inputs constrained through Streamlit components

#### 2.4.3 Security
- No sensitive user data stored permanently
- Input validation prevents invalid entries and crashes

#### 2.4.4 Test Strategy
- Unit Testing
- Integration Testing
- System Testing

#### 2.4.5 System Test
- End-to-end workflow validation from loading to prediction

#### 2.4.6 Performance Test
- Dataset loading time
- UI rendering speed
- prediction response speed

#### 2.4.7 Security Test
- Local-only execution behaviour
- invalid input handling

#### 2.4.8 Basic Test
- Page navigation
- Model artifact loading
- CSV report generation

#### 2.4.9 Stress and Volume Test
- Large dataset handling with more than 6 million rows
- Heavy chart rendering behaviour

#### 2.4.10 Recovery Test
- graceful handling when dataset or model file is missing

#### 2.4.11 Documentation Test
- verifies document clarity and code-module traceability

#### 2.4.12 User Acceptance Test
- verified through interactive dashboard usage

#### 2.4.13 System
- fully functional under normal local execution conditions

### 2.5 Graphical User Interface (GUI) Layout
Pages:
- Home
- Analytics Dashboard
- Fraud Prediction
- Model Insights

Features:
- Sidebar navigation
- KPI cards
- interactive Plotly charts
- prediction result cards
- downloadable CSV report
- performance metrics expander

### 2.6 Customer testing
The project is intended for academic and demonstration use. Informal user testing focuses on:
- ease of navigation
- clarity of fraud verdict
- usefulness of dashboard visuals
- responsiveness on a large dataset

### 2.7 Evaluation

#### 2.7.1 Table
**1: Performance**

| Metric | Result |
|---|---|
| Accuracy | 0.9988 |
| Precision | 0.5198 |
| Recall | 0.9586 |
| F1-Score | 0.6741 |
| ROC-AUC | 0.9858 |
| Recommended Threshold | 0.4917 |

The evaluation shows that the model strongly prioritizes fraud detection recall, which is appropriate for fraud screening use cases.

#### 2.7.2 Static Code Analysis
- Code follows modular structure
- Core logic is separated from UI
- Shared preprocessing reduces inconsistency risk

#### 2.7.3 Wireshark
- Not applicable in the current version because the system is a local file-based dashboard and does not depend on external network traffic for fraud scoring

#### 2.7.4 Test of Main Function
- Dataset loading verified
- Model training pipeline verified
- Prediction flow working with threshold-based decisions
- Rule engine working for inconsistent transactions

## 3. Snapshots of the Project

This section may include screenshots of:
- Home page
- Analytics Dashboard
- Fraud Prediction result
- Model Insights page

## 4. Conclusions

The project successfully:
- integrates data analytics with machine learning techniques
- provides fraud probability prediction for financial transactions
- delivers an interactive and user-friendly dashboard
- includes rule-based safety checks beyond the ML model

This project demonstrates the practical application of:
- Data Science
- Machine Learning
- Fraud Analytics
- Dashboard-based decision support

## 5. Further development or research
- Deploying the application on cloud platforms
- Integrating live transaction streams
- Adding role-based authentication
- Supporting batch scoring pipelines
- Using advanced explainability methods such as SHAP
- Adding automated model retraining and drift monitoring

## 6. References
- Streamlit Documentation
- Scikit-learn Documentation
- Pandas Documentation
- LightGBM Documentation
- Plotly Documentation

## 7. Appendix
- Dataset file name: `fraud_dataset.csv.csv`
- Model artifacts: `model.pkl`, `preprocessor.pkl`, `metrics.pkl`
- Performance module: `performance.py`
