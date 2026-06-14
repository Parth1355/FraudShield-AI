# Project Synopsis

## Title
FraudShield AI: Financial Fraud Detection Web Application

## Student Details
- Name: `[Your Name]`
- Roll No.: `[Your Roll Number]`
- Class/Division: `[Your Class]`
- Department: `[Your Department]`
- Guide Name: `[Guide Name]`
- Academic Year: `[Academic Year]`

## 1. Introduction
Digital payment systems process a large number of transactions every day, which increases the risk of fraudulent activity. Conventional rule-based systems are often rigid, difficult to scale, and unable to detect evolving fraud patterns. This project presents an intelligent fraud detection system that applies machine learning to identify suspicious transactions in real time.

The proposed system, FraudShield AI, is a Streamlit-based web application that uses a LightGBM classifier trained on a financial transactions dataset. It supports data analytics, fraud probability prediction, model interpretation, and export of filtered fraud reports.

## 2. Problem Statement
Financial organizations require a fast and accurate mechanism to detect fraudulent transactions while reducing false alarms for legitimate users. Existing manual and static rule-based approaches are not sufficient for dynamic fraud scenarios. The system must analyze transactional behavior, classify suspicious transactions, and present results through an easy-to-use interface.

## 3. Objectives
- To build a machine learning based fraud detection system for financial transactions.
- To perform preprocessing and feature engineering on transaction data.
- To train a classification model capable of detecting fraudulent patterns with high recall.
- To provide an interactive dashboard for fraud analysis and model insights.
- To support real-time prediction for a single transaction entered by the user.
- To generate downloadable reports for fraud-focused transaction analysis.

## 4. Scope
The scope of the project includes:
- Loading and cleaning financial transaction data.
- Engineering behavioral, balance-based, and time-based features.
- Training a supervised machine learning model for fraud detection.
- Visualizing fraud trends, class imbalance, and model performance.
- Predicting fraud probability for user-entered transaction details.

The project does not include direct banking integration, multi-user authentication, or production-grade transaction ingestion from live payment gateways.

## 5. Proposed Methodology
The proposed methodology follows these stages:
- Data collection from the IBM synthetic financial transaction dataset.
- Data cleaning by removing duplicates, handling missing values, and dropping administrative leakage columns.
- Feature engineering using transaction time, sender and receiver balances, balance discrepancies, customer behavior, and transaction frequency.
- Data preprocessing using `ColumnTransformer`, `StandardScaler`, and `OrdinalEncoder`.
- Model training using `LightGBM` with imbalance handling through `scale_pos_weight`.
- Performance evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, and confusion matrix.
- Deployment of the trained model in a Streamlit-based web application.

## 6. Software and Hardware Requirements
### Software Requirements
- Operating System: Windows/Linux/macOS
- Programming Language: Python 3.x
- Frontend/Application Framework: Streamlit
- Libraries: pandas, numpy, scikit-learn, lightgbm, plotly, imbalanced-learn

### Hardware Requirements
- Processor: Intel i3 or above
- RAM: Minimum 4 GB, recommended 8 GB or above
- Storage: Minimum 2 GB free space

## 7. Dataset Used
- Dataset Name: IBM Synthetic Financial Dataset (PaySim style dataset)
- Main Attributes:
  - `step`
  - `type`
  - `amount`
  - `nameOrig`
  - `oldbalanceOrg`
  - `newbalanceOrig`
  - `nameDest`
  - `oldbalanceDest`
  - `newbalanceDest`
  - `isFraud`
  - `isFlaggedFraud`

Project dataset observations:
- Total transactions: 6,362,620
- Fraud cases: 8,213
- Fraud rate: 0.1291%

## 8. Expected Outcome
The expected output is a functional fraud detection system that can:
- Detect suspicious transactions with high recall.
- Help users understand fraud distribution through charts and reports.
- Predict the fraud probability of a new transaction in real time.
- Improve decision-making for analysts through interpretable model diagnostics.

## 9. Applications
- Banking transaction monitoring
- Digital wallet fraud screening
- Payment gateway risk analysis
- Internal audit and compliance support

## 10. Conclusion
FraudShield AI is designed as an end-to-end fraud detection solution that combines machine learning, feature engineering, visualization, and real-time prediction in a single system. The project demonstrates how data-driven methods can improve fraud identification accuracy and provide a practical decision-support platform.

## 11. References
1. Lopez-Rojas, E.A., Elmir, A. and Axelsson, S., PaySim: A financial mobile money simulator for fraud detection.
2. LightGBM Documentation.
3. scikit-learn Documentation.
4. Streamlit Documentation.
