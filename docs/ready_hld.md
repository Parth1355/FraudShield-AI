# Savitribai Phule Pune University

## Project - High Level Design
### On
## FraudShield AI: Financial Fraud Detection Dashboard with ML-Based Transaction Risk Prediction

**Team members:**  
`[Student Name]` (`[Roll No.]`)

**Guided by:** `[Guide Name]`  
**Industry Mentor:** `[If Applicable]`

## Table of Contents
1. Introduction
1.1 Scope of the document
1.2 Intended Audience
1.3 System Overview
2. System Design
2.1 Application Design
2.2 Process Flow
2.3 Information Flow
2.4 Components Design
2.5 Key Design Considerations
2.6 API Catalogue
3. Data Design
3.1 Data Model
3.2 Data Access Mechanism
3.3 Data Retention Policies
3.4 Data Migration
4. Interfaces
5. State and Session Management
6. Caching
7. Non-Functional Requirements
8. References

## 1] Introduction

Financial fraud in digital payment systems creates direct monetary loss, customer dissatisfaction, and operational burden. As transaction volumes grow, manual review and static rule-based systems become insufficient for identifying suspicious patterns efficiently. A modern fraud detection system should not only detect anomalies from historical data but also assist the analyst with dashboards, model interpretation, and real-time decision support.

FraudShield AI is a machine learning-based fraud analytics and prediction platform designed for this purpose. The project analyzes a large transaction dataset, engineers fraud-sensitive features, trains a LightGBM-based classifier, calibrates prediction probabilities, and exposes the results through a Streamlit web dashboard. The application allows users to inspect fraud distribution, filter transactions, score new transactions, and review model diagnostics from a single interface.

The main objective of this project is to demonstrate how machine learning, feature engineering, and visualization can be combined to create a practical fraud-screening dashboard for academic use.

### 1.1 Scope of the Document
This document describes the High Level Design (HLD) of the FraudShield AI system. It explains the system architecture, major modules, process flow, data flow, interfaces, and non-functional design considerations.

### 1.2 Intended Audience
- Project guide and academic evaluators
- Developers and maintainers
- Data science students
- Testing and documentation stakeholders

### 1.3 System Overview
The FraudShield AI system is designed as a machine learning-powered analytics and prediction platform.

The system performs the following major functions:
1. Loads historical transaction dataset from CSV.
2. Cleans the dataset and removes invalid or duplicate entries.
3. Creates engineered features from transaction amount, balances, time, and discrepancy signals.
4. Trains a LightGBM model for fraud classification.
5. Calibrates prediction probabilities and stores trained artifacts.
6. Loads trained artifacts in a Streamlit dashboard.
7. Accepts user input for real-time transaction scoring.
8. Applies both model probability and hard consistency rules for final decision support.
9. Displays analytics, insights, charts, and downloadable reports.

The system is designed to be modular, interactive, and scalable enough for large CSV-based academic datasets.

## 2] System Design

### 2.1 Application Design
The application follows a modular layered architecture where different modules perform independent responsibilities.

| Module | Description |
|---|---|
| Data Loader Module | Loads dataset and saved artifacts |
| Data Cleaning Module | Removes duplicates and handles missing values |
| Feature Engineering Module | Builds time, ratio, and balance discrepancy features |
| Model Training Module | Trains and calibrates the fraud classification model |
| Prediction Module | Scores user-entered transactions |
| Rule Validation Module | Applies hard balance-consistency checks |
| Dashboard Module | Provides interactive Streamlit UI |
| Visualization Module | Displays analytics and model insights |
| Performance Module | Tracks timings and improves responsiveness |

This modular structure improves maintainability, readability, and future extensibility.

### 2.2 Process Flow
The high-level process followed by the system is:
1. Input transaction dataset is provided in CSV format.
2. Dataset is cleaned and engineered features are generated.
3. Preprocessing pipeline transforms numeric and categorical fields.
4. LightGBM model is trained and calibrated.
5. Model artifacts are stored as `model.pkl`, `preprocessor.pkl`, and `metrics.pkl`.
6. Streamlit dashboard loads data and artifacts.
7. User navigates analytics, prediction, or model insight pages.
8. For prediction, the system converts user input into engineered features and computes fraud probability.
9. Rule-based transaction consistency checks refine the final displayed decision.

### 2.3 Information Flow
Information flows through the system in the following stages:

**1. User Input Data**  
The user enters transaction details such as transaction type, step, amount, sender balance, and receiver balance through the Streamlit interface.

**2. Validation Layer**  
Input values are constrained through widget defaults and rule-based validation so that obviously invalid data is caught or highlighted.

**3. Data Preprocessing**  
Input is transformed using the same preprocessing pipeline used during training, ensuring consistency between training-time and run-time behaviour.

**4. Prediction Engine**  
The calibrated model generates a fraud probability score.

**5. Decision Layer**  
The app combines probability threshold logic with hard balance-consistency rules to produce one of the following outcomes:
- Fraud Detected
- Review Recommended
- Legitimate

**6. Result Presentation**  
Predicted result, probability, confidence band, triggered rules, charts, and diagnostics are displayed to the user.

### 2.4 Components Design
The system contains several key components that interact with each other.

| Component | Function |
|---|---|
| Data Loader | Loads dataset and artifacts |
| Data Preprocessor | Cleans and transforms transaction records |
| Feature Generator | Builds engineered fraud-detection features |
| Pipeline Trainer | Trains LightGBM and computes metrics |
| Model Saver | Stores model, preprocessor, and metrics |
| Prediction Engine | Produces fraud probability |
| Rule Engine | Detects impossible balance transitions |
| Visualization Engine | Displays charts and reports |
| Streamlit UI | Handles user interaction |
| Performance Tracker | Records timing of heavy operations |

Each component performs a focused role within the overall architecture.

### 2.5 Key Design Considerations
The following design principles were considered during development:
- **Modularity** – Core logic is separated across training, utilities, and UI files.
- **Consistency** – Shared feature engineering is reused for both training and inference.
- **Scalability** – The system supports a large dataset through caching and deferred heavy views.
- **Interpretability** – Metrics, feature importance, confusion matrix, and ROC curve are provided.
- **Usability** – Streamlit UI offers simple navigation and interactive controls.
- **Performance** – Lazy loading and timing instrumentation improve responsiveness.

### 2.6 API Catalogue
This project does not expose external REST APIs in the current version. Internal callable interfaces exist between modules such as:
- data loading functions
- training functions
- prediction helpers
- chart generation functions
- rule-validation functions

## 3] Data Design

### 3.1 Data Model
The transaction dataset contains fraud-related attributes used for analytics and prediction.

| Field | Description |
|---|---|
| `step` | Hour index of transaction |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender identifier |
| `oldbalanceOrg` | Sender old balance |
| `newbalanceOrig` | Sender new balance |
| `nameDest` | Receiver identifier |
| `oldbalanceDest` | Receiver old balance |
| `newbalanceDest` | Receiver new balance |
| `isFraud` | Fraud label |

Engineered fields such as `hour_of_day`, `is_night`, `orig_balance_ratio`, `balance_discrepancy_orig`, and `log_amount` are created internally for model use.

### 3.2 Data Access Mechanism
The dataset is stored in CSV format and accessed through Pandas. The trained model, preprocessor, and metrics are stored locally as Pickle artifacts and loaded during inference.

### 3.3 Data Retention Policies
The system does not permanently store user-entered prediction inputs. Input values are processed during the active session only. Dataset files and model artifacts remain stored locally for academic project use.

### 3.4 Data Migration
The current system is file-based and does not require database migration. However, it can be extended in future to work with MySQL, PostgreSQL, MongoDB, or cloud data stores if production deployment is required.

## 4] Interfaces

The system interacts with the following interfaces:

| Interface | Description |
|---|---|
| Streamlit Dashboard | User interaction and display |
| Machine Learning Model | Fraud prediction engine |
| Plotly Charts | Analytics and visual diagnostics |
| Pickle Artifacts | Stored model components |
| Performance Tracker | Execution timing summary |

## 5] State and Session Management

Streamlit manages application sessions automatically. Each user interaction creates an independent rerun context, and the application uses Streamlit caching to reduce repeated heavy loads.

## 6] Caching

Caching is implemented to improve application performance. Streamlit caching prevents repeated loading of:
- datasets
- model artifacts
- metrics
- chart summaries

Heavy charts such as correlation heatmap and top high-risk customers are deferred until the user expands them.

## 7] Non-Functional Requirements

### 7.1 Security Aspects
- Input validation prevents invalid numeric values.
- User data is not permanently stored.
- Local execution avoids unnecessary exposure of sensitive data.
- Stored artifacts are loaded from local project paths only.

### 7.2 Performance Aspects
- Cached resource loading reduces repeated overhead.
- Large dataset handling is improved through deferred visualization.
- Timing metrics help identify bottlenecks during execution.
- Prediction path remains lightweight for single-transaction scoring.

## 8] References
- Python Documentation
- Streamlit Documentation
- Scikit-learn Documentation
- LightGBM Documentation
- Plotly Documentation
