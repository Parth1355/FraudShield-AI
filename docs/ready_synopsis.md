# Savitribai Phule Pune University

## FraudShield AI: Financial Fraud Detection Dashboard with ML-Based Transaction Risk Prediction

### PROJECT SYNOPSIS

**BACHELOR OF TECHNOLOGY**  
Data Science

**SUBMITTED BY**  
`[Student Name]` (`[Roll No.]`)

**Guided by:**  
`[Guide Name]`

## Project - Synopsis Format

### Title Page
1. `[Student Name]`
2. `[Roll No.]`
3. `[Address]`  
   Email :- `[Email]`  
   Phone No :- `[Phone No.]`
4. Data Science
5. `[Batch]`
6. FraudShield AI: Financial Fraud Detection Dashboard with ML-Based Transaction Risk Prediction

## Content Page

### Table of Contents
- Introduction
- Literature Survey
- Methodology / Planning
- Facilities Required
- References

## Introduction Page

The FraudShield AI system is a web-based application developed to analyze financial transaction data and provide fraud risk prediction using machine learning. In digital payment systems, fraudulent transactions are rare but highly damaging, so there is a strong need for intelligent screening mechanisms that can identify suspicious behaviour without excessively disturbing legitimate users.

This project integrates data preprocessing, feature engineering, machine learning, visualization, and real-time prediction into a single platform. The system allows users to explore fraud trends, inspect transaction patterns, and predict whether a user-entered transaction is legitimate, review-worthy, or fraudulent. It also provides model insights such as feature importance, confusion matrix, ROC curve, and a downloadable fraud-focused report.

The application is built using Python and Streamlit for frontend interaction, while Pandas, NumPy, Scikit-learn, LightGBM, and Pickle are used for data handling, preprocessing, training, and deployment. The current implementation also includes performance tracking and lazy loading of heavy analytics views so that the application remains responsive even with a large dataset.

### Key Technologies Used
- Python
- Streamlit (UI Framework)
- Pandas and NumPy (Data Handling)
- Scikit-learn (Preprocessing and Evaluation)
- LightGBM (Machine Learning)
- Plotly (Visualization)
- Pickle (Model Artifact Storage)

### Field of Project
- Data Science
- Machine Learning
- Fraud Analytics
- Web-based Dashboard

## Literature Survey

Several fraud detection systems rely either on rigid business rules or on standalone machine learning models. Rule-based systems are fast but often fail to detect new fraud patterns. Traditional classification models improve adaptability, but many academic demos do not combine analytics, model interpretation, real-time scoring, and decision support in one interface.

Research in fraud detection commonly uses imbalance-aware learning, probability-based classification, and behaviour-derived features. Tree-based ensemble methods such as Gradient Boosting and LightGBM are frequently preferred because they can capture nonlinear interactions in transactional data. Existing systems also show the importance of recall-oriented evaluation because missed frauds are more costly than moderate false positives.

This project improves on basic academic solutions by combining:
- Dataset analytics
- Machine learning prediction
- Threshold-based decision support
- Hard transaction consistency checks
- Interactive dashboard-based presentation

## Methodology / Planning of Work

The development of the project follows a structured approach:

**Step 1: Data Collection**  
- Financial transaction dataset collected in CSV format

**Step 2: Data Preprocessing**  
- Remove duplicates  
- Fill missing values  
- Drop non-essential fields

**Step 3: Feature Engineering**  
- Create time-based features  
- Create balance-difference and discrepancy features  
- Create ratio and zero-balance indicators

**Step 4: Model Training**  
- Preprocess data using StandardScaler and OrdinalEncoder  
- Train LightGBM classifier  
- Calibrate probabilities and save metrics

**Step 5: Model Deployment**  
- Save artifacts using Pickle  
- Integrate model into Streamlit application

**Step 6: UI Development**  
- Develop Home, Analytics, Prediction, and Model Insights pages  
- Add charts, reports, and threshold controls

**Step 7: Testing and Validation**  
- Validate data loading  
- Test prediction flow  
- Verify analytics and model diagnostics

## Facilities Required for Proposed Work

### Hardware Requirements
- Minimum 4 GB RAM
- Processor: i3 or above
- Storage: 10 GB free space

### Software Requirements
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- Plotly

## References
- Python Documentation
- Streamlit Documentation
- Scikit-learn Documentation
- LightGBM Documentation
- Plotly Documentation
