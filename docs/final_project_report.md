# Final Project Report

## Title
FraudShield AI: Financial Fraud Detection Web Application

## Student Details
- Name: `[Your Name]`
- Roll No.: `[Your Roll Number]`
- Class/Division: `[Your Class]`
- Department: `[Your Department]`
- Guide Name: `[Guide Name]`
- College Name: `[College Name]`
- Academic Year: `[Academic Year]`

## Certificate
This project report entitled **FraudShield AI: Financial Fraud Detection Web Application** is submitted as a partial fulfillment of the academic requirements for the completion of the degree program. The work has been carried out by `[Your Name]` under the guidance of `[Guide Name]`.

## Acknowledgement
I would like to express my sincere gratitude to my project guide, department faculty, and institution for their support and guidance during the development of this project. I also acknowledge the use of Python open-source libraries and the IBM synthetic financial transaction dataset that supported this work.

## Abstract
Fraud detection is one of the most important applications of machine learning in financial systems. With the increase in online transactions, fraudulent activities have become more sophisticated and difficult to detect through static rule-based methods alone. This project proposes FraudShield AI, a web-based fraud detection system that uses a LightGBM classification model to identify suspicious transactions using transactional, behavioral, and balance-based features.

The system is implemented using Python, Streamlit, scikit-learn, Plotly, and LightGBM. It provides an end-to-end workflow that includes data preprocessing, feature engineering, model training, dashboard analytics, real-time transaction scoring, and model interpretation. The trained model achieved strong performance with high recall, making it suitable for fraud screening scenarios where missing a fraud case is more costly than raising a false alarm.

## 1. Introduction
Financial transaction fraud leads to heavy economic losses and operational challenges. Traditional methods based on static business rules often fail to adapt to new fraud strategies. Machine learning offers a more flexible approach by learning patterns from historical data and detecting suspicious behavior based on multiple signals.

FraudShield AI is developed as a practical fraud detection platform that combines:
- Historical fraud analytics
- Real-time prediction
- Model performance reporting
- Downloadable transaction reports

## 2. Problem Definition
Organizations processing digital payments need a system that can:
- Detect fraudulent transactions with high sensitivity
- Minimize unnecessary interruption to legitimate transactions
- Provide interpretable outputs for analysts
- Support quick analysis of transaction trends and model behavior

## 3. Objectives
- To study fraud patterns in digital payment transactions.
- To preprocess and transform raw transaction data for machine learning.
- To engineer meaningful predictive features from account balances and behavioral patterns.
- To train and evaluate a high-performance fraud detection model.
- To develop a user-friendly web interface for analytics and prediction.
- To generate project documentation aligned with academic requirements.

## 4. Literature and Background
Fraud detection systems usually rely on classification methods, anomaly detection, and behavioral analysis. In highly imbalanced financial datasets, fraud cases are rare and therefore difficult to model. Ensemble methods such as gradient-boosted decision trees are effective because they capture nonlinear relationships and interactions between features. Techniques such as class weighting, feature engineering, and recall-oriented evaluation are commonly used to improve detection performance.

## 5. Existing System and Need for Proposed System
### Existing System
- Manual monitoring is slow and difficult to scale.
- Rule-based systems require continuous maintenance.
- Static rules generate false positives and miss novel fraud patterns.

### Need for Proposed System
- Data-driven fraud detection improves adaptability.
- Machine learning can analyze multiple transactional signals together.
- Visualization and model diagnostics support practical decision-making.

## 6. Proposed System
FraudShield AI is a machine learning based fraud detection system built on a modular architecture.

### Main capabilities
- Transaction data loading and preprocessing
- Feature engineering using 24+ meaningful signals
- LightGBM model training and evaluation
- Interactive dashboard with fraud-focused charts
- Real-time single-transaction prediction
- Probability gauge, risk categorization, and CSV report export

## 7. System Requirements
### Hardware Requirements
- Processor: Intel i3 or above
- RAM: 4 GB minimum, 8 GB recommended
- Storage: 2 GB or more

### Software Requirements
- Python 3.x
- Streamlit
- LightGBM
- scikit-learn
- pandas
- numpy
- Plotly
- imbalanced-learn

## 8. Dataset Description
The project uses a financial transactions dataset with the following attributes:
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

### Dataset statistics used in the project
- Total transactions: 6,362,620
- Fraud transactions: 8,213
- Fraud rate: 0.1291%
- Average transaction amount: 179,861.90
- Maximum transaction amount: 92,445,516.64

The class imbalance in the dataset makes fraud detection a challenging problem and justifies the use of imbalance-aware model training.

## 9. Methodology
### 9.1 Data Cleaning
- Removal of duplicate records
- Median imputation for numeric missing values
- Mode imputation for categorical missing values
- Removal of `isFlaggedFraud` from training features

### 9.2 Feature Engineering
The system creates new predictive features in the following groups:
- Time-based features
- Balance-derived features
- Zero-balance fraud indicators
- Balance discrepancy features
- Customer transaction behavior features
- Destination frequency features
- Log-transformed amount feature

### 9.3 Preprocessing
- `StandardScaler` for numeric features
- `OrdinalEncoder` for categorical feature `type`
- `ColumnTransformer` to ensure a consistent pipeline

### 9.4 Model Training
- Train-test split with stratification
- LightGBM classifier training
- Class imbalance handling through `scale_pos_weight`

### 9.5 Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

## 10. System Design
### 10.1 Modules
- Data preparation module
- Model training module
- Analytics dashboard module
- Fraud prediction module
- Model insights module

### 10.2 Architecture
The application follows a layered architecture:
- Data Layer: dataset CSV and saved artifacts
- Logic Layer: preprocessing, feature engineering, prediction, reporting
- Presentation Layer: Streamlit web interface

### 10.3 File Structure
- `app.py` for user interface and page routing
- `train_model.py` for training pipeline
- `utils.py` for shared logic
- `generate_sample_data.py` for synthetic dataset generation
- `model.pkl`, `preprocessor.pkl`, `metrics.pkl` for saved artifacts

## 11. Implementation Details
### Home Page
Displays project summary, problem statement, KPIs, and technology stack.

### Analytics Dashboard
Supports transaction-type filtering, amount-range filtering, fraud-only filtering, chart visualization, raw data preview, and CSV export.

### Fraud Prediction
Allows the user to enter transaction values and get:
- Fraud or legitimate classification
- Fraud probability
- Risk level
- Feature breakdown

### Model Insights
Provides:
- Accuracy, precision, recall, F1-score, ROC-AUC
- Feature importance visualization
- Confusion matrix analysis
- ROC curve
- Classification report

## 12. Results and Discussion
### Model Performance
- Accuracy: 0.9998
- Precision: 0.8973
- Recall: 0.9842
- F1-score: 0.9388
- ROC-AUC: 0.9961

### Confusion Matrix
- True Negatives: 1,270,696
- False Positives: 185
- False Negatives: 26
- True Positives: 1,617

### Interpretation
The model performs very well on the fraud detection task. The high recall indicates that the system successfully identifies most fraudulent transactions, which is critical in fraud screening. Precision is also strong, which means the number of false alarms is relatively low. The ROC-AUC score confirms that the classifier has high discriminative ability.

## 13. Advantages
- High fraud detection capability
- Real-time prediction support
- Interactive analytics and visualization
- Modular implementation for easy maintenance
- Clear model interpretability support

## 14. Limitations
- Works on historical/static data rather than live transaction streams
- No authentication or role-based access control
- No direct deployment to cloud or banking infrastructure
- No automated retraining or drift monitoring

## 15. Future Scope
- Integration with live payment or banking APIs
- User authentication and analyst dashboards
- Cloud deployment with continuous retraining
- Use of explainable AI techniques such as SHAP
- Batch scoring and alert notification pipeline

## 16. Conclusion
FraudShield AI demonstrates how machine learning can be applied effectively to the fraud detection problem using financial transaction data. The system combines preprocessing, advanced feature engineering, model training, analytics, and real-time inference in a single application. Its strong recall and overall performance make it a useful academic prototype for fraud screening and decision support.

## 17. References
1. Lopez-Rojas, E.A., Elmir, A. and Axelsson, S. PaySim: A financial mobile money simulator for fraud detection.
2. LightGBM official documentation.
3. scikit-learn official documentation.
4. Streamlit official documentation.
5. Plotly official documentation.
