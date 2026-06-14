# Low Level Design

## Title
FraudShield AI: Low Level Design Document

## Student Details
- Name: `[Your Name]`
- Roll No.: `[Your Roll Number]`
- Guide Name: `[Guide Name]`

## 1. Introduction
This document describes the low level design of the FraudShield AI system. It explains the implementation details of each source file, the internal functions, data movement, and model workflow.

## 2. File Level Design
### 2.1 `app.py`
Responsibilities:
- Configures Streamlit page settings and custom UI theme.
- Loads dataset, model, preprocessor, and metrics.
- Provides sidebar navigation.
- Implements four application pages:
  - Home
  - Analytics Dashboard
  - Fraud Prediction
  - Model Insights

### 2.2 `train_model.py`
Responsibilities:
- Resolves dataset location.
- Loads and validates dataset.
- Cleans raw data.
- Engineers advanced features.
- Builds preprocessing pipeline.
- Trains LightGBM model.
- Evaluates model and saves artifacts.

### 2.3 `utils.py`
Responsibilities:
- Shared feature definitions.
- Cached data and artifact loaders.
- Data cleaning and feature engineering helpers.
- Prediction helper functions.
- Plot generation functions.
- CSV report generation.

### 2.4 `generate_sample_data.py`
Responsibilities:
- Generates synthetic transaction records.
- Simulates fraud-heavy patterns for selected transaction types.
- Saves synthetic dataset for testing when actual data is unavailable.

## 3. Function Level Design
### 3.1 `train_model.py`
- `resolve_dataset_path(path)`
  - Searches common dataset paths and returns the first valid file path.
- `load_data(path)`
  - Reads CSV and validates availability.
- `clean_data(df)`
  - Removes duplicates, handles missing values, and drops non-essential columns.
- `engineer_features(df)`
  - Adds time, balance, discrepancy, behavioral, and log-based features.
- `build_preprocessor()`
  - Creates `ColumnTransformer` with scaling for numeric features and ordinal encoding for categorical features.
- `train(df)`
  - Splits data, fits preprocessor, trains model, evaluates performance, and builds metrics output.

### 3.2 `utils.py`
- `load_model()`
  - Loads `model.pkl`.
- `load_preprocessor()`
  - Loads `preprocessor.pkl`.
- `load_metrics()`
  - Loads `metrics.pkl`.
- `load_and_prepare_data(path)`
  - Loads CSV and applies `_clean()` and `_engineer()`.
- `_clean(df)`
  - Performs raw data cleanup.
- `_engineer(df)`
  - Recreates the same feature engineering logic used during training.
- `build_input_row(input_dict)`
  - Converts a single user-entered transaction into the model feature schema.
- `predict(model, preprocessor, row_df, threshold)`
  - Produces fraud probability and binary class label.
- `risk_level(prob)`
  - Maps probability to low, medium, or high risk.
- Chart helper functions
  - Build dashboard and diagnostics charts using Plotly.
- `generate_csv_report(df)`
  - Exports filtered data to CSV bytes for download.

## 4. Feature Design
### Raw Features
- `step`
- `type`
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`

### Engineered Features
- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `is_night`
- `orig_balance_diff`
- `dest_balance_diff`
- `orig_balance_ratio`
- `dest_balance_ratio`
- `orig_zeroed_out`
- `dest_was_zero`
- `balance_discrepancy_orig`
- `balance_discrepancy_dest`
- `customer_tx_frequency`
- `customer_avg_amount`
- `amount_deviation`
- `amount_deviation_ratio`
- `dest_tx_frequency`
- `log_amount`

## 5. Preprocessing Design
- Numeric features are scaled using `StandardScaler`.
- Categorical feature `type` is encoded using `OrdinalEncoder`.
- Preprocessing is implemented through `ColumnTransformer`.
- The same preprocessing artifact is reused during prediction to maintain consistency.

## 6. Model Design
- Model: `LGBMClassifier`
- Key parameters:
  - `num_leaves = 63`
  - `max_depth = 8`
  - `learning_rate = 0.05`
  - `n_estimators = 500`
  - `subsample = 0.8`
  - `colsample_bytree = 0.8`
  - `reg_alpha = 0.1`
  - `reg_lambda = 0.1`
- Imbalance handling:
  - `scale_pos_weight` is computed from the ratio of legitimate to fraud transactions in the training data.

## 7. Page Level Design
### Home Page
- Displays project overview, KPIs, problem statement, technology stack, and key features.

### Analytics Dashboard
- Applies filters on transaction type, amount range, and fraud-only view.
- Displays:
  - Class balance before and after SMOTE
  - Class share before and after SMOTE
  - Fraud rate by transaction type
  - Hourly fraud trend
  - Top high-risk customers
  - Correlation heatmap
- Supports raw data preview and CSV export.

### Fraud Prediction Page
- Collects user-entered transaction details.
- Uses threshold slider for decision boundary.
- Shows fraud label, probability gauge, risk level, and engineered feature table.

### Model Insights Page
- Displays:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - Feature importance
  - Confusion matrix
  - ROC curve
  - Classification report

## 8. Data Flow at Code Level
1. Dataset path is resolved.
2. CSV is loaded and cleaned.
3. New features are generated.
4. Training script saves model and metrics.
5. App loads artifacts with caching.
6. User input is converted to engineered features.
7. Preprocessor transforms inputs.
8. Model predicts class probability.
9. UI renders results and charts.

## 9. Error Handling
- Missing dataset shows warning in sidebar and page-level error in analytics.
- Missing model artifacts show error in prediction and insights pages.
- Prediction exceptions are captured and displayed to the user.

## 10. Conclusion
The low level design reflects a clear separation between training logic, shared utilities, synthetic data generation, and user interface. This structure helps maintain consistency between offline training and online prediction.
