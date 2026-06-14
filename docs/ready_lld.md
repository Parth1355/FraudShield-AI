# Savitribai Phule Pune University

## Low Level Design
### on
## FraudShield AI: Financial Fraud Detection Dashboard with ML-Based Transaction Risk Prediction

**SUBMITTED BY**  
`[Student Name]` (`[Roll No.]`)

**Guided by:** `[Guide Name]`  
**Industry Mentor:** `[If Applicable]`

## Table of Contents
1. Introduction
1.1 Scope of the document
1.2 Intended audience
1.3 System overview
2. Low Level System Design
2.1 Sequence Diagram
2.2 Navigation Flow/UI Implementation
2.3 Screen Validations, Defaults and Attributes
2.4 Client-Side Validation Implementation
2.5 Server-Side Validation Implementation
2.6 Components Design Implementation
2.7 Configurations/Settings
2.8 Interfaces to other components
3. Data Design
3.1 List of Key Schemas/Tables in database
3.2 Details of access levels on key tables in scope
3.3 Key design considerations in data design
4. Details of other frameworks being used
4.1 Session Management
4.2 Caching
5. Unit Testing

## 1 Introduction

### 1.1 Scope of the document
This document provides a detailed Low-Level Design (LLD) for the FraudShield AI system. It covers page logic, backend function flow, data handling, model integration, validation logic, and performance-aware implementation details.

### 1.2 Intended audience
- Developers
- Data Scientists / ML Engineers
- QA Engineers
- Project guide and stakeholders

### 1.3 System overview
The system is a Streamlit-based web application that:
- Displays financial fraud analytics
- Predicts fraud probability for a single transaction
- Provides model insights and downloadable fraud reports

### Key Features
- Home page with project KPIs
- Analytics Dashboard
- Fraud Prediction
- Model Insights
- Hard transaction-rule checks
- Performance timing support

## 2. Low Level System Design

### 2.1 Sequence Diagram (Logical Flow)
1. User opens Streamlit application.
2. App loads dataset, model, preprocessor, and metrics.
3. User navigates to Analytics, Prediction, or Insights page.
4. For prediction, user enters transaction data.
5. App builds engineered input row.
6. Preprocessor transforms the row.
7. Model predicts fraud probability.
8. Rule engine checks transaction consistency.
9. UI shows verdict, probability, risk level, and diagnostics.

### 2.2 Navigation Flow/UI Implementation
Pages:
- Home
- Analytics Dashboard
- Fraud Prediction
- Model Insights

Navigation Logic:
`page = st.radio("Navigation", [...])`

UI Framework:
- Streamlit
- Custom CSS for cards, tabs, metrics, and dark visual theme

### 2.3 Screen Validations, Defaults and Attributes

| Field | Type | Validation | Default |
|---|---|---|---|
| Transaction Type | Dropdown | Must match supported transaction types | PAYMENT |
| Step | Integer | Range 1–744 | 1 |
| Amount | Float | Must be positive | 1000.0 |
| Sender Old Balance | Float | Minimum 0 | 5000.0 |
| Sender New Balance | Float | Minimum 0 | 4000.0 |
| Receiver Old Balance | Float | Minimum 0 | 0.0 |
| Receiver New Balance | Float | Minimum 0 | 0.0 |

### 2.4 Client-Side Validation Implementation
Handled using Streamlit widgets:
- `selectbox()` ensures valid categorical values
- `number_input()` enforces numeric ranges
- `slider()` constrains fraud threshold values

### 2.5 Server-Side Validation Implementation
Server-side validation is handled through helper functions:
- dataset presence check
- model artifact availability check
- input conversion into a DataFrame
- rule-based consistency validation for impossible balance behaviour
- exception handling for prediction errors

### 2.6 Components Design Implementation

**1. UI Layer (`app.py`)**  
Handles:
- page rendering
- user inputs
- chart display
- prediction output

**2. Core Feature Layer (`fraud_core.py`)**  
Handles:
- dataset path resolution
- data cleaning
- feature engineering
- building model-ready input rows

**3. Utility Layer (`utils.py`)**  
Handles:
- cached model and data loading
- prediction helper functions
- risk classification
- hard inconsistency rules
- Plotly chart generation
- CSV export

**4. Model Layer (`train_model.py`)**  
Handles:
- preprocessing pipeline creation
- LightGBM training
- probability calibration
- metrics creation and artifact saving

**5. Performance Layer (`performance.py`)**  
Handles:
- operation timing
- performance summary display
- heavy chart timing measurements

### 2.7 Configurations/Settings

| Config | Value |
|---|---|
| Page Layout | Wide |
| Initial Sidebar | Expanded |
| Theme | Custom dark theme |
| Model Artifact Files | `model.pkl`, `preprocessor.pkl`, `metrics.pkl` |
| Dataset File | `fraud_dataset.csv` or `fraud_dataset.csv.csv` |
| Threshold Source | Metrics file default or user slider |

### 2.8 Interfaces to other components
- `app.py` calls `utils.py` for charts, prediction, and artifact loading
- `utils.py` calls `fraud_core.py` for preprocessing consistency
- `pipeline.py` orchestrates data generation, training, and optional app launch
- `train_model.py` stores outputs consumed by the Streamlit UI

## 3. Data Design

### 3.1 List of Key Schemas/Tables in database
The current project does not use a database. Key data structures are:
- transaction CSV dataset
- engineered Pandas DataFrame
- serialized model artifacts
- metrics dictionary

### 3.2 Details of access levels on key tables in scope
- Read-only dataset access in UI
- Read-only artifact loading during app execution
- No write-back of prediction inputs into permanent storage

### 3.3 Key design considerations in data design
- handle missing values with numeric median and categorical mode
- remove duplicates
- drop non-essential administrative column
- preserve identical training and inference feature logic
- use engineered balance discrepancy features for fraud sensitivity

## 4. Details of other frameworks being used

### 4.1 Session Management
- Streamlit manages sessions automatically
- Each interaction triggers rerun with retained widget state where applicable

### 4.2 Caching
Examples used in the project:
- `@st.cache_resource` for model and preprocessor
- `@st.cache_data` for metrics and chart summaries
- lazy loading for expensive analytics views

## 5. Unit Testing

Suggested Tests:

| Module | Test Case |
|---|---|
| Data Loader | dataset file found and read successfully |
| Feature Engineering | engineered columns created correctly |
| Model Prediction | probability output type and range |
| Rule Engine | impossible balance cases flagged |
| UI Inputs | valid range enforcement |
| Pipeline | training and artifact save success |
