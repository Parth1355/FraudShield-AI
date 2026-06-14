"""
Shared data cleaning and feature engineering used by training and inference.
"""

import os

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "orig_balance_diff",
    "dest_balance_diff",
    "orig_balance_ratio",
    "dest_balance_ratio",
    "orig_zeroed_out",
    "dest_was_zero",
    "balance_discrepancy_orig",
    "balance_discrepancy_dest",
    "log_amount",
]

CATEGORICAL_FEATURES = ["type"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "isFraud"


def resolve_dataset_path(path: str | None = None) -> str | None:
    """
    Resolve dataset path with multiple fallback locations.
    Checks in order: provided path, root level, data/ directory.
    """
    candidates = []
    if path:
        candidates.append(path)

    candidates.extend(
        [
            "fraud_dataset.csv",
            "fraud_dataset.csv.csv",
            os.path.join("data", "fraud_dataset.csv"),
            os.path.join("data", "fraud_dataset.csv.csv"),
        ]
    )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and preprocess the dataset."""
    df = df.drop_duplicates()
    df = df.drop(columns=["isFlaggedFraud"], errors="ignore")

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object"]).columns:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 23 features from raw transaction data."""
    df = df.copy()
    df["hour_of_day"] = df["step"] % 24
    df["day_of_week"] = (df["step"] // 24) % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)

    df["orig_balance_diff"] = df["newbalanceOrig"] - df["oldbalanceOrg"]
    df["dest_balance_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["orig_balance_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["dest_balance_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1)

    df["orig_zeroed_out"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_was_zero"] = (df["oldbalanceDest"] == 0).astype(int)

    df["balance_discrepancy_orig"] = (
        df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    ).abs()
    df["balance_discrepancy_dest"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    ).abs()

    df["log_amount"] = np.log1p(df["amount"])
    return df


def build_input_row(input_dict: dict) -> pd.DataFrame:
    """Build and engineer a single input row for prediction."""
    row = pd.DataFrame([dict(input_dict)])
    row = engineer_features(clean_data(row))
    return row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
