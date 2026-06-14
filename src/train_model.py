"""
End-to-end training pipeline for the fraud detection system.
"""

import argparse
import os
import pickle
import warnings

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from fraud_core import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    clean_data,
    engineer_features,
    resolve_dataset_path,
)

warnings.filterwarnings("ignore")


def load_data(path: str) -> pd.DataFrame:
    """Load and validate dataset."""
    dataset_path = resolve_dataset_path(path)
    if dataset_path is None:
        raise FileNotFoundError(f"Dataset not found: {path}")
    print(f"[INFO] Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    return df


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean training data."""
    print("[INFO] Cleaning data...")
    df = clean_data(df)
    print(f"[INFO] Clean shape: {df.shape}")
    print(f"[INFO] Fraud rate: {df[TARGET].mean():.4%}")
    return df


def engineer_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for training."""
    print("[INFO] Engineering features...")
    df = engineer_features(df)
    print(f"[INFO] Feature-engineered shape: {df.shape}")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline."""
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline(
        [("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train(df: pd.DataFrame):
    """
    Train fraud detection model with calibration.
    
    Returns:
        Tuple of (calibrated_model, preprocessor, metrics)
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
    )
    print(
        f"[INFO] Train size: {X_train.shape}, Calibration size: {X_calib.shape}, "
        f"Test size: {X_test.shape}"
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_calib_proc = preprocessor.transform(X_calib)
    X_test_proc = preprocessor.transform(X_test)

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    print(f"[INFO] scale_pos_weight = {scale_pos_weight:.1f}")

    model = LGBMClassifier(
        num_leaves=63,
        max_depth=8,
        learning_rate=0.05,
        n_estimators=500,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    print("[INFO] Training LightGBM model...")
    model.fit(X_train_proc, y_train, eval_set=[(X_calib_proc, y_calib)], callbacks=[])

    print("[INFO] Calibrating probabilities...")
    calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated_model.fit(X_calib_proc, y_calib)

    y_prob = calibrated_model.predict_proba(X_test_proc)[:, 1]
    precision_curve, recall_curve, threshold_curve = precision_recall_curve(y_test, y_prob)
    f1_curve = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
        precision_curve[:-1] + recall_curve[:-1] + 1e-12
    )
    best_idx = int(f1_curve.argmax()) if len(f1_curve) else 0
    recommended_threshold = float(threshold_curve[best_idx]) if len(threshold_curve) else 0.5
    y_pred = (y_prob >= recommended_threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_test": y_test.tolist(),
        "y_prob": y_prob.tolist(),
        "feature_names": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "feature_importances": model.feature_importances_.tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "calibration_method": "sigmoid",
        "recommended_threshold": recommended_threshold,
    }

    print("\n[RESULTS]")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  Threshold: {metrics['recommended_threshold']:.4f}")
    return calibrated_model, preprocessor, metrics


def save_artifacts(model, preprocessor, metrics, out_dir="models"):
    """Save trained model and preprocessor to disk."""
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    prep_path = os.path.join(out_dir, "preprocessor.pkl")
    meta_path = os.path.join(out_dir, "metrics.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(prep_path, "wb") as f:
        pickle.dump(preprocessor, f)
    with open(meta_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\n[INFO] Saved to {model_path}")
    print(f"[INFO] Saved to {prep_path}")
    print(f"[INFO] Saved to {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", default="fraud_dataset.csv", help="Path to the CSV dataset")
    parser.add_argument("--out", default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    df = load_data(args.data)
    df = clean_training_data(df)
    df = engineer_training_features(df)
    model, preprocessor, metrics = train(df)
    save_artifacts(model, preprocessor, metrics, out_dir=args.out)
    print("\n[INFO] Training complete. Run: streamlit run src/app.py")
