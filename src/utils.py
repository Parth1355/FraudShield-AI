"""
Shared helper functions for the fraud detection web application.
"""

import os
import pickle
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fraud_core import build_input_row, clean_data, engineer_features, resolve_dataset_path
from performance import get_tracker, profile_function

warnings.filterwarnings("ignore")

COLORS = {
    "fraud": "#FF4B4B",
    "legit": "#00C49A",
    "primary": "#1E3A5F",
    "accent": "#F5A623",
    "bg": "#0E1117",
    "card": "#1A1F2E",
    "text": "#E8ECF0",
    "muted": "#8892A4",
    "chart_seq": px.colors.sequential.Plasma,
}


def _get_model_path() -> str:
    """Resolve model path (support both root and models/ directory)."""
    if os.path.exists("models/model.pkl"):
        return "models/model.pkl"
    return "model.pkl"


def _get_preprocessor_path() -> str:
    """Resolve preprocessor path (support both root and models/ directory)."""
    if os.path.exists("models/preprocessor.pkl"):
        return "models/preprocessor.pkl"
    return "preprocessor.pkl"


def _get_metrics_path() -> str:
    """Resolve metrics path (support both root and models/ directory)."""
    if os.path.exists("models/metrics.pkl"):
        return "models/metrics.pkl"
    return "metrics.pkl"


@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model from disk."""
    path = _get_model_path()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """Load preprocessor from disk."""
    path = _get_preprocessor_path()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Load metrics from disk."""
    path = _get_metrics_path()
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=True)
def load_and_prepare_data(path: str | None = None) -> pd.DataFrame | None:
    """Load, clean, and engineer features for dataset."""
    tracker = get_tracker()
    dataset_path = resolve_dataset_path(path)
    if dataset_path is None:
        return None
    
    # Track data loading
    tracker.start("CSV Loading")
    df = pd.read_csv(dataset_path)
    tracker.end("CSV Loading")
    
    # Track data cleaning
    tracker.start("Data Cleaning")
    df = clean_data(df)
    tracker.end("Data Cleaning")
    
    # Track feature engineering
    tracker.start("Feature Engineering")
    df = engineer_features(df)
    tracker.end("Feature Engineering")
    
    return df


def predict(model, preprocessor, row_df: pd.DataFrame, threshold: float = 0.5):
    """Make a fraud prediction on a single transaction row."""
    X = preprocessor.transform(row_df)
    prob = model.predict_proba(X)[0][1]
    label = int(prob >= threshold)
    return label, float(prob)


def risk_level(prob: float) -> tuple[str, str]:
    """Classify fraud probability into risk level."""
    if prob < 0.3:
        return "Low Risk", COLORS["legit"]
    if prob < 0.6:
        return "Medium Risk", COLORS["accent"]
    return "High Risk", COLORS["fraud"]


def review_decision(prob: float, threshold: float, margin: float = 0.1) -> tuple[str, str, str]:
    """Determine decision with margin band."""
    lower = max(0.0, threshold - margin)
    upper = min(1.0, threshold + margin)

    if lower <= prob <= upper:
        return (
            "Review Recommended",
            COLORS["accent"],
            "The score is close to the decision threshold. Manual review is safer than a hard verdict.",
        )
    if prob >= threshold:
        return (
            "Fraud Detected",
            COLORS["fraud"],
            "This transaction has been flagged as potentially fraudulent.",
        )
    return (
        "Legitimate",
        COLORS["legit"],
        "This transaction appears to be legitimate.",
    )


def detect_inconsistency_rules(input_dict: dict, tolerance: float = 1.0) -> list[str]:
    """Detect balance inconsistencies in transaction data."""
    amount = float(input_dict["amount"])
    old_orig = float(input_dict["oldbalanceOrg"])
    new_orig = float(input_dict["newbalanceOrig"])
    old_dest = float(input_dict["oldbalanceDest"])
    new_dest = float(input_dict["newbalanceDest"])
    tx_type = str(input_dict["type"])

    rules: list[str] = []
    sender_delta = new_orig - old_orig
    receiver_delta = new_dest - old_dest
    sender_spent = old_orig - new_orig
    receiver_received = new_dest - old_dest
    amount_tolerance = max(tolerance, amount * 0.01)

    if tx_type in {"TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"} and new_orig > old_orig + tolerance:
        rules.append("Sender balance increases after the transaction.")
    if tx_type in {"TRANSFER", "DEBIT", "CASH_IN"} and new_dest + tolerance < old_dest:
        rules.append("Receiver balance decreases after the transaction.")
    if tx_type in {"TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"} and amount > old_orig + tolerance and new_orig > 0:
        rules.append("Transaction amount exceeds sender balance without draining the sender account.")
    if amount <= 0:
        rules.append("Transaction amount must be positive.")

    if abs(sender_delta) > amount + amount_tolerance:
        rules.append("Sender balance changes by more than the transaction amount.")
    if receiver_delta > amount + amount_tolerance:
        rules.append("Receiver balance increases by more than the transaction amount.")

    debit_like = {"TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"}
    receiver_credit_required = {"TRANSFER", "CASH_IN", "PAYMENT"}
    receiver_credit_optional = {"CASH_OUT"}

    if tx_type in debit_like:
        expected_new_orig = max(0.0, old_orig - amount)
        if abs(new_orig - expected_new_orig) > amount_tolerance:
            rules.append("Sender ending balance does not reconcile with the transaction amount.")
        if abs(sender_spent - amount) > amount_tolerance and not (amount > old_orig and new_orig <= tolerance):
            rules.append("Sender before/after balance change does not match the transaction amount.")

    if tx_type in receiver_credit_required:
        expected_new_dest = old_dest + amount
        if abs(new_dest - expected_new_dest) > amount_tolerance:
            rules.append("Receiver ending balance does not reconcile with the transaction amount.")
        if receiver_delta < amount - amount_tolerance:
            rules.append("Receiver did not receive the transaction amount.")
        if abs(receiver_received - amount) > amount_tolerance:
            rules.append("Receiver before/after balance change does not match the transaction amount.")

    if tx_type in receiver_credit_optional and receiver_delta > tolerance:
        expected_new_dest = old_dest + amount
        if abs(new_dest - expected_new_dest) > amount_tolerance:
            rules.append("Receiver ending balance is inconsistent with the transaction amount.")
        if abs(receiver_received - amount) > amount_tolerance:
            rules.append("Receiver before/after balance change does not match the transaction amount.")

    if tx_type == "DEBIT" and receiver_delta + tolerance < amount:
        rules.append("Debit transactions should move funds to the receiver.")
    if tx_type == "CASH_IN" and sender_delta + tolerance < amount:
        rules.append("Cash-in transactions should increase the sender balance by the transaction amount.")
    if tx_type == "CASH_IN":
        if abs(sender_delta - amount) > amount_tolerance:
            rules.append("Sender before/after balance change does not match the transaction amount.")
    if tx_type == "PAYMENT" and abs(sender_spent - amount) > amount_tolerance:
        rules.append("Sender before/after balance change does not match the transaction amount.")
    if tx_type == "DEBIT":
        if abs(sender_spent - amount) > amount_tolerance:
            rules.append("Sender before/after balance change does not match the transaction amount.")
        if abs(receiver_received - amount) > amount_tolerance:
            rules.append("Receiver before/after balance change does not match the transaction amount.")

    net_flow_error = abs((sender_delta + receiver_delta))
    if tx_type in {"TRANSFER", "DEBIT", "PAYMENT"} and net_flow_error > amount_tolerance:
        rules.append("Funds do not balance between sender and receiver for this transaction.")

    return list(dict.fromkeys(rules))


def classify_transaction(
    input_dict: dict,
    prob: float,
    threshold: float,
    margin: float = 0.1,
) -> dict:
    """Classify transaction as fraud, review, or legitimate."""
    triggered_rules = detect_inconsistency_rules(input_dict)
    if triggered_rules:
        return {
            "status": "Fraud Detected",
            "color": COLORS["fraud"],
            "detail": "Hard validation rules found impossible balance transitions.",
            "confidence_band": "Rule Triggered",
            "triggered_rules": triggered_rules,
        }

    lower = max(0.0, threshold - margin)
    upper = min(1.0, threshold + margin)

    if prob >= upper:
        return {
            "status": "Fraud Detected",
            "color": COLORS["fraud"],
            "detail": "The calibrated fraud score is comfortably above the decision threshold.",
            "confidence_band": "High Confidence Fraud",
            "triggered_rules": [],
        }
    if prob >= lower:
        return {
            "status": "Review Recommended",
            "color": COLORS["accent"],
            "detail": "The score is too close to the decision threshold for a confident automated verdict.",
            "confidence_band": "Review Recommended",
            "triggered_rules": [],
        }
    return {
        "status": "Legitimate",
        "color": COLORS["legit"],
        "detail": "The calibrated fraud score is below the review band.",
        "confidence_band": "Low Confidence Legitimate" if prob >= max(0.0, lower / 2) else "Legitimate",
        "triggered_rules": [],
    }


_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="IBM Plex Mono, monospace"),
    margin=dict(t=50, b=40, l=40, r=20),
)


@st.cache_data(show_spinner=False)
def class_balance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate class balance summary."""
    tracker = get_tracker()
    tracker.start("Class Balance Summary")
    counts = df["isFraud"].astype(int).value_counts().reindex([0, 1], fill_value=0)
    total = counts.sum()
    result = pd.DataFrame(
        [
            {
                "label": "Legitimate",
                "count": int(counts.loc[0]),
                "pct": float(counts.loc[0] / total * 100) if total else 0.0,
            },
            {
                "label": "Fraud",
                "count": int(counts.loc[1]),
                "pct": float(counts.loc[1] / total * 100) if total else 0.0,
            },
        ]
    )
    tracker.end("Class Balance Summary")
    return result


def chart_class_balance(df: pd.DataFrame) -> go.Figure:
    """Create class balance bar chart."""
    summary = class_balance_summary(df)
    fig = go.Figure(
        go.Bar(
            x=summary["label"],
            y=summary["count"],
            marker_color=[COLORS["legit"], COLORS["fraud"]],
            text=[f"{v:,}" for v in summary["count"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Current Class Counts",
        xaxis_title="Class",
        yaxis_title="Transaction Count",
        showlegend=False,
        **_LAYOUT,
    )
    return fig


def chart_class_share(df: pd.DataFrame) -> go.Figure:
    """Create class share pie chart."""
    summary = class_balance_summary(df)
    fig = go.Figure(
        go.Pie(
            labels=summary["label"],
            values=summary["count"],
            hole=0.55,
            marker=dict(colors=[COLORS["legit"], COLORS["fraud"]]),
            textinfo="label+percent",
        )
    )
    fig.update_layout(title="Current Class Share", **_LAYOUT)
    return fig


def chart_category_fraud_rate(df: pd.DataFrame) -> go.Figure:
    """Create fraud rate by transaction type chart."""
    rates = (
        df.groupby("type")["isFraud"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "fraud_rate", "count": "total"})
        .sort_values("fraud_rate", ascending=True)
    )
    rates["fraud_pct"] = (rates["fraud_rate"] * 100).round(2)

    fig = go.Figure(
        go.Bar(
            x=rates["fraud_pct"],
            y=rates["type"],
            orientation="h",
            marker=dict(
                color=rates["fraud_pct"],
                colorscale=[[0, COLORS["legit"]], [1, COLORS["fraud"]]],
                showscale=True,
                colorbar=dict(title="%"),
            ),
            text=[f"{v:.2f}%" for v in rates["fraud_pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Fraud Rate by Transaction Type",
        xaxis_title="Fraud Rate (%)",
        **_LAYOUT,
    )
    return fig


def chart_hourly_trend(df: pd.DataFrame) -> go.Figure:
    """Create hourly fraud trend chart."""
    hourly = df.groupby("hour_of_day")["isFraud"].agg(["sum", "count"]).reset_index()
    hourly["fraud_rate"] = hourly["sum"] / hourly["count"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hourly["hour_of_day"],
            y=hourly["fraud_rate"],
            mode="lines+markers",
            line=dict(color=COLORS["fraud"], width=2.5),
            marker=dict(size=7, color=COLORS["accent"]),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.15)",
            name="Fraud Rate %",
        )
    )
    fig.update_layout(
        title="Fraud Rate by Hour of Day",
        xaxis_title="Hour (0-23)",
        yaxis_title="Fraud Rate (%)",
        **_LAYOUT,
    )
    return fig


def chart_top_high_risk(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Create top high-risk customers chart."""
    tracker = get_tracker()
    tracker.start("Top High-Risk Customers Chart")
    
    top = (
        df[df["isFraud"] == 1]
        .groupby("nameOrig")
        .size()
        .reset_index(name="fraud_count")
        .sort_values("fraud_count", ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(
        go.Bar(
            x=top["fraud_count"],
            y=top["nameOrig"],
            orientation="h",
            marker_color=COLORS["fraud"],
            text=top["fraud_count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} High-Risk Customers",
        xaxis_title="Fraud Transaction Count",
        **_LAYOUT,
    )
    
    tracker.end("Top High-Risk Customers Chart")
    return fig


def chart_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create correlation heatmap."""
    tracker = get_tracker()
    tracker.start("Correlation Heatmap")
    
    num_cols = [
        "amount",
        "log_amount",
        "orig_balance_ratio",
        "dest_balance_ratio",
        "balance_discrepancy_orig",
        "balance_discrepancy_dest",
        "orig_zeroed_out",
        "dest_was_zero",
        "hour_of_day",
        "isFraud",
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr().round(2)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=10),
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(title="Feature Correlation Heatmap", height=550, **_LAYOUT)
    
    tracker.end("Correlation Heatmap")
    return fig


def chart_confusion_matrix(cm: list) -> go.Figure:
    """Create confusion matrix heatmap."""
    z = pd.DataFrame(cm).values
    z_text = [[str(v) for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale=[[0, COLORS["legit"]], [1, COLORS["fraud"]]],
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix", **_LAYOUT)
    return fig


def chart_normalized_confusion_matrix(cm: list) -> go.Figure:
    """Create normalized confusion matrix heatmap."""
    z = pd.DataFrame(cm).values.astype(float)
    row_sums = z.sum(axis=1, keepdims=True)
    norm = z / row_sums.clip(min=1.0)
    z_text = [[f"{(v * 100):.2f}%" for v in row] for row in norm]

    fig = go.Figure(
        go.Heatmap(
            z=norm,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
            colorscale=[
                [0.0, "#0B132B"],
                [0.35, "#1C2541"],
                [0.6, "#3A506B"],
                [0.8, "#5BC0BE"],
                [1.0, "#CDEB8B"],
            ],
            zmin=0,
            zmax=1,
            text=z_text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="#F8FAFC"),
            colorbar=dict(title="Share", tickformat=".0%"),
        )
    )
    fig.update_layout(title="Normalized Confusion Matrix", **_LAYOUT)
    return fig


def chart_roc_curve(y_test: list, y_prob: list) -> go.Figure:
    """Create ROC curve."""
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {roc_auc:.4f})",
            line=dict(color=COLORS["accent"], width=2.5),
            fill="tozeroy",
            fillcolor="rgba(245,166,35,0.1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color=COLORS["muted"], width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        **_LAYOUT,
    )
    return fig


def chart_feature_importance(feature_names: list, importances: list, top_n: int = 20) -> go.Figure:
    """Create feature importance chart."""
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(
        go.Bar(
            x=fi["importance"],
            y=fi["feature"],
            orientation="h",
            marker=dict(
                color=fi["importance"],
                colorscale=[[0, COLORS["primary"]], [1, COLORS["accent"]]],
                showscale=False,
            ),
            text=[f"{v:.0f}" for v in fi["importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances (LightGBM)",
        xaxis_title="Importance Score",
        height=600,
        **_LAYOUT,
    )
    return fig


def chart_probability_gauge(prob: float) -> go.Figure:
    """Create probability gauge chart."""
    color = COLORS["legit"] if prob < 0.3 else (COLORS["accent"] if prob < 0.6 else COLORS["fraud"])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number=dict(suffix="%", font=dict(size=36, color=color)),
            delta=dict(reference=50, valueformat=".1f"),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=COLORS["text"]),
                bar=dict(color=color, thickness=0.3),
                bgcolor=COLORS["card"],
                borderwidth=0,
                steps=[
                    dict(range=[0, 30], color="rgba(0,196,154,0.15)"),
                    dict(range=[30, 60], color="rgba(245,166,35,0.15)"),
                    dict(range=[60, 100], color="rgba(255,75,75,0.15)"),
                ],
                threshold=dict(
                    line=dict(color=color, width=4),
                    thickness=0.75,
                    value=prob * 100,
                ),
            ),
            title=dict(text="Fraud Probability", font=dict(size=18, color=COLORS["text"])),
            domain=dict(x=[0, 1], y=[0, 1]),
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text"], family="IBM Plex Mono, monospace"),
        height=280,
        margin=dict(t=60, b=10, l=20, r=20),
    )
    return fig


def generate_csv_report(df: pd.DataFrame) -> bytes:
    """Generate fraud report as CSV bytes."""
    cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "isFraud",
    ]
    available = [c for c in cols if c in df.columns]
    fraud_df = df[df["isFraud"] == 1][available].sort_values("amount", ascending=False)
    return fraud_df.to_csv(index=False).encode("utf-8")
