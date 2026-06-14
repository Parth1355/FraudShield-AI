"""
app.py
======
Fraud Detection Web Application — Main Entry Point
Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    load_model, load_preprocessor, load_metrics,
    load_and_prepare_data, resolve_dataset_path,
    build_input_row, predict, risk_level, classify_transaction,
    chart_class_balance, chart_class_share,
    chart_category_fraud_rate, chart_hourly_trend,
    chart_top_high_risk, chart_correlation_heatmap,
    chart_confusion_matrix, chart_normalized_confusion_matrix, chart_roc_curve,
    chart_feature_importance, chart_probability_gauge,
    generate_csv_report, COLORS,
)
from performance import get_tracker

# ─────────────────────────────────────────────
# PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Imports ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --fraud:   #FF4B4B;
    --legit:   #00C49A;
    --primary: #1E3A5F;
    --accent:  #F5A623;
    --bg:      #0B0F1A;
    --card:    #131929;
    --border:  #1E2D45;
    --text:    #E8ECF0;
    --muted:   #8892A4;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1321 0%, #0B0F1A 100%);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    color: var(--muted);
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--accent);
}
div[data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.07em;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text) !important;
}

/* ── Cards ── */
.fraud-card {
    background: linear-gradient(135deg, #2D0B0B 0%, #1A0808 100%);
    border: 1px solid var(--fraud);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.legit-card {
    background: linear-gradient(135deg, #042918 0%, #031A10 100%);
    border: 1px solid var(--legit);
    border-radius: 14px;
    padding: 24px;
    text-align: center;
}
.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
}

/* ── Section headings ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 6px;
}
.page-title {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.15;
    margin-bottom: 4px;
}
.page-sub {
    font-size: 1rem;
    color: var(--muted);
    margin-bottom: 32px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
}
.stTabs [aria-selected="true"] {
    background: var(--card);
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #F5A623 0%, #E8910A 100%);
    color: #0B0F1A;
    border: none;
    border-radius: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button:hover {
    box-shadow: 0 0 20px rgba(245,166,35,0.4);
    transform: translateY(-1px);
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ── Divider ── */
hr { border-color: var(--border); }

/* ── Alert banners ── */
.stAlert {
    background: var(--card);
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* ── Probability bar ── */
.prob-bar-outer {
    background: var(--border);
    border-radius: 999px;
    height: 10px;
    margin: 8px 0 16px;
    overflow: hidden;
}
.prob-bar-inner {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 12px; text-align:center;'>
        <div style='font-size:2.4rem;'>🛡️</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:1.05rem;
                    font-weight:600; color:#E8ECF0; letter-spacing:0.04em;'>
            FraudShield AI
        </div>
        <div style='font-size:0.72rem; color:#8892A4; margin-top:2px;
                    font-family:"IBM Plex Mono",monospace; letter-spacing:0.12em;'>
            v1.0 · DETECTION ENGINE
        </div>
    </div>
    <hr style='border-color:#1E2D45; margin:0 0 20px;'/>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠  Home",
         "📊  Analytics Dashboard",
         "🔍  Fraud Prediction",
         "📈  Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1E2D45; margin:20px 0 14px;'/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.68rem;
                color:#8892A4; letter-spacing:0.08em; padding: 0 0 10px;'>
        DATASET<br>
        <span style='color:#E8ECF0;'>IBM Synthetic Financial</span><br><br>
        MODEL<br>
        <span style='color:#E8ECF0;'>LightGBM Classifier</span><br><br>
        BUILT WITH<br>
        <span style='color:#E8ECF0;'>Streamlit · Plotly · sklearn</span>
    </div>
    """, unsafe_allow_html=True)

    # Data upload
    st.markdown("<hr style='border-color:#1E2D45; margin:8px 0 14px;'/>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📁 Dataset</div>', unsafe_allow_html=True)
    data_path = resolve_dataset_path()
    if data_path:
        st.success(f"Dataset loaded: `{data_path}`")
    else:
        st.warning("No dataset found.\nPlace `fraud_dataset.csv` in the project folder or `data/`.")


# ─────────────────────────────────────────────
# LOAD SHARED RESOURCES
# ─────────────────────────────────────────────
df      = load_and_prepare_data()
model   = load_model()
prep    = load_preprocessor()
metrics = load_metrics()

DATA_OK  = df is not None
MODEL_OK = (model is not None) and (prep is not None)

# Display performance metrics in sidebar
tracker = get_tracker()
with st.sidebar:
    st.markdown("<hr style='border-color:#1E2D45; margin:20px 0 14px;'/>", unsafe_allow_html=True)
    tracker.display_in_streamlit(location="sidebar")


# ─────────────────────────────────────────────
# PAGE: HOME
# ─────────────────────────────────────────────
if page == "🏠  Home":
    st.markdown('<div class="section-title">FRAUD DETECTION PLATFORM</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">FraudShield AI 🛡️</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time financial fraud detection powered by LightGBM</div>', unsafe_allow_html=True)

    # KPI banner (live from data)
    if DATA_OK:
        c1, c2, c3, c4 = st.columns(4)
        total      = len(df)
        fraud_n    = int(df["isFraud"].sum())
        fraud_pct  = fraud_n / total * 100
        fraud_amt  = df[df["isFraud"] == 1]["amount"].sum()

        c1.metric("📋 Total Transactions",  f"{total:,}")
        c2.metric("🚨 Fraud Cases",         f"{fraud_n:,}")
        c3.metric("⚠️ Fraud Rate",          f"{fraud_pct:.3f}%")
        c4.metric("💸 Fraud Amount",        f"${fraud_amt:,.0f}")

    st.markdown("---")

    # Feature highlights
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:1.5rem; margin-bottom:8px;'>🎯 Problem Statement</div>
            <p style='color:#8892A4; font-size:0.92rem; line-height:1.7;'>
            Financial fraud costs the global economy hundreds of billions of dollars annually.
            Traditional rule-based systems miss novel fraud patterns and generate excessive
            false positives — disrupting legitimate customers. <br><br>
            FraudShield AI uses gradient-boosted trees trained on transactional and
            balance-derived signals to catch fraud with high recall while minimising
            false alarms.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div style='font-size:1.5rem; margin-bottom:8px;'>⚙️ Tech Stack</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
                        line-height:2; color:#8892A4;'>
            🟡 LightGBM &nbsp;&nbsp; → &nbsp; Primary classifier<br>
            🔵 scikit-learn → &nbsp; Preprocessing pipeline<br>
            🟠 Streamlit &nbsp;&nbsp; → &nbsp; Web application<br>
            🟣 Plotly &nbsp;&nbsp;&nbsp;&nbsp; → &nbsp; Interactive charts<br>
            🟢 Class weighting → Imbalance handling
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <div style='font-size:1.5rem; margin-bottom:8px;'>✨ Key Features</div>
            <ul style='color:#8892A4; font-size:0.92rem; line-height:2.0; padding-left:20px;'>
                <li>Shared deterministic feature engineering for training and inference</li>
                <li>Interactive analytics dashboard with Plotly</li>
                <li>Real-time single-transaction prediction</li>
                <li>Adjustable fraud probability threshold</li>
                <li>Risk scoring: Low / Medium / High</li>
                <li>ROC curve, confusion matrix, feature importances</li>
                <li>One-click fraud report CSV export</li>
                <li>Streamlit caching for fast performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <div style='font-size:1.5rem; margin-bottom:10px;'>🗺️ How to Navigate</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
                        line-height:2.1; color:#8892A4;'>
            📊 Analytics Dashboard &nbsp;→ Explore the data<br>
            🔍 Fraud Prediction &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Test a transaction<br>
            📈 Model Insights &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Understand the model
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if not DATA_OK:
        st.warning("⚠️ Place your IBM fraud CSV in the project folder or `data/` and refresh.")
    if not MODEL_OK:
        st.info("ℹ️ Run `python pipeline.py --no-app` to train and save the model.")


# ─────────────────────────────────────────────
# PAGE: ANALYTICS DASHBOARD
# ─────────────────────────────────────────────
elif page == "📊  Analytics Dashboard":
    st.markdown('<div class="section-title">EXPLORATORY ANALYSIS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analytics Dashboard 📊</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Interactive exploration of transaction patterns, class balance, and rule-relevant signals</div>', unsafe_allow_html=True)

    if not DATA_OK:
        st.error("Dataset not found. Place `fraud_dataset.csv` in the project folder or `data/`.")
        st.stop()

    # ── Filters ──
    with st.expander("🔧 Filter Options", expanded=False):
        col_f1, col_f2, col_f3 = st.columns(3)
        tx_types = sorted(df["type"].unique().tolist())
        sel_types = col_f1.multiselect("Transaction Types", tx_types, default=tx_types)

        amt_min, amt_max = float(df["amount"].min()), float(df["amount"].max())
        amt_range = col_f2.slider("Amount Range ($)", amt_min, amt_max, (amt_min, amt_max),
                                   format="$%.0f")

        show_fraud_only = col_f3.checkbox("Show Fraud Only", value=False)

    filtered = df[df["type"].isin(sel_types)]
    filtered = filtered[filtered["amount"].between(amt_range[0], amt_range[1])]
    if show_fraud_only:
        filtered = filtered[filtered["isFraud"] == 1]

    st.markdown(f"*Showing **{len(filtered):,}** transactions after filters*")
    st.info("This page describes the filtered dataset slice. Training uses class weighting and calibrated probabilities, while the app adds hard balance-consistency rules at prediction time.")

    # ── KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Rows",      f"{len(filtered):,}")
    c2.metric("Fraud Rows",      f"{int(filtered['isFraud'].sum()):,}")
    c3.metric("Fraud Rate",      f"{filtered['isFraud'].mean()*100:.3f}%")
    c4.metric("Avg Txn Amount",  f"${filtered['amount'].mean():,.2f}")
    c5.metric("Max Txn Amount",  f"${filtered['amount'].max():,.0f}")

    st.markdown("---")

    # ── Charts Row 1 ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(chart_class_balance(filtered), use_container_width=True)
    with col_b:
        st.plotly_chart(chart_class_share(filtered), use_container_width=True)

    # ── Charts Row 2 ──
    col_c, col_d = st.columns(2)
    with col_c:
        st.plotly_chart(chart_category_fraud_rate(filtered), use_container_width=True)
    with col_d:
        st.plotly_chart(chart_hourly_trend(filtered), use_container_width=True)

    # ── Charts Row 3 ──
    col_e, col_f = st.columns(2)
    with col_e:
        top_n = st.slider("Top N High-Risk Customers", 5, 30, 10)
        with st.expander("📊 Top High-Risk Customers (click to load)", expanded=False):
            st.plotly_chart(chart_top_high_risk(filtered, top_n), use_container_width=True)
    with col_f:
        with st.expander("📊 Correlation Heatmap (click to load)", expanded=False):
            st.plotly_chart(chart_correlation_heatmap(filtered), use_container_width=True)

    st.markdown("---")

    # ── Raw Data Preview ──
    with st.expander("🗃️ Raw Data Preview (first 500 rows)"):
        preview_cols = [c for c in ["step","type","amount","nameOrig","nameDest",
                                     "oldbalanceOrg","newbalanceOrig","isFraud"] if c in filtered.columns]
        st.dataframe(filtered[preview_cols].head(500), use_container_width=True)

    # ── Download Report ──
    st.markdown("---")
    st.markdown("### 📥 Download Report")
    st.caption("Exports fraud-labeled rows from the current dataset view. This is a data extract, not a model prediction report.")
    csv_bytes = generate_csv_report(filtered)
    st.download_button(
        label="⬇️ Download Fraud Transactions CSV",
        data=csv_bytes,
        file_name="fraud_transactions_report.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# PAGE: FRAUD PREDICTION
# ─────────────────────────────────────────────
elif page == "🔍  Fraud Prediction":
    st.markdown('<div class="section-title">REAL-TIME SCORING</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Fraud Prediction 🔍</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Enter transaction details to get an instant fraud risk assessment</div>', unsafe_allow_html=True)

    if not MODEL_OK:
        st.error("Model not loaded. Run `python pipeline.py --no-app` first.")
        st.stop()

    default_threshold = 0.50
    if metrics is not None and metrics.get("recommended_threshold") is not None:
        default_threshold = float(metrics["recommended_threshold"])

    # ── Threshold slider ──
    with st.expander("⚙️ Advanced Settings", expanded=False):
        threshold = st.slider(
            "Fraud Probability Threshold",
            min_value=0.01, max_value=0.99, value=float(round(default_threshold, 2)), step=0.01,
            help="Transactions with probability ≥ threshold are flagged as Fraud.",
        )
        st.caption(f"Current threshold: **{threshold:.2f}** — Lower = more sensitive (higher recall), higher = more precise")
        if metrics is not None and metrics.get("recommended_threshold") is not None:
            st.caption(f"Recommended threshold from validation: **{metrics['recommended_threshold']:.4f}**")

    st.markdown("---")
    st.markdown("### 📝 Transaction Details")

    # ── Input form ──
    col1, col2, col3 = st.columns(3)

    with col1:
        tx_type = st.selectbox("Transaction Type",
                               ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
        step    = st.number_input("Step (hour since start)", min_value=1, max_value=744, value=1)
        amount  = st.number_input("Transaction Amount ($)", min_value=0.01, value=1000.0, step=100.0)

    with col2:
        old_bal_orig = st.number_input("Sender: Old Balance ($)", min_value=0.0, value=5000.0, step=100.0)
        new_bal_orig = st.number_input("Sender: New Balance ($)", min_value=0.0, value=4000.0, step=100.0)

    with col3:
        old_bal_dest = st.number_input("Receiver: Old Balance ($)", min_value=0.0, value=0.0, step=100.0)
        new_bal_dest = st.number_input("Receiver: New Balance ($)", min_value=0.0, value=0.0, step=100.0)

    st.markdown("---")

    # ── Predict button ──
    run_col, _ = st.columns([1, 3])
    run_pred = run_col.button("🔍  Analyse Transaction", use_container_width=True)

    if run_pred:
        input_dict = {
            "type":            tx_type,
            "step":            step,
            "amount":          amount,
            "oldbalanceOrg":   old_bal_orig,
            "newbalanceOrig":  new_bal_orig,
            "oldbalanceDest":  old_bal_dest,
            "newbalanceDest":  new_bal_dest,
        }

        row_df = build_input_row(input_dict)

        try:
            _label, prob = predict(model, prep, row_df, threshold=threshold)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        risk_text, risk_color = risk_level(prob)
        decision = classify_transaction(input_dict, prob, threshold)
        decision_text = decision["status"]
        decision_color = decision["color"]
        decision_detail = decision["detail"]
        confidence_band = decision["confidence_band"]
        triggered_rules = decision["triggered_rules"]

        # ── Result card ──
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if decision_text == "Fraud Detected":
                st.markdown(f"""
                <div class="fraud-card">
                    <div style='font-size:3rem;'>🚨</div>
                    <div style='font-size:1.8rem; font-weight:700; color:#FF4B4B; margin:8px 0;'>
                        FRAUD DETECTED
                    </div>
                    <div style='font-size:1.1rem; color:#FF9090;'>
                        {decision_detail}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif decision_text == "Review Recommended":
                st.markdown(f"""
                <div class="info-card" style="text-align:center; border-color:{decision_color};">
                    <div style='font-size:3rem;'>🟡</div>
                    <div style='font-size:1.8rem; font-weight:700; color:{decision_color}; margin:8px 0;'>
                        REVIEW RECOMMENDED
                    </div>
                    <div style='font-size:1.05rem; color:#E8ECF0; line-height:1.6;'>
                        {decision_detail}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-card">
                    <div style='font-size:3rem;'>✅</div>
                    <div style='font-size:1.8rem; font-weight:700; color:#00C49A; margin:8px 0;'>
                        LEGITIMATE
                    </div>
                    <div style='font-size:1.1rem; color:#80E8D0;'>
                        {decision_detail}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if triggered_rules:
                st.warning("Hard inconsistency checks triggered:")
                for rule in triggered_rules:
                    st.write(f"- {rule}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <div class="section-title">RISK LEVEL</div>
                <div style='font-size:1.5rem; font-weight:700; color:{risk_color};'>
                    {risk_text}
                </div>
                <div style='margin-top:10px; font-family:"IBM Plex Mono",monospace;
                            font-size:0.85rem; color:{decision_color};'>
                    Confidence Band: {confidence_band}
                </div>
                <div style='margin-top:10px; font-family:"IBM Plex Mono",monospace;
                            font-size:0.85rem; color:#8892A4;'>
                    Threshold: {threshold:.2f} · Probability: {prob:.4f} · Decision: {decision_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            fig_gauge = chart_probability_gauge(prob)
            st.plotly_chart(fig_gauge, use_container_width=True)

            bar_color = "#FF4B4B" if prob >= 0.6 else ("#F5A623" if prob >= 0.3 else "#00C49A")
            st.markdown(f"""
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.78rem;
                        color:#8892A4; margin-bottom:4px;'>
                FRAUD PROBABILITY: <span style='color:{bar_color};'>{prob*100:.2f}%</span>
            </div>
            <div class="prob-bar-outer">
                <div class="prob-bar-inner"
                     style='width:{prob*100:.1f}%; background:{bar_color};'></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔬 Feature Breakdown (engineered inputs)"):
            st.dataframe(row_df.T.rename(columns={0: "Value"}), use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: MODEL INSIGHTS
# ─────────────────────────────────────────────
elif page == "📈  Model Insights":
    st.markdown('<div class="section-title">MODEL DIAGNOSTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model Insights 📈</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Calibrated LightGBM metrics, feature importance, and the current decision policy</div>', unsafe_allow_html=True)

    if not MODEL_OK or metrics is None:
        st.error("Model artifacts not found. Run `python pipeline.py --no-app` first.")
        st.stop()

    # ── KPIs ──
    st.markdown("### 🏆 Model Performance")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall",    f"{metrics['recall']:.4f}")
    c4.metric("F1-Score",  f"{metrics['f1']:.4f}")
    c5.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")
    c6.metric("Threshold", f"{metrics.get('recommended_threshold', 0.5):.4f}")

    st.markdown("---")
    st.info("These metrics come from the calibrated model on the held-out test set. The live app also applies hard balance-consistency rules before showing a final decision.")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Feature Importance",
        "🗂️  Confusion Matrix",
        "📉  ROC Curve",
        "📋  Classification Report",
    ])

    with tab1:
        top_n_fi = st.slider("Show top N features", 5, len(metrics["feature_names"]),
                             min(20, len(metrics["feature_names"])))
        fig_fi = chart_feature_importance(
            metrics["feature_names"],
            metrics["feature_importances"],
            top_n=top_n_fi,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        fi_df = pd.DataFrame({
            "Feature":    metrics["feature_names"],
            "Importance": metrics["feature_importances"],
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        fi_df.index += 1
        with st.expander("📋 Full Feature Importance Table"):
            st.dataframe(fi_df, use_container_width=True)

    with tab2:
        col_cm, col_info = st.columns([1, 1])
        with col_cm:
            fig_cm = chart_confusion_matrix(metrics["confusion_matrix"])
            st.plotly_chart(fig_cm, use_container_width=True)
            fig_norm_cm = chart_normalized_confusion_matrix(metrics["confusion_matrix"])
            st.plotly_chart(fig_norm_cm, use_container_width=True)
        with col_info:
            cm = np.array(metrics["confusion_matrix"])
            tn, fp, fn, tp = cm.ravel()
            actual_legit = tn + fp
            actual_fraud = fn + tp
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="info-card">
                <div class="section-title">CONFUSION MATRIX BREAKDOWN</div>
                <div style='font-family:"IBM Plex Mono",monospace; font-size:0.88rem;
                            line-height:2.3; color:#8892A4;'>
                ✅ True Negatives (TN)  : <span style='color:#E8ECF0;'>{tn:,}</span><br>
                🚨 True Positives (TP)  : <span style='color:#00C49A;'>{tp:,}</span><br>
                ⚠️  False Positives (FP) : <span style='color:#F5A623;'>{fp:,}</span><br>
                ❌ False Negatives (FN) : <span style='color:#FF4B4B;'>{fn:,}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card">
                <div class="section-title">DERIVED METRICS</div>
                <div style='font-family:"IBM Plex Mono",monospace; font-size:0.88rem;
                            line-height:2.3; color:#8892A4;'>
                Specificity   : <span style='color:#E8ECF0;'>{(tn/(tn+fp) if (tn+fp) else 0):.4f}</span><br>
                Miss Rate (FNR): <span style='color:#FF4B4B;'>{(fn/(fn+tp) if (fn+tp) else 0):.4f}</span><br>
                Fall-out (FPR): <span style='color:#F5A623;'>{(fp/(fp+tn) if (fp+tn) else 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card">
                <div class="section-title">NORMALIZED VIEW</div>
                <div style='font-family:"IBM Plex Mono",monospace; font-size:0.88rem;
                            line-height:2.3; color:#8892A4;'>
                Legit correctly cleared : <span style='color:#E8ECF0;'>{((tn/actual_legit) if actual_legit else 0):.4f}</span><br>
                Legit falsely flagged   : <span style='color:#F5A623;'>{((fp/actual_legit) if actual_legit else 0):.4f}</span><br>
                Fraud correctly caught  : <span style='color:#00C49A;'>{((tp/actual_fraud) if actual_fraud else 0):.4f}</span><br>
                Fraud missed            : <span style='color:#FF4B4B;'>{((fn/actual_fraud) if actual_fraud else 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        fig_roc = chart_roc_curve(metrics["y_test"], metrics["y_prob"])
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info(f"**AUC = {metrics['roc_auc']:.4f}** — This reflects the calibrated model's ranking quality on the held-out test set.")
        if metrics.get("calibration_method"):
            st.caption(f"Probabilities are calibrated using `{metrics['calibration_method']}` scaling.")

    with tab4:
        rep = metrics.get("classification_report", {})
        if rep:
            report_rows = []
            for cls, vals in rep.items():
                if isinstance(vals, dict):
                    report_rows.append({
                        "Class":     cls,
                        "Precision": f"{vals.get('precision',0):.4f}",
                        "Recall":    f"{vals.get('recall',0):.4f}",
                        "F1-Score":  f"{vals.get('f1-score',0):.4f}",
                        "Support":   int(vals.get("support", 0)),
                    })
            rep_df = pd.DataFrame(report_rows)
            st.dataframe(rep_df, use_container_width=True, hide_index=True)
        else:
            st.info("Classification report not available in saved metrics.")

    st.markdown("---")

    with st.expander("🧭 Decision Policy"):
        st.markdown(
            f"""
            - Model probabilities are calibrated with `{metrics.get('calibration_method', 'none')}`.
            - Default app threshold: `{metrics.get('recommended_threshold', 0.5):.4f}`.
            - Hard balance-consistency rules override the model for impossible transactions.
            - The prediction UI surfaces confidence bands instead of only a binary verdict.
            """
        )

    with st.expander("🔧 Model Configuration"):
        st.markdown("""
        ```
        LGBMClassifier(
            num_leaves       = 63,
            max_depth        = 8,
            learning_rate    = 0.05,
            n_estimators     = 500,
            scale_pos_weight = <computed from class ratio>,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            reg_alpha        = 0.1,
            reg_lambda       = 0.1,
            min_child_samples= 20,
            random_state     = 42,
        )

        Probability calibration:
            method           = sigmoid
            threshold        = learned from validation PR curve
        ```
        """)
