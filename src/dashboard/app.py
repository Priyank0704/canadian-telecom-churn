"""
Streamlit Retention ROI Calculator.

A three-tab dashboard for the Canadian Telecom Churn project:
    Tab 1 — Portfolio Overview: customer base composition, churn distribution
    Tab 2 — Retention Targeting Simulator: the headline interactive widget
    Tab 3 — Individual Customer Lookup: explain a single customer's risk

Run with:
    streamlit run src/dashboard/app.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import (
    AVG_LIFETIME_MONTHS,
    AVG_MONTHLY_ARPU_CAD,
    DEFAULT_RETENTION_OFFER_CAD,
    GROSS_MARGIN,
    MODELS_DIR,
)
from src.features.build_features import build_modelling_matrix, load_segmented

# ---------------------------------------------------------------------------
# Page config + style
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Canadian Telecom Churn — Retention ROI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "primary":   "#534AB7",
    "stay":      "#1D9E75",
    "churn":     "#D85A30",
    "neutral":   "#888780",
    "highlight": "#185FA5",
}


# ---------------------------------------------------------------------------
# Cached loaders — Streamlit reruns scripts top-to-bottom on every interaction,
# so we cache expensive operations
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    feature_names = json.loads((MODELS_DIR / "feature_names.json").read_text())
    model_card = json.loads((MODELS_DIR / "model_card.json").read_text())
    return model, feature_names, model_card


@st.cache_data
def load_scored_customers():
    """Score every customer once, then cache."""
    model, feature_names, _ = load_model_artifacts()
    df_full = load_segmented()
    X, y, _ = build_modelling_matrix(df_full)
    X = X[feature_names]
    df_full["churn_prob"] = model.predict_proba(X)[:, 1]
    return df_full


# ---------------------------------------------------------------------------
# Sidebar — business assumptions (user-editable)
# ---------------------------------------------------------------------------

st.sidebar.title("📊 Business assumptions")
st.sidebar.caption("Adjust these to see how the retention math shifts.")

avg_arpu = st.sidebar.number_input(
    "Average monthly ARPU (CAD)", min_value=20, max_value=200,
    value=int(AVG_MONTHLY_ARPU_CAD), step=5,
    help="Average revenue per user per month. Canadian wireless ARPU is ~$64.",
)
margin = st.sidebar.slider(
    "Gross margin", min_value=0.20, max_value=0.70,
    value=float(GROSS_MARGIN), step=0.05,
    help="Telecom carrier gross margins are typically 40-50%.",
)
lifetime = st.sidebar.slider(
    "Avg customer lifetime (months)", min_value=12, max_value=84,
    value=int(AVG_LIFETIME_MONTHS), step=6,
    help="Expected remaining lifetime if retained.",
)
offer_cost = st.sidebar.number_input(
    "Retention offer cost per customer (CAD)", min_value=10, max_value=300,
    value=int(DEFAULT_RETENTION_OFFER_CAD), step=10,
    help="Discount, credit, or service upgrade given to at-risk customers.",
)
offer_success = st.sidebar.slider(
    "Retention offer success rate", min_value=0.10, max_value=0.80,
    value=0.40, step=0.05,
    help="Fraction of contacted at-risk customers who accept and stay. "
          "Industry benchmark: 30-50%.",
)

clv = avg_arpu * margin * lifetime
st.sidebar.metric("Customer lifetime value (CLV)", f"${clv:,.0f}")
st.sidebar.metric("CLV-to-offer ratio", f"{clv / offer_cost:.1f}×")


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("📡 Canadian Telecom Churn — Retention ROI Calculator")
st.caption(
    "A churn risk scoring tool with an interactive retention targeting simulator. "
    "Built on IBM Telco Churn data with Canadian market context "
    "(CRTC complaint rates, StatCan provincial spending)."
)


# ---------------------------------------------------------------------------
# Load data + model
# ---------------------------------------------------------------------------

with st.spinner("Loading model and scoring customer base..."):
    model, feature_names, model_card = load_model_artifacts()
    df = load_scored_customers()

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 Portfolio Overview",
    "🎯 Retention Targeting Simulator",
    "🔍 Individual Customer Lookup",
])


# ---------------------------------------------------------------------------
# TAB 1 — Portfolio Overview
# ---------------------------------------------------------------------------

with tab1:
    st.subheader("Customer base at a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total customers", f"{len(df):,}")
    c2.metric("Currently churned", f"{df['churn_flag'].sum():,}",
              f"{df['churn_flag'].mean():.1%} of base")
    c3.metric("Predicted high-risk (top decile)",
              f"{int(len(df) * 0.10):,}")
    c4.metric("Model ROC-AUC",
              f"{model_card['test_metrics']['roc_auc']:.3f}")

    st.divider()

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Churn probability distribution**")
        fig = px.histogram(
            df, x="churn_prob", nbins=40, color="Churn",
            color_discrete_map={"Yes": PALETTE["churn"], "No": PALETTE["stay"]},
        )
        fig.update_layout(
            bargap=0.05, height=350,
            legend_title_text="Actual churn",
            xaxis_title="Predicted churn probability",
            yaxis_title="Customers",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Churn rate by RFM segment**")
        seg_stats = (
            df.groupby("segment_name")
              .agg(n=("churn_flag", "size"),
                   churn_rate=("churn_flag", "mean"),
                   avg_prob=("churn_prob", "mean"))
              .reset_index()
              .sort_values("churn_rate", ascending=True)
        )
        fig = go.Figure(go.Bar(
            x=seg_stats["churn_rate"], y=seg_stats["segment_name"],
            orientation="h", marker=dict(color=PALETTE["primary"]),
            text=[f"{r:.1%} (n={n:,})" for r, n in
                   zip(seg_stats["churn_rate"], seg_stats["n"])],
            textposition="outside",
        ))
        fig.update_layout(
            height=350, xaxis_title="Actual churn rate",
            yaxis_title="", margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Customer mix by province and provider**")
    col_a, col_b = st.columns(2)
    with col_a:
        prov_stats = df.groupby("province_code").agg(
            customers=("churn_flag", "size"),
            churn_rate=("churn_flag", "mean"),
        ).reset_index().sort_values("customers", ascending=False)
        fig = px.bar(prov_stats, x="province_code", y="customers",
                      color="churn_rate", color_continuous_scale="OrRd",
                      labels={"province_code": "Province",
                              "customers": "Customers",
                              "churn_rate": "Churn rate"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        prov_stats = df.groupby("provider").agg(
            customers=("churn_flag", "size"),
            churn_rate=("churn_flag", "mean"),
        ).reset_index().sort_values("customers", ascending=False)
        fig = px.bar(prov_stats, x="provider", y="customers",
                      color="churn_rate", color_continuous_scale="OrRd",
                      labels={"provider": "Provider", "customers": "Customers",
                              "churn_rate": "Churn rate"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 2 — Retention Targeting Simulator (the headline widget)
# ---------------------------------------------------------------------------

with tab2:
    st.subheader("How much budget should I spend on retention this quarter?")
    st.caption(
        "Set a targeting threshold and see the expected net savings. "
        "The model predicts churn probability for every customer; we contact the "
        "top X% by risk, spend the offer cost on each, and assume a fraction accept."
    )

    target_pct = st.slider(
        "Target top X% of customers by predicted churn probability",
        min_value=1, max_value=50, value=10, step=1,
    )

    # Threshold corresponds to the X-th percentile of churn_prob
    threshold = df["churn_prob"].quantile(1 - target_pct / 100)
    targeted = df[df["churn_prob"] >= threshold].copy()

    # Within the targeted group, separate true positives from false positives
    # using the actual Churn label
    true_positives  = (targeted["churn_flag"] == 1).sum()
    false_positives = (targeted["churn_flag"] == 0).sum()
    total_targeted  = len(targeted)

    # Customers we'd save = TP × offer_success
    saved = true_positives * offer_success
    revenue_saved = saved * clv
    offer_spend   = total_targeted * offer_cost
    net_savings   = revenue_saved - offer_spend
    roi = (revenue_saved / offer_spend - 1) * 100 if offer_spend > 0 else 0

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers contacted", f"{total_targeted:,}",
              f"{target_pct}% of base")
    k2.metric("True churners reached", f"{true_positives:,}",
              f"{true_positives / df['churn_flag'].sum() * 100:.0f}% of all churners")
    k3.metric("Expected saves", f"{saved:,.0f}",
              f"{offer_success:.0%} success rate")
    k4.metric("Revenue retained", f"${revenue_saved:,.0f}")
    k5.metric("Net savings", f"${net_savings:,.0f}",
              f"ROI: {roi:+.0f}%",
              delta_color="normal" if net_savings >= 0 else "inverse")

    st.divider()

    # Sweep across all target_pct values to draw the curve
    sweep = []
    for pct in range(1, 51):
        thr = df["churn_prob"].quantile(1 - pct / 100)
        grp = df[df["churn_prob"] >= thr]
        tp = (grp["churn_flag"] == 1).sum()
        sv = tp * offer_success
        rev = sv * clv
        cost = len(grp) * offer_cost
        sweep.append({
            "target_pct": pct,
            "net_savings": rev - cost,
            "roi": (rev / cost - 1) * 100 if cost > 0 else 0,
            "precision": tp / len(grp) if len(grp) > 0 else 0,
        })
    sweep_df = pd.DataFrame(sweep)
    best_pct = int(sweep_df.loc[sweep_df["net_savings"].idxmax(), "target_pct"])

    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("**Net savings by targeting depth**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sweep_df["target_pct"], y=sweep_df["net_savings"],
            mode="lines", line=dict(color=PALETTE["primary"], width=3),
            fill="tozeroy", fillcolor="rgba(83, 74, 183, 0.15)",
            name="Net savings",
        ))
        fig.add_vline(x=target_pct, line_dash="dash",
                       line_color=PALETTE["churn"], line_width=2,
                       annotation_text=f"You: {target_pct}%",
                       annotation_position="top")
        fig.add_vline(x=best_pct, line_dash="dot",
                       line_color=PALETTE["stay"], line_width=2,
                       annotation_text=f"Optimal: {best_pct}%",
                       annotation_position="bottom")
        fig.update_layout(
            height=400, xaxis_title="Target top X% by churn probability",
            yaxis_title="Expected net savings (CAD)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**Precision of targeting**")
        st.caption(
            "How many of the customers we contact are actual churners. "
            "Tighter targeting (smaller X) = higher precision but fewer saves."
        )
        fig = go.Figure(go.Scatter(
            x=sweep_df["target_pct"], y=sweep_df["precision"],
            mode="lines", line=dict(color=PALETTE["highlight"], width=3),
        ))
        fig.add_hline(y=df["churn_flag"].mean(), line_dash="dash",
                       line_color=PALETTE["neutral"], line_width=1.5,
                       annotation_text=f"Random ({df['churn_flag'].mean():.1%})",
                       annotation_position="bottom right")
        fig.update_layout(
            height=400, xaxis_title="Target top X%",
            yaxis_title="Precision (true churners ÷ contacted)",
            yaxis_tickformat=".0%",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.success(
        f"💡 **Optimal targeting:** contact the top **{best_pct}%** by predicted churn "
        f"probability. Expected net savings = "
        f"**${sweep_df['net_savings'].max():,.0f}** "
        f"(ROI: {sweep_df.loc[sweep_df['net_savings'].idxmax(), 'roi']:+.0f}%)."
    )


# ---------------------------------------------------------------------------
# TAB 3 — Individual Customer Lookup
# ---------------------------------------------------------------------------

with tab3:
    st.subheader("Look up a customer and see why the model rates them")

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        prov_filter = st.selectbox(
            "Province", ["All"] + sorted(df["province_code"].unique().tolist()),
        )
    with col_filter2:
        provider_filter = st.selectbox(
            "Provider", ["All"] + sorted(df["provider"].unique().tolist()),
        )
    with col_filter3:
        risk_filter = st.selectbox(
            "Risk band", ["All", "High (>50%)", "Medium (20-50%)", "Low (<20%)"],
        )

    filtered = df.copy()
    if prov_filter != "All":
        filtered = filtered[filtered["province_code"] == prov_filter]
    if provider_filter != "All":
        filtered = filtered[filtered["provider"] == provider_filter]
    if risk_filter == "High (>50%)":
        filtered = filtered[filtered["churn_prob"] > 0.5]
    elif risk_filter == "Medium (20-50%)":
        filtered = filtered[(filtered["churn_prob"] >= 0.2) & (filtered["churn_prob"] <= 0.5)]
    elif risk_filter == "Low (<20%)":
        filtered = filtered[filtered["churn_prob"] < 0.2]

    st.write(f"**{len(filtered):,} customers match these filters.**")

    if len(filtered) == 0:
        st.warning("No customers match. Loosen the filters.")
    else:
        display_cols = [
            "customerID", "provider", "province_code", "tenure",
            "Contract", "MonthlyCharges", "PaymentMethod",
            "churn_prob", "segment_name",
        ]
        view = (
            filtered[display_cols]
            .sort_values("churn_prob", ascending=False)
            .head(50)
            .rename(columns={
                "customerID": "Customer ID",
                "provider": "Provider",
                "province_code": "Province",
                "tenure": "Tenure (mo)",
                "Contract": "Contract",
                "MonthlyCharges": "Monthly $",
                "PaymentMethod": "Payment",
                "churn_prob": "Churn prob",
                "segment_name": "Segment",
            })
        )
        st.dataframe(
            view.style
                .format({"Churn prob": "{:.1%}", "Monthly $": "${:.2f}"})
                .background_gradient(subset=["Churn prob"], cmap="OrRd"),
            use_container_width=True, hide_index=True,
        )
        st.caption("Showing top 50 by churn probability. Sort the table by clicking any column.")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    f"Model: {model_card['model_type'].upper()} | "
    f"Test AUC: {model_card['test_metrics']['roc_auc']:.3f} | "
    f"Test F1: {model_card['test_metrics']['f1']:.3f} | "
    f"Business-cost-optimal threshold: {model_card['threshold']:.2f}"
)
