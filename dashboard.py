import os

import duckdb
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
MOTHERDUCK_DB    = os.getenv("MOTHERDUCK_DB", "titanic_db")
MD_CONN_STR      = f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}"

st.set_page_config(
    page_title="🚢 Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide",
)


@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")
    df = conn.execute("SELECT * FROM predictions").df()
    conn.close()
    return df


@st.cache_data(ttl=60)
def load_monitoring() -> pd.DataFrame:
    conn = duckdb.connect(MD_CONN_STR)
    conn.execute(f"USE {MOTHERDUCK_DB}")
    df = conn.execute("SELECT * FROM monitoring_results").df()
    conn.close()
    return df


# ── Header ─────────────────────────────────────────────────
st.title("🚢 Titanic Survival Predictor — ML Dashboard")
st.caption("Batch predictions + Evidently monitoring results from MotherDuck")

# ── Load data ──────────────────────────────────────────────
with st.spinner("Loading data from MotherDuck..."):
    try:
        predictions_df = load_predictions()
        monitoring_df  = load_monitoring()
        st.success(f"✅ Loaded {len(predictions_df)} predictions and {len(monitoring_df)} monitoring metrics")
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        st.stop()

# ── Tab layout ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "🔍 Data Drift", "📋 Raw Data"])

# ── Tab 1: Predictions ─────────────────────────────────────
with tab1:
    st.subheader("Prediction Summary")

    survived     = int((predictions_df["prediction"] == 1).sum())
    not_survived = int((predictions_df["prediction"] == 0).sum())
    total        = len(predictions_df)
    avg_prob     = predictions_df["survival_probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Passengers", total)
    col2.metric("✅ Survived",       survived,     f"{survived/total:.1%}")
    col3.metric("❌ Did Not Survive", not_survived, f"{not_survived/total:.1%}")
    col4.metric("Avg Survival Prob", f"{avg_prob:.3f}")

    st.divider()

    # Survival distribution bar chart
    st.subheader("Survival Distribution")
    dist = predictions_df["survived_label"].value_counts().reset_index()
    dist.columns = ["Label", "Count"]
    st.bar_chart(dist.set_index("Label"))

    st.divider()

    # Probability histogram
    st.subheader("Survival Probability Distribution")
    st.bar_chart(
        predictions_df["survival_probability"]
        .round(1)
        .value_counts()
        .sort_index()
    )

    st.divider()

    # Model info
    st.subheader("Model Info")
    model_name = predictions_df["model_used"].iloc[0]
    st.info(f"🤖 Model in use: **{model_name}**")

# ── Tab 2: Data Drift ──────────────────────────────────────
with tab2:
    st.subheader("Evidently Monitoring Results")

    if monitoring_df.empty:
        st.warning("No monitoring results found.")
    else:
        # Latest run
        latest_run  = monitoring_df["run_id"].iloc[0]
        latest_time = monitoring_df["timestamp"].iloc[0]
        st.caption(f"Run ID: `{latest_run}` | Timestamp: `{latest_time}`")

        # Drift metrics
        drift_metrics = monitoring_df[
            monitoring_df["metric_group"].str.contains("Drift", case=False, na=False)
        ]

        quality_metrics = monitoring_df[
            monitoring_df["metric_group"].str.contains("Quality", case=False, na=False)
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Drift Metrics")
            if not drift_metrics.empty:
                st.dataframe(
                    drift_metrics[["metric_name", "metric_value"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No drift metrics found.")

        with col2:
            st.subheader("🧹 Quality Metrics")
            if not quality_metrics.empty:
                st.dataframe(
                    quality_metrics[["metric_name", "metric_value"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No quality metrics found.")

        st.divider()

        # Full metrics table
        st.subheader("All Metrics")
        st.dataframe(
            monitoring_df[["metric_group", "metric_name", "metric_value", "timestamp"]],
            use_container_width=True,
            hide_index=True,
        )

        # Link to HTML report
        reports = [f for f in os.listdir("reports") if f.endswith(".html")] \
            if os.path.exists("reports") else []
        if reports:
            latest_report = sorted(reports)[-1]
            st.divider()
            st.subheader("📄 Full Evidently HTML Report")
            with open(f"reports/{latest_report}", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)

# ── Tab 3: Raw Data ────────────────────────────────────────
with tab3:
    st.subheader("Raw Predictions Table")

    col1, col2 = st.columns(2)
    with col1:
        label_filter = st.selectbox(
            "Filter by label",
            ["All", "Survived", "Did not survive"],
        )
    with col2:
        prob_min = st.slider("Min survival probability", 0.0, 1.0, 0.0, 0.05)

    filtered = predictions_df.copy()
    if label_filter != "All":
        filtered = filtered[filtered["survived_label"] == label_filter]
    filtered = filtered[filtered["survival_probability"] >= prob_min]

    st.caption(f"Showing {len(filtered)} of {len(predictions_df)} rows")
    st.dataframe(filtered, use_container_width=True, hide_index=True)