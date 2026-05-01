import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SQL & Data Engineering Pipeline", page_icon="🗄️", layout="wide", initial_sidebar_state="collapsed")
TEMPLATE = "plotly_dark"

TAB_TAKEAWAYS = {
    "overview": "Row-count transparency validates reporting completeness so stakeholders can trust the KPI denominator.",
    "quality": "Data-quality exceptions quantify governance risk and indicate where remediation has the highest reporting impact.",
    "kpis": "Monthly KPI movement identifies where growth and margin pressure diverge, enabling earlier course correction.",
    "segment": "Segment and zone performance surfaces concentration risk and where commercial levers can shift revenue mix.",
    "weather": "Weather-linked performance shows exogenous demand sensitivity, improving operational planning resilience.",
    "raw": "Raw-vs-staging comparisons prove transformation value and improve confidence in downstream analytics outputs.",
}


def render_takeaway(key: str) -> None:
    st.info(f"Shareholder Takeaway: {TAB_TAKEAWAYS[key]}")


@st.cache_resource
def load_artifacts():
    return joblib.load("models/artifacts.pkl")


def main():
    arts = load_artifacts()
    st.title("🗄️ SQL & Data Engineering Pipeline")
    dataset_key = arts.get("dataset_key", "unknown")
    st.markdown("Warehouse-style pipeline with raw, staging, fact, and mart layers built in DuckDB with data-quality checks and KPI outputs.")
    st.caption(f"Dataset mode: {dataset_key}")

    tabs = st.tabs(["Overview", "Data Quality", "Monthly KPIs", "Segment Performance", "Weather Impact", "Raw vs Staging"])

    with tabs[0]:
        render_takeaway("overview")
        counts = arts["row_counts"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Customers", f"{counts['raw_customers']:,}")
        c2.metric("Raw Orders", f"{counts['raw_orders']:,}")
        c3.metric("Staged Orders", f"{counts['stg_orders']:,}")
        c4.metric("Fact Orders", f"{counts['fct_orders']:,}")

    with tabs[1]:
        render_takeaway("quality")
        test_df = pd.DataFrame(list(arts["tests"].items()), columns=["test", "value"])
        st.dataframe(test_df, use_container_width=True, hide_index=True)
        fig = px.bar(test_df, x="test", y="value", color="value", color_continuous_scale="RdYlGn_r", title="Data Quality Test Results", template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        render_takeaway("kpis")
        monthly = arts["monthly_kpis"]
        st.markdown("**Volume KPIs (linear scale)**")
        fig_counts = px.line(
            monthly,
            x="order_month",
            y=["orders", "active_customers"],
            title="Orders & Active Customers",
            template=TEMPLATE,
        )
        st.plotly_chart(fig_counts, use_container_width=True)

        st.markdown("**Revenue KPI (linear scale)**")
        fig_revenue = px.line(
            monthly,
            x="order_month",
            y=["revenue"],
            title="Revenue Trend",
            template=TEMPLATE,
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
        st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tabs[3]:
        render_takeaway("segment")
        seg = arts["segment_performance"]
        if "segment" in seg.columns:
            fig = px.treemap(seg, path=["segment", "country", "product_category"], values="revenue", color="avg_order_value", color_continuous_scale="Blues")
        else:
            fig = px.treemap(seg, path=["payment_channel", "pickup_zone"], values="revenue", color="avg_order_value", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)

    with tabs[4]:
        render_takeaway("weather")
        weather = arts.get("weather_impact", pd.DataFrame())
        if weather is None or weather.empty:
            st.info("Weather join output is unavailable in fallback mode.")
        else:
            fig_wx = px.line(
                weather,
                x="order_month",
                y=["revenue", "avg_precip_mm"],
                title="Revenue vs Precipitation by Month",
                template=TEMPLATE,
            )
            st.plotly_chart(fig_wx, use_container_width=True)
            st.dataframe(weather, use_container_width=True, hide_index=True)

    with tabs[5]:
        render_takeaway("raw")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw Orders Sample**")
            st.dataframe(arts["raw_orders_sample"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Staged Orders Sample**")
            st.dataframe(arts["stg_orders_sample"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
