import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SQL & Data Engineering Pipeline", page_icon="🗄️", layout="wide", initial_sidebar_state="collapsed")
TEMPLATE = "plotly_dark"


@st.cache_resource
def load_artifacts():
    return joblib.load("models/artifacts.pkl")


def main():
    arts = load_artifacts()
    st.title("🗄️ SQL & Data Engineering Pipeline")
    st.markdown("Warehouse-style pipeline with raw, staging, fact, and mart layers built in DuckDB with data-quality checks and KPI outputs.")

    tabs = st.tabs(["Overview", "Data Quality", "Monthly KPIs", "Segment Performance", "Raw vs Staging"])

    with tabs[0]:
        counts = arts["row_counts"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Raw Customers", f"{counts['raw_customers']:,}")
        c2.metric("Raw Orders", f"{counts['raw_orders']:,}")
        c3.metric("Staged Orders", f"{counts['stg_orders']:,}")
        c4.metric("Fact Orders", f"{counts['fct_orders']:,}")

    with tabs[1]:
        test_df = pd.DataFrame(list(arts["tests"].items()), columns=["test", "value"])
        st.dataframe(test_df, use_container_width=True, hide_index=True)
        fig = px.bar(test_df, x="test", y="value", color="value", color_continuous_scale="RdYlGn_r", title="Data Quality Test Results", template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        monthly = arts["monthly_kpis"]
        fig = px.line(monthly, x="order_month", y=["revenue", "orders", "active_customers"], title="Monthly KPI Trends", template=TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(monthly, use_container_width=True, hide_index=True)

    with tabs[3]:
        seg = arts["segment_performance"]
        fig = px.treemap(seg, path=["segment", "country", "product_category"], values="revenue", color="avg_order_value", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)

    with tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Raw Orders Sample**")
            st.dataframe(arts["raw_orders_sample"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Staged Orders Sample**")
            st.dataframe(arts["stg_orders_sample"], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
