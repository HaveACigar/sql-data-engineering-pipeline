import os
import joblib
import duckdb
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def create_raw_data():
    n_customers = 5000
    n_orders = 30000
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "signup_date": pd.to_datetime("2022-01-01") + pd.to_timedelta(RNG.integers(0, 730, size=n_customers), unit="D"),
        "segment": RNG.choice(["consumer", "small_business", "enterprise"], size=n_customers, p=[0.72, 0.22, 0.06]),
        "country": RNG.choice(["US", "Canada", "UK", "Germany", "France"], size=n_customers, p=[0.58, 0.14, 0.12, 0.08, 0.08]),
    })

    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": RNG.integers(1, n_customers + 1, size=n_orders),
        "order_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(RNG.integers(0, 365, size=n_orders), unit="D"),
        "product_category": RNG.choice(["Accessories", "Software", "Hardware", "Services"], size=n_orders, p=[0.28, 0.21, 0.33, 0.18]),
        "channel": RNG.choice(["web", "mobile", "sales_rep"], size=n_orders, p=[0.62, 0.25, 0.13]),
        "quantity": RNG.integers(1, 6, size=n_orders),
        "unit_price": np.round(RNG.uniform(12, 850, size=n_orders), 2),
    })
    orders["gross_revenue"] = np.round(orders["quantity"] * orders["unit_price"], 2)
    orders["discount_rate"] = np.round(RNG.choice([0, 0.05, 0.10, 0.15], size=n_orders, p=[0.62, 0.18, 0.14, 0.06]), 2)
    orders["net_revenue"] = np.round(orders["gross_revenue"] * (1 - orders["discount_rate"]), 2)

    # Inject a few data-quality issues intentionally
    dupes = orders.sample(120, random_state=42)
    orders = pd.concat([orders, dupes], ignore_index=True)
    orders.loc[orders.sample(85, random_state=7).index, "net_revenue"] = np.nan
    orders.loc[orders.sample(40, random_state=9).index, "quantity"] = -1
    return customers, orders


def main():
    os.makedirs("warehouse", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    customers, orders = create_raw_data()
    con = duckdb.connect("warehouse/analytics.duckdb")
    con.register("customers_df", customers)
    con.register("orders_df", orders)

    con.execute("create or replace table raw_customers as select * from customers_df")
    con.execute("create or replace table raw_orders as select * from orders_df")

    con.execute("""
        create or replace table stg_orders as
        select distinct
            order_id,
            customer_id,
            cast(order_date as date) as order_date,
            product_category,
            channel,
            case when quantity < 1 then 1 else quantity end as quantity,
            unit_price,
            gross_revenue,
            discount_rate,
            coalesce(net_revenue, gross_revenue * (1 - discount_rate)) as net_revenue
        from raw_orders
    """)

    con.execute("""
        create or replace table dim_customers as
        select
            customer_id,
            signup_date,
            segment,
            country
        from raw_customers
    """)

    con.execute("""
        create or replace table fct_orders as
        select
            o.order_id,
            o.customer_id,
            o.order_date,
            date_trunc('month', o.order_date) as order_month,
            o.product_category,
            o.channel,
            o.quantity,
            o.unit_price,
            o.discount_rate,
            o.net_revenue,
            c.segment,
            c.country
        from stg_orders o
        left join dim_customers c using (customer_id)
    """)

    con.execute("""
        create or replace table mart_kpis as
        select
            order_month,
            count(distinct order_id) as orders,
            count(distinct customer_id) as active_customers,
            round(sum(net_revenue), 2) as revenue,
            round(avg(net_revenue), 2) as avg_order_value
        from fct_orders
        group by 1
        order by 1
    """)

    con.execute("""
        create or replace table mart_segment_performance as
        select
            segment,
            country,
            product_category,
            round(sum(net_revenue), 2) as revenue,
            count(distinct order_id) as orders,
            round(avg(net_revenue), 2) as avg_order_value
        from fct_orders
        group by 1,2,3
        order by revenue desc
    """)

    tests = {
        "raw_order_duplicates": con.execute("select count(*) from raw_orders") .fetchone()[0] - con.execute("select count(distinct order_id) from raw_orders").fetchone()[0],
        "staging_null_net_revenue": con.execute("select count(*) from stg_orders where net_revenue is null").fetchone()[0],
        "negative_quantities_after_cleaning": con.execute("select count(*) from stg_orders where quantity < 1").fetchone()[0],
        "orphaned_customer_keys": con.execute("select count(*) from fct_orders where customer_id is null").fetchone()[0],
    }

    artifacts = {
        "monthly_kpis": con.execute("select * from mart_kpis").df(),
        "segment_performance": con.execute("select * from mart_segment_performance limit 100").df(),
        "raw_orders_sample": con.execute("select * from raw_orders limit 20").df(),
        "stg_orders_sample": con.execute("select * from stg_orders limit 20").df(),
        "tests": tests,
        "row_counts": {
            "raw_customers": con.execute("select count(*) from raw_customers").fetchone()[0],
            "raw_orders": con.execute("select count(*) from raw_orders").fetchone()[0],
            "stg_orders": con.execute("select count(*) from stg_orders").fetchone()[0],
            "fct_orders": con.execute("select count(*) from fct_orders").fetchone()[0],
        },
    }
    joblib.dump(artifacts, "models/artifacts.pkl", compress=3)
    con.close()


if __name__ == "__main__":
    main()
