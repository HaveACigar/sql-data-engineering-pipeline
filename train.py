import os
import joblib
import duckdb
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
NYC_TAXI_URLS = [
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
]
WEATHER_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude=40.7128&longitude=-74.0060"
    "&start_date=2023-01-01&end_date=2023-02-28"
    "&daily=temperature_2m_mean,precipitation_sum,snowfall_sum"
    "&timezone=America%2FNew_York"
)


def create_fallback_data():
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

    dupes = orders.sample(120, random_state=42)
    orders = pd.concat([orders, dupes], ignore_index=True)
    orders.loc[orders.sample(85, random_state=7).index, "net_revenue"] = np.nan
    orders.loc[orders.sample(40, random_state=9).index, "quantity"] = -1
    weather = pd.DataFrame()
    return customers, orders, weather, "synthetic_fallback"


def load_weather_daily() -> pd.DataFrame:
    payload = pd.read_json(WEATHER_URL)
    daily = pd.DataFrame(payload["daily"][0])
    daily["pickup_date"] = pd.to_datetime(daily["time"]).dt.date
    daily = daily.rename(columns={
        "temperature_2m_mean": "temp_c_mean",
        "precipitation_sum": "precip_mm",
        "snowfall_sum": "snow_mm",
    })[["pickup_date", "temp_c_mean", "precip_mm", "snow_mm"]]
    return daily


def load_nyc_taxi_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    frames = []
    for url in NYC_TAXI_URLS:
        frames.append(pd.read_parquet(url))
    raw = pd.concat(frames, ignore_index=True)

    keep_cols = [
        "VendorID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "PULocationID",
        "DOLocationID",
        "payment_type",
        "fare_amount",
        "total_amount",
    ]
    raw = raw[keep_cols].copy()
    raw = raw.rename(columns={
        "tpep_pickup_datetime": "order_date",
        "PULocationID": "pu_location_id",
        "DOLocationID": "do_location_id",
    })

    max_rows = int(os.getenv("NYC_SAMPLE_ROWS", "300000"))
    if len(raw) > max_rows:
        raw = raw.sample(max_rows, random_state=42).reset_index(drop=True)

    raw["order_date"] = pd.to_datetime(raw["order_date"], errors="coerce")
    raw["pickup_date"] = raw["order_date"].dt.date

    weather = load_weather_daily()

    customers = pd.DataFrame({
        "customer_id": raw["VendorID"].fillna(0).astype(int),
        "signup_date": pd.to_datetime("2020-01-01"),
        "segment": raw["payment_type"].fillna(0).astype(int).map({1: "card", 2: "cash"}).fillna("other"),
        "country": "US",
    }).drop_duplicates()

    orders = pd.DataFrame({
        "order_id": np.arange(1, len(raw) + 1),
        "customer_id": raw["VendorID"].fillna(0).astype(int),
        "order_date": raw["order_date"],
        "product_category": "Taxi Ride",
        "channel": raw["payment_type"].fillna(0).astype(int).map({1: "card", 2: "cash"}).fillna("other"),
        "quantity": 1,
        "unit_price": raw["fare_amount"].fillna(0),
        "gross_revenue": raw["fare_amount"].fillna(0),
        "discount_rate": 0.0,
        "net_revenue": raw["total_amount"].fillna(0),
        "trip_distance": raw["trip_distance"].fillna(0),
        "pu_location_id": raw["pu_location_id"].fillna(-1).astype(int),
        "do_location_id": raw["do_location_id"].fillna(-1).astype(int),
        "pickup_date": raw["pickup_date"],
    })

    # Inject additional quality issues to showcase wrangling discipline.
    dupes = orders.sample(min(300, len(orders) // 20), random_state=9)
    orders = pd.concat([orders, dupes], ignore_index=True)
    if len(orders) > 200:
        orders.loc[orders.sample(120, random_state=11).index, "net_revenue"] = np.nan
        orders.loc[orders.sample(80, random_state=12).index, "trip_distance"] = -1

    return customers, orders, weather, "nyc_taxi_weather"


def main():
    os.makedirs("warehouse", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    try:
        customers, orders, weather_daily, dataset_key = load_nyc_taxi_data()
    except Exception as exc:
        print(f"NYC TLC load failed ({exc}); using synthetic fallback")
        customers, orders, weather_daily, dataset_key = create_fallback_data()

    con = duckdb.connect("warehouse/analytics.duckdb")
    con.register("customers_df", customers)
    con.register("orders_df", orders)
    if len(weather_daily):
        con.register("weather_df", weather_daily)

    con.execute("create or replace table raw_customers as select * from customers_df")
    con.execute("create or replace table raw_orders as select * from orders_df")

    con.execute("""
        create or replace table stg_orders as
        select distinct
            order_id,
            customer_id,
            cast(order_date as date) as order_date,
            cast(order_date as date) as pickup_date,
            product_category,
            channel,
            case when quantity < 1 then 1 else quantity end as quantity,
            case when coalesce(trip_distance, 0) < 0 then null else trip_distance end as trip_distance,
            unit_price,
            gross_revenue,
            discount_rate,
            coalesce(net_revenue, gross_revenue * (1 - discount_rate)) as net_revenue,
            pu_location_id,
            do_location_id
        from raw_orders
        where order_date is not null
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
            date_part('hour', o.order_date) as pickup_hour,
            o.product_category,
            o.channel,
            o.quantity,
            o.trip_distance,
            o.unit_price,
            o.discount_rate,
            o.net_revenue,
            o.pu_location_id,
            o.do_location_id,
            c.segment,
            c.country
        from stg_orders o
        left join dim_customers c using (customer_id)
    """)

    if len(weather_daily):
        con.execute("create or replace table dim_weather_daily as select * from weather_df")
        con.execute("""
            create or replace table mart_weather_impact as
            select
                f.order_month,
                avg(w.temp_c_mean) as avg_temp_c,
                avg(w.precip_mm) as avg_precip_mm,
                count(*) as rides,
                round(sum(f.net_revenue), 2) as revenue,
                round(avg(f.trip_distance), 3) as avg_trip_distance
            from fct_orders f
            left join dim_weather_daily w
              on cast(f.order_date as date) = w.pickup_date
            group by 1
            order by 1
        """)

    con.execute("""
        create or replace table mart_kpis as
        select
            order_month,
            count(distinct order_id) as orders,
            count(distinct customer_id) as active_customers,
            round(sum(net_revenue), 2) as revenue,
            round(avg(net_revenue), 2) as avg_order_value,
            round(avg(trip_distance), 3) as avg_trip_distance
        from fct_orders
        group by 1
        order by 1
    """)

    con.execute("""
        create or replace table mart_segment_performance as
        select
            channel as payment_channel,
            cast(pu_location_id as varchar) as pickup_zone,
            round(sum(net_revenue), 2) as revenue,
            count(distinct order_id) as orders,
            round(avg(net_revenue), 2) as avg_order_value,
            round(avg(trip_distance), 3) as avg_trip_distance
        from fct_orders
        group by 1,2
        order by revenue desc
    """)

    tests = {
        "raw_order_duplicates": con.execute("select count(*) from raw_orders") .fetchone()[0] - con.execute("select count(distinct order_id) from raw_orders").fetchone()[0],
        "staging_null_net_revenue": con.execute("select count(*) from stg_orders where net_revenue is null").fetchone()[0],
        "negative_distance_after_cleaning": con.execute("select count(*) from stg_orders where trip_distance < 0").fetchone()[0],
        "null_order_date_after_cleaning": con.execute("select count(*) from stg_orders where order_date is null").fetchone()[0],
    }

    artifacts = {
        "dataset_key": dataset_key,
        "monthly_kpis": con.execute("select * from mart_kpis").df(),
        "segment_performance": con.execute("select * from mart_segment_performance limit 100").df(),
        "weather_impact": con.execute("select * from mart_weather_impact").df() if len(weather_daily) else pd.DataFrame(),
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
