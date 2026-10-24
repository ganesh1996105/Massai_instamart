"""Clean customers and orders, then write CSV files and data/instamart.db."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CUSTOMER_COLUMNS = ["customer_id", "city", "signup_date", "age", "gender"]
PRODUCT_COLUMNS = ["product_id", "product_name", "category", "price_inr"]
ORDER_COLUMNS = [
    "order_id", "customer_id", "product_id", "quantity", "order_ts",
    "status", "discount_pct", "order_amount", "delivery_minutes",
]


def money(value):
    return round(float(value), 2)


def clean_customers(raw):
    customers = raw.copy()
    original_city = customers["city"]
    customers["city"] = customers["city"].str.strip().str.title()
    city_changed = int((original_city != customers["city"]).sum())

    original_gender = customers["gender"]
    blank_gender = int(original_gender.str.strip().eq("").sum())

    def map_gender(value):
        text = str(value).strip().lower()
        if text in {"m", "male"}:
            return "male"
        if text in {"f", "female"}:
            return "female"
        return "unknown"

    customers["gender"] = customers["gender"].map(map_gender)
    gender_changed = int((original_gender != customers["gender"]).sum())

    age = pd.to_numeric(customers["age"], errors="coerce")
    blank_age = int(customers["age"].str.strip().eq("").sum())
    invalid_age = age.notna() & ((age < 10) | (age > 100))
    invalid_age_count = int(invalid_age.sum())
    age = age.mask(invalid_age)
    customers["age"] = age
    customers["age"] = customers.groupby("city")["age"].transform(lambda s: s.fillna(s.median()))
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])

    stats = {
        "city_values_changed": city_changed,
        "blank_gender_set_to_unknown": blank_gender,
        "gender_values_changed": gender_changed,
        "blank_age": blank_age,
        "invalid_age_set_missing": invalid_age_count,
        "ages_imputed": blank_age + invalid_age_count,
    }
    return customers[CUSTOMER_COLUMNS], stats


def clean_orders(raw_orders, products):
    orders = raw_orders.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    timestamps_parsed = len(orders)

    before = len(orders)
    orders = orders.drop_duplicates()
    exact_removed = before - len(orders)

    orders = orders.sort_values(["order_id", "order_ts"], kind="mergesort")
    before = len(orders)
    orders = orders.drop_duplicates(subset=["order_id"], keep="last")
    older_removed = before - len(orders)

    catalogue = products[["product_id", "category", "price_inr"]].copy()
    catalogue["price_inr"] = pd.to_numeric(catalogue["price_inr"])
    orders = orders.merge(catalogue, on="product_id", how="left")
    orders["quantity"] = pd.to_numeric(orders["quantity"])
    orders["discount_pct"] = pd.to_numeric(orders["discount_pct"])
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    missing_amount = orders["order_amount"].isna()
    amounts_filled = int(missing_amount.sum())
    repaired = (
        orders.loc[missing_amount, "price_inr"]
        * orders.loc[missing_amount, "quantity"]
        * (1 - orders.loc[missing_amount, "discount_pct"] / 100)
    )
    orders.loc[missing_amount, "order_amount"] = repaired.map(money)
    orders["order_amount"] = orders["order_amount"].map(money)

    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    negative = orders["delivery_minutes"] < 0
    negatives_cleared = int(negative.sum())
    orders.loc[negative, "delivery_minutes"] = pd.NA
    completed = orders["status"].eq("completed")
    missing_completed = int(orders.loc[completed, "delivery_minutes"].isna().sum())
    orders.loc[completed, "delivery_minutes"] = (
        orders.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform(lambda s: s.fillna(s.median()))
    )

    orders = orders[ORDER_COLUMNS].reset_index(drop=True)
    stats = {
        "timestamps_parsed": timestamps_parsed,
        "exact_duplicates_removed": exact_removed,
        "older_rows_removed": older_removed,
        "amounts_filled": amounts_filled,
        "negative_delivery_cleared": negatives_cleared,
        "completed_delivery_imputed": missing_completed,
    }
    return orders, stats


def store(customers, products, orders):
    CLEAN.mkdir(parents=True, exist_ok=True)
    customers_out = customers.copy()
    customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
    orders_out = orders.copy()
    orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    products_out = products.copy()
    products_out["price_inr"] = pd.to_numeric(products_out["price_inr"])

    customers_out.to_csv(CLEAN / "customers.csv", index=False)
    products_out.to_csv(CLEAN / "products.csv", index=False)
    orders_out.to_csv(CLEAN / "orders.csv", index=False)

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as conn:
        customers_out.to_sql("customers", conn, if_exists="replace", index=False)
        products_out.to_sql("products", conn, if_exists="replace", index=False)
        orders_out.to_sql("orders", conn, if_exists="replace", index=False)


def main():
    customers_raw = pd.read_csv(RAW / "customers.csv", dtype=str, keep_default_na=False)
    products = pd.read_csv(RAW / "products.csv", dtype=str, keep_default_na=False)
    orders_raw = pd.read_csv(RAW / "orders.csv", dtype=str, keep_default_na=False)

    customers, customer_stats = clean_customers(customers_raw)
    orders, order_stats = clean_orders(orders_raw, products)
    store(customers, products, orders)

    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    print("CUSTOMER RULES")
    for key, value in customer_stats.items():
        print(f"  {key}: {value}")
    print("  city_counts:", customers["city"].value_counts().to_dict())
    print("  gender_counts:", customers["gender"].value_counts().to_dict())
    print("  missing_age:", int(customers["age"].isna().sum()))
    print("  age_min:", float(customers["age"].min()), "age_max:", float(customers["age"].max()))
    print("ORDER RULES")
    for key, value in order_stats.items():
        print(f"  {key}: {value}")
    print("  rows:", len(orders))
    print("  completed:", int(completed.sum()), "cancelled:", int(cancelled.sum()))
    print("  pending:", int(orders["status"].eq("pending").sum()))
    print("  missing_amount:", int(orders["order_amount"].isna().sum()))
    print("  missing_delivery_completed:", int(orders.loc[completed, "delivery_minutes"].isna().sum()))
    print("  missing_delivery_cancelled:", int(orders.loc[cancelled, "delivery_minutes"].isna().sum()))
    print("  completed_amount_sum:", f"{money(orders.loc[completed, 'order_amount'].sum()):.2f}")


if __name__ == "__main__":
    main()
