"""Audit the raw customer, product, and order exports."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def load(name):
    return pd.read_csv(RAW / name, dtype=str, keep_default_na=False)


def missing_table(frame):
    rows = []
    n = len(frame)
    for column in frame.columns:
        missing = int(frame[column].str.strip().eq("").sum())
        pct = 100.0 * missing / n if n else 0.0
        rows.append((column, str(frame[column].dtype), missing, round(pct, 2)))
    return rows


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def main():
    customers = load("customers.csv")
    products = load("products.csv")
    orders = load("orders.csv")

    print("CUSTOMERS", customers.shape)
    for column, dtype, missing, pct in missing_table(customers):
        print(f"  {column}: dtype={dtype} missing={missing} ({pct}%)")
    print("PRODUCTS", products.shape)
    for column, dtype, missing, pct in missing_table(products):
        print(f"  {column}: dtype={dtype} missing={missing} ({pct}%)")
    print("ORDERS", orders.shape)
    for column, dtype, missing, pct in missing_table(orders):
        print(f"  {column}: dtype={dtype} missing={missing} ({pct}%)")

    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    age = pd.to_numeric(customers["age"], errors="coerce")
    gender = customers["gender"].str.strip()

    checks = {
        "order_rows": (len(orders), 13084),
        "exact_duplicate_rows": (int(orders.duplicated().sum()), 119),
        "pending_rows": (int(orders["status"].eq("pending").sum()), 656),
        "distinct_order_id": (int(orders["order_id"].nunique()), 12309),
        "non_numeric_amount": (int(amount.isna().sum()), 120),
        "negative_delivery": (int((delivery < 0).sum()), 106),
        "blank_delivery": (int(orders["delivery_minutes"].str.strip().eq("").sum()), 2541),
        "blank_age": (int(customers["age"].str.strip().eq("").sum()), 61),
        "age_outside_10_100": (int((age.notna() & ((age < 10) | (age > 100))).sum()), 6),
        "raw_city_spellings": (int(customers["city"].nunique()), 24),
        "raw_gender_spellings": (int(gender.loc[gender.ne("")].nunique()), 10),
        "blank_gender": (int(gender.eq("").sum()), 30),
    }
    print("AUDIT CHECKS")
    for name, (got, expected) in checks.items():
        print(f"  {name}: {got}")
        check(name, got, expected)
    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
