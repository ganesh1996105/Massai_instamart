"""Assert the Part A cleaning checkpoints."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CITY_COUNTS = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184,
}
GENDER_COUNTS = {"female": 610, "male": 560, "unknown": 30}


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def main():
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")

    check("customer_rows", len(customers), 1200)
    check("product_rows", len(products), 40)
    check("cities", customers["city"].value_counts().to_dict(), CITY_COUNTS)
    check("genders", customers["gender"].value_counts().to_dict(), GENDER_COUNTS)
    check("missing_age", int(customers["age"].isna().sum()), 0)
    check("age_min", float(customers["age"].min()) >= 18, True)
    check("age_max", float(customers["age"].max()) <= 60, True)

    check("order_rows", len(orders), 12309)
    check("distinct_order_id", int(orders["order_id"].nunique()), 12309)
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    check("completed", int(completed.sum()), 11110)
    check("cancelled", int(cancelled.sum()), 1199)
    check("pending", int(orders["status"].eq("pending").sum()), 0)
    check("missing_amount", int(orders["order_amount"].isna().sum()), 0)
    check("missing_delivery_completed", int(orders.loc[completed, "delivery_minutes"].isna().sum()), 0)
    check("missing_delivery_cancelled", int(orders.loc[cancelled, "delivery_minutes"].isna().sum()), 1199)
    total = round(float(orders.loc[completed, "order_amount"].sum()), 2)
    check("completed_amount_sum", total, 2998668.80)

    with sqlite3.connect(DB_PATH) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        check("db_tables", tables, {"customers", "products", "orders"})
        for name, expected in (("customers", 1200), ("products", 40), ("orders", 12309)):
            count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            check(f"db_{name}", count, expected)
        db_sum = conn.execute(
            "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'"
        ).fetchone()[0]
        check("db_completed_amount_sum", float(db_sum), 2998668.80)

    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
