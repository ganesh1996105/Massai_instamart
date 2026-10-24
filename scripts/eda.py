"""Pandas cross-check of city-category revenue, plus distribution and correlation."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
DB_PATH = ROOT / "data" / "instamart.db"

CITY_REFERENCE = {
    "Bengaluru": (523643.40, 276.47, 1894),
    "Mumbai": (479058.50, 297.00, 1613),
}


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def main():
    orders = pd.read_csv(CLEAN / "orders.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    completed = orders.loc[orders["status"].eq("completed")].merge(
        customers[["customer_id", "city"]], on="customer_id", how="left"
    ).merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )

    pivot = pd.pivot_table(
        completed,
        values="order_amount",
        index="city",
        columns="category",
        aggfunc="sum",
        margins=True,
        margins_name="All",
    )
    grand = round(float(pivot.loc["All", "All"]), 2)
    with sqlite3.connect(DB_PATH) as conn:
        sql_total = pd.read_sql_query(
            """
            SELECT SUM(o.order_amount) AS revenue
            FROM orders AS o
            WHERE o.status = 'completed'
            """,
            conn,
        )["revenue"].iloc[0]
    sql_total = round(float(sql_total), 2)
    check("pivot_grand_total", grand, 2998668.80)
    check("pivot_matches_sql", grand, sql_total)
    print(f"Pivot grand total INR {grand:.2f}; SQL total INR {sql_total:.2f}")

    by_city = completed.groupby("city").agg(
        sum=("order_amount", "sum"),
        mean=("order_amount", "mean"),
        count=("order_amount", "count"),
    )
    by_city["sum"] = by_city["sum"].map(lambda value: round(float(value), 2))
    by_city["mean"] = by_city["mean"].map(lambda value: round(float(value), 2))
    print("CITY AGGREGATION")
    print(by_city.to_string())
    for city, (total, mean, count) in CITY_REFERENCE.items():
        check(f"{city}_sum", float(by_city.loc[city, "sum"]), total)
        check(f"{city}_mean", float(by_city.loc[city, "mean"]), mean)
        check(f"{city}_count", int(by_city.loc[city, "count"]), count)

    amount = completed["order_amount"]
    q1 = round(float(amount.quantile(0.25)), 2)
    q3 = round(float(amount.quantile(0.75)), 2)
    iqr = q3 - q1
    lower = round(q1 - 1.5 * iqr, 2)
    upper = round(q3 + 1.5 * iqr, 2)
    below = int((amount < lower).sum())
    above = int((amount > upper).sum())
    mean = round(float(amount.mean()), 2)
    median = round(float(amount.median()), 2)
    skew = round(float(amount.skew()), 2)
    check("mean", mean, 269.91)
    check("median", median, 135.00)
    check("q1", q1, 64.80)
    check("q3", q3, 297.00)
    check("upper_fence", upper, 645.30)
    check("outliers_above", above, 897)
    check("outliers_below", below, 0)
    print(
        f"DISTRIBUTION mean={mean:.2f} median={median:.2f} skew={skew:.2f} "
        f"q1={q1:.2f} q3={q3:.2f} lower={lower:.2f} upper={upper:.2f} "
        f"below={below} above={above}"
    )

    columns = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
    corr = completed[columns].corr()
    print("CORRELATION")
    print(corr.round(4).to_string())
    pairs = []
    for left_i, left in enumerate(columns):
        for right in columns[left_i + 1:]:
            pairs.append((abs(float(corr.loc[left, right])), left, right, float(corr.loc[left, right])))
    pairs.sort(reverse=True)
    _, left, right, value = pairs[0]
    check("strongest_pair", (left, right), ("order_amount", "discount_pct"))
    check("strongest_rounded", round(value, 3), -0.095)
    quantity_corr = round(float(corr.loc["order_amount", "quantity"]), 3)
    check("quantity_corr", quantity_corr, 0.087)
    print(f"STRONGEST {left} {right} {value:.4f}")
    print("EDA PASSED")


if __name__ == "__main__":
    main()
