"""Run sql/queries.sql against data/instamart.db and check the reference results."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "instamart.db"
SQL_PATH = ROOT / "sql" / "queries.sql"

TOP_CUSTOMERS = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10),
}


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def load_queries(path):
    text = path.read_text(encoding="utf-8")
    chunks = re.split(r"(?m)^-- Q([1-7])\s*$", text)
    queries = {}
    pairs = chunks[1:]
    for number, body in zip(pairs[0::2], pairs[1::2]):
        queries[int(number)] = body.strip().rstrip(";").strip()
    if set(queries) != set(range(1, 8)):
        raise SystemExit(f"FAIL query file: found {sorted(queries)}")
    return queries


def money(series):
    return round(float(series), 2)


def main():
    queries = load_queries(SQL_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        frames = {
            number: pd.read_sql_query(sql, conn)
            for number, sql in queries.items()
        }

    q1 = frames[1]
    check("q1_customers", len(q1), 30)
    print(f"Q1 customers: {len(q1)}")

    q2 = frames[2]
    q2["revenue"] = q2["revenue"].map(lambda value: round(float(value), 2))
    check("q2_rows", len(q2), 30)
    check("q2_total", money(q2["revenue"].sum()), 2998668.80)
    top = q2.loc[q2["revenue"].idxmax()]
    check("q2_top_city", top["city"], "Bengaluru")
    check("q2_top_category", top["category"], "Electronics")
    check("q2_top_revenue", float(top["revenue"]), 233909.45)
    print(f"Q2 rows: {len(q2)} total: {money(q2['revenue'].sum()):.2f}")
    print(f"Q2 largest: {top['city']} / {top['category']} INR {float(top['revenue']):.2f}")

    q3 = frames[3]
    check("q3_products", len(q3), 11)
    print(f"Q3 products: {len(q3)}")
    print(q3.to_string(index=False))

    q4 = frames[4]
    check("q4_customers", len(q4), 25)
    print(f"Q4 customers with no completed order: {len(q4)}")

    q5 = frames[5]
    check("q5_customers", len(q5), 550)
    average = round(float(q5["avg_total"].iloc[0]), 2)
    check("q5_average", average, 2552.06)
    print(f"Q5 customers: {len(q5)} average INR {average:.2f}")

    q6 = frames[6]
    q6["total_spend"] = q6["total_spend"].map(lambda value: round(float(value), 2))
    tops = q6.loc[q6["rank_pos"].eq(1), ["city", "customer_id", "total_spend"]]
    found = {
        row.city: (row.customer_id, float(row.total_spend))
        for row in tops.itertuples(index=False)
    }
    check("q6_tops", found, TOP_CUSTOMERS)
    rank_differs = int((q6["rank_pos"] != q6["row_num"]).sum())
    check("q6_rank_vs_row_number", rank_differs, 8)
    print("Q6 top customer per city:")
    print(tops.sort_values("city").to_string(index=False))
    print(f"Q6 rows where RANK differs from ROW_NUMBER: {rank_differs}")

    q7 = frames[7]
    check("q7_days", len(q7), 183)
    last = round(float(q7["cumulative_revenue"].iloc[-1]), 2)
    check("q7_cumulative", last, 2998668.80)
    print(f"Q7 days: {len(q7)} final cumulative INR {last:.2f}")
    print("QUERIES PASSED")


if __name__ == "__main__":
    main()
