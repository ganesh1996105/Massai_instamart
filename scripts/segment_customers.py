"""Build the RFM table, choose K-means segments, and write customer_segments.csv."""
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "customer_segments.csv"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]
CHOSEN_K = 4

# Inertia and silhouette from the brief. Inertia tolerance is ±5%; silhouette is ±0.05.
REFERENCE = {
    2: (1820.8, 0.465),
    3: (1268.2, 0.445),
    4: (743.1, 0.526),
    5: (617.4, 0.481),
    6: (503.2, 0.482),
    7: (408.5, 0.416),
    8: (366.5, 0.412),
}
ACTIONS = {
    "Premium": "Offer a personal outreach and a high-value retention reward before the customer goes quiet.",
    "Frequent": "Send refill reminders for the grocery and dairy baskets these customers already reorder.",
    "At risk": "Send a win-back coupon while the gap since the last order is still recoverable.",
    "Occasional": "Use a small basket-building offer to lift a second order without a deep discount.",
}


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def within(name, got, expected, tolerance):
    if abs(got - expected) > tolerance:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected} ± {tolerance}")


def build_rfm(orders):
    completed = orders.loc[orders["status"].eq("completed")].copy()
    completed["order_ts"] = pd.to_datetime(completed["order_ts"])
    rfm = completed.groupby("customer_id", as_index=False).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm["monetary"] = rfm["monetary"].map(lambda value: round(float(value), 2))
    return rfm


def name_segments(profile):
    """Cluster ids are unordered labels. Names come from the RFM profile."""
    names = {}
    remaining = profile.copy()
    premium = remaining["monetary_mean"].idxmax()
    names[premium] = "Premium"
    remaining = remaining.drop(index=premium)
    frequent = remaining["frequency_mean"].idxmax()
    names[frequent] = "Frequent"
    remaining = remaining.drop(index=frequent)
    at_risk = remaining["recency_mean"].idxmax()
    names[at_risk] = "At risk"
    names[remaining.drop(index=at_risk).index[0]] = "Occasional"
    return names


def main():
    orders = pd.read_csv(CLEAN / "orders.csv")
    rfm = build_rfm(orders)
    check("rfm_customers", len(rfm), 1175)
    check("recency_median", float(rfm["recency"].median()), 19.0)
    check("recency_min", int(rfm["recency"].min()), 0)
    check("recency_max", int(rfm["recency"].max()), 180)
    check("frequency_mean", round(float(rfm["frequency"].mean()), 2), 9.46)
    check("frequency_max", int(rfm["frequency"].max()), 33)
    check("monetary_mean", round(float(rfm["monetary"].mean()), 2), 2552.06)
    check("monetary_max", round(float(rfm["monetary"].max()), 2), 14766.55)
    print(
        f"RFM customers={len(rfm)} recency median={rfm['recency'].median():.0f} "
        f"min={rfm['recency'].min()} max={rfm['recency'].max()} "
        f"frequency mean={rfm['frequency'].mean():.2f} max={rfm['frequency'].max()} "
        f"monetary mean={rfm['monetary'].mean():.2f} max={rfm['monetary'].max():.2f}"
    )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[RFM_COLUMNS])
    print("K INERTIA SILHOUETTE")
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled, labels))
        expected_inertia, expected_score = REFERENCE[k]
        within(f"k{k}_inertia", float(model.inertia_), expected_inertia, abs(expected_inertia) * 0.05)
        within(f"k{k}_silhouette", score, expected_score, 0.05)
        print(f"{k} {model.inertia_:.1f} {score:.3f}")

    final = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=42)
    rfm["segment_id"] = final.fit_predict(scaled)
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "size"),
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean"),
    )
    names = name_segments(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_name"].eq("Premium")
    check("premium_segments", int(rfm.loc[rfm["is_premium"], "segment_id"].nunique()), 1)
    premium_id = rfm.loc[rfm["is_premium"], "segment_id"].iloc[0]
    check(
        "premium_is_highest_monetary",
        int(premium_id),
        int(profile["monetary_mean"].idxmax()),
    )

    print("SEGMENT PROFILE")
    for segment_id, row in profile.sort_index().iterrows():
        name = names[segment_id]
        print(
            f"  id={segment_id} name={name} size={int(row['size'])} "
            f"recency={row['recency_mean']:.2f} frequency={row['frequency_mean']:.2f} "
            f"monetary={row['monetary_mean']:.2f}"
        )
        print(f"    action: {ACTIONS[name]}")

    pca = PCA(n_components=2, random_state=42)
    pca.fit(scaled)
    ratios = pca.explained_variance_ratio_
    within("pca_1", float(ratios[0]), 0.634, 0.02)
    within("pca_2", float(ratios[1]), 0.218, 0.02)
    within("pca_total", float(ratios.sum()), 0.852, 0.02)
    print(f"PCA {ratios[0]:.3f} {ratios[1]:.3f} total={ratios.sum():.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv(
        OUT, index=False
    )
    print(f"WROTE {OUT}")
    print("SEGMENTS PASSED")


if __name__ == "__main__":
    main()
