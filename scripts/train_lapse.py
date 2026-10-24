"""Train a leakage-free lapse model on orders placed before 2025-09-01."""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
SEGMENTS = ROOT / "outputs" / "customer_segments.csv"
MODEL_PATH = ROOT / "outputs" / "lapse_model.joblib"
RETENTION_PATH = ROOT / "outputs" / "retention_list.csv"
CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")

NUMERIC = [
    "recency", "frequency", "monetary", "avg_discount", "cancel_rate",
    "avg_delivery", "age", "tenure_days",
]
CATEGORICAL = ["city", "gender", "favourite_category"]
FEATURES = NUMERIC + CATEGORICAL

READINGS = {
    "recency": {
        1: "A longer gap since the last order raises the lapse chance, so a quiet customer should be contacted first.",
        -1: "A longer gap since the last order lowers the lapse chance in this fit, so the gap is not the risk signal here.",
    },
    "frequency": {
        1: "Ordering more often raises the lapse chance in this fit, so frequency is not acting as a loyalty signal.",
        -1: "Customers who order more often are less likely to lapse, so a drop in order count is a warning.",
    },
    "monetary": {
        1: "Higher completed spend raises the lapse chance in this fit, so spend alone is not a loyalty signal.",
        -1: "Higher completed spend lowers the lapse chance, so the bigger buyers are the more stable ones.",
    },
}


def check(name, got, expected):
    if got != expected:
        raise SystemExit(f"FAIL {name}: got {got}, expected {expected}")


def favourite_category(categories):
    counts = categories.value_counts()
    winners = counts[counts == counts.max()].index.tolist()
    return sorted(winners)[0]


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first"), CATEGORICAL),
    ])


def build_features(orders, customers, products):
    orders = orders.copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers = customers.copy()
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    before = orders.loc[orders["order_ts"] < CUTOFF].copy()
    completed = before.loc[before["status"].eq("completed")].copy()
    if int(completed["delivery_minutes"].isna().sum()) != 0:
        raise SystemExit("FAIL completed delivery has missing values before the cutoff")

    completed = completed.merge(products[["product_id", "category"]], on="product_id", how="left")
    behaviour = completed.groupby("customer_id", as_index=False).agg(
        last_order=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount=("discount_pct", "mean"),
        avg_delivery=("delivery_minutes", "mean"),
        favourite_category=("category", favourite_category),
    )
    behaviour["recency"] = (CUTOFF - behaviour["last_order"]).dt.days

    before["is_cancelled"] = before["status"].eq("cancelled").astype(int)
    rates = before.groupby("customer_id", as_index=False).agg(
        n_orders=("order_id", "size"),
        n_cancelled=("is_cancelled", "sum"),
    )
    rates["cancel_rate"] = rates["n_cancelled"] / rates["n_orders"]

    in_september = set(orders.loc[
        orders["order_ts"].ge(CUTOFF)
        & orders["order_ts"].lt(LABEL_END)
        & orders["status"].eq("completed"),
        "customer_id",
    ])
    frame = behaviour.merge(rates[["customer_id", "cancel_rate"]], on="customer_id", how="left")
    frame = frame.merge(
        customers[["customer_id", "city", "age", "gender", "signup_date"]],
        on="customer_id",
        how="left",
    )
    frame["tenure_days"] = (CUTOFF - frame["signup_date"]).dt.days
    frame["lapsed"] = (~frame["customer_id"].isin(in_september)).astype(int)
    return frame


def class_metrics(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def print_metrics(title, metrics):
    matrix = metrics["matrix"]
    print(
        f"{title} precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )
    print(f"  confusion [tn fp; fn tp] = [{matrix[0, 0]} {matrix[0, 1]}; {matrix[1, 0]} {matrix[1, 1]}]")


def coefficient(pipeline, feature):
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    coefs = pipeline.named_steps["model"].coef_[0]
    key = f"num__{feature}"
    return float(coefs[names.index(key)])


def main():
    orders = pd.read_csv(CLEAN / "orders.csv")
    customers = pd.read_csv(CLEAN / "customers.csv")
    products = pd.read_csv(CLEAN / "products.csv")
    frame = build_features(orders, customers, products)
    missing = int(frame[FEATURES + ["lapsed"]].isna().sum().sum())
    check("missing_feature_values", missing, 0)
    check("cohort", len(frame), 1156)
    lapsed_count = int(frame["lapsed"].sum())
    check("lapsed", lapsed_count, 443)
    check("lapsed_pct", round(100 * lapsed_count / len(frame), 2), 38.32)
    print(f"COHORT {len(frame)} lapsed={lapsed_count} ({100 * lapsed_count / len(frame):.2f}%)")

    features = frame[FEATURES]
    target = frame["lapsed"]
    index_train, index_test = train_test_split(
        frame.index, test_size=0.2, stratify=target, random_state=42
    )
    x_train, x_test = features.loc[index_train], features.loc[index_test]
    y_train, y_test = target.loc[index_train], target.loc[index_test]
    check("train_rows", len(x_train), 924)
    check("test_rows", len(x_test), 232)

    baseline_pred = np.zeros(len(y_test), dtype=int)
    baseline_accuracy = float((baseline_pred == y_test.to_numpy()).mean())
    baseline_f1 = float(f1_score(y_test, baseline_pred, pos_label=1, zero_division=0))
    check("baseline_accuracy", round(100 * baseline_accuracy, 1), 61.6)
    check("baseline_f1", baseline_f1, 0.0)
    print(f"BASELINE accuracy={100 * baseline_accuracy:.1f}% lapsed_f1={baseline_f1:.1f}")

    logistic = Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    logistic_search = GridSearchCV(
        logistic,
        {"model__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="f1",
    )
    logistic_search.fit(x_train, y_train)
    print(f"LOGISTIC best={logistic_search.best_params_} cv_f1={logistic_search.best_score_:.4f}")

    forest = Pipeline([
        ("prep", make_preprocessor()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    forest_search = GridSearchCV(
        forest,
        {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]},
        cv=5,
        scoring="f1",
    )
    forest_search.fit(x_train, y_train)
    print(f"FOREST best={forest_search.best_params_} cv_f1={forest_search.best_score_:.4f}")

    fitted_scaler = logistic_search.best_estimator_.named_steps["prep"].named_transformers_["num"]
    scaler_frequency = float(fitted_scaler.mean_[NUMERIC.index("frequency")])
    cohort_frequency = float(frame["frequency"].mean())
    print(f"SCALER frequency mean={scaler_frequency:.4f} cohort frequency mean={cohort_frequency:.4f}")
    print("The two means differ because the scaler was fit on the training rows alone and never saw the test rows.")
    if abs(scaler_frequency - cohort_frequency) == 0:
        raise SystemExit("FAIL scaler mean unexpectedly matches the full cohort")

    logistic_test = class_metrics(y_test, logistic_search.predict(x_test))
    forest_test = class_metrics(y_test, forest_search.predict(x_test))
    print_metrics("LOGISTIC TEST", logistic_test)
    print_metrics("FOREST TEST", forest_test)
    if forest_test["f1"] < 0.55 and logistic_test["f1"] < 0.55:
        raise SystemExit("FAIL neither model reached lapsed-class F1 of 0.55")
    selected_name = "random_forest" if forest_test["f1"] >= logistic_test["f1"] else "logistic_regression"
    selected = forest_search.best_estimator_ if selected_name == "random_forest" else logistic_search.best_estimator_
    selected_f1 = forest_test["f1"] if selected_name == "random_forest" else logistic_test["f1"]
    check("selected_f1_at_least_0.55", selected_f1 >= 0.55, True)
    print(f"SELECTED {selected_name} test_f1={selected_f1:.4f}")

    rng = np.random.RandomState(42)
    positive_positions = np.flatnonzero(y_train.to_numpy() == 1)
    n_positive = len(positive_positions)
    n_negative = len(y_train) - n_positive
    check("train_not_lapsed", n_negative, 570)
    drawn = rng.choice(positive_positions, size=n_negative - n_positive, replace=True)
    x_balanced = pd.concat([x_train, x_train.iloc[drawn]], axis=0)
    y_balanced = pd.concat([y_train, y_train.iloc[drawn]], axis=0)
    check("balanced_lapsed", int((y_balanced == 1).sum()), 570)
    check("balanced_not_lapsed", int((y_balanced == 0).sum()), 570)
    balanced_logistic = Pipeline([
        ("prep", make_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, C=logistic_search.best_params_["model__C"])),
    ])
    balanced_logistic.fit(x_balanced, y_balanced)
    balanced_test = class_metrics(y_test, balanced_logistic.predict(x_test))
    print_metrics("BALANCED LOGISTIC TEST", balanced_test)
    moves = {
        "precision": balanced_test["precision"] - logistic_test["precision"],
        "recall": balanced_test["recall"] - logistic_test["recall"],
        "f1": balanced_test["f1"] - logistic_test["f1"],
    }
    moved_most = max(moves, key=lambda name: abs(moves[name]))
    print(
        f"BALANCE MOVE precision={moves['precision']:+.4f} recall={moves['recall']:+.4f} "
        f"f1={moves['f1']:+.4f} most={moved_most}"
    )

    forest_model = forest_search.best_estimator_
    importance_names = list(forest_model.named_steps["prep"].get_feature_names_out())
    importances = forest_model.named_steps["model"].feature_importances_
    ranking = sorted(zip(importance_names, importances), key=lambda item: item[1], reverse=True)
    print("FOREST IMPORTANCE")
    for name, weight in ranking[:5]:
        print(f"  {name} {weight:.4f}")

    print("LOGISTIC SIGNS")
    for feature in ("recency", "frequency", "monetary"):
        value = coefficient(logistic_search.best_estimator_, feature)
        sign = 1 if value > 0 else -1
        print(f"  {feature} {value:+.4f} {READINGS[feature][sign]}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    reloaded = joblib.load(MODEL_PATH)
    original_pred = selected.predict(x_test)
    reloaded_pred = reloaded.predict(x_test)
    if not np.array_equal(original_pred, reloaded_pred):
        raise SystemExit("FAIL reloaded model predictions differ")
    print("RELOAD predictions match")

    segments = pd.read_csv(SEGMENTS)
    scored = frame.loc[index_test, ["customer_id", "lapsed"]].copy()
    scored["lapse_probability"] = selected.predict_proba(x_test)[:, 1]
    premium = scored.merge(
        segments.loc[segments["is_premium"], ["customer_id", "is_premium"]],
        on="customer_id",
        how="inner",
    )
    premium = premium.sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True]
    ).head(10)
    retention = premium.rename(columns={"lapsed": "actually_lapsed"})[
        ["customer_id", "lapse_probability", "actually_lapsed"]
    ]
    retention.to_csv(RETENTION_PATH, index=False)
    print(f"RETENTION rows={len(retention)} actually_lapsed={int(retention['actually_lapsed'].sum())}")
    print(retention.to_string(index=False))
    print("LAPSE PASSED")


if __name__ == "__main__":
    main()
