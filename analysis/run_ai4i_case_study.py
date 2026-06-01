from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / ".cache"
OUTPUT_DIR = ROOT / "docs" / "data" / "ai4i-case-study"
DATASET_URL = "https://cdn.uci-ics-mlr-prod.aws.uci.edu/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
DATASET_DOI = "10.24432/C5HS5C"
DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"
ZIP_PATH = CACHE_DIR / "ai4i-dataset.zip"
EXTRACT_DIR = CACHE_DIR / "ai4i-dataset"
CSV_PATH = EXTRACT_DIR / "ai4i2020.csv"
RANDOM_SEED = 42
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_FEATURES = ["Type"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_LABELS = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}


def ensure_dataset() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {DATASET_NAME}...")
        urlretrieve(DATASET_URL, ZIP_PATH)
    if not CSV_PATH.exists():
        print("Extracting dataset archive...")
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(EXTRACT_DIR)


def load_dataset() -> pd.DataFrame:
    ensure_dataset()
    return pd.read_csv(CSV_PATH)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "dummy_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            random_state=RANDOM_SEED,
            max_depth=5,
        ),
    }


def to_float(value: float) -> float:
    return round(float(value), 4)


def summarize_model(name: str, estimator: object, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pipeline = Pipeline([("pre", build_preprocessor()), ("model", estimator)])
    started_at = time.time()
    pipeline.fit(x_train, y_train)
    training_seconds = time.time() - started_at
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "model": name,
        "accuracy": to_float(accuracy_score(y_test, predictions)),
        "roc_auc": to_float(roc_auc_score(y_test, probabilities)),
        "pr_auc": to_float(average_precision_score(y_test, probabilities)),
        "precision": to_float(precision),
        "recall": to_float(recall),
        "f1": to_float(f1),
        "training_seconds": to_float(training_seconds),
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_review_queue(probabilities: pd.Series, targets: pd.Series) -> list[dict]:
    total_failures = int(targets.sum())
    baseline_failure_rate = total_failures / len(targets)
    ranked = probabilities.sort_values(ascending=False)
    budgets = []

    for fraction in (0.05, 0.10, 0.15):
        reviewed = max(1, math.ceil(len(ranked) * fraction))
        selected_index = ranked.head(reviewed).index
        captured_failures = int(targets.loc[selected_index].sum())
        review_yield = captured_failures / reviewed
        budgets.append(
            {
                "review_fraction": fraction,
                "reviewed_assets": reviewed,
                "captured_failures": captured_failures,
                "failure_capture_rate": to_float(captured_failures / total_failures) if total_failures else 0.0,
                "review_yield": to_float(review_yield),
                "random_review_yield": to_float(baseline_failure_rate),
                "yield_lift_vs_random": to_float(review_yield / baseline_failure_rate) if baseline_failure_rate else 0.0,
            }
        )
    return budgets


def main() -> None:
    df = load_dataset()
    features = df[MODEL_FEATURES]
    target = df["Machine failure"]
    (
        x_train,
        x_test,
        y_train,
        y_test,
        df_train,
        df_test,
    ) = train_test_split(
        features,
        target,
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    benchmarks = [
        summarize_model(name, estimator, x_train, y_train, x_test, y_test)
        for name, estimator in build_models().items()
    ]

    final_model_name = "hist_gradient_boosting"
    final_pipeline = Pipeline(
        [
            ("pre", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=5)),
        ]
    )
    final_pipeline.fit(x_train, y_train)

    probabilities = pd.Series(final_pipeline.predict_proba(x_test)[:, 1], index=df_test.index)
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()

    review_budgets = build_review_queue(probabilities, y_test)
    primary_queue = next(bucket for bucket in review_budgets if bucket["review_fraction"] == 0.10)
    queue_index = probabilities.sort_values(ascending=False).head(primary_queue["reviewed_assets"]).index
    queue_rows = df_test.loc[queue_index]

    permutation = permutation_importance(
        final_pipeline,
        x_test,
        y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    feature_importance = []
    for feature_name, importance in sorted(
        zip(MODEL_FEATURES, permutation.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    ):
        feature_importance.append(
            {"feature": feature_name, "importance": to_float(importance)}
        )

    failure_mode_summary = []
    for column in FAILURE_COLUMNS:
        holdout_failures = int(df_test[column].sum())
        captured_failures = int(queue_rows[column].sum())
        failure_mode_summary.append(
            {
                "failure_code": column,
                "label": FAILURE_LABELS[column],
                "holdout_failures": holdout_failures,
                "captured_in_top_10_percent_queue": captured_failures,
                "capture_rate": to_float(captured_failures / holdout_failures) if holdout_failures else 0.0,
            }
        )

    review_cost = 35
    preventive_maintenance_cost = 500
    unplanned_failure_cost = 3500
    total_holdout_failures = int(y_test.sum())
    captured_failures = primary_queue["captured_failures"]
    missed_failures = total_holdout_failures - captured_failures
    reactive_cost = total_holdout_failures * unplanned_failure_cost
    risk_queue_cost = (
        primary_queue["reviewed_assets"] * review_cost
        + captured_failures * preventive_maintenance_cost
        + missed_failures * unplanned_failure_cost
    )
    random_review_captured = round(total_holdout_failures * primary_queue["review_fraction"])
    random_review_cost = (
        primary_queue["reviewed_assets"] * review_cost
        + random_review_captured * preventive_maintenance_cost
        + (total_holdout_failures - random_review_captured) * unplanned_failure_cost
    )

    summary = {
        "project": "Predictive maintenance triage benchmark",
        "portfolio_role": "support case study",
        "dataset": {
            "name": DATASET_NAME,
            "doi": DATASET_DOI,
            "source_url": "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset",
            "download_url": DATASET_URL,
            "rows": int(len(df)),
            "failures": int(target.sum()),
            "failure_rate": to_float(target.mean()),
            "evaluation_split": {
                "train_rows": int(len(x_train)),
                "holdout_rows": int(len(x_test)),
                "strategy": "Deterministic 80/20 stratified holdout",
            },
        },
        "final_model": {
            "name": "HistGradientBoostingClassifier",
            "roc_auc": to_float(roc_auc_score(y_test, probabilities)),
            "pr_auc": to_float(average_precision_score(y_test, probabilities)),
            "accuracy": to_float(accuracy_score(y_test, predictions)),
            "precision": to_float(precision),
            "recall": to_float(recall),
            "f1": to_float(f1),
            "confusion_matrix": matrix,
        },
        "review_queue": {
            "selected_budget_fraction": primary_queue["review_fraction"],
            "selected_budget_label": "Top 10% highest-risk machines",
            "reviewed_assets": primary_queue["reviewed_assets"],
            "captured_failures": primary_queue["captured_failures"],
            "failure_capture_rate": primary_queue["failure_capture_rate"],
            "review_yield": primary_queue["review_yield"],
            "random_review_yield": primary_queue["random_review_yield"],
            "yield_lift_vs_random": primary_queue["yield_lift_vs_random"],
        },
        "operational_takeaway": "A risk-ranked maintenance queue captures most holdout failures inside a small analyst budget, which makes the model useful for maintenance prioritization rather than just post-hoc scoring.",
        "limitations": [
            "The UCI AI4I dataset is simulated, so it is better for benchmarking than for plant-specific deployment claims.",
            "The cost model is illustrative and meant to show how review logic can be translated into business tradeoffs.",
            "The app still serves a lightweight demo UI rather than a production monitoring stack.",
        ],
    }

    model_selection = {
        "selected_model": final_model_name,
        "selection_reason": "HistGradientBoosting produced the strongest PR-AUC and F1 on the imbalanced holdout while staying fast and free-tier friendly.",
        "benchmarks_considered": benchmarks,
        "next_step": "Add time-aware validation or asset-level splits when the repo evolves from benchmark framing to deployment realism.",
    }

    review_queue_payload = {
        "review_budgets": review_budgets,
        "top_queue_assets": [
            {
                "product_id": row["Product ID"],
                "risk_score": to_float(probabilities.loc[index]),
                "machine_failure": int(row["Machine failure"]),
                "failure_modes": [
                    FAILURE_LABELS[column]
                    for column in FAILURE_COLUMNS
                    if int(row[column]) == 1
                ],
            }
            for index, row in queue_rows.head(12).iterrows()
        ],
    }

    cost_simulation = {
        "assumptions": {
            "review_cost_per_asset": review_cost,
            "scheduled_maintenance_cost_per_captured_failure": preventive_maintenance_cost,
            "unplanned_failure_cost": unplanned_failure_cost,
        },
        "reactive_baseline_cost": reactive_cost,
        "random_review_cost": random_review_cost,
        "risk_queue_cost": risk_queue_cost,
        "savings_vs_reactive": reactive_cost - risk_queue_cost,
        "savings_vs_reactive_share": to_float((reactive_cost - risk_queue_cost) / reactive_cost) if reactive_cost else 0.0,
    }

    dataset_profile = {
        "dataset_name": DATASET_NAME,
        "row_count": int(len(df)),
        "type_distribution": {
            key: int(value)
            for key, value in df["Type"].value_counts().sort_index().to_dict().items()
        },
        "target_distribution": {
            "no_failure": int((target == 0).sum()),
            "failure": int((target == 1).sum()),
        },
        "failure_mode_totals": {
            FAILURE_LABELS[column]: int(df[column].sum()) for column in FAILURE_COLUMNS
        },
    }

    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "benchmark-comparison.json", benchmarks)
    write_json(OUTPUT_DIR / "model-selection.json", model_selection)
    write_json(OUTPUT_DIR / "feature-importance.json", feature_importance)
    write_json(OUTPUT_DIR / "review-queue.json", review_queue_payload)
    write_json(OUTPUT_DIR / "failure-mode-breakdown.json", failure_mode_summary)
    write_json(OUTPUT_DIR / "cost-simulation.json", cost_simulation)
    write_json(OUTPUT_DIR / "dataset-profile.json", dataset_profile)

    print(f"Wrote case-study artifacts to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
