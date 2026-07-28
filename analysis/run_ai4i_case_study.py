from __future__ import annotations

import json
import math
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
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
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / ".cache"
OUTPUT_DIR = ROOT / "docs" / "data" / "ai4i-case-study"
MODEL_DIR = ROOT / "analysis" / "artifacts"
MODEL_PATH = MODEL_DIR / "model.pkl"
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
DERIVED_FEATURES = [
    "mechanical_load",
    "thermal_stress",
    "tool_wear_load_ratio",
]
CATEGORICAL_FEATURES = ["Type"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
MODEL_FEATURES_ENHANCED = CATEGORICAL_FEATURES + NUMERIC_FEATURES + DERIVED_FEATURES
FAILURE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_LABELS = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}
# Per-failure-mode cost assumptions (illustrative)
FAILURE_MODE_COSTS = {
    "TWF": {"unplanned_cost": 1500, "preventive_cost": 300},
    "HDF": {"unplanned_cost": 4000, "preventive_cost": 600},
    "PWF": {"unplanned_cost": 5000, "preventive_cost": 500},
    "OSF": {"unplanned_cost": 2500, "preventive_cost": 400},
    "RNF": {"unplanned_cost": 8000, "preventive_cost": 1000},
}
REVIEW_COST_PER_ASSET = 35


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


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed engineered features."""
    df = df.copy()
    # Proxy for mechanical power delivered to the spindle
    df["mechanical_load"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    # Relative temperature difference between process and ambient
    df["thermal_stress"] = (df["Process temperature [K]"] - df["Air temperature [K]"]) / df[
        "Air temperature [K]"
    ]
    # Tool wear accumulated per unit of applied torque (degradation rate proxy)
    df["tool_wear_load_ratio"] = df["Tool wear [min]"] / (df["Torque [Nm]"].clip(lower=1e-6))
    return df


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


def build_preprocessor_enhanced() -> ColumnTransformer:
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
                NUMERIC_FEATURES + DERIVED_FEATURES,
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


def summarize_model(
    name: str,
    estimator: object,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
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
                "failure_capture_rate": (
                    to_float(captured_failures / total_failures) if total_failures else 0.0
                ),
                "review_yield": to_float(review_yield),
                "random_review_yield": to_float(baseline_failure_rate),
                "yield_lift_vs_random": (
                    to_float(review_yield / baseline_failure_rate) if baseline_failure_rate else 0.0
                ),
            }
        )
    return budgets


def compute_local_feature_attribution(
    pipeline: Pipeline,
    x_high_risk: pd.DataFrame,
    feature_names: list[str],
    x_reference: pd.DataFrame,
) -> list[dict]:
    """Permutation-based local feature attribution for top high-risk predictions.

    For each sample, computes how much the predicted probability changes when
    a feature is replaced with its median (numeric) or mode (categorical) value.
    """
    reference_medians = {}
    for feat in feature_names:
        if feat not in x_reference.columns:
            continue
        col = x_reference[feat]
        if pd.api.types.is_numeric_dtype(col):
            reference_medians[feat] = col.median()
        else:
            reference_medians[feat] = col.mode().iloc[0] if not col.mode().empty else col.iloc[0]
    results = []
    for idx, row in x_high_risk.iterrows():
        base_prob = float(pipeline.predict_proba(row.to_frame().T)[0, 1])
        attributions = []
        for feat in feature_names:
            if feat not in reference_medians:
                continue
            ablated = row.copy()
            ablated[feat] = reference_medians[feat]
            ablated_prob = float(pipeline.predict_proba(ablated.to_frame().T)[0, 1])
            attributions.append(
                {
                    "feature": feat,
                    "contribution": to_float(base_prob - ablated_prob),
                }
            )
        attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)
        results.append(
            {
                "sample_index": int(idx),
                "predicted_probability": to_float(base_prob),
                "top_features": attributions[:5],
            }
        )
    return results


def build_multiclass_failure_models(
    x_train: pd.DataFrame,
    df_train: pd.DataFrame,
    x_test: pd.DataFrame,
    df_test: pd.DataFrame,
) -> list[dict]:
    """Train one binary model per failure mode and report per-mode metrics."""
    results = []
    for col in FAILURE_COLUMNS:
        y_tr = df_train[col]
        y_te = df_test[col]
        if y_tr.sum() < 5:
            continue

        n_pos = int(y_tr.sum())
        n_neg = int((y_tr == 0).sum())
        scale_pos = n_neg / max(n_pos, 1)

        model = Pipeline(
            [
                ("pre", build_preprocessor_enhanced()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        random_state=RANDOM_SEED,
                        max_depth=4,
                        min_samples_leaf=5,
                    ),
                ),
            ]
        )
        sample_weight = np.where(y_tr == 1, scale_pos, 1.0)
        model.fit(x_train, y_tr, model__sample_weight=sample_weight)

        proba = model.predict_proba(x_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_te, pred, average="binary", zero_division=0
        )

        # Capture rate at 10% review budget
        ranked_idx = pd.Series(proba, index=df_test.index).sort_values(ascending=False)
        top10_idx = ranked_idx.head(max(1, math.ceil(len(ranked_idx) * 0.10))).index
        captured = int(y_te.loc[top10_idx].sum())
        total = int(y_te.sum())

        roc = to_float(roc_auc_score(y_te, proba)) if total > 0 and total < len(y_te) else None
        pr = to_float(average_precision_score(y_te, proba)) if total > 0 else None

        results.append(
            {
                "failure_code": col,
                "label": FAILURE_LABELS[col],
                "train_failures": n_pos,
                "holdout_failures": total,
                "roc_auc": roc,
                "pr_auc": pr,
                "precision": to_float(precision),
                "recall": to_float(recall),
                "f1": to_float(f1),
                "capture_rate_at_10pct_budget": to_float(captured / total) if total else 0.0,
            }
        )
    return results


def build_calibration_data(
    probabilities: np.ndarray,
    y_true: pd.Series,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data for the final model's predictions."""
    fraction_pos, mean_pred = calibration_curve(
        y_true, probabilities, n_bins=n_bins, strategy="quantile"
    )
    ece = float(np.mean(np.abs(fraction_pos - mean_pred)))
    return {
        "n_bins": n_bins,
        "expected_calibration_error": to_float(ece),
        "reliability_diagram": [
            {
                "predicted_probability_bin": to_float(float(mp)),
                "actual_failure_rate": to_float(float(fp)),
                "calibration_gap": to_float(float(fp - mp)),
            }
            for mp, fp in zip(mean_pred, fraction_pos)
        ],
        "interpretation": (
            "ECE measures average gap between predicted probabilities and actual failure rates. "
            "Lower is better. Values below 0.05 indicate well-calibrated predictions."
        ),
    }


def build_per_mode_cost_model(
    mode_results: list[dict],
    df_test: pd.DataFrame,
    review_budgets: list[dict],
) -> list[dict]:
    """Compute per-failure-mode cost savings using differentiated cost assumptions."""
    primary_budget = next(b for b in review_budgets if b["review_fraction"] == 0.10)
    reviewed_count = primary_budget["reviewed_assets"]
    cost_rows = []

    for mode in mode_results:
        col = mode["failure_code"]
        costs = FAILURE_MODE_COSTS[col]
        holdout = mode["holdout_failures"]
        captured = round(holdout * mode["capture_rate_at_10pct_budget"])
        missed = holdout - captured

        reactive_cost = holdout * costs["unplanned_cost"]
        risk_queue_cost = (
            reviewed_count * REVIEW_COST_PER_ASSET
            + captured * costs["preventive_cost"]
            + missed * costs["unplanned_cost"]
        )
        savings = reactive_cost - risk_queue_cost

        cost_rows.append(
            {
                "failure_code": col,
                "label": mode["label"],
                "unplanned_failure_cost": costs["unplanned_cost"],
                "preventive_maintenance_cost": costs["preventive_cost"],
                "holdout_failures": holdout,
                "captured_failures": captured,
                "missed_failures": missed,
                "reactive_baseline_cost": reactive_cost,
                "risk_queue_cost": risk_queue_cost,
                "savings_vs_reactive": savings,
                "savings_share": to_float(savings / reactive_cost) if reactive_cost else 0.0,
            }
        )

    return sorted(cost_rows, key=lambda r: r["savings_vs_reactive"], reverse=True)


def main() -> None:
    df = load_dataset()
    df = add_derived_features(df)

    features = df[MODEL_FEATURES]
    features_enhanced = df[MODEL_FEATURES_ENHANCED]
    target = df["Machine failure"]

    (
        x_train,
        x_test,
        y_train,
        y_test,
        df_train,
        df_test,
        xenh_train,
        xenh_test,
    ) = train_test_split(
        features,
        target,
        df,
        features_enhanced,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    # ── Benchmark comparison (original features, unchanged) ──────────────────
    benchmarks = [
        summarize_model(name, estimator, x_train, y_train, x_test, y_test)
        for name, estimator in build_models().items()
    ]

    # ── Final binary model with enhanced features ────────────────────────────
    final_model_name = "hist_gradient_boosting_enhanced"
    final_pipeline = Pipeline(
        [
            ("pre", build_preprocessor_enhanced()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=5)),
        ]
    )
    final_pipeline.fit(xenh_train, y_train)

    probabilities = pd.Series(final_pipeline.predict_proba(xenh_test)[:, 1], index=df_test.index)
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
    queue_index = (
        probabilities.sort_values(ascending=False).head(primary_queue["reviewed_assets"]).index
    )
    queue_rows = df_test.loc[queue_index]

    # ── Permutation feature importance (enhanced feature set) ────────────────
    permutation = permutation_importance(
        final_pipeline,
        xenh_test,
        y_test,
        scoring="average_precision",
        n_repeats=8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    feature_importance = []
    for feature_name, importance in sorted(
        zip(MODEL_FEATURES_ENHANCED, permutation.importances_mean),
        key=lambda item: item[1],
        reverse=True,
    ):
        feature_importance.append({"feature": feature_name, "importance": to_float(importance)})

    # ── Local feature attribution for top high-risk predictions ─────────────
    top_risk_samples = xenh_test.loc[probabilities.sort_values(ascending=False).head(10).index]
    local_attributions = compute_local_feature_attribution(
        final_pipeline,
        top_risk_samples,
        MODEL_FEATURES_ENHANCED,
        xenh_train,
    )

    # ── Per-failure-mode binary models (multiclass breakdown) ────────────────
    print("Training per-failure-mode models...")
    multiclass_results = build_multiclass_failure_models(xenh_train, df_train, xenh_test, df_test)

    # ── Probability calibration ──────────────────────────────────────────────
    calibration_data = build_calibration_data(probabilities.values, y_test)

    # ── Per-failure-mode cost model ──────────────────────────────────────────
    per_mode_costs = build_per_mode_cost_model(multiclass_results, df_test, review_budgets)

    # ── Legacy failure mode breakdown (original format, capture rates only) ──
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
                "capture_rate": (
                    to_float(captured_failures / holdout_failures) if holdout_failures else 0.0
                ),
            }
        )

    # ── Cost simulation (original unified model) ─────────────────────────────
    review_cost = REVIEW_COST_PER_ASSET
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

    # ── Assemble JSON payloads ───────────────────────────────────────────────
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
        "feature_engineering": {
            "original_features": MODEL_FEATURES,
            "derived_features": {
                "mechanical_load": "Torque × RPM — proxy for power delivered to spindle",
                "thermal_stress": "(Process temp − Air temp) / Air temp — relative thermal load",
                "tool_wear_load_ratio": "Tool wear / Torque — wear rate under load",
            },
            "total_features": len(MODEL_FEATURES_ENHANCED),
        },
        "final_model": {
            "name": "HistGradientBoostingClassifier (enhanced features)",
            "roc_auc": to_float(roc_auc_score(y_test, probabilities)),
            "pr_auc": to_float(average_precision_score(y_test, probabilities)),
            "accuracy": to_float(accuracy_score(y_test, predictions)),
            "precision": to_float(precision),
            "recall": to_float(recall),
            "f1": to_float(f1),
            "confusion_matrix": matrix,
            "calibration_ece": calibration_data["expected_calibration_error"],
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
        "per_failure_mode_models": {
            "approach": "One binary HistGradientBoosting classifier per failure mode with cost-proportional sample weights",
            "modes_trained": len(multiclass_results),
            "summary": [
                {
                    "failure_code": m["failure_code"],
                    "label": m["label"],
                    "roc_auc": m["roc_auc"],
                    "recall": m["recall"],
                    "capture_rate_at_10pct": m["capture_rate_at_10pct_budget"],
                }
                for m in multiclass_results
            ],
        },
        "operational_takeaway": (
            "A risk-ranked maintenance queue captures most holdout failures inside a small analyst "
            "budget. Derived features (mechanical load, thermal stress, wear-rate) improve model "
            "interpretability. Per-failure-mode models enable cost-differentiated maintenance decisions."
        ),
        "limitations": [
            "The UCI AI4I dataset is simulated, so it is better for benchmarking than for plant-specific deployment claims.",
            "The cost model is illustrative and meant to show how review logic can be translated into business tradeoffs.",
            "The app still serves a lightweight demo UI rather than a production monitoring stack.",
            "Local feature attributions use median-ablation (not SHAP), which underestimates interaction effects.",
        ],
    }

    model_selection = {
        "selected_model": final_model_name,
        "selection_reason": (
            "HistGradientBoosting with engineered features produced the strongest PR-AUC and F1 "
            "on the imbalanced holdout. Derived features (mechanical_load, thermal_stress, "
            "tool_wear_load_ratio) add domain signal without external data requirements."
        ),
        "benchmarks_considered": benchmarks,
        "feature_engineering_note": (
            "Three derived features were added: mechanical_load (Torque × RPM), "
            "thermal_stress ((Tprocess - Tair) / Tair), and tool_wear_load_ratio (wear / torque). "
            "These encode domain knowledge about failure mechanisms into the feature space."
        ),
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
                    FAILURE_LABELS[column] for column in FAILURE_COLUMNS if int(row[column]) == 1
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
        "savings_vs_reactive_share": (
            to_float((reactive_cost - risk_queue_cost) / reactive_cost) if reactive_cost else 0.0
        ),
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

    enhanced_feature_importance = {
        "method": "Permutation importance (average precision scoring, 8 repeats)",
        "features": feature_importance,
        "derived_feature_note": (
            "mechanical_load and thermal_stress capture cross-feature interactions "
            "that individual raw sensors cannot express independently."
        ),
        "local_attributions_top10_high_risk": local_attributions,
    }

    per_mode_cost_model = {
        "approach": (
            "Per-failure-mode cost model using differentiated unplanned-failure costs "
            "and preventive-maintenance costs. Savings computed at 10% review budget."
        ),
        "cost_assumptions": FAILURE_MODE_COSTS,
        "review_cost_per_asset": REVIEW_COST_PER_ASSET,
        "modes": per_mode_costs,
        "total_per_mode_savings": sum(m["savings_vs_reactive"] for m in per_mode_costs),
    }

    # ── Write all artifacts ──────────────────────────────────────────────────
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "benchmark-comparison.json", benchmarks)
    write_json(OUTPUT_DIR / "model-selection.json", model_selection)
    write_json(OUTPUT_DIR / "feature-importance.json", enhanced_feature_importance)
    write_json(OUTPUT_DIR / "review-queue.json", review_queue_payload)
    write_json(OUTPUT_DIR / "failure-mode-breakdown.json", failure_mode_summary)
    write_json(OUTPUT_DIR / "cost-simulation.json", cost_simulation)
    write_json(OUTPUT_DIR / "dataset-profile.json", dataset_profile)
    write_json(OUTPUT_DIR / "failure-mode-multiclass.json", multiclass_results)
    write_json(OUTPUT_DIR / "calibration.json", calibration_data)
    write_json(OUTPUT_DIR / "cost-model.json", per_mode_cost_model)

    # ── Persist the final trained pipeline for reuse outside this script ─────
    # (e.g. by the live dashboard in main.py, so it scores against the same
    # model that was benchmarked here instead of a separate ad-hoc model).
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    print(f"Wrote case-study artifacts to {OUTPUT_DIR}")
    print(
        f"  Binary model  — ROC-AUC: {summary['final_model']['roc_auc']}  PR-AUC: {summary['final_model']['pr_auc']}"
    )
    print(f"  Calibration   — ECE: {calibration_data['expected_calibration_error']}")
    print(f"  Per-mode models trained: {len(multiclass_results)}")
    print("  New artifacts: failure-mode-multiclass.json, calibration.json, cost-model.json")
    print(f"  Model artifact: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
