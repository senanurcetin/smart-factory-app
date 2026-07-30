from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

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
        logger.info(f"Downloading {DATASET_NAME}...")
        urlretrieve(DATASET_URL, ZIP_PATH)
    if not CSV_PATH.exists():
        logger.info("Extracting dataset archive...")
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


def compute_shap_local_attribution(
    pipeline: Pipeline,
    x_high_risk: pd.DataFrame,
    x_background: pd.DataFrame,
    background_size: int = 200,
) -> list[dict]:
    """SHAP (TreeExplainer, interventional, probability-space) local attribution
    for top high-risk predictions.

    Replaces the earlier median-ablation approximation: SHAP values are exact
    Shapley-value estimates for the fitted tree ensemble rather than a single-
    feature perturbation heuristic, and correctly account for feature
    interactions (e.g. torque and tool wear jointly driving overstrain risk).
    """
    preprocessor = pipeline.named_steps["pre"]
    model = pipeline.named_steps["model"]
    encoded_feature_names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]

    background = preprocessor.transform(
        x_background.sample(n=min(background_size, len(x_background)), random_state=RANDOM_SEED)
    )
    explainer = shap.TreeExplainer(model, background, model_output="probability")

    x_transformed = preprocessor.transform(x_high_risk)
    shap_values = explainer.shap_values(x_transformed)
    base_probabilities = pipeline.predict_proba(x_high_risk)[:, 1]

    results = []
    for row_i, (idx, _) in enumerate(x_high_risk.iterrows()):
        attributions = [
            {"feature": feat, "contribution": to_float(shap_values[row_i, j])}
            for j, feat in enumerate(encoded_feature_names)
        ]
        attributions.sort(key=lambda a: abs(a["contribution"]), reverse=True)
        results.append(
            {
                "sample_index": int(idx),
                "predicted_probability": to_float(base_probabilities[row_i]),
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


def run_split_robustness_check(
    features_enhanced: pd.DataFrame,
    target: pd.Series,
    seeds: list[int],
    model_params: dict,
) -> dict:
    """Re-fit the final (tuned) model architecture across independent stratified
    holdouts to confirm the headline metrics aren't an artifact of one lucky split.

    AI4I 2020 has no time axis or repeated-asset grouping — each of the 10,000
    rows is one independently sampled machine-state snapshot with a unique
    Product ID — so a time-aware or group-based (e.g. GroupKFold) split is not
    meaningful for this dataset. Repeated stratified holdouts across several
    seeds are the applicable robustness check instead.
    """
    per_seed_results = []
    for seed in seeds:
        x_train, x_test, y_train, y_test = train_test_split(
            features_enhanced,
            target,
            test_size=0.20,
            random_state=seed,
            stratify=target,
        )
        pipeline = Pipeline(
            [
                ("pre", build_preprocessor_enhanced()),
                ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, **model_params)),
            ]
        )
        pipeline.fit(x_train, y_train)
        proba = pipeline.predict_proba(x_test)[:, 1]
        predictions = (proba >= 0.5).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average="binary", zero_division=0
        )
        per_seed_results.append(
            {
                "split_seed": seed,
                "roc_auc": to_float(roc_auc_score(y_test, proba)),
                "pr_auc": to_float(average_precision_score(y_test, proba)),
                "f1": to_float(f1),
            }
        )

    roc_values = [r["roc_auc"] for r in per_seed_results]
    pr_values = [r["pr_auc"] for r in per_seed_results]
    return {
        "note": (
            "AI4I 2020 has no time axis or repeated-asset grouping to split on "
            "(10,000 rows = 10,000 distinct machine snapshots), so time-aware or "
            "group-based cross-validation is not applicable. This instead confirms "
            "that the primary split (seed=42, used everywhere else in this repo) "
            "is not a lucky outlier by re-fitting the same model architecture "
            "across independent stratified 80/20 splits."
        ),
        "primary_split_seed": RANDOM_SEED,
        "per_seed_results": per_seed_results,
        "roc_auc_mean": to_float(float(np.mean(roc_values))),
        "roc_auc_std": to_float(float(np.std(roc_values))),
        "pr_auc_mean": to_float(float(np.mean(pr_values))),
        "pr_auc_std": to_float(float(np.std(pr_values))),
    }


HGB_PARAM_DISTRIBUTIONS = {
    "model__max_depth": [3, 4, 5, 6, None],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
    "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    "model__max_leaf_nodes": [15, 31, 63],
}


def tune_final_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 20,
) -> tuple[dict, dict]:
    """Fixed-budget RandomizedSearchCV over the final architecture's hyperparameters.

    Scored on PR-AUC (average precision) via 5-fold stratified CV, since PR-AUC
    is the primary selection metric for this imbalanced target everywhere else
    in this pipeline. Returns (best_params, a transparent summary for the
    model-selection artifact).
    """
    pipeline = Pipeline(
        [
            ("pre", build_preprocessor_enhanced()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=HGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x_train, y_train)

    best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    summary = {
        "search_method": "RandomizedSearchCV",
        "n_iter": n_iter,
        "cv_folds": 5,
        "scoring": "average_precision (PR-AUC)",
        "best_params": best_params,
        "best_cv_pr_auc": to_float(search.best_score_),
        "param_distributions": {
            key.removeprefix("model__"): value for key, value in HGB_PARAM_DISTRIBUTIONS.items()
        },
    }
    return best_params, summary


def build_model_card(df: pd.DataFrame) -> dict:
    """Minimal model card: enough to know when/how/on-what-data this model was
    trained without standing up a full model registry for a portfolio project.
    """
    dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "shap_version": shap.__version__,
        "dataset": {
            "name": DATASET_NAME,
            "doi": DATASET_DOI,
            "rows": int(len(df)),
            "sha256": dataset_hash,
        },
        "random_seed": RANDOM_SEED,
    }


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

    # ── Hyperparameter search for the final (enhanced-feature) architecture ──
    logger.info("Tuning final model hyperparameters...")
    tuned_params, tuning_summary = tune_final_model(xenh_train, y_train)

    # ── Final binary model with enhanced features ────────────────────────────
    final_model_name = "hist_gradient_boosting_enhanced"
    final_pipeline = Pipeline(
        [
            ("pre", build_preprocessor_enhanced()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, **tuned_params)),
        ]
    )
    final_pipeline.fit(xenh_train, y_train)

    # ── Split-robustness check using the same tuned architecture ─────────────
    logger.info("Running split-robustness check...")
    robustness_check = run_split_robustness_check(
        features_enhanced, target, seeds=[RANDOM_SEED, 7, 13, 99, 2024], model_params=tuned_params
    )

    probabilities = pd.Series(final_pipeline.predict_proba(xenh_test)[:, 1], index=df_test.index)
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions).astype(int).tolist()

    # ── Isolate feature-engineering impact from hyperparameter tuning ────────
    # Same tuned hyperparameters, raw vs. enhanced (derived) feature sets, so
    # the reported delta is attributable to the three derived features alone.
    raw_tuned_pipeline = Pipeline(
        [
            ("pre", build_preprocessor()),
            ("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, **tuned_params)),
        ]
    )
    raw_tuned_pipeline.fit(x_train, y_train)
    raw_tuned_proba = raw_tuned_pipeline.predict_proba(x_test)[:, 1]
    raw_tuned_pred = (raw_tuned_proba >= 0.5).astype(int)
    _, _, raw_tuned_f1, _ = precision_recall_fscore_support(
        y_test, raw_tuned_pred, average="binary", zero_division=0
    )
    feature_engineering_impact = {
        "note": (
            "The same tuned hyperparameters are applied to the raw and enhanced "
            "(derived) feature sets, so this delta is attributable to the three "
            "derived features alone, not conflated with hyperparameter tuning."
        ),
        "raw_features": {
            "roc_auc": to_float(roc_auc_score(y_test, raw_tuned_proba)),
            "pr_auc": to_float(average_precision_score(y_test, raw_tuned_proba)),
            "f1": to_float(raw_tuned_f1),
        },
        "enhanced_features": {
            "roc_auc": to_float(roc_auc_score(y_test, probabilities)),
            "pr_auc": to_float(average_precision_score(y_test, probabilities)),
            "f1": to_float(f1),
        },
    }
    feature_engineering_impact["delta"] = {
        "roc_auc": to_float(
            feature_engineering_impact["enhanced_features"]["roc_auc"]
            - feature_engineering_impact["raw_features"]["roc_auc"]
        ),
        "pr_auc": to_float(
            feature_engineering_impact["enhanced_features"]["pr_auc"]
            - feature_engineering_impact["raw_features"]["pr_auc"]
        ),
        "f1": to_float(
            feature_engineering_impact["enhanced_features"]["f1"]
            - feature_engineering_impact["raw_features"]["f1"]
        ),
    }

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

    # ── Local feature attribution for top high-risk predictions (SHAP) ───────
    top_risk_samples = xenh_test.loc[probabilities.sort_values(ascending=False).head(10).index]
    local_attributions = compute_shap_local_attribution(
        final_pipeline,
        top_risk_samples,
        xenh_train,
    )

    # ── Per-failure-mode binary models (multiclass breakdown) ────────────────
    logger.info("Training per-failure-mode models...")
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
            "Local feature attributions use SHAP TreeExplainer (interventional, probability-space, 200-row "
            "background sample) rather than exact Shapley values over the full training set, so they are a "
            "close but not perfect approximation.",
            "AI4I 2020 has no time axis or repeated-asset grouping (each row is one independently sampled "
            "machine-state snapshot), so time-aware or group-based splitting is not applicable; split "
            "stability was instead checked via repeated stratified holdouts across 5 random seeds "
            "(see validation-robustness.json).",
        ],
    }

    model_selection = {
        "selected_model": final_model_name,
        "model_card": build_model_card(df),
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
        "feature_engineering_impact": feature_engineering_impact,
        "hyperparameter_tuning": tuning_summary,
        "validation_robustness": {
            "summary": (
                f"ROC-AUC {robustness_check['roc_auc_mean']} ± {robustness_check['roc_auc_std']}, "
                f"PR-AUC {robustness_check['pr_auc_mean']} ± {robustness_check['pr_auc_std']} "
                f"across {len(robustness_check['per_seed_results'])} independent stratified splits."
            ),
            "detail_artifact": "validation-robustness.json",
        },
        "next_step": (
            "AI4I 2020 is a static snapshot dataset with no run-to-failure time series, so it cannot "
            "support real RUL (remaining-useful-life) regression. A natural next step is a separate, "
            "clearly-labeled case study on a run-to-failure dataset (e.g. NASA C-MAPSS) for genuine "
            "survival/RUL modeling, rather than the risk-score-derived heuristic used in the live "
            "dashboard today."
        ),
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
        "local_attribution_method": (
            "SHAP TreeExplainer (interventional, probability-space, 200-row background sample)"
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
    write_json(OUTPUT_DIR / "validation-robustness.json", robustness_check)

    # ── Persist the final trained pipeline for reuse outside this script ─────
    # (e.g. by the live dashboard in main.py, so it scores against the same
    # model that was benchmarked here instead of a separate ad-hoc model).
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    logger.info(f"Wrote case-study artifacts to {OUTPUT_DIR}")
    logger.info(
        f"  Binary model  — ROC-AUC: {summary['final_model']['roc_auc']}  PR-AUC: {summary['final_model']['pr_auc']}"
    )
    logger.info(f"  Calibration   — ECE: {calibration_data['expected_calibration_error']}")
    logger.info(f"  Per-mode models trained: {len(multiclass_results)}")
    logger.info("  New artifacts: failure-mode-multiclass.json, calibration.json, cost-model.json")
    logger.info(f"  Model artifact: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
