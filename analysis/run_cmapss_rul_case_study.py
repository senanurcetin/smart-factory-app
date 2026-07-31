"""NASA C-MAPSS (FD001) Remaining Useful Life (RUL) regression case study.

Unlike AI4I 2020 (a static snapshot dataset — one row per independently
sampled machine, no time axis), C-MAPSS is a genuine run-to-failure dataset:
each engine ("unit") is observed over many operating cycles until failure.
That structure is what makes real RUL regression possible here, and also
what makes a proper *grouped* train/validation split possible (rows from the
same engine never span both sides of a fold) — the exact rigor technique
AI4I's pipeline explicitly could not apply.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

try:
    from analysis._common import build_model_card, to_float, write_json
except ImportError:
    from _common import build_model_card, to_float, write_json

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "analysis" / ".cache"
OUTPUT_DIR = ROOT / "docs" / "data" / "cmapss-rul-case-study"
MODEL_DIR = ROOT / "analysis" / "artifacts"
MODEL_PATH = MODEL_DIR / "rul_model.pkl"

DATASET_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
DATASET_NAME = "NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)"
DATASET_REFERENCE = (
    "A. Saxena, K. Goebel, D. Simon, N. Eklund, "
    '"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", '
    "PHM08, Denver CO, 2008 (NASA Prognostics Center of Excellence Data Repository)"
)
ZIP_PATH = CACHE_DIR / "cmapss-dataset.zip"
EXTRACT_DIR = CACHE_DIR / "cmapss-dataset"
INNER_ZIP_ENTRY = "6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData.zip"
TRAIN_PATH = EXTRACT_DIR / "train_FD001.txt"
TEST_PATH = EXTRACT_DIR / "test_FD001.txt"
RUL_TRUTH_PATH = EXTRACT_DIR / "RUL_FD001.txt"

RANDOM_SEED = 42
RUL_CAP = 125  # standard C-MAPSS practice: early-life RUL is non-linear/uninformative
ROLLING_WINDOW = 5
SENSOR_STD_THRESHOLD = 1e-5  # drop sensors that are ~constant under FD001's single condition

ID_COLUMNS = ["unit_number", "time_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

HGB_PARAM_DISTRIBUTIONS = {
    "model__max_depth": [3, 4, 5, 6, None],
    "model__learning_rate": [0.03, 0.05, 0.1, 0.15, 0.2],
    "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
    "model__max_leaf_nodes": [15, 31, 63],
}


def ensure_dataset() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        logger.info(f"Downloading {DATASET_NAME}...")
        try:
            urlretrieve(DATASET_URL, ZIP_PATH)
        except (URLError, OSError) as exc:
            raise RuntimeError(
                f"Could not download {DATASET_NAME} from {DATASET_URL} ({exc}). "
                "NASA's prognostics data repository has changed hosts before. If this URL "
                "is now stale, download the '6. Turbofan Engine Degradation Simulation Data "
                f"Set' zip manually and save it as {ZIP_PATH}, then re-run this script."
            ) from exc
    if not TRAIN_PATH.exists():
        logger.info("Extracting dataset archive...")
        with zipfile.ZipFile(ZIP_PATH, "r") as outer_zip:
            inner_zip_bytes = outer_zip.read(INNER_ZIP_ENTRY)
        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as inner_zip:
            inner_zip.extractall(EXTRACT_DIR)


def _load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, sep=r"\s+", header=None, names=RAW_COLUMNS, usecols=range(len(RAW_COLUMNS))
    )


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    ensure_dataset()
    train_df = _load_raw(TRAIN_PATH)
    test_df = _load_raw(TEST_PATH)
    rul_truth = pd.read_csv(RUL_TRUTH_PATH, sep=r"\s+", header=None, names=["RUL"])["RUL"]
    return train_df, test_df, rul_truth


def select_informative_sensors(
    train_df: pd.DataFrame, std_threshold: float = SENSOR_STD_THRESHOLD
) -> list[str]:
    """Drop sensors with ~zero variance under FD001's single operating condition —
    they carry no signal and would just add noise to the feature set."""
    stds = train_df[SENSOR_COLUMNS].std()
    return [col for col in SENSOR_COLUMNS if stds[col] > std_threshold]


def add_rolling_features(
    df: pd.DataFrame, sensor_cols: list[str], window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """Add per-unit rolling mean/std over the last `window` cycles for each sensor.

    A single raw reading is noisy; a short rolling window captures the trend
    direction and volatility that better tracks degradation. `min_periods=1`
    means early cycles (before a full window has accumulated) still get a
    value computed from whatever history exists so far.
    """
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True).copy()
    grouped = df.groupby("unit_number")
    for col in sensor_cols:
        rolling = grouped[col].rolling(window=window, min_periods=1)
        df[f"{col}_roll_mean"] = rolling.mean().reset_index(level=0, drop=True)
        df[f"{col}_roll_std"] = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
    return df


def add_train_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit_number")["time_cycles"].transform("max")
    df["RUL"] = (max_cycle - df["time_cycles"]).clip(upper=cap)
    return df


def build_official_test_rows(test_df: pd.DataFrame) -> pd.DataFrame:
    """The C-MAPSS evaluation protocol only truths the RUL at each test engine's
    final observed cycle — not every cycle — so evaluation uses exactly one
    row per test unit (its last), matched against RUL_FD001.txt in unit order.
    """
    return (
        test_df.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number")
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, dtype=float) - np.asarray(y_pred)) ** 2)))


def phm08_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA's PHM08 challenge scoring function (Saxena et al., 2008).

    Asymmetric by design: predicting *more* RUL than actually remains (a late/
    optimistic prediction, d >= 0) is penalized more heavily than predicting
    less (an early/conservative prediction, d < 0) — an engine taken out of
    service too early is a cost; one flown too long on an over-optimistic
    prediction is a safety risk.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    penalty = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalty))


def build_pipeline(params: dict | None = None) -> Pipeline:
    model_kwargs = dict(random_state=RANDOM_SEED)
    if params:
        model_kwargs.update(params)
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**model_kwargs)),
        ]
    )


def tune_model(
    x_train: pd.DataFrame, y_train: pd.Series, groups: pd.Series, n_iter: int = 15
) -> tuple[dict, dict]:
    """RandomizedSearchCV with GroupKFold grouped by unit_number — engines are
    never split across train/validation within a fold, unlike a plain KFold
    which would let a model "cheat" by seeing other cycles of the same engine
    it's being validated on.
    """
    search = RandomizedSearchCV(
        build_pipeline(),
        param_distributions=HGB_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=GroupKFold(n_splits=5),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=False,
    )
    search.fit(x_train, y_train, groups=groups)

    best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
    summary = {
        "search_method": "RandomizedSearchCV",
        "n_iter": n_iter,
        "cv_folds": 5,
        "cv_strategy": (
            "GroupKFold grouped by unit_number — rows from the same engine never appear "
            "in both the train and validation side of a fold. This is the grouped/"
            "time-aware split AI4I's static, one-row-per-machine structure cannot support."
        ),
        "scoring": "neg_root_mean_squared_error",
        "best_params": best_params,
        "best_cv_rmse": to_float(-search.best_score_),
        "param_distributions": {
            key.removeprefix("model__"): value for key, value in HGB_PARAM_DISTRIBUTIONS.items()
        },
    }
    return best_params, summary


def build_sample_trajectories(
    test_df: pd.DataFrame,
    official_test_rows: pd.DataFrame,
    rul_truth: pd.Series,
    pipeline: Pipeline,
    feature_cols: list[str],
    sample_units: tuple[int, ...] = (1, 2, 3),
) -> list[dict]:
    """Per-cycle predicted RUL for a few sample engines, for the degradation-curve
    chart. Only the final cycle of each test engine has an official ground-truth
    RUL; earlier cycles' "true" RUL here is a linear back-projection from that
    single truthed point, for visualization only — never used for scoring.
    """
    final_cycle_by_unit = dict(
        zip(official_test_rows["unit_number"], official_test_rows["time_cycles"])
    )
    truth_by_unit = dict(zip(official_test_rows["unit_number"], rul_truth.values))

    trajectories = []
    for unit in sample_units:
        unit_df = test_df[test_df["unit_number"] == unit].sort_values("time_cycles")
        if unit_df.empty:
            continue
        predicted = pipeline.predict(unit_df[feature_cols])
        final_cycle = final_cycle_by_unit[unit]
        true_rul_at_final = truth_by_unit[unit]
        true_rul_extrapolated = (final_cycle - unit_df["time_cycles"]) + true_rul_at_final
        trajectories.append(
            {
                "unit_number": int(unit),
                "cycles": unit_df["time_cycles"].astype(int).tolist(),
                "predicted_rul": [to_float(v) for v in predicted],
                "true_rul_extrapolated": [to_float(v) for v in true_rul_extrapolated],
            }
        )
    return trajectories


def main() -> None:
    train_df, test_df, rul_truth = load_dataset()

    informative_sensors = select_informative_sensors(train_df)
    dropped_sensors = sorted(set(SENSOR_COLUMNS) - set(informative_sensors))
    logger.info(
        f"Kept {len(informative_sensors)}/{len(SENSOR_COLUMNS)} sensors "
        f"(dropped near-constant: {dropped_sensors})"
    )

    raw_feature_cols = SETTING_COLUMNS + informative_sensors
    rolling_feature_cols = [f"{col}_roll_mean" for col in informative_sensors] + [
        f"{col}_roll_std" for col in informative_sensors
    ]
    enhanced_feature_cols = raw_feature_cols + rolling_feature_cols

    train_enhanced = add_train_rul(add_rolling_features(train_df, informative_sensors))
    test_enhanced = add_rolling_features(test_df, informative_sensors)
    official_test_rows = build_official_test_rows(test_enhanced)

    if len(official_test_rows) != len(rul_truth):
        raise RuntimeError(
            f"Official test set has {len(official_test_rows)} units but "
            f"RUL_FD001.txt has {len(rul_truth)} truth values — dataset mismatch."
        )
    y_test_true = rul_truth.values

    # ── Hyperparameter search (enhanced features, GroupKFold by unit) ────────
    logger.info("Tuning RUL model hyperparameters...")
    tuned_params, tuning_summary = tune_model(
        train_enhanced[enhanced_feature_cols], train_enhanced["RUL"], train_enhanced["unit_number"]
    )

    # ── Final model on enhanced features ──────────────────────────────────────
    final_pipeline = build_pipeline(tuned_params)
    final_pipeline.fit(train_enhanced[enhanced_feature_cols], train_enhanced["RUL"])
    final_predictions = final_pipeline.predict(official_test_rows[enhanced_feature_cols])
    final_rmse = rmse(y_test_true, final_predictions)
    final_score = phm08_score(y_test_true, final_predictions)

    # ── Naive baseline: always predict the median training RUL ───────────────
    median_train_rul = float(train_enhanced["RUL"].median())
    baseline_predictions = np.full_like(y_test_true, fill_value=median_train_rul, dtype=float)
    baseline_rmse = rmse(y_test_true, baseline_predictions)
    baseline_score = phm08_score(y_test_true, baseline_predictions)

    # ── Isolate rolling-window feature impact (same tuned params, raw vs enhanced) ──
    raw_pipeline = build_pipeline(tuned_params)
    raw_pipeline.fit(train_enhanced[raw_feature_cols], train_enhanced["RUL"])
    raw_predictions = raw_pipeline.predict(official_test_rows[raw_feature_cols])
    raw_rmse = rmse(y_test_true, raw_predictions)
    raw_score = phm08_score(y_test_true, raw_predictions)

    # ── SHAP global feature importance on the official test set ──────────────
    logger.info("Computing SHAP feature importance...")
    imputer = final_pipeline.named_steps["impute"]
    model = final_pipeline.named_steps["model"]
    test_imputed = imputer.transform(official_test_rows[enhanced_feature_cols])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_imputed)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        (
            {"feature": feat, "mean_abs_shap": to_float(val)}
            for feat, val in zip(enhanced_feature_cols, mean_abs_shap)
        ),
        key=lambda row: row["mean_abs_shap"],
        reverse=True,
    )

    # ── Sample degradation trajectories for charting ──────────────────────────
    sample_trajectories = build_sample_trajectories(
        test_enhanced, official_test_rows, rul_truth, final_pipeline, enhanced_feature_cols
    )

    # ── Assemble JSON payloads ────────────────────────────────────────────────
    summary = {
        "project": "RUL regression on a genuine run-to-failure dataset",
        "portfolio_role": (
            "second, separate case study — demonstrates real remaining-useful-life "
            "regression, which AI4I 2020's static snapshot structure cannot support"
        ),
        "dataset": {
            "name": DATASET_NAME,
            "reference": DATASET_REFERENCE,
            "subset": "FD001 (one operating condition, one fault mode)",
            "train_units": int(train_df["unit_number"].nunique()),
            "test_units": int(test_df["unit_number"].nunique()),
            "train_rows": int(len(train_df)),
        },
        "target": {
            "definition": "RUL = max_cycle_for_unit - current_cycle, capped at "
            f"{RUL_CAP} cycles (standard C-MAPSS practice — early-life RUL is "
            "non-linear/uninformative)",
            "evaluation_protocol": (
                "Official held-out test set: exactly one row per test engine (its final "
                "observed cycle), scored against NASA's provided RUL_FD001.txt ground "
                "truth — the dataset's own designed evaluation protocol, not a custom split."
            ),
        },
        "sensors": {
            "total": len(SENSOR_COLUMNS),
            "kept_informative": len(informative_sensors),
            "dropped_near_constant": dropped_sensors,
        },
        "final_model": {
            "name": "HistGradientBoostingRegressor (rolling-window features, tuned)",
            "rmse": to_float(final_rmse),
            "phm08_score": to_float(final_score),
        },
        "baseline": {
            "name": f"Naive: always predict median training RUL ({to_float(median_train_rul)})",
            "rmse": to_float(baseline_rmse),
            "phm08_score": to_float(baseline_score),
        },
        "lift_vs_baseline": {
            "rmse_reduction_pct": to_float((baseline_rmse - final_rmse) / baseline_rmse * 100),
            "phm08_score_reduction_pct": to_float(
                (baseline_score - final_score) / baseline_score * 100
            ),
        },
        "operational_takeaway": (
            "A tuned gradient-boosting regressor on rolling-window sensor features "
            "substantially beats a naive median-RUL baseline on NASA's own held-out "
            "test protocol, using a grouped cross-validation strategy that AI4I's "
            "static dataset structure cannot support."
        ),
        "limitations": [
            "FD001 is the simplest C-MAPSS subset (one operating condition, one fault "
            "mode) — FD002/FD004 (six conditions, two fault modes) are harder and not "
            "covered here.",
            "The RUL cap (125 cycles) follows common practice but is a modeling choice, "
            "not a physical constant.",
            "Per-cycle 'true RUL' shown in sample trajectory charts before an engine's "
            "final test cycle is a linear back-projection from the single officially "
            "truthed point, not an independently verified value.",
            "This is a simulated benchmark (PHM08 challenge data), not flight data from "
            "a real fleet.",
        ],
    }

    model_selection = {
        "selected_model": "hist_gradient_boosting_regressor",
        "selection_reason": (
            "Consistent with the AI4I case study's model choice: a tree ensemble needs "
            "no feature scaling, trains in seconds on this dataset size, and supports "
            "exact/fast SHAP explanations via TreeExplainer."
        ),
        "model_card": build_model_card(train_df, DATASET_NAME, DATASET_REFERENCE, RANDOM_SEED),
        "hyperparameter_tuning": tuning_summary,
        "feature_engineering_impact": {
            "note": (
                "Same tuned hyperparameters applied to raw sensor readings vs. "
                "rolling-window (mean/std) engineered features, so the delta is "
                "attributable to the rolling-window features alone."
            ),
            "raw_features": {"rmse": to_float(raw_rmse), "phm08_score": to_float(raw_score)},
            "enhanced_features": {
                "rmse": to_float(final_rmse),
                "phm08_score": to_float(final_score),
            },
            "delta": {
                "rmse": to_float(raw_rmse - final_rmse),
                "phm08_score": to_float(raw_score - final_score),
            },
        },
        "next_step": (
            "Extend to FD002/FD004 (multiple operating conditions) to test whether the "
            "rolling-window feature set generalizes beyond a single flight regime."
        ),
    }

    feature_importance_payload = {
        "method": "SHAP TreeExplainer (exact for tree ensembles) on the official test set",
        "features": feature_importance,
    }

    trajectories_payload = {"sample_units": sample_trajectories}

    # ── Write artifacts ────────────────────────────────────────────────────
    write_json(OUTPUT_DIR / "summary.json", summary)
    write_json(OUTPUT_DIR / "model-selection.json", model_selection)
    write_json(OUTPUT_DIR / "feature-importance.json", feature_importance_payload)
    write_json(OUTPUT_DIR / "sample-trajectories.json", trajectories_payload)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_PATH)

    logger.info(f"Wrote RUL case-study artifacts to {OUTPUT_DIR}")
    logger.info(
        f"  Final model — RMSE: {to_float(final_rmse)}  PHM08 score: {to_float(final_score)}"
    )
    logger.info(
        f"  Baseline    — RMSE: {to_float(baseline_rmse)}  PHM08 score: {to_float(baseline_score)}"
    )
    logger.info(f"  Model artifact: {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
