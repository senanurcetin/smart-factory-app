"""Generate visualization assets for the C-MAPSS RUL case study.

Reads pre-computed JSON outputs from docs/data/cmapss-rul-case-study/ and
writes PNG charts to docs/assets/. No dataset download or model training
required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "cmapss-rul-case-study"
ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#2563EB",
    "accent": "#16A34A",
    "warn": "#D97706",
    "danger": "#DC2626",
    "neutral": "#6B7280",
    "highlight": "#7C3AED",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
}

TRAJECTORY_COLORS = [PALETTE["primary"], PALETTE["accent"], PALETTE["highlight"]]


def _save(fig, name):
    out = ASSETS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  saved -> {out.relative_to(ROOT)}")
    return out


def predicted_vs_actual():
    trajectories = json.loads((DATA_DIR / "sample-trajectories.json").read_text())
    summary = json.loads((DATA_DIR / "summary.json").read_text())

    # Use the final-cycle point of every sample trajectory plus the headline
    # RMSE to frame the scatter; the full official test set predictions
    # aren't persisted per-unit, so this chart illustrates trend using the
    # sample units alongside the reported RMSE/PHM08 numbers.
    true_vals, pred_vals = [], []
    for unit in trajectories["sample_units"]:
        true_vals.append(unit["true_rul_extrapolated"][-1])
        pred_vals.append(unit["predicted_rul"][-1])

    fig, ax = plt.subplots(figsize=(6.5, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    lims = [0, max(true_vals + pred_vals) * 1.15]
    ax.plot(
        lims,
        lims,
        linestyle="--",
        color=PALETTE["neutral"],
        linewidth=1.2,
        label="Perfect prediction",
    )
    ax.scatter(true_vals, pred_vals, color=PALETTE["primary"], s=90, alpha=0.85, zorder=3)
    for unit in trajectories["sample_units"]:
        ax.annotate(
            f"unit {unit['unit_number']}",
            (unit["true_rul_extrapolated"][-1], unit["predicted_rul"][-1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="#1E293B",
        )

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True RUL (cycles)", fontsize=10)
    ax.set_ylabel("Predicted RUL (cycles)", fontsize=10)
    ax.set_title(
        f"Sample Predicted vs. True RUL — Test RMSE {summary['final_model']['rmse']:.1f} cycles",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.7, loc="upper left")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "cmapss-predicted-vs-actual.png")


def degradation_trajectories():
    trajectories = json.loads((DATA_DIR / "sample-trajectories.json").read_text())["sample_units"]

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    for i, unit in enumerate(trajectories):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        ax.plot(
            unit["cycles"],
            unit["true_rul_extrapolated"],
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=0.55,
        )
        ax.plot(
            unit["cycles"],
            unit["predicted_rul"],
            color=color,
            linewidth=2.2,
            label=f"Unit {unit['unit_number']} (solid = predicted, dashed = true*)",
        )

    ax.set_xlabel("Operating cycle", fontsize=10)
    ax.set_ylabel("RUL (cycles)", fontsize=10)
    ax.set_title(
        "Predicted RUL Over Time — Sample Test Engines",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=8, framealpha=0.7)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.text(
        0.01,
        -0.02,
        "*True RUL before the final test cycle is a linear back-projection from the one officially truthed point — illustrative, not scored.",
        fontsize=7.5,
        color=PALETTE["neutral"],
    )

    _save(fig, "cmapss-degradation-trajectories.png")


def model_vs_baseline():
    summary = json.loads((DATA_DIR / "summary.json").read_text())

    labels = ["Naive\n(median RUL)", "Tuned\nHistGradientBoosting"]
    rmse_vals = [summary["baseline"]["rmse"], summary["final_model"]["rmse"]]

    fig, ax = plt.subplots(figsize=(6, 5.5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.bar(
        labels, rmse_vals, color=[PALETTE["neutral"], PALETTE["primary"]], alpha=0.88, width=0.55
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            fontsize=10,
            color="#1E293B",
        )

    ax.set_ylabel("RMSE (cycles) — lower is better", fontsize=10)
    ax.set_title(
        f"Model vs. Naive Baseline on the Official Test Set\n"
        f"({summary['lift_vs_baseline']['rmse_reduction_pct']:.0f}% RMSE reduction)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "cmapss-model-vs-baseline.png")


def feature_importance():
    data = json.loads((DATA_DIR / "feature-importance.json").read_text())
    top = data["features"][:12][::-1]
    names = [row["feature"] for row in top]
    values = [row["mean_abs_shap"] for row in top]

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.barh(names, values, color=PALETTE["highlight"], alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (cycles)", fontsize=10)
    ax.set_title(
        "RUL Model — Top Features by SHAP Importance",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "cmapss-feature-importance.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Generating C-MAPSS RUL case-study visualization assets...")
    predicted_vs_actual()
    degradation_trajectories()
    model_vs_baseline()
    feature_importance()
    logger.info("Done.")
