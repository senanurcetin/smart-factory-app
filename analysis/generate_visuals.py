"""Generate portfolio visualization assets from JSON artifacts.

Reads pre-computed JSON outputs from docs/data/ai4i-case-study/ and writes
PNG charts to docs/assets/.  No dataset download or model training required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "ai4i-case-study"
ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "primary": "#2563EB",
    "accent": "#16A34A",
    "warn": "#D97706",
    "neutral": "#6B7280",
    "highlight": "#7C3AED",
    "bg": "#F8FAFC",
    "grid": "#E2E8F0",
}


def _save(fig, name):
    out = ASSETS_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"  saved -> {out.relative_to(ROOT)}")
    return out


def model_comparison():
    data = json.loads((DATA_DIR / "benchmark-comparison.json").read_text())

    labels = {
        "dummy_baseline": "Dummy\nBaseline",
        "logistic_regression": "Logistic\nRegression",
        "random_forest": "Random\nForest",
        "hist_gradient_boosting": "Hist Gradient\nBoosting *",
    }
    models = [labels[d["model"]] for d in data]
    roc = [d["roc_auc"] for d in data]
    pr = [d["pr_auc"] for d in data]
    f1 = [d["f1"] for d in data]

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    b1 = ax.bar(x - width, roc, width, label="ROC-AUC", color=PALETTE["primary"], alpha=0.85)
    b2 = ax.bar(x, pr, width, label="PR-AUC", color=PALETTE["accent"], alpha=0.85)
    b3 = ax.bar(x + width, f1, width, label="F1", color=PALETTE["highlight"], alpha=0.85)

    for bar in list(b1) + list(b2) + list(b3):
        h = bar.get_height()
        if h > 0.01:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.012,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#1E293B",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(
        "Model Benchmark Comparison — AI4I 2020 Dataset\n(* = selected model)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "model-comparison.png")


def feature_importance():
    data = json.loads((DATA_DIR / "feature-importance.json").read_text())
    feats = data["features"]

    names = [f["feature"] for f in feats]
    values = [f["importance"] for f in feats]
    colors = [PALETTE["primary"] if v >= 0 else PALETTE["warn"] for v in values]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    for bar, val in zip(bars, values[::-1]):
        offset = 0.008 if val >= 0 else -0.008
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha=ha,
            fontsize=8,
        )

    ax.axvline(0, color="#64748B", linewidth=0.8)
    ax.set_xlabel("Permutation Importance (avg precision scoring, 8 repeats)", fontsize=9)
    ax.set_title(
        "Feature Importance — HistGradientBoosting on AI4I", fontsize=12, fontweight="bold", pad=12
    )
    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pos_patch = mpatches.Patch(color=PALETTE["primary"], alpha=0.85, label="Positive impact")
    neg_patch = mpatches.Patch(color=PALETTE["warn"], alpha=0.85, label="Near-zero / negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8, framealpha=0.7)

    _save(fig, "feature-importance.png")


def review_queue_curve():
    data = json.loads((DATA_DIR / "review-queue.json").read_text())
    budgets = data["review_budgets"]

    fracs = [0.0] + [b["review_fraction"] for b in budgets]
    capture = [0.0] + [b["failure_capture_rate"] for b in budgets]
    random_line = [b["review_fraction"] for b in budgets]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.plot(
        fracs,
        capture,
        "o-",
        color=PALETTE["primary"],
        linewidth=2.2,
        markersize=7,
        label="Model (ranked queue)",
    )
    ax.plot(
        [0] + random_line,
        [0] + random_line,
        "--",
        color=PALETTE["neutral"],
        linewidth=1.5,
        label="Random review baseline",
    )

    for b in budgets:
        ax.annotate(
            f"{b['failure_capture_rate']:.1%}  (x{b['yield_lift_vs_random']:.1f} lift)",
            xy=(b["review_fraction"], b["failure_capture_rate"]),
            xytext=(b["review_fraction"] + 0.006, b["failure_capture_rate"] - 0.045),
            fontsize=8,
            color="#1E293B",
        )

    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel("Fraction of assets reviewed", fontsize=10)
    ax.set_ylabel("Failure capture rate", fontsize=10)
    ax.set_title(
        "Maintenance Review Queue — Failure Capture vs Budget",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.xaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    _save(fig, "review-queue-curve.png")


def cost_model_bars():
    data = json.loads((DATA_DIR / "cost-model.json").read_text())
    breakdown = data.get("modes", [])
    if not breakdown:
        logger.info("  skipped cost-model (no modes key)")
        return

    modes = [b["label"] for b in breakdown]
    unplanned = [b["unplanned_failure_cost"] for b in breakdown]
    preventive = [b["preventive_maintenance_cost"] for b in breakdown]

    x = np.arange(len(modes))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    ax.bar(
        x - width / 2,
        unplanned,
        width,
        label="Unplanned cost ($)",
        color=PALETTE["warn"],
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        preventive,
        width,
        label="Preventive cost ($)",
        color=PALETTE["accent"],
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(modes, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Illustrative cost ($)", fontsize=10)
    ax.set_title(
        "Cost Model by Failure Mode — Illustrative Maintenance Economics",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.yaxis.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "cost-model.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Generating portfolio visualization assets...")
    model_comparison()
    feature_importance()
    review_queue_curve()
    cost_model_bars()
    logger.info("Done.")
