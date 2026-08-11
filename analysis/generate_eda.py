"""Generate EDA (exploratory data analysis) visualization assets.

Reads pre-computed JSON artifacts — no dataset download required.
Outputs PNG charts to docs/assets/, styled to match the site's dark theme.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

try:
    from analysis._common import CHART_PALETTE, apply_dark_chart_style, save_chart_figure
except ImportError:
    from _common import CHART_PALETTE, apply_dark_chart_style, save_chart_figure

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data" / "ai4i-case-study"
ASSETS_DIR = ROOT / "docs" / "assets"

apply_dark_chart_style()


def _save(fig, name):
    out = save_chart_figure(fig, ASSETS_DIR, name)
    logger.info(f"  saved -> {out.relative_to(ROOT)}")
    return out


def eda_class_balance():
    """Class imbalance: failure vs normal."""
    profile = json.loads((DATA_DIR / "dataset-profile.json").read_text())
    dist = profile["target_distribution"]
    normal = dist["no_failure"]
    failure = dist["failure"]
    total = normal + failure
    failure_rate = failure / total

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: counts
    ax = axes[0]
    bars = ax.bar(
        ["Normal", "Failure"],
        [normal, failure],
        color=[CHART_PALETTE["primary"], CHART_PALETTE["danger"]],
        width=0.5,
    )
    for bar, val in zip(bars, [normal, failure]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{val:,}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=CHART_PALETTE["text"],
        )
    ax.set_ylim(0, normal * 1.12)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Class Distribution (full dataset)", fontsize=11, fontweight="bold")
    ax.yaxis.grid(True, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: pie
    ax2 = axes[1]
    wedge_colors = [CHART_PALETTE["primary"], CHART_PALETTE["danger"]]
    explode = (0, 0.08)
    wedges, texts, autotexts = ax2.pie(
        [normal, failure],
        labels=["Normal (96.6%)", "Failure (3.4%)"],
        colors=wedge_colors,
        explode=explode,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9, "color": CHART_PALETTE["text"]},
        wedgeprops={"edgecolor": CHART_PALETTE["bg"], "linewidth": 1.5},
    )
    autotexts[1].set_color(CHART_PALETTE["bg"])
    autotexts[1].set_fontweight("bold")
    ax2.set_title(
        f"Class Balance\n(failure rate = {failure_rate:.2%})", fontsize=11, fontweight="bold"
    )

    fig.suptitle("EDA — Class Imbalance: AI4I 2020 Dataset", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, "eda-class-balance.png")


def eda_failure_modes():
    """Failure mode distribution — full dataset vs holdout."""
    profile = json.loads((DATA_DIR / "dataset-profile.json").read_text())
    breakdown = json.loads((DATA_DIR / "failure-mode-breakdown.json").read_text())

    mode_totals = profile["failure_mode_totals"]
    labels = list(mode_totals.keys())
    full_counts = list(mode_totals.values())

    holdout_counts = {b["label"]: b["holdout_failures"] for b in breakdown}
    holdout = [holdout_counts.get(label, 0) for label in labels]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(
        x - width / 2,
        full_counts,
        width,
        label="Full dataset (n=10,000)",
        color=CHART_PALETTE["primary"],
    )
    b2 = ax.bar(
        x + width / 2,
        holdout,
        width,
        label="Holdout set (n=2,000)",
        color=CHART_PALETTE["accent"],
    )

    for bar, val in zip(list(b1) + list(b2), full_counts + holdout):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
            color=CHART_PALETTE["text"],
        )

    short_labels = [
        "Tool Wear\n(TWF)",
        "Heat Dissipation\n(HDF)",
        "Power\n(PWF)",
        "Overstrain\n(OSF)",
        "Random\n(RNF)",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylabel("Failure count", fontsize=10)
    ax.set_title(
        "EDA — Failure Mode Distribution (Full Dataset vs Holdout)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(fontsize=9, framealpha=0.9)
    ax.yaxis.grid(True, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "eda-failure-modes.png")


def eda_product_type():
    """Product type distribution and failure rate by type."""
    profile = json.loads((DATA_DIR / "dataset-profile.json").read_text())
    type_dist = profile["type_distribution"]

    types = list(type_dist.keys())
    counts = list(type_dist.values())
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    type_colors = [CHART_PALETTE["highlight"], CHART_PALETTE["primary"], CHART_PALETTE["accent"]]

    fig, ax = plt.subplots(figsize=(7, 4))

    bars = ax.bar(types, counts, color=type_colors, width=0.5)
    for bar, val, pct in zip(bars, counts, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            color=CHART_PALETTE["text"],
        )

    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_xlabel("Product Type", fontsize=10)
    ax.set_ylabel("Record count", fontsize=10)
    ax.set_title(
        "EDA — Product Type Distribution\n(H = High quality, L = Low quality, M = Medium quality)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.yaxis.grid(True, linewidth=0.8, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _save(fig, "eda-type-distribution.png")


def eda_confusion_matrix():
    """Confusion matrix of the final model (from summary.json)."""
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    cm = summary["final_model"]["confusion_matrix"]
    # cm = [[TN, FP], [FN, TP]]
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp

    matrix = np.array([[tn, fp], [fn, tp]])
    labels = np.array(
        [
            [f"TN\n{tn:,}\n({tn/total:.1%})", f"FP\n{fp:,}\n({fp/total:.2%})"],
            [f"FN\n{fn:,}\n({fn/total:.2%})", f"TP\n{tp:,}\n({tp/total:.2%})"],
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    cmap = LinearSegmentedColormap.from_list(
        "dark_blues", [CHART_PALETTE["bg"], CHART_PALETTE["primary"]]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=tn)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels[i, j],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=CHART_PALETTE["text"],
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Failure"], fontsize=10)
    ax.set_yticklabels(["Actual Normal", "Actual Failure"], fontsize=10)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.set_title(
        f"Confusion Matrix — HistGradientBoosting (Enhanced Features)\n"
        f"Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  Holdout n={total:,}",
        fontsize=11,
        fontweight="bold",
        pad=14,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color=CHART_PALETTE["muted"])
    plt.setp(cbar.ax.get_yticklabels(), color=CHART_PALETTE["muted"])
    _save(fig, "eda-confusion-matrix.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Generating EDA visualization assets...")
    eda_class_balance()
    eda_failure_modes()
    eda_product_type()
    eda_confusion_matrix()
    logger.info("Done.")
