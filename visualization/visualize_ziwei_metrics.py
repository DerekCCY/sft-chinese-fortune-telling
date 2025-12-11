# visualize_ziwei_metrics.py
"""
Visualization utilities for Zi Wei Dou Shu evaluation metrics.

This script generates:
1. Radar chart from summary metrics
2. Histograms of metric distributions from per-case metrics
3. Bar chart of career sub-topic F1 scores

Inputs:
- per-case metrics JSONL (from ZiWeiEvaluator)
- summary metrics JSON (from ZiWeiEvaluator)

Outputs (saved images):
- radar_chart.jpg
- distribution_chart.jpg
- topic_chart.jpg
"""

import json
import math
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os

import configs

# -----------------------------
# Matplotlib global settings
# -----------------------------
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 7
plt.rcParams["axes.unicode_minus"] = False

LIGHT = "#9ecae1"
DARK = "#3182bd"

CAREER_TOPICS = ["career_role", "career_wealth", "career_location"]


# =======================================================
# 1. Data Loading
# =======================================================

def load_per_case_metrics(path: str) -> pd.DataFrame:
    """Load per-case JSONL metrics into a pandas DataFrame."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_summary_metrics(path: str) -> Dict[str, float]:
    """Load aggregated summary metrics (JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =======================================================
# 2. Radar Chart
# =======================================================

def plot_radar_from_summary(
    summary: Dict[str, float],
    title: str = "Model Overview (Radar Chart)",
    metrics: List[str] = None,
    save_path: str = None,
):
    """Plot a radar chart from summary metrics."""
    if metrics is None:
        metrics = [
            "chart_star_jaccard_overall",
            "daxian_range_iou",
            "daxian_ganzhi_accuracy",
            "star_palace_pair_f1",
            "topic_f1",
            "cosine_similarity",
            "bertscore",
        ]

    values = [summary.get(m, 0.0) for m in metrics]
    num_vars = len(metrics)

    # close the loop
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    values += values[:1]

    labels = [
        "Chart_Star_Jaccard",
        "Daxian_Range_IoU",
        "Daxian_Ganzhi",
        "Star_Palace_Pair_F1",
        "Topic_F1",
        "Cosine_Similarity",
        "BERTScore",
    ]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2, marker="o", color=DARK)
    ax.fill(angles, values, alpha=0.25, color=LIGHT)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.tick_params(axis="x", pad=25)

    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # annotate values
    for angle, v in zip(angles[:-1], values[:-1]):
        ax.text(angle, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(title, fontsize=14, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# =======================================================
# 3. Metric Histogram Plots
# =======================================================

def plot_metric_histograms(
    df: pd.DataFrame,
    metric_names: List[str],
    bins: int = 20,
    title_prefix: str = "",
    save_path: str = None,
):
    """Plot histograms for selected metrics from the per-case DataFrame."""
    num_metrics = len(metric_names)
    cols = 3
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(16, 9))
    axes = axes.flatten()

    last_idx = -1

    for i, metric in enumerate(metric_names):
        ax = axes[i]
        if metric not in df.columns:
            ax.set_visible(False)
            continue

        values = df[metric].dropna()
        if values.empty:
            ax.set_visible(False)
            continue

        ax.hist(values, bins=bins, alpha=0.7, color=LIGHT)
        ax.set_title(metric, fontsize=7)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.set_xlim(0.0, 1.0)

        mean_v = values.mean()
        ax.axvline(mean_v, linestyle="--")
        ax.text(
            mean_v,
            ax.get_ylim()[1] * 0.9,
            f"mean={mean_v:.2f}",
            rotation=90,
            ha="right",
            va="top",
            fontsize=5,
        )

        ax.grid(True, axis="y", alpha=0.3)
        last_idx = i

    for j in range(last_idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title_prefix + "Metric Distributions", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# =======================================================
# 4. Per-Career-Topic F1
# =======================================================

def compute_per_career_topic_f1_from_stats(
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """Compute average F1 for each career_* subtopic across all records."""
    accum = {t: {"f1_sum": 0.0, "count": 0} for t in CAREER_TOPICS}

    for _, row in df.iterrows():
        per_topic = row.get("per_topic_stats", {}) or {}
        for topic in CAREER_TOPICS:
            stats_t = per_topic.get(topic)
            if not stats_t:
                continue
            f1 = stats_t.get("f1")
            if f1 is None:
                continue
            accum[topic]["f1_sum"] += float(f1)
            accum[topic]["count"] += 1

    final = {}
    for topic in CAREER_TOPICS:
        cnt = accum[topic]["count"]
        avg_f1 = accum[topic]["f1_sum"] / cnt if cnt else 0.0
        final[topic] = {"f1": avg_f1}

    return final


def plot_career_topic_f1_bars(
    topic_stats: Dict[str, Dict[str, float]],
    title: str = "Career Sub-Topic F1 Scores",
    save_path: str = None,
):
    """Plot bar chart for career sub-topic F1 scores."""
    topics = list(topic_stats.keys())
    f1_scores = [topic_stats[t]["f1"] for t in topics]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(topics, f1_scores, color=LIGHT)
    ax.set_ylim(0.0, max(f1_scores + [0.6]) + 0.2)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)

    ax.set_xticklabels(topics, rotation=20, ha="right")

    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# =======================================================
# 5. Main Example
# =======================================================

if __name__ == "__main__":
    # Input paths from configs
    per_case_path = os.path.join(configs.RESULT_FINETUNE, "finetune_metrics_per_cases.jsonl")
    summary_path = os.path.join(configs.RESULT_FINETUNE, "finetune_metrics_summary.jsonl")

    per_case_df = load_per_case_metrics(per_case_path)
    summary_metrics = load_summary_metrics(summary_path)

    # Ensure output directory exists
    os.makedirs(configs.VIS_PATH, exist_ok=True)

    # 5.1 Radar Chart
    plot_radar_from_summary(
        summary_metrics,
        title="Baseline Model – Radar Chart",
        metrics=[
            "chart_star_jaccard_overall",
            "daxian_range_iou",
            "daxian_ganzhi_accuracy",
            "star_palace_pair_f1",
            "topic_f1",
            "cosine_similarity",
            "bertscore",
        ],
        save_path=os.path.join(configs.VIS_PATH, "radar_chart.jpg"),
    )

    # 5.2 Metric Distributions
    plot_metric_histograms(
        per_case_df,
        metric_names=[
            "chart_star_jaccard_overall",
            "daxian_range_iou",
            "daxian_ganzhi_accuracy",
            "star_palace_pair_f1",
            "topic_f1",
            "overall_structural_score",
            "overall_interpretation_content_score",
            "overall_text_similarity_score",
            "cosine_similarity",
            "bertscore",
        ],
        title_prefix="Baseline – ",
        save_path=os.path.join(configs.VIS_PATH, "distribution_chart.jpg"),
    )

    # 5.3 Per-Career-Topic F1 Bar Chart
    per_topic_stats = compute_per_career_topic_f1_from_stats(per_case_df)
    print("Per-career-topic F1 stats:")
    print(json.dumps(per_topic_stats, ensure_ascii=False, indent=2))

    plot_career_topic_f1_bars(
        per_topic_stats,
        title="Baseline – Career Topic F1",
        save_path=os.path.join(configs.VIS_PATH, "topic_chart.jpg"),
    )
