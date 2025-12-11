import json
import math
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import configs


# ==========================
# Matplotlib Global Settings
# ==========================
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 7
plt.rcParams["axes.unicode_minus"] = False


# ==========================
# 1. Data Loading
# ==========================

def load_per_case_metrics(path: str) -> pd.DataFrame:
    """
    Load per-case evaluation metrics stored as JSONL into a pandas DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def load_summary_metrics(path: str) -> Dict[str, float]:
    """
    Load aggregated summary metrics stored as JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================
# 2. Colors for Four Models
# ==========================

COLORS = {
    "benchmark_light": "#9ecae1",
    "benchmark_dark":  "#3182bd",

    "baseline_light":  "#c7e9c0",
    "baseline_dark":   "#31a354",

    "finetune_light":  "#fdbb84",
    "finetune_dark":   "#e34a33",

    "rag_light":       "#dadaeb",
    "rag_dark":        "#756bb1",
}


# ==========================
# 3. Radar Chart (4-Model Comparison)
# ==========================

def plot_radar_compare(
    benchmark_summary: Dict[str, float],
    baseline_summary: Dict[str, float],
    finetune_summary: Dict[str, float],
    rag_summary: Dict[str, float],
    title: str = "Model Comparison (Radar)",
    metrics: List[str] = None,
    save_path: str = None,
):
    """
    Plot an overlapping radar chart comparing four models:

        - benchmark (Gemini-2.5-Flash)
        - baseline  (Qwen3-4B)
        - finetune  (Ziwei)
        - rag       (Ziwei-RAG)

    metrics: a list of metric keys to display in clockwise order.
    save_path: optional file path to save the figure.
    """
    if metrics is None:
        metrics = [
            "chart_accuracy_exact",
            "chart_star_jaccard_overall",
            "daxian_range_iou",
            "daxian_ganzhi_accuracy",
            "star_palace_pair_f1",
            "topic_f1",
            "cosine_similarity",
            "bertscore",
        ]

    # Values in the order of metrics
    benchmark_vals = [benchmark_summary.get(m, 0.0) for m in metrics]
    baseline_vals  = [baseline_summary.get(m, 0.0)  for m in metrics]
    finetune_vals  = [finetune_summary.get(m, 0.0)  for m in metrics]
    rag_vals       = [rag_summary.get(m, 0.0)       for m in metrics]

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Close the polygon for radar chart
    benchmark_vals += benchmark_vals[:1]
    baseline_vals  += baseline_vals[:1]
    finetune_vals  += finetune_vals[:1]
    rag_vals       += rag_vals[:1]
    angles         += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # Display names used on chart
    display_labels = [
        "Chart_Accuracy",
        "Chart_Star_Jaccard",
        "Daxian_Range_IoU",
        "Daxian_Ganzhi_Accuracy",
        "Star_Palace_Pair_f1",
        "Topic_f1",
        "Cosine_Similarity",
        "BERTScore",
    ]

    # Benchmark
    ax.plot(angles, benchmark_vals, marker="o",
            color=COLORS["benchmark_dark"], linewidth=2,
            label="Gemini-2.5-Flash")
    ax.fill(angles, benchmark_vals, alpha=0.20, color=COLORS["benchmark_light"])

    # Baseline
    ax.plot(angles, baseline_vals, marker="o",
            color=COLORS["baseline_dark"], linewidth=2,
            label="Qwen3-4B")
    ax.fill(angles, baseline_vals, alpha=0.20, color=COLORS["baseline_light"])

    # Finetuned model
    ax.plot(angles, finetune_vals, marker="o",
            color=COLORS["finetune_dark"], linewidth=2,
            label="Ziwei")
    ax.fill(angles, finetune_vals, alpha=0.20, color=COLORS["finetune_light"])

    # RAG model
    ax.plot(angles, rag_vals, marker="o",
            color=COLORS["rag_dark"], linewidth=2,
            label="Ziwei-RAG")
    ax.fill(angles, rag_vals, alpha=0.20, color=COLORS["rag_light"])

    # Metric labels on outer ring
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(display_labels, fontsize=12)
    ax.tick_params(axis="x", pad=40)

    # Radius 0–1
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])

    # Annotate benchmark values (optional)
    for angle, value in zip(angles[:-1], benchmark_vals[:-1]):
        ax.text(angle, value + 0.05, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_title(title, fontsize=18, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.20))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# ==========================
# 4. Example Main
# ==========================

if __name__ == "__main__":

    # Input files
    summary_fin   = os.path.join(configs.RESULT_FINETUNE, "finetune_metrics_summary.jsonl")
    summary_bench = os.path.join(configs.RESULT_BENCHMARK, "benchmark_metrics_summary.jsonl")
    summary_base  = os.path.join(configs.RESULT_BASELINE,  "baseline_metrics_summary.jsonl")
    summary_rag   = os.path.join(configs.RESULT_RAG,       "finetune_rag_metrics_summary.jsonl")

    # Load summaries
    finetune_summary = load_summary_metrics(summary_fin)
    benchmark_summary = load_summary_metrics(summary_bench)
    baseline_summary = load_summary_metrics(summary_base)
    rag_summary = load_summary_metrics(summary_rag)

    # Ensure output folder exists
    os.makedirs(configs.VIS_PATH, exist_ok=True)

    metrics_for_radar = [
        "chart_accuracy_exact",
        "chart_star_jaccard_overall",
        "daxian_range_iou",
        "daxian_ganzhi_accuracy",
        "star_palace_pair_f1",
        "topic_f1",
        "cosine_similarity",
        "bertscore",
    ]

    save_path = os.path.join(configs.VIS_PATH, "radar_compare_4models.jpg")

    # Plot the radar chart
    plot_radar_compare(
        benchmark_summary,
        baseline_summary,
        finetune_summary,
        rag_summary,
        title="Gemini / Qwen / Ziwei / Ziwei-RAG – Radar Comparison",
        metrics=metrics_for_radar,
        save_path=save_path,
    )