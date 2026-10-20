#!/usr/bin/env python3
"""Reproduce the public Vanilla-versus-Adversarial ROC/PR figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from reproduce_prediction_statistics import delong_ci, pr_auc


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/predictions/external252.csv"
PROTOCOL = ROOT / "configs/final_protocol.json"
OUTPUT = ROOT / "results/figures/vanilla_vs_adversarial_roc_pr.png"


def threshold_point(y, score, threshold):
    predicted = score > threshold
    tp = np.sum(predicted & (y == 1))
    fp = np.sum(predicted & (y == 0))
    tn = np.sum(~predicted & (y == 0))
    fn = np.sum(~predicted & (y == 1))
    return tp / (tp + fn), fp / (fp + tn), tp / (tp + fp)


def style(axis):
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.05)
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.grid(True, color="#B8B8B8", alpha=0.32, linewidth=0.8)
    axis.set_axisbelow(True)


def main():
    data = pd.read_csv(DATA)
    protocol = json.loads(PROTOCOL.read_text())
    thresholds = protocol["external252_youden_thresholds"]
    y = data["label"].to_numpy(int)
    series = [
        ("ndpi_vanilla", data["ndpi_vanilla"].to_numpy(float), "#0072B2", thresholds["ndpi_vanilla"]),
        ("ndpi_adversarial", data["ndpi_adversarial"].to_numpy(float), "#009E73",
         thresholds["ndpi_adversarial"]),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.75))
    axes[0].plot([0, 1], [0, 1], "--", color="#2647B8", linewidth=1.15,
                 label="Chance")
    axes[1].axhline(y.mean(), linestyle="--", color="#6B6B6B", linewidth=1.15,
                    label=f"Prevalence {y.mean():.3f}")
    for label, score, color, threshold in series:
        fpr, tpr, _ = roc_curve(y, score)
        metric = delong_ci(y, score)
        axes[0].step(
            fpr, tpr, where="post", color=color, linewidth=2,
            label=(f"{label}: AUC {metric['roc_auc']:.3f} "
                   f"({metric['roc_auc_ci_lower']:.3f}–{metric['roc_auc_ci_upper']:.3f})"),
        )
        sensitivity, false_positive_rate, precision_at_threshold = threshold_point(
            y, score, threshold
        )
        axes[0].scatter(false_positive_rate, sensitivity, s=43, color=color,
                        edgecolor="white", linewidth=0.9, zorder=5,
                        label=f"{label}: Youden threshold {threshold:.3f}")

        precision, recall, _ = precision_recall_curve(y, score)
        axes[1].step(recall[::-1], precision[::-1], where="post", color=color,
                     linewidth=2, label=f"{label}: PR-AUC {pr_auc(y, score):.3f}")
        axes[1].scatter(sensitivity, precision_at_threshold, s=43, color=color,
                        edgecolor="white", linewidth=0.9, zorder=5,
                        label=f"{label}: Youden threshold {threshold:.3f}")

    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate",
                title="ROC curve")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR curve")
    for axis, location in zip(axes, ("lower right", "upper right")):
        style(axis)
        axis.legend(loc=location, frameon=True, framealpha=0.92, fontsize=8.2)
    figure.subplots_adjust(top=0.91, wspace=0.24)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white",
                   metadata={"Software": "Matplotlib"})
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
