#!/usr/bin/env python3
"""Recompute the released prediction metrics, confidence intervals, and tests."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "predictions"
N_BOOT = 10_000
SEED = 20260905


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(y, score)
    return float(auc(recall, precision))


def delong_components(y: np.ndarray, score: np.ndarray):
    positive, negative = score[y == 1], score[y == 0]
    matrix = np.where(
        positive[:, None] > negative[None, :],
        1.0,
        np.where(positive[:, None] == negative[None, :], 0.5, 0.0),
    )
    return float(matrix.mean()), matrix.mean(axis=1), matrix.mean(axis=0)


def delong_ci(y: np.ndarray, score: np.ndarray) -> dict:
    estimate, v10, v01 = delong_components(y, score)
    variance = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    se = math.sqrt(max(float(variance), 0.0))
    z = 1.95996398454
    return {
        "roc_auc": estimate,
        "roc_auc_ci_lower": max(0.0, estimate - z * se),
        "roc_auc_ci_upper": min(1.0, estimate + z * se),
        "pr_auc_trapezoidal": pr_auc(y, score),
    }


def paired_delong(y: np.ndarray, on: np.ndarray, off: np.ndarray) -> dict:
    auc_on, on10, on01 = delong_components(y, on)
    auc_off, off10, off01 = delong_components(y, off)
    delta = auc_on - auc_off
    d10, d01 = on10 - off10, on01 - off01
    variance = np.var(d10, ddof=1) / len(d10) + np.var(d01, ddof=1) / len(d01)
    se = math.sqrt(max(float(variance), 0.0))
    p_value = 1.0 if se == 0 and delta == 0 else (
        0.0 if se == 0 else float(2 * stats.norm.sf(abs(delta / se)))
    )
    z = 1.95996398454
    return {
        "delta_roc_auc": delta,
        "ci_lower": delta - z * se,
        "ci_upper": delta + z * se,
        "p_value_two_sided": p_value,
    }




def arm_metrics(data: pd.DataFrame, columns: list[str]) -> dict:
    output = {}
    for column in columns:
        selected = data.loc[data[column].notna(), ["label", column]]
        ys = selected["label"].to_numpy(dtype=int)
        scores = selected[column].to_numpy(dtype=float)
        output[column] = {
            "n": int(len(selected)),
            "n_positive": int(ys.sum()),
            "n_negative": int((1 - ys).sum()),
            **delong_ci(ys, scores),
        }
    return output


def summarize_frozen_bootstrap(
    replicates: pd.DataFrame, key: str, observed_roc: float, observed_pr: float
) -> dict:
    def summarize(metric: str, observed: float) -> dict:
        values = replicates[f"{key}_{metric}"].to_numpy(dtype=float)
        left = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
        right = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
        return {
            "observed": observed,
            "ci_lower": float(np.percentile(values, 2.5)),
            "ci_upper": float(np.percentile(values, 97.5)),
            "p_value_two_sided": float(min(1.0, 2 * min(left, right))),
        }

    return {
        "delta_roc_auc": summarize("roc_auc", observed_roc),
        "delta_pr_auc": summarize("pr_auc", observed_pr),
        "resamples": int(len(replicates)),
        "seed": SEED,
    }


def compare(
    data: pd.DataFrame,
    on_name: str,
    off_name: str,
    replicates: pd.DataFrame,
    bootstrap_key: str,
) -> dict:
    selected = data.loc[data[[on_name, off_name]].notna().all(axis=1)]
    y = selected["label"].to_numpy(dtype=int)
    on = selected[on_name].to_numpy(dtype=float)
    off = selected[off_name].to_numpy(dtype=float)
    return {
        "n": int(len(selected)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "difference": f"{on_name} minus {off_name}",
        "paired_delong": paired_delong(y, on, off),
        "paired_stratified_bootstrap": summarize_frozen_bootstrap(
            replicates,
            bootstrap_key,
            roc_auc_score(y, on) - roc_auc_score(y, off),
            pr_auc(y, on) - pr_auc(y, off),
        ),
    }


def compute() -> dict:
    external = pd.read_csv(DATA / "external252.csv")
    tcga = pd.read_csv(DATA / "tcga324.csv")
    replicates = pd.read_csv(ROOT / "results" / "bootstrap_replicates.csv")
    return {
        "methods": {
            "roc_auc_ci": "DeLong structural components; normal 95% interval",
            "roc_auc_comparison": "paired DeLong structural-component covariance",
            "pr_auc": "trapezoidal integration of the precision-recall curve",
            "sensitivity": "paired label-stratified percentile bootstrap",
            "bootstrap_resamples": N_BOOT,
            "bootstrap_seed": SEED,
        },
        "metrics": {
            "external252": arm_metrics(
                external, ["ndpi_vanilla", "ndpi_adversarial", "macenko"]
            ),
            "tcga324": arm_metrics(
                tcga,
                ["internal_vanilla", "zero_shot", "resnet", "plip", "few_shot"],
            ),
        },
        "comparisons": {
            "macenko_vs_ndpi_vanilla": compare(
                external, "macenko", "ndpi_vanilla", replicates, "macenko_vs_ndpi_vanilla"
            ),
            "ndpi_adversarial_vs_ndpi_vanilla": compare(
                external, "ndpi_adversarial", "ndpi_vanilla", replicates,
                "ndpi_adversarial_vs_ndpi_vanilla",
            ),
            "zero_shot_vs_resnet": compare(
                tcga, "zero_shot", "resnet", replicates, "zero_shot_vs_resnet"
            ),
            "zero_shot_vs_plip": compare(
                tcga, "zero_shot", "plip", replicates, "zero_shot_vs_plip"
            ),
            "few_shot_vs_zero_shot": compare(
                tcga, "few_shot", "zero_shot", replicates,
                "few_shot_vs_zero_shot",
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "prediction_statistics.json")
    args = parser.parse_args()
    result = compute()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
