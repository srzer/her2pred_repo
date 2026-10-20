#!/usr/bin/env python3
"""Evaluate a five-fold checkpoint ensemble using the TCGA manifest."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_utils import binary_metrics, youden_threshold
from model_from_ckpt import build_from_ckpt


def rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, data):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument("--result-dir", action="append", required=True)
    parser.add_argument("--valid-pred", action="append", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-root", action="append", required=True)
    parser.add_argument("--cohort", choices=["full", "heldout"], default="full")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not (len(args.ckpt) == len(args.result_dir) == 5):
        parser.error("exactly five checkpoints and result directories are required")

    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    oof = []
    for path in args.valid_pred:
        oof.extend(row for row in rows(path)
                   if str(row.get("is_aug", "0")).lower() in {"0", "false", "no"})
    by_slide = {}
    for row in oof:
        if row["slide_id"] in by_slide:
            raise RuntimeError(f"duplicate OOF prediction: {row['slide_id']}")
        by_slide[row["slide_id"]] = row
    oof = list(by_slide.values())
    valid_y = np.asarray([int(row["label"]) for row in oof])
    valid_p = np.asarray([float(row["prob_oof"]) for row in oof])
    threshold = float(youden_threshold(valid_y, valid_p))

    feature_index = {}
    for root in args.feature_root:
        for path in Path(root).rglob("*.pt"):
            feature_index[path.stem] = path.resolve()
    manifest = []
    for row in rows(args.manifest):
        if row.get("eligible", "1") != "1":
            continue
        if args.cohort == "heldout" and row.get("cohort") != "heldout":
            continue
        path = feature_index.get(row["slide_id"])
        if path is None:
            raise FileNotFoundError(f"feature missing for {row['slide_id']}")
        manifest.append({**row, "label": int(row["label"]),
                         "feature_path": str(path)})
    if len(manifest) != len({row["slide_id"] for row in manifest}):
        raise RuntimeError("duplicate slide_id in manifest")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = [build_from_ckpt(ckpt, result, [], device)[0]
              for ckpt, result in zip(args.ckpt, args.result_dir)]
    predictions = []
    with torch.no_grad():
        for index, row in enumerate(manifest, 1):
            data = torch.load(row["feature_path"], map_location="cpu")
            x = data["features"].float()
            if x.dim() == 3:
                x = x.squeeze(0)
            probs = [float(model(x.to(device), instance_eval=False)[0].item())
                     for model in models]
            item = dict(row)
            for fold, prob in enumerate(probs, 1):
                item[f"prob_fold{fold}"] = prob
            item["prob_ensemble"] = float(np.mean(probs))
            predictions.append(item)
            if index % 25 == 0 or index == len(manifest):
                print(f"predicted {index}/{len(manifest)}", flush=True)

    y = np.asarray([row["label"] for row in predictions])
    p = np.asarray([row["prob_ensemble"] for row in predictions])
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "cohort": args.cohort,
        "n": len(predictions),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "threshold_source": "pooled non-augmented OOF validation; Youden J",
        "valid_threshold": threshold,
        "valid_oof": binary_metrics(valid_y, valid_p, threshold),
        "test": {
            "at_0.5": binary_metrics(y, p, 0.5),
            "at_valid_threshold": binary_metrics(y, p, threshold),
        },
        "checkpoints": [str(Path(path).resolve()) for path in args.ckpt],
    }
    write_rows(out / "predictions_valid_oof.csv", oof)
    write_rows(out / "predictions_test.csv", predictions)
    with (out / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={summary['test']['at_0.5']['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    axes[1].plot(recall, precision,
                 label=f"PR-AUC={summary['test']['at_0.5']['pr_auc']:.3f}")
    axes[1].axhline(float(y.mean()), linestyle="--", color="grey")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(out / "roc_pr_curves.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
