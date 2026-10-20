#!/usr/bin/env python3
"""Five-checkpoint ensemble evaluation on original SVS/KFB features."""
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
from dataset_paths import pt_leaf_dirs


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True)
    parser.add_argument(
        "--result-dir", action="append",
        help="Optional architecture-name hint for each checkpoint (defaults to its parent directory)",
    )
    parser.add_argument(
        "--valid-pred", action="append",
        help="Per-fold OOF validation predictions used to derive the Youden threshold",
    )
    parser.add_argument(
        "--valid-threshold", type=float,
        help="Validation-derived threshold (alternative to --valid-pred files)",
    )
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if len(args.ckpt) != 5:
        parser.error("exactly five checkpoints are required")
    if args.result_dir is not None and len(args.result_dir) != 5:
        parser.error("when supplied, exactly five --result-dir values are required")
    if (args.valid_pred is None) == (args.valid_threshold is None):
        parser.error("supply either five --valid-pred files or --valid-threshold")
    if args.valid_pred is not None and len(args.valid_pred) != 5:
        parser.error("exactly five --valid-pred files are required")
    out = Path(args.out_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)

    valid_y = valid_p = None
    if args.valid_pred is not None:
        oof = []
        for path in args.valid_pred:
            oof.extend(row for row in read_csv(path)
                       if str(row.get("is_aug", "0")).lower() in {"0", "false", "no"})
        if len(oof) != len({row["slide_id"] for row in oof}):
            raise RuntimeError("duplicate original-slide OOF predictions")
        valid_y = np.asarray([int(row["label"]) for row in oof])
        valid_p = np.asarray([float(row["prob_oof"]) for row in oof])
        threshold = float(youden_threshold(valid_y, valid_p))
        threshold_source = "pooled non-augmented OOF validation; Youden J"
    else:
        threshold = float(args.valid_threshold)
        threshold_source = "frozen pooled-OOF Youden threshold supplied by caller"

    samples = []
    for source in ("svs", "kfb"):
        collection = Path(args.root_dir) / f"{source}_data"
        # The delivered PT bundle keeps original/ and augmented/ together;
        # a training view points directly at original/.  Accept both roots.
        if (collection / "original").is_dir():
            collection = collection / "original"
        directories = pt_leaf_dirs(str(collection))
        if not directories:
            raise FileNotFoundError(
                f"no original {source.upper()} PT collection under {collection}"
            )
        seen = set()
        for directory in directories:
            for path in sorted(Path(directory).glob("*.pt")):
                if path.stem in seen:
                    continue
                seen.add(path.stem)
                data = torch.load(path, map_location="cpu")
                samples.append({"slide_id": path.stem, "source": source,
                                "label": int(data["label"]), "feature_path": str(path.resolve())})
        if not seen:
            raise RuntimeError(f"no original {source.upper()} PT files under {collection}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    result_dirs = args.result_dir or [str(Path(path).parent) for path in args.ckpt]
    models = [build_from_ckpt(ckpt, result, [], device)[0]
              for ckpt, result in zip(args.ckpt, result_dirs)]
    with torch.no_grad():
        for index, row in enumerate(samples, 1):
            data = torch.load(row["feature_path"], map_location="cpu")
            x = data["features"].float()
            if x.dim() == 3:
                x = x.squeeze(0)
            probs = [float(model(x.to(device), instance_eval=False)[0].item())
                     for model in models]
            for fold, prob in enumerate(probs, 1):
                row[f"prob_fold{fold}"] = prob
            row["prob_ensemble"] = float(np.mean(probs))
            if index % 25 == 0 or index == len(samples):
                print(f"predicted {index}/{len(samples)}", flush=True)

    def metric(group):
        y = np.asarray([row["label"] for row in group])
        p = np.asarray([row["prob_ensemble"] for row in group])
        return {"at_0.5": binary_metrics(y, p, 0.5),
                "at_valid_threshold": binary_metrics(y, p, threshold)}

    results = {source: metric([row for row in samples if row["source"] == source])
               for source in ("svs", "kfb")}
    results["all"] = metric(samples)
    summary = {"valid_threshold": threshold,
               "threshold_source": threshold_source,
               "valid_oof": (binary_metrics(valid_y, valid_p, threshold)
                             if valid_y is not None else None),
               "results": results}
    with (out / "predictions_test.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    with (out / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)

    y = np.asarray([row["label"] for row in samples])
    p = np.asarray([row["prob_ensemble"] for row in samples])
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={results['all']['at_0.5']['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[1].plot(recall, precision, label=f"PR-AUC={results['all']['at_0.5']['pr_auc']:.3f}")
    axes[1].axhline(float(y.mean()), linestyle="--", color="grey")
    axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall")
    for axis in axes: axis.legend()
    fig.tight_layout(); fig.savefig(out / "roc_pr_curves.png", dpi=180); plt.close(fig)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
