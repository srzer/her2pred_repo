#!/usr/bin/env python3
"""Train one fold of an experiment using the final protocol."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def validate_existing_view(
    view: Path, data_root: Path, manifest: Path, ndpi_only: bool
):
    metadata_path = view / ".her2pred_training_view.json"
    if not metadata_path.is_file():
        raise RuntimeError(
            f"existing training view has no provenance metadata: {view}; "
            "use a new --view path"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "data_root": str(data_root.resolve()),
        "view_mode": "ndpi_only" if ndpi_only else "full",
        "manifest": None if ndpi_only else str(manifest.resolve()),
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"existing training view does not match requested inputs: {mismatches}; "
            "use a new --view path"
        )


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("ndpi_adversarial", "zero_shot", "few_shot",
                 "ndpi-adversarial", "zero", "few"),
        required=True,
        help="Experiment name (aliases: ndpi-adversarial, zero, few)",
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Clean dataset variant: filtered or unfiltered")
    parser.add_argument(
        "--manifest", type=Path,
        default=repo / "manifests/tcga_standard_b_few_shot.csv",
        help="TCGA 324-case manifest (default: bundled few_shot 36/288 split)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--view", type=Path, default=Path("training_view"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.mode = {
        "ndpi-adversarial": "ndpi_adversarial",
        "zero": "zero_shot",
        "few": "few_shot",
    }.get(args.mode, args.mode)

    ndpi_only = args.mode == "ndpi_adversarial"
    view = args.view.resolve()
    if not view.exists():
        view_command = [
            args.python, str(repo / "scripts/create_training_view.py"),
            "--data-root", str(args.data_root), "--out", str(view),
        ]
        if ndpi_only:
            view_command.append("--ndpi-only")
        else:
            view_command += ["--manifest", str(args.manifest.resolve())]
        subprocess.run(view_command, check=True)
    else:
        validate_existing_view(view, args.data_root, args.manifest, ndpi_only)
    name = f"final_{args.mode}_fold{args.fold}"
    command = [
        args.python, str(repo / "her2pred/run_ndpi_cv_dann.py"),
        "--gpu", args.gpu, "--root_dir", str(view),
        "--epochs", "16", "--lr", "1e-4", "--weight_decay", "1e-4",
        "--keep_ckpt", "--ckpt_select", "last",
        "--domain_mode", "appearregfull", "--lam_max", "15",
        "--constant_lam", "--dom_weight", "1",
        "--disc_hidden", "256", "--n_patch", "256",
        "--fold_ids", str(args.fold),
        "--out_dir", str(args.out), "--name", name,
    ]
    if args.mode == "ndpi_adversarial":
        # Only NDPI samples enter ndpi_adversarial optimization.
        # There is no SVS/KFB/TCGA loader and no site-adversarial objective.
        command += ["--internal_only_no_test"]
    else:
        held = view / "tcga_data/heldout"
        expected = {
            "candidate": 324, "eligible": 324, "positive": 66,
            "train_n": 36, "train_positive": 18,
            "heldout_n": 288, "heldout_positive": 48,
        }
        command += [
            "--mixed_pool", "--dual_loader", "--external_source_balanced",
            "--external_site_only", "--conditional_site_adv",
            "--site_adv_w", "0.01", "--site_adv_pos_mult", "1",
            "--site_adv_neg_mult", "1", "--site_slide_only",
            "--svs_task_w", "0.1", "--kfb_task_w", "0.01",
            "--extra_tcga_dir", str(held),
            "--standard_b_manifest", str(args.manifest.resolve()),
            "--standard_b_expected_counts",
            json.dumps(expected, separators=(",", ":")),
        ]
    if args.mode == "zero_shot":
        command += [
            "--test_tcga", "--external_steps_by_source",
            json.dumps({"kfb": 368, "svs": 40}, separators=(",", ":")),
        ]
    elif args.mode == "few_shot":
        tcga_steps = 28 if args.fold <= 3 else 30
        command += [
            "--fewshot_cn275", "--tcga_adv_w", "1", "--tcga_pos_weight", "1",
            "--external_steps_by_source",
            json.dumps({"kfb": 368, "svs": 40, "tcga": tcga_steps},
                       separators=(",", ":")),
        ]
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    subprocess.run(command, check=True, cwd=repo)


if __name__ == "__main__":
    main()
