#!/usr/bin/env python3
"""Prepare feature directories and TCGA splits for training."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path


def link(source: Path, destination: Path):
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def link_file(source: Path, destination: Path):
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve())


def read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"slide_id", "case_id", "cohort", "eligible", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest requires columns {sorted(required)}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path to the delivered PT bundle's filtered or unfiltered directory")
    parser.add_argument("--manifest", type=Path,
                        help="TCGA 324-case training/evaluation manifest")
    parser.add_argument(
        "--ndpi-only", action="store_true",
        help="Create only NDPI links; do not inspect or require SVS/KFB/TCGA data",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = args.data_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    link(data / "ndpi_data/original", out / "ndpi_data")
    link(data / "ndpi_data/augmented", out / "ndpi_data_aug")
    if args.ndpi_only:
        (out / ".her2pred_training_view.json").write_text(
            json.dumps({
                "data_root": str(data),
                "view_mode": "ndpi_only",
                "manifest": None,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(out)
        return

    if args.manifest is None:
        parser.error("--manifest is required unless --ndpi-only is used")
    for modality in ("svs_data", "kfb_data"):
        link(data / modality / "original", out / modality)
        link(data / modality / "augmented", out / f"{modality}_aug")

    # Assign TCGA labels and splits using the manifest.
    # This makes the fixed 36-case few-shot selection auditable.
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    eligible = [row for row in manifest if str(row["eligible"]) == "1"]
    if len(eligible) != 324 or len({row["case_id"] for row in eligible}) != 324:
        raise RuntimeError("expected 324 eligible unique TCGA cases")
    expected = Counter({("train", "0"): 18, ("train", "1"): 18,
                        ("heldout", "0"): 240, ("heldout", "1"): 48})
    counts = Counter((row["cohort"], row["label"]) for row in eligible)
    if counts != expected:
        raise RuntimeError(f"unexpected TCGA split counts: {counts}")

    feature_index = {}
    tcga_root = data / "tcga_data"
    for source_dir in (tcga_root / "tcga_data_few_shot",
                       tcga_root / "tcga_data_held_out"):
        for path in source_dir.glob("*.pt"):
            if path.stem in feature_index:
                raise RuntimeError(f"duplicate TCGA feature: {path.stem}")
            feature_index[path.stem] = path
    if len(feature_index) != 324:
        raise RuntimeError(f"expected 324 TCGA feature files, found {len(feature_index)}")

    for row in eligible:
        source = feature_index.get(row["slide_id"])
        if source is None:
            raise RuntimeError(f"TCGA feature missing: {row['slide_id']}")
        link_file(source, out / "tcga_data" / row["cohort"] / source.name)
    (out / ".her2pred_training_view.json").write_text(
        json.dumps({
            "data_root": str(data),
            "view_mode": "full",
            "manifest": str(manifest_path),
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TCGA split: {dict(sorted(counts.items()))}")
    print(out)


if __name__ == "__main__":
    main()
