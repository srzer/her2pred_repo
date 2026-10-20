#!/usr/bin/env python3
"""Filter feature files using tumor-patch coordinates from a reference dataset.

Matches coordinates between the source and reference features and saves the
selected source features to --out_root.

Example:
    python her2pred/build_filtered.py --src_root features_resnet \
        --filtered_ref features_gigapath --out_root features_resnet_filtered
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def index_rn50(src_root: Path) -> dict:
    """
    建立 {(source_dir, slide_id): path} 和 {slide_id: [path, ...]} 两级索引。
    source_dir 形如 'ndpi_data' / 'ndpi_data_aug'。
    """
    by_src_id = {}
    by_id     = {}
    for pt in src_root.rglob("*.pt"):
        rel = pt.relative_to(src_root)
        if len(rel.parts) < 2:
            continue
        source_dir = rel.parts[0]
        slide_id   = pt.stem
        by_src_id[(source_dir, slide_id)] = pt
        by_id.setdefault(slide_id, []).append(pt)
    return by_src_id, by_id


def coord_key(coords: np.ndarray):
    """把 (N,2) 坐标数组变成可 hash 的 tuple 列表。"""
    return [(int(a), int(b)) for a, b in coords]


def main(args):
    src_root = Path(args.src_root)
    ref_root  = Path(args.filtered_ref)
    out_root  = Path(args.out_root)

    if not src_root.is_dir():
        raise SystemExit(f"--src_root not found: {src_root}")
    if not ref_root.is_dir():
        raise SystemExit(f"--filtered_ref not found: {ref_root}")

    print(f"Indexing ResNet50 features under {src_root} ...")
    by_src_id, by_id = index_rn50(src_root)
    print(f"  {len(by_src_id)} ResNet50 .pt files indexed\n")

    ref_files = sorted(ref_root.rglob("*.pt"))
    if args.only:
        ref_files = [p for p in ref_files
                     if p.relative_to(ref_root).parts[0] in set(args.only)]
    print(f"Reference (GigaPath filtered): {len(ref_files)} .pt files under {ref_root}\n")

    n_done = n_skip = n_missing = n_mismatch = 0
    kept_total = src_total = 0

    for ref_path in tqdm(ref_files, desc="Slides"):
        rel        = ref_path.relative_to(ref_root)
        source_dir = rel.parts[0]
        slide_id   = ref_path.stem
        out_path   = out_root / rel

        if out_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        # ── 定位对应的 源特征 ──────────────────────────────
        src_path = by_src_id.get((source_dir, slide_id))
        if src_path is None:
            cands = by_id.get(slide_id, [])
            # 退而求其次：按 slide_id 全局找，但 aug / 非 aug 绝不能混
            want_aug = source_dir.endswith('_aug')
            cands = [c for c in cands
                     if c.relative_to(src_root).parts[0].endswith('_aug') == want_aug]
            if len(cands) == 1:
                src_path = cands[0]
            else:
                tqdm.write(f"[missing] {source_dir}/{slide_id}: "
                           f"no unique ResNet50 counterpart ({len(cands)} candidates)")
                n_missing += 1
                continue

        ref_data = torch.load(ref_path, map_location="cpu")
        src_data  = torch.load(src_path,  map_location="cpu")

        ref_coords = np.asarray(ref_data["coords"])
        src_coords  = np.asarray(src_data["coords"])

        if len(ref_coords) == 0:
            tqdm.write(f"[skip] {source_dir}/{slide_id}: reference has no coords")
            n_skip += 1
            continue

        # ── 按坐标对齐 ────────────────────────────────────────────
        lookup = {c: i for i, c in enumerate(coord_key(src_coords))}
        idx, missed = [], 0
        for c in coord_key(ref_coords):
            j = lookup.get(c)
            if j is None:
                missed += 1
            else:
                idx.append(j)

        if missed:
            tqdm.write(
                f"[mismatch] {source_dir}/{slide_id}: {missed}/{len(ref_coords)} "
                f"reference coords not found in the ResNet50 grid "
                f"(rn50 has {len(src_coords)} tiles). "
                f"两边 tiling 参数不一致 —— 检查 target_mpp / tissue_thresh / mask_level。"
            )
            n_mismatch += 1
            if not args.allow_partial:
                continue
        if not idx:
            n_mismatch += 1
            continue

        if int(ref_data["label"]) != int(src_data["label"]):
            tqdm.write(f"[warn] {source_dir}/{slide_id}: label differs "
                       f"(ref={int(ref_data['label'])}, rn50={int(src_data['label'])}) "
                       f"— 用 reference 的 label")

        sel = np.array(idx, dtype=np.int64)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": src_data["features"][sel],
                "label":    ref_data["label"],
                "slide_id": ref_data["slide_id"],
                "coords":   src_coords[sel],
            },
            out_path,
        )
        kept_total += len(sel)
        src_total  += len(src_coords)
        n_done += 1

    print(f"\nDone.  saved={n_done}  skipped={n_skip}  "
          f"missing_src={n_missing}  coord_mismatch={n_mismatch}")
    if n_done:
        print(f"Kept {kept_total} / {src_total} tiles "
              f"({100.0 * kept_total / max(src_total, 1):.1f}%)")
    print(f"Output → {out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Derive the ResNet50 tumor-filtered dataset from the existing "
                    "GigaPath filtered dataset by coordinate matching.")
    ap.add_argument("--src_root",    default="./slide_datasets_resnet50",
                    help="未过滤的新特征根目录（见 dataset_paths.py）")
    ap.add_argument("--filtered_ref", default="./slide_datasets_filtered",
                    help="Existing GigaPath filtered dataset (read-only reference)")
    ap.add_argument("--out_root",     default="./slide_datasets_resnet50_filtered",
                    help="过滤后特征的输出目录（见 dataset_paths.py）")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only process these source dirs (e.g. ndpi_data ndpi_data_aug)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite files that already exist in --out_root")
    ap.add_argument("--allow_partial", action="store_true",
                    help="Save even when some reference coords are missing "
                         "(default: skip such slides so the comparison stays exact)")
    main(ap.parse_args())
