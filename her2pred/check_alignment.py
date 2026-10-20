#!/usr/bin/env python3
"""Check alignment between two feature datasets.

Compares file membership, train/valid splits, coordinates, and labels.
Feature dimensions may differ between encoders.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def main(args):
    src = Path(args.src)
    ref  = Path(args.ref)

    if not src.is_dir():
        raise SystemExit(f"--src 不存在: {src}")
    if not ref.is_dir():
        raise SystemExit(f"--ref 不存在: {ref}")

    src_files = sorted(src.rglob("*.pt"))
    if args.only:
        keep = set(args.only)
        src_files = [p for p in src_files if p.relative_to(src).parts[0] in keep]

    if not src_files:
        raise SystemExit(f"{src} 下没有 .pt 文件"
                         + (f"（--only {' '.join(args.only)}）" if args.only else ""))

    print(f"ResNet50 : {src}   ({len(src_files)} files)")
    print(f"Reference: {ref}\n")

    n_ok = 0
    problems = []          # (相对路径, 原因)
    ref_ids  = {p.stem for p in ref.rglob("*.pt")}
    seen_ids = set()

    for src_path in tqdm(src_files, desc="Checking", disable=args.verbose):
        rel      = src_path.relative_to(src)
        ref_path = ref / rel
        seen_ids.add(src_path.stem)

        if not ref_path.exists():
            # 同名文件可能被放进了另一个 split
            alt = [p for p in ref.rglob(src_path.name)
                   if p.relative_to(ref).parts[0] == rel.parts[0]]
            if alt:
                problems.append((str(rel), f"split 不一致：reference 在 "
                                           f"{alt[0].relative_to(ref)}"))
            else:
                problems.append((str(rel), "reference 里没有这个文件"))
            continue

        a = torch.load(src_path,  map_location="cpu")
        b = torch.load(ref_path, map_location="cpu")

        ac, bc = np.asarray(a["coords"]), np.asarray(b["coords"])
        issues = []

        if ac.shape != bc.shape:
            issues.append(f"coords 形状不同 {ac.shape} vs {bc.shape}")
        elif not np.array_equal(ac, bc):
            n_diff = int((ac != bc).any(axis=1).sum())
            issues.append(f"coords 有 {n_diff}/{len(ac)} 行不同")

        if a["features"].shape[0] != b["features"].shape[0]:
            issues.append(f"tile 数不同 {a['features'].shape[0]} vs "
                          f"{b['features'].shape[0]}")
        if a["features"].shape[0] != len(ac):
            issues.append(f"features 行数 {a['features'].shape[0]} != "
                          f"coords 行数 {len(ac)}")

        if int(a["label"]) != int(b["label"]):
            issues.append(f"label 不同 {int(a['label'])} vs {int(b['label'])}")
        if str(a["slide_id"]) != str(b["slide_id"]):
            issues.append(f"slide_id 不同 {a['slide_id']} vs {b['slide_id']}")

        f = a["features"]
        if torch.isnan(f).any():
            issues.append(f"features 里有 {int(torch.isnan(f).sum())} 个 nan")
        n_zero = int((f.abs().sum(1) == 0).sum())
        if n_zero:
            issues.append(f"{n_zero} 行全零")

        if issues:
            problems.append((str(rel), "; ".join(issues)))
            if args.verbose:
                print(f"FAIL {rel}  {'; '.join(issues)}")
        else:
            n_ok += 1
            if args.verbose:
                print(f"OK   {str(rel):44s} src={tuple(a['features'].shape)} "
                      f"ref={tuple(b['features'].shape)}")

    # reference 里有、源侧还没提取的（= 还没跑完）
    missing = sorted(ref_ids - seen_ids) if not args.only else []

    print(f"\n{'='*78}")
    print(f"  通过 {n_ok} / {len(src_files)}")
    if problems:
        print(f"  不一致 {len(problems)} 个：")
        for rel, why in problems[:args.max_report]:
            print(f"    {rel}  —  {why}")
        if len(problems) > args.max_report:
            print(f"    ... 还有 {len(problems) - args.max_report} 个")
        print(f"\n  coords 对不上通常意味着两边的 tiling 参数不一致，检查：")
        print(f"    target_mpp(0.5) / tissue_thresh(0.5) / mask_level(ndpi,kfb=4, svs=2)")
        print(f"    以及 aug 侧的 grid_offset(112)")
    if missing:
        print(f"\n  reference 里有、源侧还没有的 slide：{len(missing)} 个"
              f"（提取还没跑完就是正常的）")
        for s in missing[:10]:
            print(f"    {s}")
        if len(missing) > 10:
            print(f"    ... 还有 {len(missing) - 10} 个")
    print(f"{'='*78}\n")

    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Verify the ResNet50 feature set is coordinate-identical "
                    "to the GigaPath one.")
    ap.add_argument("--src", default="./slide_datasets_resnet50",
                    help="源特征根目录")
    ap.add_argument("--ref",  default="./slide_datasets",
                    help="对应的 GigaPath 特征根目录")
    ap.add_argument("--only", nargs="+", default=None,
                    help="只查这些 source 目录（如 ndpi_data ndpi_data_aug）")
    ap.add_argument("--verbose", action="store_true", help="每个文件打印一行")
    ap.add_argument("--max_report", type=int, default=20,
                    help="最多列出多少个有问题的文件")
    main(ap.parse_args())
