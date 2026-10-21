#!/usr/bin/env python3
"""Extract WSI tile features with GigaPath, ResNet50, or PLIP.

Supports NDPI, SVS, and KFB slides, optional stain normalization, augmentation,
and extraction at coordinates supplied by a reference feature dataset.
Use --slide_dir for the input slides and --save_dir for output features.
"""

import os
import sys

# --gpu 必须在 import torch 之前生效
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
        break

import argparse
import copy

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slide_io import (                                                # noqa: E402
    FORMAT_CFG, find_slides, prepare_slide, close_slide, resolve_label,
)
from encoders import build_encoder, build_transform                   # noqa: E402
from dataset_paths import source_dir                                  # noqa: E402
from stain_norm import (                                              # noqa: E402
    build_normalizer, load_target, sample_pixels_from_tiles,
)


# ---------------------------------------------------------------------------
# Tile 读取（可选多进程；每个 worker 自己开一个 slide handle）
# ---------------------------------------------------------------------------

def _worker_init(_):
    """Limit OpenCV and PyTorch threads within each DataLoader worker."""
    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass
    torch.set_num_threads(1)


class TileDataset(Dataset):
    def __init__(self, slide_path, backend, coords, level, read_size, transform):
        self.slide_path = slide_path
        self.backend    = backend
        self.coords     = coords
        self.level      = level
        self.read_size  = read_size
        self.transform  = transform
        self._slide     = None      # lazily opened, per-process

    def __len__(self):
        return len(self.coords)

    def _handle(self):
        if self._slide is None:
            from slide_io import open_slide
            self._slide = open_slide(self.slide_path, self.backend)
        return self._slide

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        tile = self._handle().read_region(
            (int(x), int(y)), self.level, (self.read_size, self.read_size)
        ).convert("RGB")
        return self.transform(tile)


# ---------------------------------------------------------------------------
# train/valid 划分对齐
# ---------------------------------------------------------------------------

def build_split_map(layout_from: str) -> dict:
    """
    扫描已有 GigaPath 数据目录（如 ./slide_datasets/ndpi_data），
    返回 {slide_id: split_subdir}，让新数据落到完全相同的 train/valid 划分里。
    """
    split_map = {}
    if not layout_from:
        return split_map
    if not os.path.isdir(layout_from):
        print(f"[warn] --layout_from '{layout_from}' not found; everything goes to train/")
        return split_map
    for split in sorted(os.listdir(layout_from)):
        d = os.path.join(layout_from, split)
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith('.pt'):
                split_map[os.path.splitext(fname)[0]] = split
    print(f"  Layout reference: {layout_from}  →  {len(split_map)} slide ids mapped")
    return split_map


def build_coords_map(coords_from: str) -> dict:
    """Load reference coordinates, splits, and labels keyed by slide ID.

    Returns {slide_id: {'split': str, 'coords': ndarray, 'label': int}}.
    """
    cmap = {}
    if not os.path.isdir(coords_from):
        raise SystemExit(f"--coords_from 目录不存在: {coords_from}")
    # 常规 source 是 train/valid 等子目录；后补的 TCGA pos_2 是扁平目录。
    # 先收根目录 .pt，并统一放到 train，避免它们被旧的仅子目录扫描逻辑漏掉。
    root_pt = [f for f in sorted(os.listdir(coords_from)) if f.endswith('.pt')]
    for fname in root_pt:
        sid = os.path.splitext(fname)[0]
        try:
            ref = torch.load(os.path.join(coords_from, fname), map_location='cpu')
            cmap[sid] = {'split': 'train',
                         'coords': np.asarray(ref['coords']),
                         'label': int(ref['label'])}
        except Exception as e:
            print(f"[warn] 读不了 {coords_from}/{fname}: {e}")

    for split in sorted(os.listdir(coords_from)):
        d = os.path.join(coords_from, split)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.pt'):
                continue
            sid = os.path.splitext(fname)[0]
            try:
                ref = torch.load(os.path.join(d, fname), map_location='cpu')
                cmap[sid] = {'split': split,
                             'coords': np.asarray(ref['coords']),
                             'label': int(ref['label'])}
            except Exception as e:
                print(f"[warn] 读不了 {d}/{fname}: {e}")
    n_tiles = sum(len(v['coords']) for v in cmap.values())
    print(f"  Coords reference: {coords_from}  →  {len(cmap)} slides, "
          f"{n_tiles:,} tiles")
    return cmap


def default_save_dir(encoder: str, stain: str, fmt: str, augment: bool) -> str:
    """目录命名走 dataset_paths.py，和训练侧保持一致。"""
    return source_dir(encoder, stain, fmt, augment=augment, filtered=False)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def process_slides(
    slide_dir:     str,
    save_dir:      str,
    fmt:           str,
    label:         int,
    encoder:       str   = 'resnet50',
    target_mpp:    float = 0.5,
    batch_size:    int   = 256,
    augment:       bool  = False,
    tissue_thresh: float = 0.5,
    grid_offset:   int   = 0,
    mask_level:    int   = None,
    weights:       str   = 'imagenet_v1',
    encoder_layer: str   = 'layer4',
    workers:       int   = 4,
    amp:           bool  = False,
    split_map:     dict  = None,
    default_split: str   = 'train',
    coords_map:    dict  = None,
    stain_method:  str   = 'none',
    stain_target:  str   = None,
    stain_fit_tiles: int = 100,
    on_stain_fail: str   = 'skip',
    color_jitter:  bool  = True,
    shard:         tuple = None,
    missing_mpp_fallback: float = None,
):
    cfg       = FORMAT_CFG[fmt]
    ext       = cfg['ext']
    split_map = split_map or {}
    use_ref_coords = coords_map is not None
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Format  : {fmt}  (backend={cfg['backend']}, "
          f"mask_level={cfg['mask_level'] if mask_level is None else mask_level})")
    if label < 0:
        print(f"Label   : 逐片取自 --coords_from 参考集")
    else:
        print(f"Label   : {label}  ({'pos' if label else 'neg'})")
    print(f"Device  : {device}")
    print(f"Augment : {augment}"
          + (f"  (color_jitter={'on' if color_jitter else 'off'})" if augment else ""))

    # --- 染色归一化的 target ---
    stain_proto = None      # 只带 target 的模板，每张切片 deepcopy 一份再 fit source
    if stain_method != 'none':
        if stain_target:
            stain_proto = load_target(stain_target)
            if stain_proto.name != stain_method:
                raise SystemExit(f"--stain_target 文件是 {stain_proto.name} 的，"
                                 f"和 --stain_norm {stain_method} 不匹配")
            print(f"Stain   : {stain_method}   target ← {stain_target}")
        elif stain_method == 'macenko':
            stain_proto = build_normalizer('macenko')      # 标准参考矩阵
            print(f"Stain   : macenko   target ← Macenko 标准参考矩阵")
        else:
            raise ValueError(f"unsupported stain method: {stain_method}")
    else:
        print(f"Stain   : none")

    # --- Model ---
    print(f"Encoder : {encoder}"
          + (f" (weights={weights}, layer={encoder_layer})" if encoder == 'resnet50' else ""))
    model, out_dim = build_encoder(encoder, device=device,
                                   weights=weights, layer=encoder_layer)
    print(f"Feature dim: {out_dim}")

    # 不做染色归一化时 transform 是全局固定的，只建一次
    base_transform = (build_transform(augment, None, color_jitter,
                                      encoder_name=encoder)
                      if stain_proto is None else None)

    all_slide_paths = find_slides(slide_dir, ext)
    print(f"Found {len(all_slide_paths)} {ext} files under {slide_dir}")
    if shard is not None and shard[1] > 1:
        i, n = shard
        all_slide_paths = all_slide_paths[i::n]
        print(f"  shard {i}/{n} → 本进程处理 {len(all_slide_paths)} 张")
    print()

    n_done = n_skip = n_fail = n_stain_fail = 0

    for slide_path in all_slide_paths:
        slide_id  = os.path.splitext(os.path.basename(slide_path))[0]

        ref = coords_map.get(slide_id) if use_ref_coords else None
        if use_ref_coords and ref is None:
            # 参考集里没有这张片 —— GigaPath 当初也跳过了它，保持一致
            print(f"[skip] {slide_id}: 不在 --coords_from 参考集里。")
            n_skip += 1
            continue

        split     = ref['split'] if ref else split_map.get(slide_id, default_split)
        out_dir   = os.path.join(save_dir, split)
        save_path = os.path.join(out_dir, f"{slide_id}.pt")

        if os.path.exists(save_path):
            print(f"[skip] {slide_id} already processed.")
            n_skip += 1
            continue

        prep = prepare_slide(slide_path, fmt, target_mpp, grid_offset,
                             tissue_thresh, mask_level,
                             need_coords=not use_ref_coords,
                             missing_mpp_fallback=missing_mpp_fallback)
        if prep is None:
            n_fail += 1
            continue

        slide = prep['slide']
        if ref:
            # 坐标直接抄参考集，连组织 mask 都不用算
            coords = [tuple(int(v) for v in c) for c in ref['coords']]
            if label >= 0 and ref['label'] != label:
                print(f"[warn] {slide_id}: 参考集 label={ref['label']} 与 "
                      f"--label {label} 不符 —— 采用参考集的。")
        else:
            coords = prep['coords']
        slide_label = ref['label'] if ref else label

        # --- 按整张切片 fit 染色矩阵（不是逐 tile）---
        transform = base_transform
        if stain_proto is not None:
            def read_tile(x, y):
                tile = slide.read_region(
                    (x, y), prep['best_level'],
                    (prep['read_size'], prep['read_size'])).convert("RGB")
                return np.asarray(tile)

            px = sample_pixels_from_tiles(read_tile, coords,
                                          n_tiles=stain_fit_tiles)
            norm = copy.deepcopy(stain_proto)
            if not norm.fit(px):
                n_stain_fail += 1
                if on_stain_fail == 'skip':
                    print(f"[skip] {slide_id}: 染色矩阵 fit 失败"
                          f"（组织像素太少 / SVD 退化），不保存。"
                          f" 用 --on_stain_fail identity 可改为不做归一化直接存。")
                    close_slide(slide)
                    continue
                print(f"[warn] {slide_id}: 染色矩阵 fit 失败，该片按未归一化处理。")
            transform = build_transform(augment, norm, color_jitter,
                                        encoder_name=encoder)

        print(f"Processing {slide_id}: {len(coords)} tiles | "
              f"best_level={prep['best_level']} | offset_l0={prep['offset_l0']}px | "
              f"split={split}")

        # workers>0 时每个 worker 自开 handle；主进程这个先关掉免得句柄冲突
        close_slide(slide)

        ds = TileDataset(slide_path, prep['backend'], coords,
                         prep['best_level'], prep['read_size'], transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers,
                            worker_init_fn=_worker_init if workers > 0 else None,
                            pin_memory=(device.type == 'cuda'))

        embs = []
        try:
            with torch.inference_mode():
                for batch in tqdm(loader, desc=slide_id, leave=False):
                    batch = batch.to(device, non_blocking=True)
                    if amp:
                        with torch.autocast(device_type=device.type, dtype=torch.float16):
                            feats = model(batch)
                        feats = feats.float()
                    else:
                        feats = model(batch)
                    embs.append(feats.cpu())
        except Exception as e:
            print(f"[error] {slide_id}: extraction failed: {e}")
            n_fail += 1
            continue

        if embs:
            all_embs = torch.cat(embs, dim=0)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(
                {
                    "features": all_embs,
                    "label":    torch.tensor(slide_label),
                    "slide_id": slide_id,
                    "coords":   np.array(coords),
                },
                save_path,
            )
            print(f"Saved {save_path}  shape={tuple(all_embs.shape)}")
            n_done += 1

    print(f"\nDone. saved={n_done}  skipped={n_skip}  failed={n_fail}"
          + (f"  stain_fit_failed={n_stain_fail}" if stain_method != 'none' else ""))
    if n_stain_fail:
        print(f"  [!] 有 {n_stain_fail} 张切片染色 fit 失败。"
              f"跑 check_alignment.py 会把它们列成 missing，属于预期内。")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Extract tile features: format × encoder × stain normalization.")
    p.add_argument("--format", required=True, choices=sorted(FORMAT_CFG))
    p.add_argument("--slide-dir", "--slide_dir", required=True, help="输入目录（递归搜索）")
    p.add_argument("--encoder", default='resnet50',
                   choices=['resnet50', 'gigapath', 'plip'])
    p.add_argument("--out", "--save-dir", "--save_dir", dest="save_dir", default=None,
                   help="source 级输出目录（默认 "
                        "./slide_datasets_<encoder>[_<stain>]/<fmt>_data[_aug]）")
    p.add_argument("--gpu", default=None, help="CUDA device id")

    p.add_argument("--stain-norm", "--stain_norm", default="none", choices=["none", "macenko"])
    p.add_argument("--stain-target", "--stain_target", default=None,
                   help='Reference stain-target JSON. Use the same target for all slide formats.')
    p.add_argument("--stain-fit-tiles", "--stain_fit_tiles", type=int, default=100,
                   help="每张切片抽多少 tile 来 fit 染色矩阵（默认 100）")
    p.add_argument("--on-stain-fail", "--on_stain_fail", default='skip', choices=['skip', 'identity'],
                   help='Action when stain fitting fails: skip the slide or use the identity transform.')
    p.add_argument("--shard", default=None,
                   help='Process shard i/n with zero-based indexing. Omit to process all slides.')
    p.add_argument("--keep-color-jitter", "--keep_color_jitter", action="store_true",
                   help="开染色归一化时仍然保留 ColorJitter。"
                        "默认关掉，因为颜色抖动会把刚归一化掉的色差又加回去")

    p.add_argument("--label", default=None, choices=['auto', '0', '1'],
                   help="该目录所有切片的标签；auto=从路径推断（默认按 format）")
    p.add_argument("--target-mpp", "--target_mpp", type=float, default=0.5)
    p.add_argument("--missing-mpp-fallback", "--missing_mpp_fallback", type=float, default=None,
                   help="仅在切片缺失 MPP 元数据时使用的显式兜底值；"
                        "默认不启用并跳过该切片")
    p.add_argument("--batch-size", "--batch_size", type=int, default=None, help="默认按 format")
    p.add_argument("--augment", action="store_true",
                   help="随机增广 + 半个 patch 的网格错位")
    p.add_argument("--tissue-thresh", "--tissue_thresh", type=float, default=0.5)
    p.add_argument("--mask-level", "--mask_level", type=int, default=None, help="默认按 format")
    p.add_argument("--weights", default='imagenet_v1',
                   choices=['imagenet_v1', 'imagenet_v2', 'random'],
                   help="ResNet50 权重（对 gigapath 无效）")
    p.add_argument("--encoder-layer", "--encoder_layer", default='layer4', choices=['layer3', 'layer4'],
                   help="ResNet50: layer4→2048-d（默认），layer3→1024-d（CLAM 式）")
    p.add_argument("--workers", type=int, default=4,
                   help="读 tile 的进程数，默认 4，0 表示串行读取")
    p.add_argument("--amp", action="store_true", help="fp16 前向（特征仍存 fp32）")
    p.add_argument("--coords-from", "--coords_from", default=None,
                   help='Reference feature directory providing extraction coordinates, labels, and splits.')
    p.add_argument("--layout-from", "--layout_from", default=None,
                   help="已有 GigaPath source 目录，照抄其 train/valid 划分")
    p.add_argument("--default-split", "--default_split", default='train')
    return p.parse_args(argv)


def main():
    args = parse_args()
    cfg  = FORMAT_CFG[args.format]

    batch_size = args.batch_size if args.batch_size is not None else cfg['batch_size']

    # --coords_from 模式下 label 逐片取自参考集，不需要（也不该）从路径推断；
    # 显式给了 --label 就仍然解析出来，好在不一致时报 warn。
    if args.coords_from and args.label is None:
        label = -1                       # 占位，实际会被参考集的 label 覆盖
    else:
        label_arg = args.label if args.label is not None else cfg['label']
        label     = resolve_label(label_arg, args.slide_dir)
    grid_offset = 112 if args.augment else 0     # half a patch for augmentation

    if args.save_dir is None:
        args.save_dir = default_save_dir(args.encoder, args.stain_norm,
                                         args.format, args.augment)

    # Report the selected baseline encoder.
    if args.encoder == 'gigapath' and args.stain_norm == 'none':
        print("[!] encoder=gigapath 且 stain_norm=none —— 这就是 ./slide_datasets/ 里"
              "已有的基线特征。\n"
              "    除非是要复现验证，否则直接用已有数据即可，不必重跑。\n")

    # 染色归一化开着时，ColorJitter 默认关掉
    color_jitter = args.keep_color_jitter or args.stain_norm == 'none'

    shard = None
    if args.shard:
        try:
            _i, _n = (int(x) for x in str(args.shard).split('/'))
        except Exception:
            raise SystemExit(f"--shard 格式应为 i/n，收到 {args.shard!r}")
        if not (0 <= _i < _n):
            raise SystemExit(f"--shard i 必须在 [0, n)，收到 {args.shard!r}")
        shard = (_i, _n)

    print(f"Output     → {args.save_dir}/<split>/")
    print(f"Grid offset: {grid_offset} patch-px  "
          f"(≈ {grid_offset * args.target_mpp:.1f} µm at target MPP {args.target_mpp})")

    coords_map = build_coords_map(args.coords_from) if args.coords_from else None
    # --coords_from 已经自带 split 信息，不需要再 --layout_from
    split_map  = {} if coords_map else build_split_map(args.layout_from)

    process_slides(
        slide_dir     = args.slide_dir,
        save_dir      = args.save_dir,
        fmt           = args.format,
        label         = label,
        encoder       = args.encoder,
        target_mpp    = args.target_mpp,
        batch_size    = batch_size,
        augment       = args.augment,
        tissue_thresh = args.tissue_thresh,
        grid_offset   = grid_offset,
        mask_level    = args.mask_level,
        weights       = args.weights,
        encoder_layer = args.encoder_layer,
        workers       = args.workers,
        amp           = args.amp,
        split_map     = split_map,
        default_split = args.default_split,
        coords_map    = coords_map,
        stain_method  = args.stain_norm,
        stain_target  = args.stain_target,
        stain_fit_tiles = args.stain_fit_tiles,
        on_stain_fail = args.on_stain_fail,
        color_jitter  = color_jitter,
        shard         = shard,
        missing_mpp_fallback = args.missing_mpp_fallback,
    )


if __name__ == "__main__":
    main()
