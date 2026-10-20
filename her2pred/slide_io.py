"""WSI reading and tiling for NDPI, SVS, and KFB.

Provides slide discovery, tissue masking, MPP-based tile sizing, coordinate
generation, and label resolution. NDPI and SVS use OpenSlide. KFB uses ASlide.
"""

import os

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Defaults for each WSI format.
# ---------------------------------------------------------------------------

FORMAT_CFG = {
    'ndpi': {'ext': '.ndpi', 'backend': 'openslide', 'mask_level': 4,
             'batch_size': 256, 'label': 'auto'},
    'svs' : {'ext': '.svs',  'backend': 'openslide', 'mask_level': 2,
             'batch_size': 128, 'label': '1'},
    'kfb' : {'ext': '.kfb',  'backend': 'aslide',    'mask_level': 4,
             'batch_size': 256, 'label': '0'},
}


# ---------------------------------------------------------------------------
# Slide 后端
# ---------------------------------------------------------------------------

def open_slide(path: str, backend: str):
    """openslide 用于 ndpi/svs，Aslide 用于 kfb（只在需要时 import）。"""
    if backend == 'openslide':
        import openslide
        return openslide.OpenSlide(path)
    elif backend == 'aslide':
        from Aslide import Slide
        return Slide(path)
    raise ValueError(f"unknown backend {backend!r}")


def read_mpp(slide, backend: str, slide_id: str,
             missing_mpp_fallback: float = None):
    """level-0 的 microns-per-pixel；读不到时可显式启用光学分辨率兜底。"""
    try:
        if backend == 'openslide':
            import openslide
            val = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        else:
            val = slide.properties.get("openslide.mpp-x")
            if val is None:
                raise ValueError("MPP key missing")
        return float(val)
    except (KeyError, ValueError, TypeError, AttributeError):
        if missing_mpp_fallback is not None:
            if missing_mpp_fallback <= 0:
                raise ValueError("missing_mpp_fallback must be positive")
            print(f"[warn] Cannot read MPP for {slide_id}; using explicit "
                  f"fallback {missing_mpp_fallback:g} um/px.")
            return float(missing_mpp_fallback)
        print(f"[warn] Cannot read MPP for {slide_id}, skipping.")
        return None


def close_slide(slide):
    if hasattr(slide, "close"):
        slide.close()


def find_slides(slide_dir: str, ext: str):
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(slide_dir)
        for f in files
        if f.lower().endswith(ext)
    )


# ---------------------------------------------------------------------------
# 组织 mask
# ---------------------------------------------------------------------------

def get_tissue_mask(slide, low_res_level: int):
    """
    低倍层上 Otsu 出组织二值 mask。
    返回 (mask_uint8, scale_factor_from_level0)。
    """
    level = min(low_res_level, len(slide.level_dimensions) - 1)
    dims  = slide.level_dimensions[level]

    img    = slide.read_region((0, 0), level, dims).convert("L")
    img_np = np.array(img)

    _, mask = cv2.threshold(img_np, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    return mask, slide.level_downsamples[level]


# ---------------------------------------------------------------------------
# 几何
# ---------------------------------------------------------------------------

def compute_geometry(slide, target_mpp: float, mpp0: float, grid_offset: int):
    """返回 (best_level, read_size_on_level, step_size_l0, offset_l0)。"""
    total_downsample = target_mpp / mpp0
    step_size_l0     = int(round(224 * total_downsample))
    offset_l0        = int(round(grid_offset * total_downsample))

    best_level = 0
    for i, ds in enumerate(slide.level_downsamples):
        if ds <= total_downsample:
            best_level = i
    current_ds         = slide.level_downsamples[best_level]
    read_size_on_level = int(round(step_size_l0 / current_ds))

    return best_level, read_size_on_level, step_size_l0, offset_l0


def collect_valid_coords(slide, mask, mask_scale, step_size_l0, offset_l0,
                         tissue_thresh):
    """遍历顺序 / 边界条件 / 取整方式全部照抄已有脚本，保证坐标逐元素一致。"""
    mask_h, mask_w      = mask.shape
    width_l0, height_l0 = slide.dimensions

    valid_coords = []
    for y in range(offset_l0, height_l0 - step_size_l0, step_size_l0):
        for x in range(offset_l0, width_l0 - step_size_l0, step_size_l0):
            m_x = int(x / mask_scale)
            m_y = int(y / mask_scale)
            m_w = max(1, int(step_size_l0 / mask_scale))

            m_x2 = min(m_x + m_w, mask_w)
            m_y2 = min(m_y + m_w, mask_h)
            if m_x >= mask_w or m_y >= mask_h:
                continue

            mask_roi = mask[m_y:m_y2, m_x:m_x2]
            if mask_roi.size == 0:
                continue
            if (np.count_nonzero(mask_roi) / mask_roi.size) >= tissue_thresh:
                valid_coords.append((x, y))
    return valid_coords


def prepare_slide(slide_path: str, fmt: str, target_mpp: float,
                  grid_offset: int, tissue_thresh: float,
                  mask_level: int = None, need_coords: bool = True,
                  missing_mpp_fallback: float = None):
    """
    一站式：开片 → 读 MPP → 算几何 →（可选）出组织 mask → 收集合法坐标。

    need_coords=False 时跳过组织 mask 和坐标遍历，只返回几何信息，
    coords 为 None —— 用于坐标已经从别处（既有 .pt）拿到的场景。

    返回 dict(slide, backend, best_level, read_size, step_size_l0,
              offset_l0, coords, mpp0)；
    失败（打不开 / 读不到 MPP / 没有组织）返回 None，并已把 slide 关掉。
    """
    cfg      = FORMAT_CFG[fmt]
    backend  = cfg['backend']
    mask_lvl = cfg['mask_level'] if mask_level is None else mask_level
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]

    try:
        slide = open_slide(slide_path, backend)
    except Exception as e:
        print(f"[error] Failed to open slide {slide_id}: {e}")
        return None

    mpp0 = read_mpp(slide, backend, slide_id, missing_mpp_fallback)
    if mpp0 is None:
        close_slide(slide)
        return None

    best_level, read_size, step_size_l0, offset_l0 = compute_geometry(
        slide, target_mpp, mpp0, grid_offset)

    coords = None
    if need_coords:
        mask, mask_scale = get_tissue_mask(slide, mask_lvl)
        coords = collect_valid_coords(slide, mask, mask_scale,
                                      step_size_l0, offset_l0, tissue_thresh)
        if not coords:
            print(f"[skip] {slide_id}: no tissue tiles found.")
            close_slide(slide)
            return None

    return {
        'slide': slide, 'backend': backend, 'slide_id': slide_id,
        'best_level': best_level, 'read_size': read_size,
        'step_size_l0': step_size_l0, 'offset_l0': offset_l0,
        'coords': coords, 'mpp0': mpp0,
    }


# ---------------------------------------------------------------------------
# Label
# ---------------------------------------------------------------------------

def infer_label_from_path(slide_dir: str) -> int:
    """从路径里最后一个含 pos/neg 的分量推断标签（与 data_preprocess_ndpi.py 一致）。"""
    parts = os.path.normpath(os.path.abspath(slide_dir)).split(os.sep)
    for part in reversed(parts):
        pl = part.lower()
        if "pos" in pl:
            return 1
        if "neg" in pl:
            return 0
    raise ValueError(
        f"Cannot infer label from path '{slide_dir}'. "
        "Ensure a path component contains 'pos' or 'neg', or pass --label 0/1.")


def resolve_label(label_arg: str, slide_dir: str) -> int:
    return infer_label_from_path(slide_dir) if label_arg == 'auto' else int(label_arg)
