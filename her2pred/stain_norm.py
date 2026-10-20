"""H&E stain normalization and color transforms.

Provides Macenko normalization using a shared target across slide formats. Fit source parameters from sampled pixels once per slide,
then apply the transform to its tiles.
"""

import json

import numpy as np


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# Macenko 论文 / staintools 里广泛使用的标准参考染色矩阵。
# 列 = [Hematoxylin, Eosin]，行 = [R, G, B] 的光密度分量。
STANDARD_HE_REF = np.array([[0.5626, 0.2159],
                            [0.7201, 0.8012],
                            [0.4062, 0.5581]], dtype=np.float64)
STANDARD_MAXC_REF = np.array([1.9705, 1.0308], dtype=np.float64)


def _od(rgb: np.ndarray, Io: float) -> np.ndarray:
    """RGB(uint8) → optical density。+1 是为了避免 log(0)。"""
    return -np.log((rgb.astype(np.float64) + 1.0) / Io)


def _tissue_pixels(rgb: np.ndarray, Io: float, beta: float) -> np.ndarray:
    """
    丢掉近乎透明的像素（背景 / 玻璃），只留组织。
    rgb: (N, 3) uint8  →  返回 (M, 3) 的 OD，M <= N。
    """
    od = _od(rgb, Io)
    keep = ~np.any(od < beta, axis=1)
    return od[keep]


# ---------------------------------------------------------------------------
# Macenko
# ---------------------------------------------------------------------------

class MacenkoNormalizer:
    """
    用法：
        n = MacenkoNormalizer()            # target 默认用标准参考矩阵
        n.fit_target(ref_pixels)           # 可选：改用自己的参考切片
        n.fit(slide_pixels)                # 每张切片 fit 一次
        out = n.transform(tile_rgb_uint8)  # 每个 tile 调一次
    """

    name = 'macenko'

    def __init__(self, Io: float = 240.0, alpha: float = 1.0, beta: float = 0.15,
                 min_tissue_px: int = 1000):
        self.Io    = float(Io)
        self.alpha = float(alpha)          # 取角度分位数时的百分比
        self.beta  = float(beta)           # OD 阈值，低于它算背景
        self.min_tissue_px = int(min_tissue_px)

        self.target_HE   = STANDARD_HE_REF.copy()
        self.target_maxC = STANDARD_MAXC_REF.copy()

        self.source_HE   = None
        self.source_maxC = None
        self._pinv_T     = None            # pinv(source_HE).T，float32，transform 里复用
        self._scale      = None            # target_maxC / source_maxC，float32
        self._tgt_HE_T   = None            # target_HE.T，float32
        self.fitted      = False

    # ---- 估计染色矩阵 -----------------------------------------------------

    def _estimate(self, pixels: np.ndarray):
        """
        pixels: (N, 3) uint8。返回 (HE (3,2), maxC (2,))，失败返回 (None, None)。
        """
        od = _tissue_pixels(pixels, self.Io, self.beta)
        if od.shape[0] < self.min_tissue_px:
            return None, None

        try:
            # 协方差最大的两个方向张成「染色平面」
            _, V = np.linalg.eigh(np.cov(od.T))
            V = V[:, 1:3]                                    # (3, 2)

            proj = od @ V                                    # (N, 2)
            phi  = np.arctan2(proj[:, 1], proj[:, 0])

            min_phi = np.percentile(phi, self.alpha)
            max_phi = np.percentile(phi, 100.0 - self.alpha)

            v_min = V @ np.array([np.cos(min_phi), np.sin(min_phi)])
            v_max = V @ np.array([np.cos(max_phi), np.sin(max_phi)])

            # 第一列固定是 Hematoxylin（R 分量更小的那个）
            if v_min[0] > v_max[0]:
                HE = np.stack([v_min, v_max], axis=1)
            else:
                HE = np.stack([v_max, v_min], axis=1)        # (3, 2)

            # 归一化列向量，符号统一为正
            norms = np.linalg.norm(HE, axis=0, keepdims=True)
            if np.any(norms < 1e-8):
                return None, None
            HE = HE / norms
            HE = HE * np.sign(HE.sum(axis=0, keepdims=True))

            # 每个像素在 H/E 上的浓度，取 99 分位当作「最大浓度」
            C = od @ np.linalg.pinv(HE).T                    # (N, 2)
            maxC = np.percentile(C, 99, axis=0)
            if np.any(~np.isfinite(maxC)) or np.any(maxC < 1e-6):
                return None, None
            return HE, maxC
        except np.linalg.LinAlgError:
            return None, None

    def fit_target(self, pixels: np.ndarray) -> bool:
        """从参考切片的像素重新定义 target（不调用则用标准参考矩阵）。"""
        HE, maxC = self._estimate(pixels)
        if HE is None:
            return False
        self.target_HE, self.target_maxC = HE, maxC
        return True

    def fit(self, pixels: np.ndarray) -> bool:
        """
        对一张切片 fit source 染色矩阵。
        返回 False 表示 fit 失败（组织太少 / SVD 退化），
        此时 transform() 退化成恒等映射，调用方应当告警。
        """
        HE, maxC = self._estimate(pixels)
        if HE is None:
            self.fitted = False
            return False
        self.source_HE, self.source_maxC = HE, maxC
        # transform 走 float32：OD 值域只有 0~3，float32 精度绰绰有余，
        # 而内存带宽减半 —— 逐 tile 的开销主要在访存不在算术。
        self._pinv_T   = np.linalg.pinv(HE).T.astype(np.float32)          # (3, 2)
        self._scale    = (self.target_maxC / maxC).astype(np.float32).reshape(1, 2)
        self._tgt_HE_T = self.target_HE.T.astype(np.float32)              # (2, 3)
        self.fitted = True
        return True

    # ---- 应用 -------------------------------------------------------------

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        img: (H, W, 3) uint8 RGB  →  归一化后的 (H, W, 3) uint8 RGB。
        未 fit 成功时原样返回。
        每个 tile 只做两次小矩阵乘 + 一次 log/exp，没有逐 tile 的 SVD。
        """
        if not self.fitted:
            return img

        h, w, _ = img.shape

        # OD = -log((I + 1) / Io)，全程 float32 且尽量原地，少分配临时数组
        od = img.reshape(-1, 3).astype(np.float32)           # (N, 3)
        od += 1.0
        od *= np.float32(1.0 / self.Io)
        np.log(od, out=od)
        od *= np.float32(-1.0)

        C = od @ self._pinv_T                                # (N, 2)
        C *= self._scale                                     # 缩放到 target 浓度

        out = C @ self._tgt_HE_T                             # (N, 3) = OD_norm
        out *= np.float32(-1.0)
        np.exp(out, out=out)
        out *= np.float32(self.Io)

        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8).reshape(h, w, 3)

    # ---- 持久化 -----------------------------------------------------------

    def target_dict(self) -> dict:
        return {'method': self.name,
                'Io': self.Io, 'alpha': self.alpha, 'beta': self.beta,
                'target_HE': self.target_HE.tolist(),
                'target_maxC': self.target_maxC.tolist()}

    def load_target(self, d: dict):
        if d.get('method') != self.name:
            raise ValueError(f"target 文件是 {d.get('method')} 的，"
                             f"和当前方法 {self.name} 不匹配")
        self.Io    = float(d['Io'])
        self.alpha = float(d['alpha'])
        self.beta  = float(d['beta'])
        self.target_HE   = np.asarray(d['target_HE'],   dtype=np.float64)
        self.target_maxC = np.asarray(d['target_maxC'], dtype=np.float64)


NORMALIZERS = {'macenko': MacenkoNormalizer}


def build_normalizer(method: str, **kw):
    """method='none' 返回 None（调用方据此跳过归一化）。"""
    if method in (None, 'none'):
        return None
    if method not in NORMALIZERS:
        raise ValueError(f"unknown stain norm method {method!r}; "
                         f"choose from {sorted(NORMALIZERS)} or 'none'")
    return NORMALIZERS[method](**kw)


def save_target(normalizer, path: str):
    with open(path, 'w') as f:
        json.dump(normalizer.target_dict(), f, indent=2)


def load_target(path: str):
    """读 json 并返回配好 target 的 normalizer（方法由文件内容决定）。"""
    with open(path) as f:
        d = json.load(f)
    n = build_normalizer(d['method'])
    n.load_target(d)
    return n


# ---------------------------------------------------------------------------
# 从切片抽样像素（fit 用）
# ---------------------------------------------------------------------------

def sample_pixels_from_tiles(read_tile, coords, n_tiles: int = 50,
                             max_px_per_tile: int = 20000, seed: int = 0):
    """Sample pixels from evenly spaced slide tiles for normalizer fitting.

    read_tile(x, y) returns an (H, W, 3) uint8 RGB array. coords contains tile
    origins. Returns an (N, 3) uint8 array.
    """
    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    n_tiles = min(n_tiles, len(coords))
    idx = np.linspace(0, len(coords) - 1, n_tiles).astype(int)

    rng   = np.random.default_rng(seed)
    chunks = []
    for i in idx:
        x, y = coords[i]
        try:
            tile = read_tile(int(x), int(y))
        except Exception:
            continue
        px = np.asarray(tile).reshape(-1, 3)
        if px.shape[0] > max_px_per_tile:
            px = px[rng.choice(px.shape[0], max_px_per_tile, replace=False)]
        chunks.append(px)

    if not chunks:
        return np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(chunks, axis=0).astype(np.uint8)
