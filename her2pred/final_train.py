"""Shared CLAM training engine configured through ExperimentConfig.

Includes datasets, model construction, cross-validation, and evaluation.
"""
from scipy import stats
import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
from tqdm import tqdm


# ==============================================================
# 实验配置
# ==============================================================
@dataclass
class ExperimentConfig:
    name: str

    # CV 池（source-stratified 5-fold 切分的来源）
    cv_pool_dirs: List[str]

    # 固定 held-out 测试集（不参与 CV，只在最终 ensemble 后打分）
    test_dirs: List[str] = field(default_factory=list)

    # Exp3 专用：固定 val 模式
    use_fixed_val: bool = False
    fixed_val_dirs: List[str] = field(default_factory=list)

    # 输出根目录
    out_dir: str = 'results'

    # 模型超参
    input_dim: int      = 1536
    size_arg: str       = 'tiny'
    k_sample: int       = 32

    # 训练超参
    seed: int           = 42
    n_folds: int        = 5
    num_epochs: int     = 64
    pos_weight_val: int = 3
    lr: float           = 1e-4
    weight_decay: float = 1e-4
    accum_steps: int    = 8
    label_smooth: float = 0.9
    bag_weight: float   = 0.8
    inst_weight: float  = 0.2


# ==============================================================
# 工具
# ==============================================================

def delong_auc_ci(labels, probs, alpha=0.05):
    """
    用 DeLong 方法计算 AUC 的 (1-alpha) 置信区间。
 
    参考：
      DeLong et al. (1988). Comparing the areas under two or more
      correlated receiver operating characteristic curves: a
      nonparametric approach. Biometrics, 44(3), 837-845.
 
    返回 (auc_val, ci_lower, ci_upper)。
    若正/负样本数 < 2，退化为只返回点估计，CI = (auc, auc)。
    """
    labels = np.asarray(labels, dtype=int)
    probs  = np.asarray(probs,  dtype=float)
 
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
 
    if n_pos < 2 or n_neg < 2:
        # 退化情形：无法估计方差
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0
        return auc_val, auc_val, auc_val
 
    pos_scores = probs[pos_idx]
    neg_scores = probs[neg_idx]
 
    # Mann-Whitney 统计量 = AUC
    # V_pos[i] = (1/n_neg) * Σ_j ψ(pos_i, neg_j)
    # V_neg[j] = (1/n_pos) * Σ_i ψ(pos_i, neg_j)
    # ψ(x, y) = 1 if x>y, 0.5 if x==y, 0 if x<y
    def psi(x, y):
        return np.where(x > y, 1.0, np.where(x == y, 0.5, 0.0))
 
    # 矩阵化计算，避免双重循环
    # shape: (n_pos, n_neg)
    mat = psi(pos_scores[:, None], neg_scores[None, :])   # (n_pos, n_neg)
 
    auc_val = mat.mean()
 
    V_pos = mat.mean(axis=1)   # 每个正样本的结构分量, shape (n_pos,)
    V_neg = mat.mean(axis=0)   # 每个负样本的结构分量, shape (n_neg,)
 
    S10 = np.var(V_pos, ddof=1)   # 正样本方差
    S01 = np.var(V_neg, ddof=1)   # 负样本方差
 
    var_auc = S10 / n_pos + S01 / n_neg
    se_auc  = np.sqrt(max(var_auc, 0.0))
 
    z = stats.norm.ppf(1 - alpha / 2)   # 1.96 for 95% CI
    ci_lo = np.clip(auc_val - z * se_auc, 0.0, 1.0)
    ci_hi = np.clip(auc_val + z * se_auc, 0.0, 1.0)
 
    return float(auc_val), float(ci_lo), float(ci_hi)
 
 
# ──────────────────────────────────────────────────────────────
# 3a. 替换 _plot_roc（单曲线图，标题栏显示 CI）
# ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc
 
 
def _plot_roc(labels, probs, save_path: str, title: str) -> float:
    if len(set(labels)) < 2:
        return 0.0
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc              = sk_auc(fpr, tpr)
    youden_idx           = np.argmax(tpr - fpr)
 
    _, ci_lo, ci_hi = delong_auc_ci(labels, probs)
 
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'AUC = {roc_auc:.4f}\n95% CI [{ci_lo:.4f}, {ci_hi:.4f}]')
    plt.plot([0, 1], [0, 1], 'navy', lw=1, linestyle='--')
    plt.scatter([fpr[youden_idx]], [tpr[youden_idx]], color='red', zorder=5,
                label=f"Youden thresh = {thresholds[youden_idx]:.3f}")
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title)
    plt.legend(loc='lower right'); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()
    return roc_auc
 
 
# ──────────────────────────────────────────────────────────────
# 3b. 替换 plot_combined_roc（多曲线对比图，每条曲线带 CI）
# ──────────────────────────────────────────────────────────────
def plot_combined_roc(roc_data: list, save_path: str):
    colors = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
              '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990']
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'gray', lw=1, linestyle='--', label='Random')
 
    for i, item in enumerate(roc_data):
        if len(set(item['labels'])) < 2:
            continue
        fpr, tpr, _ = roc_curve(item['labels'], item['probs'])
        roc_auc     = sk_auc(fpr, tpr)
        youden_idx  = np.argmax(tpr - fpr)
        _, ci_lo, ci_hi = delong_auc_ci(item['labels'], item['probs'])
 
        is_summary  = any(k in item['name'] for k in ('Ensemble', 'OOF', 'Val'))
        lw = 2.5 if is_summary else 1.5
        ls = '-'  if is_summary else '--'
        c  = colors[i % len(colors)]
 
        label = (
            f"{item['name']}  "
            f"AUC={roc_auc:.4f} "
            f"[{ci_lo:.4f}, {ci_hi:.4f}]"
        )
        plt.plot(fpr, tpr, color=c, lw=lw, linestyle=ls, label=label)
        plt.scatter([fpr[youden_idx]], [tpr[youden_idx]],
                    color=c, s=40, zorder=5)
 
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC comparison')
    plt.legend(loc='lower right', fontsize=7.5)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()
    
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================
# 模型
# ==============================================================
class GatedAttention(nn.Module):
    def __init__(self, L=512, D=256, dropout=True):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_w = nn.Linear(D, 1)
        self.dropout = nn.Dropout(0.25) if dropout else nn.Identity()

    def forward(self, H):
        A = self.attention_w(self.attention_V(H) * self.attention_U(H))
        A = self.dropout(A)
        A = F.softmax(A.transpose(1, 0), dim=1)
        return A


class CLAM_SB(nn.Module):
    def __init__(self, input_dim=1536, size_arg='small', k_sample=8):
        super().__init__()
        self.k_sample = k_sample
        size_dict = {
            'tiny' : [input_dim, 256, 128],
            'small': [input_dim, 512, 256],
            'big'  : [input_dim, 512, 384],
        }
        L, D = size_dict[size_arg][1], size_dict[size_arg][2]
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, L), nn.ReLU(), nn.Dropout(0.25),
        )
        self.attention           = GatedAttention(L=L, D=D)
        self.classifier          = nn.Linear(L, 1)
        self.inst_classifier_pos = nn.Linear(L, 2)
        self.inst_classifier_neg = nn.Linear(L, 2)

    @staticmethod
    def _inst_loss(A, H, inst_clf, k_sample, device):
        top_ids = torch.topk(A,  k_sample)[1][-1]
        bot_ids = torch.topk(-A, k_sample)[1][-1]
        logits  = torch.cat([inst_clf(H[top_ids]), inst_clf(H[bot_ids])], dim=0)
        targets = torch.cat([
            torch.ones(k_sample,  dtype=torch.long, device=device),
            torch.zeros(k_sample, dtype=torch.long, device=device),
        ])
        return F.cross_entropy(logits, targets)

    def forward(self, x, label=None, instance_eval=False):
        device    = x.device
        H         = self.encoder(x)
        A         = self.attention(H)
        M         = torch.mm(A, H)
        wsi_prob  = torch.sigmoid(self.classifier(M)).squeeze()
        inst_loss = torch.tensor(0.0, device=device)
        if instance_eval and label is not None:
            is_pos   = int(label.item()) == 1
            inst_clf = self.inst_classifier_pos if is_pos else self.inst_classifier_neg
            inst_loss = self._inst_loss(A, H, inst_clf, self.k_sample, device)
        return wsi_prob, A.squeeze(), inst_loss


def build_model(cfg: ExperimentConfig, device):
    return CLAM_SB(
        input_dim=cfg.input_dim, size_arg=cfg.size_arg, k_sample=cfg.k_sample
    ).to(device)


# ==============================================================
# 数据
# ==============================================================
def _infer_source(dirpath: str) -> str:
    """从路径推断 source 名，如 ndpi / svs / kfb / tcga。"""
    parts = os.path.normpath(dirpath).split(os.sep)
    for part in reversed(parts):
        if '_data' in part:
            return part.replace('_data_aug', '').replace('_data', '')
    return 'unknown'


def _infer_root(dirpath: str) -> str:
    """
    从路径推断数据集根目录名。
    例如：
        ./slide_datasets_filtered/ndpi_data/train  →  slide_datasets_filtered
        ./slide_datasets/ndpi_data/train            →  slide_datasets
    取路径中倒数第三个分量（source_dir 的父目录）。
    """
    parts = os.path.normpath(dirpath).split(os.sep)
    # 找 _data 所在层级，其父目录就是 root
    for i in reversed(range(len(parts))):
        if '_data' in parts[i]:
            return parts[i - 1] if i > 0 else 'unknown_root'
    return 'unknown_root'


def scan_files(data_dirs: List[str]) -> List[dict]:
    """
    扫描所有 .pt 文件，每条记录包含：
        path, label, source, slide_id, is_aug, root
    root：slide_datasets 或 slide_datasets_filtered，用于 test 分组评估。
    is_aug：来自 _aug 目录。
    """
    records = []
    for d in data_dirs:
        if not os.path.exists(d):
            print(f"  [warn] not found: {d}")
            continue
        source = _infer_source(d)
        is_aug = '_aug' in os.path.normpath(d)
        root   = _infer_root(d)
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.pt'):
                continue
            path = os.path.join(d, fname)
            try:
                data     = torch.load(path, map_location='cpu')
                label    = int(data['label'].item())
                slide_id = str(data['slide_id'])
                records.append({
                    'path'    : path,
                    'label'   : label,
                    'source'  : source,
                    'slide_id': slide_id,
                    'is_aug'  : is_aug,
                    'root'    : root,
                })
            except Exception as e:
                print(f"  [warn] failed to load {path}: {e}")
    return records


class FileListDataset(Dataset):
    def __init__(self, records: List[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec  = self.records[idx]
        data = torch.load(rec['path'])
        return data['features'], data['label'].float(), data['slide_id'], data.get('coords', None)


# ==============================================================
# Source × label 双层分层 5-fold 切分（slide-level，防止泄露）
# ==============================================================
def make_source_stratified_folds(records: List[dict], n_folds: int, seed: int):
    """
    切分单位是 slide（用 slide_id 去重），保证同一 slide 的所有文件
    （orig/aug/filtered/unfiltered）永远在同一折。

    规则：
    - KFold 在 slide 级别做，以 (source, label) 双层分层
    - val 放该折 slide 的全部文件（orig + aug），两个 ROOT 都包含
    - train 放其余折 slide 的全部文件（orig + aug）
    - 切分单位是 slide，同一 slide 的所有文件永远在同一折
    """

    # 1. 以 (source, slide_id) 为 key 聚合所有文件
    slide_map: Dict[Tuple, dict] = {}
    for rec in records:
        key = (rec['source'], rec['slide_id'])
        if key not in slide_map:
            slide_map[key] = {
                'label'   : rec['label'],
                'source'  : rec['source'],
                'slide_id': rec['slide_id'],
                'files'   : [],
            }
        slide_map[key]['files'].append(rec)

    slides = list(slide_map.values())

    # 2. 按 source × label 分组，在 slide 层面做 KFold
    src_label_groups: Dict[Tuple, list] = defaultdict(list)
    for sl in slides:
        src_label_groups[(sl['source'], sl['label'])].append(sl)

    # 打印 slide 级别分布
    print(f"\n  Data pool (slide level):")
    src_summary: dict = defaultdict(lambda: [0, 0])
    for sl in slides:
        src_summary[sl['source']][sl['label']] += 1
    for src, (neg, pos) in sorted(src_summary.items()):
        total_files = sum(len(sl['files']) for sl in slides if sl['source'] == src)
        print(f"    {src:12s}  slides: pos={pos:4d}  neg={neg:4d}"
              f"  |  files (all versions): {total_files:4d}")

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_val_slides: List[List[dict]] = [[] for _ in range(n_folds)]

    for (src, lbl), group_slides in src_label_groups.items():
        arr = np.array(group_slides, dtype=object)
        y   = np.zeros(len(arr), dtype=int)
        if len(arr) < n_folds:
            for i, sl in enumerate(arr):
                fold_val_slides[i % n_folds].append(sl)
        else:
            for fold_idx, (_, val_idx) in enumerate(kf.split(arr, y)):
                for sl in arr[val_idx]:
                    fold_val_slides[fold_idx].append(sl)

    # 3. 构建每折的 (train_records, val_records)
    folds = []
    for k in range(n_folds):
        val_records   = [rec for sl in fold_val_slides[k]
                             for rec in sl['files']]
        train_records = [rec for j in range(n_folds) if j != k
                             for sl in fold_val_slides[j]
                             for rec in sl['files']]
        folds.append((train_records, val_records))

    # 4. 打印折统计
    print(f"\n  Per-fold split (slides / files):")
    for k, (tr, va) in enumerate(folds):
        tr_pos = sum(r['label'] for r in tr)
        va_pos = sum(r['label'] for r in va)
        va_slides = len(fold_val_slides[k])
        tr_slides = sum(len(fold_val_slides[j]) for j in range(n_folds) if j != k)
        src_cnt: dict = defaultdict(lambda: [0, 0])
        for r in va:
            src_cnt[r['source']][r['label']] += 1
        detail = '  '.join(
            f"{s}(+{v[1]}/-{v[0]})" for s, v in sorted(src_cnt.items()))
        print(f"    Fold {k+1}"
              f"  train {tr_slides:3d} slides / {len(tr):4d} files"
              f"  (+{tr_pos}/-{len(tr)-tr_pos})"
              f"  |  val {va_slides:3d} slides / {len(va):3d} files"
              f"  (+{va_pos}/-{len(va)-va_pos})"
              f"  [{detail}]")

    return folds
