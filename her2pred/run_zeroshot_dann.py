"""CLAM training with domain adversaries and gradient reversal.

Supports patch-level and slide-level domain classification, appearance
regression, site adversaries, and configurable encoder depth. The gradient
reversal coefficient follows a DANN schedule or a constant value.
Use scripts/train_final.py for the experiment configurations.
"""

import os
import sys

# --gpu 必须在 import torch 之前生效
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
        break

import json
import math
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ndpi_cv as R                                              # noqa: E402
import run_zeroshot_tcga as Z                                       # noqa: E402
from final_train import (                                           # noqa: E402
    ExperimentConfig, CLAM_SB, FileListDataset, set_seed, GatedAttention,
)
from metrics_utils import binary_metrics, bce_loss                  # noqa: E402
from parse_slide_ids import norm_key                               # noqa: E402
from appearance_cache_scope import validate_coverage              # noqa: E402

APPEAR_FULL_CACHE = os.environ.get(
    'HER2PRED_APPEAR_FULL_CACHE',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'assets', 'appearance_cache.npz'))

# 本进程内共享状态
STATE = {'dom_of': {}, 'domains': [], 'cfg_dann': None, 'log': [],
         'domain_mode': 'appearregfull',
         'colour_vec': None, 'reg_dim': 0,
         'appearance_selected_ids': None, 'appearance_missing_ids': None,
         'appearance_cache_info': None,
         # 站点（source）对抗 —— 见 build_site_map 的说明
         'site_of': {}, 'site_names': [], 'site_acc': []}


# ==============================================================
# 梯度反转层
# ==============================================================
class _GradReverse(torch.autograd.Function):
    """前向恒等，反向把梯度乘 −λ。"""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = float(lam)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam):
    return _GradReverse.apply(x, lam)


# ==============================================================
# CDA-CLAM：CLAM_SB + 两级域判别器
# ==============================================================
class CDA_CLAM_SB(CLAM_SB):
    """
    CLAM with patch-level and slide-level domain heads.

    forward() performs classification. forward_dann() also returns domain losses.
    """

    def __init__(self, input_dim=1536, size_arg='tiny', k_sample=32,
                 n_domains=3, disc_hidden=128, dropout=0.25, reg_dim=0,
                 n_sites=0):
        super().__init__(input_dim=input_dim, size_arg=size_arg,
                         k_sample=k_sample)
        L = self.classifier.in_features          # 从父类拿隐层宽度，避免重复 size_dict
        self.n_domains = int(n_domains)
        self.n_sites = int(n_sites)

        def mlp():
            return nn.Sequential(
                nn.Linear(L, disc_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(disc_hidden, self.n_domains),
            )
        self.slide_disc = mlp()      # global-level slide-wise alignment
        self.patch_disc = mlp()      # local-level patch-wise alignment

        # 连续多维混杂（如颜色向量）的对抗**回归**头。
        # 分类式判别器要求把连续量分桶，而分桶必须二选一：
        #   按 PC1 分  → 抓不住捷径（颜色 PCA2→标签仅 0.626）
        #   按倾向性分 → 与标签完全共线，压它等于压标签
        # 回归形式不需要做这个选择，直接要求「表示里回归不出该向量」。
        self.reg_dim = int(reg_dim)
        if self.reg_dim > 0:
            def reg():
                return nn.Sequential(
                    nn.Linear(L, disc_hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(disc_hidden, self.reg_dim),
                )
            self.slide_reg = reg()
            self.patch_reg = reg()

        if self.n_sites > 0:
            def sdisc():
                return nn.Sequential(
                    nn.Linear(L, disc_hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(disc_hidden, self.n_sites),
                )
            self.slide_sdisc = sdisc()
            self.patch_sdisc = sdisc()
            # 站点截距（固定效应）：训练时 logit 加 site_bias[s]，推理时不加。
            # 作用 —— svs 全阳 / kfb 全阴，BCE 只要「认出格式」就能归零，
            # 这份梯度是纯捷径燃料，而且会和站点判别器拔河。
            # 把站点的**流行率**交给一个每站点的标量去解释后，
            # 特征分支只需要解释**站点内**的标签变异，拔河消失。
            # 退化站点上 σ(f+b)→y 使 f 的梯度自动衰减到 ~0。
            self.site_bias = nn.Parameter(torch.zeros(self.n_sites))

    def forward_dann(self, x, label, domain, lam,
                     instance_eval=True, n_patch=256, use_patch=True,
                     use_slide=True, reg_target=None,
                     site=None, site_adv_w=0.0, use_site_bias=False,
                     conditional_site_adv=False, site_class_ids=None,
                     site_slide_only=False,
                     use_domain=True, return_repr=False,
                     domain_lam=None, site_lam=None):
        device = x.device
        H = (self._encode(x) if hasattr(self, '_encode') else self.encoder(x))
        A = self.attention(H)                     # (1, N)
        M = torch.mm(A, H)                        # (1, L)
        logit = self.classifier(M)
        if use_site_bias and site is not None and self.n_sites > 0:
            logit = logit + self.site_bias[int(site)]
        wsi_prob = torch.sigmoid(logit).squeeze()

        inst_loss = torch.tensor(0.0, device=device)
        if instance_eval and label is not None:
            is_pos = int(label.item()) == 1
            clf = self.inst_classifier_pos if is_pos else self.inst_classifier_neg
            inst_loss = self._inst_loss(A, H, clf, self.k_sample, device)

        d = torch.tensor([int(domain)], dtype=torch.long, device=device)
        zero = torch.tensor(0.0, device=device)
        domain_lam = lam if domain_lam is None else float(domain_lam)
        site_lam = lam if site_lam is None else float(site_lam)

        dom_slide = zero
        acc_slide = float('nan')
        if use_slide and use_domain:
            logit_s = self.slide_disc(grad_reverse(M, domain_lam))   # (1, K)
            dom_slide = F.cross_entropy(logit_s, d)
            acc_slide = float((logit_s.argmax(1) == d).float().mean().item())

        dom_patch = zero
        acc_patch = float('nan')
        if use_patch and use_domain:
            n = min(int(n_patch), H.shape[0])
            idx = torch.randperm(H.shape[0], device=device)[:n]
            logit_p = self.patch_disc(grad_reverse(H[idx], domain_lam))  # (n, K)
            dom_patch = F.cross_entropy(logit_p, d.expand(n))
            acc_patch = float((logit_p.argmax(1) == d).float().mean().item())

        # 对抗回归（连续多维混杂）
        if reg_target is not None and self.reg_dim > 0:
            t = reg_target.to(device).view(1, -1)
            if use_slide and use_domain:
                pred_s = self.slide_reg(grad_reverse(M, domain_lam))
                dom_slide = dom_slide + F.mse_loss(pred_s, t)
                # 「域 acc」在回归下改用 1 − 相对残差，同样是越低越对齐
                acc_slide = float(1.0 - (F.mse_loss(pred_s, t)
                                         / (t.pow(2).mean() + 1e-8)).clamp(0, 1))
            if use_patch and use_domain:
                n = min(int(n_patch), H.shape[0])
                idx = torch.randperm(H.shape[0], device=device)[:n]
                pred_p = self.patch_reg(grad_reverse(H[idx], domain_lam))
                dom_patch = dom_patch + F.mse_loss(pred_p, t.expand(n, -1))
                acc_patch = float(1.0 - (F.mse_loss(pred_p, t.expand(n, -1))
                                         / (t.pow(2).mean() + 1e-8)).clamp(0, 1))

        if site_adv_w > 0 and site is not None and self.n_sites > 0:
            sv = torch.tensor([int(site)], dtype=torch.long, device=device)

            def conditional_logits(logits):
                """Restrict source discrimination to sources seen in this class."""
                if not conditional_site_adv or label is None or not site_class_ids:
                    return logits, sv
                allowed = list(site_class_ids.get(int(label.item()), []))
                if len(allowed) < 2 or int(site) not in allowed:
                    return None, None
                cols = torch.tensor(allowed, dtype=torch.long, device=device)
                target = torch.tensor([allowed.index(int(site))],
                                      dtype=torch.long, device=device)
                return logits.index_select(1, cols), target

            if use_slide:
                ls = self.slide_sdisc(grad_reverse(M, site_lam))
                ls_use, sv_use = conditional_logits(ls)
                if ls_use is not None:
                    dom_slide = (dom_slide + site_adv_w
                                 * F.cross_entropy(ls_use, sv_use))
                    STATE['site_acc'].append(float(
                        (ls_use.argmax(1) == sv_use).float().mean().item()))
            if use_patch and not site_slide_only:
                n = min(int(n_patch), H.shape[0])
                idx = torch.randperm(H.shape[0], device=device)[:n]
                lp = self.patch_sdisc(grad_reverse(H[idx], site_lam))
                lp_use, sv_use = conditional_logits(lp)
                if lp_use is not None:
                    dom_patch = dom_patch + site_adv_w * F.cross_entropy(
                        lp_use, sv_use.expand(n))

        result = (wsi_prob, A.squeeze(), inst_loss,
                  dom_slide, dom_patch, acc_slide, acc_patch)
        # Opt-in only: expose the attention-pooled slide representation for a
        # parameter-free same-label cross-source alignment loss.  Keeping the
        # default tuple unchanged preserves every historical call path.
        return result + (M,) if return_repr else result


# ==============================================================
# λ 调度（DANN 标准）
# ==============================================================
def lam_at(progress, lam_max, gamma=10.0):
    p = float(np.clip(progress, 0.0, 1.0))
    return lam_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


def external_adain_to_ndpi(external_features, ndpi_features, eps=1e-6):
    """Match external patch-feature means and standard deviations to an NDPI slide.

    Inputs have shape (n_patches, feature_dim). Statistics are calculated across
    patches per feature channel. The reference is the shuffled main-loader item.
    """
    if external_features.ndim != 2 or ndpi_features.ndim != 2:
        raise ValueError('external AdaIN expects 2-D patch-feature tensors')
    if external_features.shape[1] != ndpi_features.shape[1]:
        raise ValueError(
            'external and NDPI feature dimensions differ: '
            f'{external_features.shape[1]} vs {ndpi_features.shape[1]}')
    ext_mean = external_features.mean(dim=0, keepdim=True)
    ref_mean = ndpi_features.mean(dim=0, keepdim=True)
    ext_std = external_features.var(
        dim=0, unbiased=False, keepdim=True).add(eps).sqrt()
    ref_std = ndpi_features.var(
        dim=0, unbiased=False, keepdim=True).add(eps).sqrt()
    return (external_features - ext_mean) / ext_std * ref_std + ref_mean


def estimate_scanner_subspace(records, input_dim):
    """Estimate two same-label scanner directions from training-fold data only.

    Each slide contributes one mean feature vector, so large slides do not get
    more weight. Augmented duplicates are excluded from the estimator; they are
    still projected and used by training. No TCGA record is used.
    """
    groups = defaultdict(list)
    seen = set()
    for rec in records:
        src = str(rec.get('source', ''))
        label = int(rec['label'])
        if src not in ('ndpi', 'svs', 'kfb'):
            continue
        path = str(rec['path'])
        if '_aug' in path:
            continue
        key = (src, str(rec['slide_id']))
        if key in seen:
            continue
        seen.add(key)
        data = torch.load(path, map_location='cpu')
        feat = data['features'].float()
        if feat.ndim != 2 or feat.shape[1] != int(input_dim):
            raise ValueError(f'invalid feature shape for scanner projection: '
                             f'{path}: {tuple(feat.shape)}')
        mean = feat.mean(dim=0)
        if not torch.isfinite(mean).all():
            raise ValueError(f'non-finite slide mean for scanner projection: {path}')
        groups[(src, label)].append(mean)

    required = [('ndpi', 0), ('ndpi', 1), ('svs', 1), ('kfb', 0)]
    missing = [key for key in required if not groups[key]]
    if missing:
        raise ValueError(f'scanner projection missing training groups: {missing}')

    means = {key: torch.stack(groups[key]).mean(dim=0) for key in required}
    delta_pos = means[('svs', 1)] - means[('ndpi', 1)]
    delta_neg = means[('kfb', 0)] - means[('ndpi', 0)]
    deltas = torch.stack([delta_pos, delta_neg])
    _, singular, vh = torch.linalg.svd(deltas, full_matrices=False)
    tol = torch.finfo(singular.dtype).eps * max(deltas.shape) * singular.max()
    rank = max(1, int((singular > tol).sum().item()))
    basis = vh[:rank].T.contiguous()
    center = torch.stack(groups[('ndpi', 0)] + groups[('ndpi', 1)]).mean(dim=0)
    counts = {f'{src}_{label}': len(groups[(src, label)])
              for src, label in required}
    return dict(center=center, basis=basis, singular=singular,
                deltas=deltas, counts=counts, rank=rank)


# ==============================================================
# 域标签
# ==============================================================
def build_domain_map(records, mode='appearregfull'):
    """Map slides to source classes or appearance-regression targets."""
    if mode == 'appearregfull':
        vec = STATE.get('colour_vec') or {}
        return ({r['slide_id']: (0 if norm_key(str(r['slide_id'])) in vec else -1)
                 for r in records}, ['colour_reg'])
    if mode != 'source':
        raise ValueError(f"unsupported domain mode: {mode}")
    srcs = sorted({r['source'] for r in records})
    idx = {source: i for i, source in enumerate(srcs)}
    return ({r['slide_id']: idx[r['source']] for r in records}, srcs)


def build_site_map(records):
    """Return slide-to-site IDs and source names for the site discriminator."""
    srcs = sorted({str(r['source']) for r in records})
    idx = {t: i for i, t in enumerate(srcs)}
    return ({norm_key(str(r['slide_id'])): idx[str(r['source'])]
             for r in records}, srcs)






def build_appearance_vectors_full():
    """Load the appearance-regression cache and return targets keyed by slide ID.

    Uses prestandardized targets directly and standardizes other targets once.
    """
    import numpy as _np
    if not os.path.exists(APPEAR_FULL_CACHE):
        raise FileNotFoundError(f"Appearance cache missing: {APPEAR_FULL_CACHE}")
    z = _np.load(APPEAR_FULL_CACHE, allow_pickle=True)
    X, sid = z['X'], z['sid']
    ok = ~_np.isnan(X).any(0)
    X = _np.nan_to_num(X[:, ok])
    prestandardized = (
        bool(_np.asarray(z['prestandardized']).reshape(-1)[0])
        if 'prestandardized' in z.files else False
    )
    STATE['appearance_selected_ids'] = None
    STATE['appearance_missing_ids'] = None
    STATE['appearance_cache_info'] = {
        'prestandardized': prestandardized,
        'target_n': int(len(sid)),
        'dimension_n': int(X.shape[1]),
    }
    if prestandardized:
        required = {'selected_sid', 'missing_sid', 'expected_selected_n',
                    'expected_target_n'}
        absent = required.difference(z.files)
        if absent:
            raise ValueError(
                f'prestandardized appearance cache lacks metadata: {sorted(absent)}'
            )
        selected = {norm_key(str(value)) for value in z['selected_sid']}
        missing = {norm_key(str(value)) for value in z['missing_sid']}
        targets = {norm_key(str(value)) for value in sid}
        expected_selected = int(
            _np.asarray(z['expected_selected_n']).reshape(-1)[0])
        expected_targets = int(
            _np.asarray(z['expected_target_n']).reshape(-1)[0])
        if (len(selected) != expected_selected
                or len(targets) != expected_targets
                or targets | missing != selected
                or targets & missing):
            raise ValueError('inconsistent public appearance-cache metadata')
        STATE['appearance_selected_ids'] = selected
        STATE['appearance_missing_ids'] = missing
        STATE['appearance_cache_info'].update({
            'selected_n': len(selected),
            'declared_missing_n': len(missing),
        })
    if not prestandardized:
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return ({norm_key(str(s)): X[i].astype('float32')
             for i, s in enumerate(sid)}, X.shape[1])








# ==============================================================
# 注入点：带对抗训练的单折训练
# ==============================================================

class DeepCDA_CLAM_SB(CDA_CLAM_SB):
    """CLAM with adversarial heads, residual encoder blocks, and an optional bottleneck.

    enc_depth controls encoder depth. legacy_act=True selects ReLU and False
    selects GELU. Additional blocks apply LayerNorm, Linear, activation, and
    Dropout with a residual connection.
    """

    def __init__(self, input_dim=1536, size_arg='tiny', k_sample=32,
                 n_domains=3, disc_hidden=128, dropout=0.25, reg_dim=0,
                 enc_depth=1, legacy_act=True, bottleneck=0, n_sites=0,
                 scanner_projection=None, scanner_proj_alpha=0.0):
        super().__init__(input_dim=input_dim, size_arg=size_arg,
                         k_sample=k_sample, n_domains=n_domains,
                         disc_hidden=disc_hidden, dropout=dropout,
                         reg_dim=reg_dim, n_sites=n_sites)
        L = self.classifier.in_features
        self.enc_depth = int(enc_depth)
        self.bottleneck = int(bottleneck)
        self.scanner_proj_alpha = float(scanner_proj_alpha)
        if scanner_projection is not None and self.scanner_proj_alpha > 0:
            self.register_buffer(
                'scanner_proj_center', scanner_projection['center'].float())
            self.register_buffer(
                'scanner_proj_basis', scanner_projection['basis'].float())
        act = (lambda: nn.ReLU()) if legacy_act else (lambda: nn.GELU())
        if self.enc_depth > 1:
            self.encoder = nn.Sequential(
                nn.LayerNorm(input_dim), nn.Linear(input_dim, L),
                act(), nn.Dropout(dropout))
            self.enc_blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(L), nn.Linear(L, L),
                              act(), nn.Dropout(dropout))
                for _ in range(self.enc_depth - 1)])
        if self.bottleneck > 0:
            # Project the slide representation to the bottleneck dimension.
            W = self.bottleneck
            self.enc_proj = nn.Sequential(nn.LayerNorm(L), nn.Linear(L, W), act())
            D_att = max(W // 2, 8)
            self.attention = GatedAttention(L=W, D=D_att)
            self.classifier = nn.Linear(W, 1)
            self.inst_classifier_pos = nn.Linear(W, 2)
            self.inst_classifier_neg = nn.Linear(W, 2)

            def mlp(out):
                return nn.Sequential(
                    nn.Linear(W, disc_hidden), act(), nn.Dropout(dropout),
                    nn.Linear(disc_hidden, out))
            self.slide_disc = mlp(self.n_domains)
            self.patch_disc = mlp(self.n_domains)
            if self.reg_dim > 0:
                self.slide_reg = mlp(self.reg_dim)
                self.patch_reg = mlp(self.reg_dim)
            if self.n_sites > 0:
                self.slide_sdisc = mlp(self.n_sites)
                self.patch_sdisc = mlp(self.n_sites)

    def _encode(self, x):
        if hasattr(self, 'scanner_proj_basis') and self.scanner_proj_alpha > 0:
            centered = x - self.scanner_proj_center
            nuisance = torch.mm(
                torch.mm(centered, self.scanner_proj_basis),
                self.scanner_proj_basis.T)
            x = x - self.scanner_proj_alpha * nuisance
        h = self.encoder(x)
        if getattr(self, 'enc_depth', 1) > 1:
            for blk in self.enc_blocks:
                h = h + blk(h)                       # 残差，训深了不退化
        if getattr(self, 'bottleneck', 0) > 0:
            h = self.enc_proj(h)
        return h

    # forward / forward_dann 都改走 _encode；depth=1 时它等价于 self.encoder
    def forward(self, x, label=None, instance_eval=False):
        H = self._encode(x)
        A = self.attention(H)
        M = torch.mm(A, H)
        wsi_prob = torch.sigmoid(self.classifier(M)).squeeze()
        inst_loss = torch.tensor(0.0, device=x.device)
        if instance_eval and label is not None:
            is_pos = int(label.item()) == 1
            clf = self.inst_classifier_pos if is_pos else self.inst_classifier_neg
            inst_loss = self._inst_loss(A, H, clf, self.k_sample, x.device)
        return wsi_prob, A.squeeze(), inst_loss


def train_one_fold_dann(fold_idx, train_records, val_records, cfg, ckpt_dir,
                        ckpt_select='last'):
    """
    结构与 run_ndpi_cv.train_one_fold 一致（相同的 loss 组合、accum、
    ckpt 选择、evaluate 时机），只多加两级域对抗损失。
    """
    dc = STATE['cfg_dann']
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = STATE['domain_mode']
    all_train = list(train_records)

    dom_of, dom_names = build_domain_map(all_train, mode)
    STATE['dom_of'], STATE['domains'] = dom_of, dom_names

    # The public cache declares the 229 NDPI cases and three target omissions.
    # In the final zero/few-shot protocol, external_site_only disables the
    # appearance loss for SVS/KFB/TCGA, so those sources need no targets.
    # Audit only records that can actually contribute appearance gradients.
    if mode in ('appearregfull',):
        validate_coverage(
            all_train, val_records, dom_of,
            STATE.get('appearance_selected_ids'),
            STATE.get('appearance_missing_ids'),
            external_site_only=bool(dc.get('external_site_only', False)),
            site_only_adv=bool(dc.get('site_only_adv', False)),
        )

    # 站点（source）对抗：只有 --site_adv_w>0 或 --site_intercept 时才启用
    site_adv_w = float(dc.get('site_adv_w', 0.0) or 0.0)
    tcga_adv_w = dc.get('tcga_adv_w')
    use_site_bias = bool(dc.get('site_intercept', False))
    if site_adv_w > 0 or use_site_bias:
        # A supervised-only few-shot target must not define a discriminator
        # class or receive GRL gradients.  Filtering here also keeps the
        # adversary architecture identical to the zero-shot three-source run.
        site_records = [
            r for r in all_train
            if not (str(r.get('source', '')) == 'tcga'
                    and tcga_adv_w is not None
                    and float(tcga_adv_w) == 0.0)]
        site_of, site_names = build_site_map(site_records)
        n_sites = len(site_names)
        STATE['site_of'], STATE['site_names'] = site_of, site_names
        print(f"    [site] 站点判别器：{n_sites} 站 {site_names}"
              f"  site_adv_w={site_adv_w}  站点截距={'开' if use_site_bias else '关'}")
    else:
        site_of, site_names, n_sites = {}, [], 0
    STATE['site_acc'] = []

    conditional_site_adv = bool(dc.get('conditional_site_adv', False))
    site_adv_pos_mult = float(dc.get('site_adv_pos_mult', 1.0))
    site_adv_neg_mult = float(dc.get('site_adv_neg_mult', 1.0))
    site_only_adv = bool(dc.get('site_only_adv', False))
    external_site_only = bool(dc.get('external_site_only', False))
    site_slide_only = bool(dc.get('site_slide_only', False))
    same_label_align_w = float(dc.get('same_label_align_w', 0.0) or 0.0)
    external_adain = bool(dc.get('external_adain', False))
    scanner_proj_alpha = float(dc.get('scanner_proj_alpha', 0.0) or 0.0)
    scanner_proj_path = dc.get('scanner_proj_path')
    sequential_external_epochs = int(
        dc.get('sequential_external_epochs', 0) or 0)
    requested_external_sources = dc.get('external_sources')
    if requested_external_sources:
        requested_external_sources = {
            str(value) for value in requested_external_sources}
    external_step_prob = float(dc.get('external_step_prob', 1.0))
    external_steps_per_epoch = dc.get('external_steps_per_epoch')
    if external_steps_per_epoch is not None:
        external_steps_per_epoch = int(external_steps_per_epoch)
    external_steps_by_source = dc.get('external_steps_by_source')
    if external_steps_by_source is not None:
        external_steps_by_source = {
            str(source): int(count)
            for source, count in external_steps_by_source.items()}
    external_source_balanced = bool(
        dc.get('external_source_balanced', False))
    external_ratio_per_step = bool(dc.get('external_ratio_per_step', False))
    alternating_adv = bool(dc.get('alternating_adv', False))
    if site_only_adv and not conditional_site_adv:
        raise ValueError('site_only_adv requires --conditional_site_adv')
    if external_site_only and site_adv_w > 0 and not conditional_site_adv:
        raise ValueError(
            'external_site_only with an active site adversary requires '
            '--conditional_site_adv')
    if alternating_adv and site_adv_w <= 0:
        raise ValueError('alternating_adv requires --site_adv_w > 0')
    if site_adv_pos_mult < 0 or site_adv_neg_mult < 0:
        raise ValueError('site adversarial class multipliers must be non-negative')

    def _site_adv_for(sample_label):
        """Return the label-specific site loss strength."""
        if sample_label is None:
            return site_adv_w
        mult = (site_adv_pos_mult if int(sample_label.item()) == 1
                else site_adv_neg_mult)
        return site_adv_w * mult
    if same_label_align_w < 0:
        raise ValueError('same_label_align_w must be non-negative')
    if same_label_align_w > 0 and external_adain:
        raise ValueError(
            'same_label_align_w and external_adain are separate single-factor '
            'protocols; do not enable both')
    if same_label_align_w > 0 and external_source_balanced:
        raise ValueError(
            'same_label_align_w uses label-matched external sampling and cannot '
            'be combined with source-balanced external sampling')
    if not 0.0 <= scanner_proj_alpha <= 1.0:
        raise ValueError('scanner_proj_alpha must be in [0, 1]')
    if sequential_external_epochs < 0:
        raise ValueError('sequential_external_epochs must be non-negative')
    if sequential_external_epochs >= cfg.num_epochs:
        raise ValueError(
            'sequential_external_epochs must leave at least one NDPI epoch')
    if not 0.0 <= external_step_prob <= 1.0:
        raise ValueError('external_step_prob must be in [0, 1]')
    if external_steps_per_epoch is not None and external_steps_per_epoch < 0:
        raise ValueError('external_steps_per_epoch must be non-negative')
    if (external_steps_by_source is not None
            and any(count < 0 for count in external_steps_by_source.values())):
        raise ValueError('external_steps_by_source counts must be non-negative')
    if external_source_balanced and not bool(dc.get('dual_loader', False)):
        raise ValueError('external_source_balanced requires --dual_loader')
    if external_ratio_per_step and not (
            bool(dc.get('dual_loader', False)) and external_source_balanced):
        raise ValueError(
            'external_ratio_per_step requires dual_loader and '
            'external_source_balanced')
    if external_ratio_per_step and external_steps_per_epoch is not None:
        raise ValueError(
            'external_ratio_per_step and external_steps_per_epoch are '
            'mutually exclusive')
    if external_ratio_per_step and external_steps_by_source is not None:
        raise ValueError(
            'external_ratio_per_step and external_steps_by_source are '
            'mutually exclusive')
    if external_steps_per_epoch is not None and not bool(dc.get('dual_loader', False)):
        raise ValueError('external_steps_per_epoch requires --dual_loader')
    if external_steps_by_source is not None and not (
            bool(dc.get('dual_loader', False)) and external_source_balanced):
        raise ValueError(
            'external_steps_by_source requires dual_loader and '
            'external_source_balanced')
    if (external_steps_by_source is not None
            and external_steps_per_epoch is not None):
        raise ValueError(
            'external_steps_by_source and external_steps_per_epoch are '
            'mutually exclusive')
    site_class_ids = None
    if conditional_site_adv:
        if not n_sites or site_adv_w <= 0:
            raise ValueError('conditional_site_adv requires site_adv_w > 0')
        present = {0: set(), 1: set()}
        for r in all_train:
            sid_site = site_of.get(norm_key(str(r['slide_id'])))
            if sid_site is not None:
                present[int(r['label'])].add(int(sid_site))
        site_class_ids = {label: sorted(ids) for label, ids in present.items()}
        readable = {label: [site_names[i] for i in ids]
                    for label, ids in site_class_ids.items()}
        print(f"    [site] conditional adversary：{readable}")

    # Task-loss weight for sources containing a single class.
    degen_w = float(dc.get('degen_site_w', 1.0))
    # Independent instance-loss weight for single-class sources.
    degen_inst_w = float(dc.get('degen_instance_w', 1.0))
    # Domain-loss weight for single-class sources.
    degen_adv_w = float(dc.get('degen_adv_w', 1.0))
    # 与 degen_adv_w 不同：这里只缩放 GRL 回到 encoder 的梯度，域判别器
    # 自身仍以完整损失学习。这样可以得到“强判别器 + 温和去域”的标准
    # adversarial setup；直接缩放 loss 会把判别器也一起饿死。
    degen_grl_w = float(dc.get('degen_grl_w', 1.0))
    svs_task_w = dc.get('svs_task_w')
    kfb_task_w = dc.get('kfb_task_w')
    svs_adv_w = dc.get('svs_adv_w')
    kfb_adv_w = dc.get('kfb_adv_w')
    svs_appearance_grl_w = dc.get('svs_appearance_grl_w')
    kfb_appearance_grl_w = dc.get('kfb_appearance_grl_w')
    svs_pos_weight = dc.get('svs_pos_weight')
    tcga_pos_weight = dc.get('tcga_pos_weight')
    source_task_override = {'svs': svs_task_w, 'kfb': kfb_task_w}
    source_adv_override = {
        'svs': svs_adv_w, 'kfb': kfb_adv_w, 'tcga': tcga_adv_w}
    appearance_grl_override = {
        'svs': svs_appearance_grl_w,
        'kfb': kfb_appearance_grl_w,
    }
    task_w_of = {}
    inst_w_of = {}
    adv_w_of = {}
    grl_w_of = {}
    appearance_grl_w_of = {}
    if (degen_w != 1.0 or degen_inst_w != 1.0
            or degen_adv_w != 1.0 or degen_grl_w != 1.0
            or any(v is not None for v in source_task_override.values())
            or any(v is not None for v in source_adv_override.values())
            or any(v is not None for v in appearance_grl_override.values())):
        from collections import Counter as _C
        cnt_ps = defaultdict(lambda: [0, 0])
        for r in all_train:
            cnt_ps[str(r['source'])][int(r['label'])] += 1
        for src_name, (nneg, npos) in sorted(cnt_ps.items()):
            degen = (nneg == 0 or npos == 0)
            task_w_of[src_name] = degen_w if degen else 1.0
            inst_w_of[src_name] = degen_inst_w if degen else 1.0
            adv_w_of[src_name] = degen_adv_w if degen else 1.0
            grl_w_of[src_name] = degen_grl_w if degen else 1.0
            appearance_grl_w_of[src_name] = 1.0
            if source_task_override.get(src_name) is not None:
                task_w_of[src_name] = float(source_task_override[src_name])
            if source_adv_override.get(src_name) is not None:
                adv_w_of[src_name] = float(source_adv_override[src_name])
            if appearance_grl_override.get(src_name) is not None:
                appearance_grl_w_of[src_name] = float(
                    appearance_grl_override[src_name])
            print(f"    [degen] {src_name:6s} pos={npos:4d} neg={nneg:4d}"
                  f"  总任务权重={task_w_of[src_name]:.2f}"
                  f"  instance权重={inst_w_of[src_name]:.2f}"
                  f"  对齐权重={adv_w_of[src_name]:.4g}"
                  f"  GRL权重={grl_w_of[src_name]:.4g}"
                  f"  外观GRL权重={appearance_grl_w_of[src_name]:.4g}"
                  + ("   <-- 站内单类，无站内判别信号" if degen else ""))

    if svs_pos_weight is not None and float(svs_pos_weight) < 0:
        raise ValueError('svs_pos_weight must be non-negative')
    if tcga_pos_weight is not None and float(tcga_pos_weight) < 0:
        raise ValueError('tcga_pos_weight must be non-negative')

    def _pos_weight_for(src_name):
        if src_name == 'svs' and svs_pos_weight is not None:
            return float(svs_pos_weight)
        if src_name == 'tcga' and tcga_pos_weight is not None:
            return float(tcga_pos_weight)
        return float(cfg.pos_weight_val)

    print(f"\n  [Fold {fold_idx}] 域定义 = {mode}，共 {len(dom_names)} 个域")
    cnt = defaultdict(lambda: [0, 0])      # [negative, positive]
    for r in all_train:
        d = dom_of.get(str(r['slide_id']), -1)
        if d < 0:
            continue
        cnt[d][r['label']] += 1
    for d, nm in enumerate(dom_names):
        neg, pos = cnt[d]
        flag = ''
        if (pos == 0 or neg == 0) and (pos + neg) > 0:
            flag = '   <-- 单一类别，与标签部分共线'
        print(f"      {nm:12s} (域 {d})  pos={pos:4d}  neg={neg:4d}"
              f"{flag}")
    skipped = sum(1 for r in all_train if dom_of.get(str(r['slide_id']), -1) < 0)
    if skipped:
        print(f"      [note] {skipped} 个文件无外观回归目标；"
              f"跳过外观损失，但有 site 标签时仍参与站点对抗")

    _src_of = {str(r['slide_id']): str(r['source']) for r in all_train}

    # 混合池不能直接沿用 ndpi-only 的 accumulation：加入 svs/kfb 后每个
    # epoch 的文件数约翻倍，Adam 的 step 数以及每个 step 中 ndpi 梯度的
    # 占比都会改变。该选项用更大的 accumulation 保持每 epoch 的优化步数
    # 接近 ndpi-only，并重标定 ndpi 梯度，使其在每个 step 中仍保持基线
    # 的平均量；外部来源则只按其显式 task/adv 权重贡献梯度。
    dual_loader = bool(dc.get('dual_loader', False))
    if dual_loader and bool(dc.get('source_balanced_optim', False)):
        raise ValueError('dual_loader already preserves NDPI optimizer cadence; '
                         'do not combine it with source_balanced_optim')
    if dual_loader and alternating_adv:
        raise ValueError('dual_loader is the simple one-stage protocol and does '
                         'not combine with alternating_adv')

    optim_accum_steps = int(cfg.accum_steps)
    optim_grad_w = defaultdict(lambda: 1.0)
    if bool(dc.get('source_balanced_optim', False)):
        supervised = [r for r in all_train
                     ]
        n_total = len(supervised)
        n_ndpi = sum(str(r.get('source', '')) == 'ndpi'
                     for r in supervised)
        if n_ndpi <= 0:
            raise ValueError('source_balanced_optim requires ndpi records')
        mix_ratio = n_total / n_ndpi
        optim_accum_steps = max(1, round(cfg.accum_steps * mix_ratio))
        optim_grad_w['ndpi'] = mix_ratio
        print(f"    [optim-balance] supervised={n_total} ndpi={n_ndpi} "
              f"ratio={mix_ratio:.4f}; accum {cfg.accum_steps}→"
              f"{optim_accum_steps}; ndpi gradient ×{mix_ratio:.4f}")

    if dual_loader:
        primary_train = [r for r in train_records
                         if str(r.get('source', '')) == 'ndpi']
        # External loaders use SVS/KFB and the fixed TCGA few-shot subset.
        external_source_names = sorted({
            str(r.get('source', '')) for r in train_records
            if str(r.get('source', '')) != 'ndpi'
            and (not requested_external_sources
                 or str(r.get('source', '')) in requested_external_sources)
        })
        if requested_external_sources:
            missing_sources = requested_external_sources.difference(
                external_source_names)
            if missing_sources:
                raise ValueError(
                    f'requested external sources absent: {sorted(missing_sources)}')
        external_records = [
            r for r in train_records
            if str(r.get('source', '')) in external_source_names]
        if not primary_train:
            raise ValueError('dual_loader requires NDPI records in every fold')
        # Keep this loader byte-for-byte equivalent to the historical NDPI-only
        # path: same record order, global RNG, shuffle=True, batch_size=1.
        train_loader = DataLoader(FileListDataset(primary_train),
                                  batch_size=1, shuffle=True, num_workers=0)
        # KFB/SVS (and TCGA-train in few-shot) stay in one ordinary pooled
        # loader, so sampling follows the actual number of slides.  A 1:1
        # source alternation would repeat the tiny all-positive SVS cohort
        # about 4.6 times per epoch; together with pos_weight=3 that makes its
        # gradient larger than all NDPI positives and is no longer a normal
        # joint-training baseline.  The pooled external loader plus the
        # existing pos_weight gives nearly balanced effective class mass.
        generator = torch.Generator()
        generator.manual_seed(int(cfg.seed) + 1000 * (fold_idx + 1))
        external_loader = (DataLoader(
            FileListDataset(external_records), batch_size=1, shuffle=True,
            num_workers=0, generator=generator) if external_records else None)
        external_source_loaders = None
        if external_source_balanced:
            external_source_loaders = {}
            for source_index, source_name in enumerate(external_source_names):
                source_records = [
                    r for r in external_records
                    if str(r.get('source', '')) == source_name]
                if not source_records:
                    raise ValueError(
                        f'external source {source_name!r} has no records')
                source_generator = torch.Generator()
                source_generator.manual_seed(
                    int(cfg.seed) + 1000 * (fold_idx + 1)
                    + 100 * (source_index + 1))
                external_source_loaders[source_name] = DataLoader(
                    FileListDataset(source_records), batch_size=1,
                    shuffle=True, num_workers=0, generator=source_generator)
            print('    [external-source-balance] uniform round-robin source '
                  'sampling; ordinary HER2 task loss retained for every source; '
                  + ', '.join(
                      f'{source}={len(loader.dataset)}'
                      for source, loader in external_source_loaders.items()))
            if (external_steps_by_source is not None
                    and set(external_steps_by_source)
                    != set(external_source_loaders)):
                raise ValueError(
                    'external_steps_by_source keys must exactly match '
                    f'external sources {sorted(external_source_loaders)}')
        external_label_loaders = None
        if same_label_align_w > 0:
            external_label_loaders = {}
            for class_label in (0, 1):
                class_records = [r for r in external_records
                                 if int(r['label']) == class_label]
                if not class_records:
                    raise ValueError(
                        'same-label alignment requires both labels in the external pool; '
                        f'class {class_label} is missing')
                class_generator = torch.Generator()
                class_generator.manual_seed(
                    int(cfg.seed) + 1000 * (fold_idx + 1) + 100 * class_label)
                external_label_loaders[class_label] = DataLoader(
                    FileListDataset(class_records), batch_size=1, shuffle=True,
                    num_workers=0, generator=class_generator)
            print('    [same-label-align] parameter-free cosine alignment; '
                  f'weight={same_label_align_w:g}; '
                  + ', '.join(
                      f'y={label}:{len(loader.dataset)}'
                      for label, loader in external_label_loaders.items()))
        external_task_active = any(
            task_w_of.get(src, 1.0) > 0 for src in external_source_names)
        external_active = external_loader is not None and (
            external_task_active
            or (site_adv_w > 0 and float(dc['lam_max']) != 0.0)
            or (not external_site_only
                and float(dc['dom_weight']) > 0
                and float(dc['lam_max']) != 0.0)
            or same_label_align_w > 0)
        print(f"    [dual-loader] NDPI={len(primary_train)} files, "
              + f"external_pool={len(external_records)} ("
              + ', '.join(f"{src}={sum(str(r.get('source','')) == src for r in external_records)}"
                          for src in external_source_names) + ")"
              + ("; source_sampling=uniform" if external_source_balanced
                 else "; source_sampling=proportional")
              + (f"; exact_external_steps={external_steps_per_epoch}"
                 if external_steps_per_epoch is not None
                 else (f"; exact_source_quotas={external_steps_by_source}"
                       if external_steps_by_source is not None
                       else ("; every_source_each_step=dataset_ratio"
                             if external_ratio_per_step
                             else f"; step_prob={external_step_prob:g}")))
              + "; "
              + f"accum={optim_accum_steps}; external_active={external_active}")
        if (external_steps_per_epoch is not None
                and external_steps_per_epoch > len(primary_train)):
            raise ValueError(
                f'external_steps_per_epoch={external_steps_per_epoch} exceeds '
                f'NDPI steps={len(primary_train)}')
        # Attach multiple external records per NDPI step to meet source quotas.
        # Optimizer updates follow the NDPI loader.
        if external_adain:
            print('    [external-adain] pooled external feature content -> '
                  'independently shuffled NDPI slide statistics; ordinary '
                  'task supervision remains active')
    else:
        train_loader = DataLoader(FileListDataset(all_train),
                                  batch_size=1, shuffle=True, num_workers=0)
        external_loader = None
        external_source_loaders = None
        external_label_loaders = None
        external_active = False

    sequential_loaders = None
    if sequential_external_epochs > 0:
        if dual_loader:
            raise ValueError(
                'sequential source curriculum and dual_loader are mutually exclusive')
        sequential_ndpi = [
            r for r in all_train if str(r.get('source', '')) == 'ndpi']
        sequential_external = [
            r for r in all_train
            if str(r.get('source', '')) in ('svs', 'kfb')]
        if not sequential_ndpi or not sequential_external:
            raise ValueError(
                'sequential source curriculum requires NDPI and SVS/KFB records')
        ext_generator = torch.Generator()
        ext_generator.manual_seed(int(cfg.seed) + 1000 * (fold_idx + 1))
        ndpi_generator = torch.Generator()
        ndpi_generator.manual_seed(int(cfg.seed) + 2000 * (fold_idx + 1))
        sequential_loaders = {
            'external': DataLoader(
                FileListDataset(sequential_external), batch_size=1,
                shuffle=True, num_workers=0, generator=ext_generator),
            'ndpi': DataLoader(
                FileListDataset(sequential_ndpi), batch_size=1,
                shuffle=True, num_workers=0, generator=ndpi_generator),
        }
        print(f"    [sequential] identical objective/optimizer: "
              f"SVS+KFB {sequential_external_epochs} epoch(s) "
              f"({len(sequential_external)} files) → NDPI "
              f"{cfg.num_epochs - sequential_external_epochs} epoch(s) "
              f"({len(sequential_ndpi)} files)")

    if same_label_align_w > 0 and not dual_loader:
        raise ValueError('same_label_align_w requires --dual_loader')
    if external_adain and not dual_loader:
        raise ValueError('external_adain requires --dual_loader')

    external_rng = np.random.RandomState(
        int(cfg.seed) + 3000 * (fold_idx + 1))

    scanner_projection = None
    if scanner_proj_alpha > 0:
        if scanner_proj_path:
            scanner_projection = torch.load(
                str(scanner_proj_path), map_location='cpu')
            required = {'center', 'basis', 'rank', 'singular', 'counts'}
            missing = required.difference(scanner_projection)
            if missing:
                raise ValueError(
                    f'scanner projection artifact missing keys: {sorted(missing)}')
            if scanner_projection['center'].numel() != cfg.input_dim:
                raise ValueError('scanner projection input dimension mismatch')
            print(f"    [scanner-proj] reused fixed artifact: {scanner_proj_path}")
        else:
            scanner_projection = estimate_scanner_subspace(
                all_train, cfg.input_dim)
        os.makedirs(ckpt_dir, exist_ok=True)
        projection_path = os.path.join(
            os.path.dirname(ckpt_dir), f'scanner_projection_fold_{fold_idx}.pt')
        torch.save({**scanner_projection, 'alpha': scanner_proj_alpha,
                    'fold': int(fold_idx)}, projection_path)
        print(f"    [scanner-proj] alpha={scanner_proj_alpha:g} "
              f"rank={scanner_projection['rank']} "
              f"singular={[float(v) for v in scanner_projection['singular']]} "
              f"counts={scanner_projection['counts']} → {projection_path}")

    model = DeepCDA_CLAM_SB(input_dim=cfg.input_dim, size_arg=cfg.size_arg,
                            k_sample=cfg.k_sample, n_domains=len(dom_names),
                            disc_hidden=dc['disc_hidden'],
                            reg_dim=STATE.get('reg_dim', 0),
                            enc_depth=dc.get('enc_depth', 1),
                            legacy_act=dc.get('legacy_act', True),
                            bottleneck=dc.get('bottleneck', 0),
                            n_sites=n_sites,
                            scanner_projection=scanner_projection,
                            scanner_proj_alpha=scanner_proj_alpha).to(device)
    init_ckpt_dir = dc.get('init_ckpt_dir')
    if init_ckpt_dir:
        init_path = os.path.join(str(init_ckpt_dir), f'fold_{fold_idx}.pth')
        if not os.path.isfile(init_path):
            raise FileNotFoundError(f'warm-start checkpoint missing: {init_path}')
        init_state = torch.load(init_path, map_location=device)
        if dc.get('init_allow_missing_heads', False):
            allowed = ('slide_sdisc.', 'patch_sdisc.', 'site_bias')
            current_state = model.state_dict()
            # strict=False still rejects same-name tensors with different
            # shapes (for example a 3-site zero-shot head warming a 4-site
            # few-shot model). Drop only explicitly allowed head mismatches.
            init_state = {
                key: value for key, value in init_state.items()
                if not (key.startswith(allowed)
                        and key in current_state
                        and value.shape != current_state[key].shape)}
            incompatible = model.load_state_dict(init_state, strict=False)
            bad_missing = [key for key in incompatible.missing_keys
                           if not key.startswith(allowed)]
            if bad_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    f'unsafe warm-start mismatch: missing={bad_missing}, '
                    f'unexpected={incompatible.unexpected_keys}')
            print(f"    [warm-start] initialized new heads: "
                  f"{incompatible.missing_keys}")
        else:
            model.load_state_dict(init_state)
        print(f"    [warm-start] loaded {init_path}")
    if dc.get('enc_depth', 1) > 1 or dc.get('bottleneck', 0) > 0:
        _np_tr = sum(q.numel() for q in model.parameters() if q.requires_grad)
        print(f"    [deep] encoder 深度 = {dc.get('enc_depth',1)}  "
              f"瓶颈 = {dc.get('bottleneck',0) or '无'}  "
              f"可训练参数 {_np_tr:,}")
    criterion = nn.BCELoss(reduction='mean')

    # ---- 站点截距的两种「让机制真的启动」的模式 ----
    # 实测问题：默认 lr=1e-4 下 site_bias 只学到 kfb −0.138 / ndpi +0.057 /
    # svs +0.086，而要吸收流行率需要 ±2.94（label_smooth=0.9 → 目标 0.95/0.05）。
    # 单个标量在 39 万参数的特征分支面前跑不过 —— 站点差异早被特征吸收了。
    # 排序对（kfb<ndpi<svs 正是流行率序）但量级差一个数量级，
    # 所以「截距剥离流行率」这个设计意图在默认设置下并没有发生。
    sb_fixed = bool(dc.get('site_bias_fixed', False))
    sb_mult = float(dc.get('site_bias_lr_mult', 1.0) or 1.0)
    if n_sites and (sb_fixed or sb_mult != 1.0):
        import numpy as _np
        cnt = defaultdict(lambda: [0, 0])
        for r in all_train:
            cnt[str(r['source'])][int(r['label'])] += 1
        lo, hi = (1 - cfg.label_smooth) / 2, 1 - (1 - cfg.label_smooth) / 2
        with torch.no_grad():
            for nm, i in [(n, i) for i, n in enumerate(site_names)]:
                neg, pos = cnt[nm]
                pv = pos / max(pos + neg, 1)
                pv = min(max(pv, lo), hi)          # 与 label_smooth 的上下限一致
                model.site_bias[i] = float(_np.log(pv / (1 - pv)))
        if sb_fixed:
            model.site_bias.requires_grad_(False)
        print(f"    [site_bias] {'固定' if sb_fixed else '初始化'}为 logit(站内流行率)："
              + "  ".join(f"{n}={float(model.site_bias[i]):+.3f}"
                          for i, n in enumerate(site_names))
              + (f"   lr×{sb_mult:g}" if not sb_fixed and sb_mult != 1 else ""))

    disc_prefixes = ('slide_disc.', 'patch_disc.', 'slide_reg.', 'patch_reg.',
                     'slide_sdisc.', 'patch_sdisc.')
    disc_named = [(name, value) for name, value in model.named_parameters()
                  if name.startswith(disc_prefixes)]
    if alternating_adv:
        if use_site_bias or sb_mult != 1.0:
            raise ValueError('alternating_adv currently excludes site intercept/bias')
        main_params = [value for name, value in model.named_parameters()
                       if value.requires_grad and not name.startswith(disc_prefixes)]
        disc_params = [value for _, value in disc_named if value.requires_grad]
        if not disc_params:
            raise ValueError('alternating_adv found no discriminator parameters')
        optimizer = optim.AdamW(main_params, lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        disc_optimizer = optim.AdamW(
            disc_params, lr=float(dc.get('disc_lr', 1e-4)),
            weight_decay=cfg.weight_decay)
        print(f"    [alternating] main={sum(p.numel() for p in main_params):,} "
              f"disc={sum(p.numel() for p in disc_params):,} "
              f"main_lr={cfg.lr:g} disc_lr={float(dc.get('disc_lr',1e-4)):g}")
    elif n_sites and not sb_fixed and sb_mult != 1.0:
        # 单独一组：放大学习率、且不加 weight decay（固定效应不该被收缩）
        other = [q for n, q in model.named_parameters() if n != 'site_bias']
        optimizer = optim.AdamW(
            [dict(params=other, lr=cfg.lr, weight_decay=cfg.weight_decay),
             dict(params=[model.site_bias], lr=cfg.lr * sb_mult,
                  weight_decay=0.0)])
    else:
        optimizer = optim.AdamW(
            [q for q in model.parameters() if q.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [],
               'val_pos_prob': [], 'val_neg_prob': [], 'val_epochs': [],
               'dom_loss': [], 'dom_acc_slide': [], 'dom_acc_patch': [],
               'site_acc_slide': [],
               'lam': []}
    if same_label_align_w > 0:
        history['same_label_align_loss'] = []
    ckpt_path = os.path.join(ckpt_dir, f'fold_{fold_idx}.pth')
    best_key = None
    tag = f"Fold {fold_idx}/{cfg.n_folds}"
    print(f"  [{tag}]  train={len(train_records)} files  val={len(val_records)} files"
          f"  λ_max={dc['lam_max']}  dom_weight={dc['dom_weight']}"
          f"  levels={'patch' if dc['use_patch'] else ''}"
          f"{'+slide' if dc['use_slide'] else ''}")

    def evaluate(epoch):
        # Evaluate with dropout disabled and restore model/RNG state afterward.
        was_training = model.training
        cuda_devices = ([torch.cuda.current_device()]
                        if device.type == 'cuda' else [])
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            labels, probs = R.predict(model, val_records, device)
        model.train(was_training)
        vl = bce_loss(labels, probs)
        m = binary_metrics(labels, probs, 0.5)
        pos = [pr for l, pr in zip(labels, probs) if l == 1]
        neg = [pr for l, pr in zip(labels, probs) if l == 0]
        history['val_loss'].append(vl)
        history['val_auc'].append(m['auc'])
        history['val_pos_prob'].append(float(np.mean(pos)) if pos else 0.0)
        history['val_neg_prob'].append(float(np.mean(neg)) if neg else 0.0)
        history['val_epochs'].append(epoch)
        return vl, m['auc']

    model.eval()
    evaluate(0)
    R.save_fold_history(history, os.path.dirname(ckpt_dir), fold_idx)

    def _disc_step(features, label, sid, dom, site_id, has_dom,
                   has_site_adv):
        """One discriminator-only update; GRL=0 blocks encoder gradients."""
        if not alternating_adv or not (has_dom or has_site_adv):
            return 0.0
        disc_optimizer.zero_grad()
        (_, _, _, ds, dp, _, _) = model.forward_dann(
            features, label=label,
            domain=max(dom, 0), lam=0.0, instance_eval=False,
            n_patch=dc['n_patch'],
            use_patch=dc['use_patch'], use_slide=dc['use_slide'],
            reg_target=(torch.from_numpy(STATE['colour_vec'][norm_key(sid)])
                        if has_dom and mode in ('appearregfull',) else None),
            site=site_id, site_adv_w=_site_adv_for(
                label),
            use_site_bias=False,
            conditional_site_adv=conditional_site_adv,
            site_class_ids=site_class_ids,
            site_slide_only=site_slide_only,
            use_domain=has_dom)
        levels = int(dc['use_patch']) + int(dc['use_slide'])
        value = (ds + dp) / max(levels, 1)
        value.backward()
        disc_optimizer.step()
        return float(value.item())

    pretrain_epochs = int(dc.get('disc_pretrain_epochs', 0) or 0)
    if alternating_adv and pretrain_epochs > 0:
        print(f"    [alternating] discriminator pretrain {pretrain_epochs} epoch(s)")
        model.train()
        for pre_ep in range(pretrain_epochs):
            losses = []
            for features, label, slide_id, _ in train_loader:
                features = features.squeeze(0).to(device)
                label = label[0].to(device)
                sid = str(slide_id[0] if isinstance(slide_id, (list, tuple))
                          else slide_id)
                dom = dom_of.get(sid, -1)
                raw_has_dom = dom >= 0
                has_dom = raw_has_dom and not site_only_adv
                site_id = site_of.get(norm_key(sid)) if n_sites else None
                has_site = site_adv_w > 0 and site_id is not None
                value = _disc_step(features, label, sid, dom, site_id,
                                   has_dom, has_site)
                if value:
                    losses.append(value)
            print(f"      pretrain {pre_ep+1}: disc_loss="
                  f"{float(np.mean(losses)) if losses else float('nan'):.4f}")

    for epoch in tqdm(range(cfg.num_epochs), desc=tag, leave=False):
        model.train()
        epoch_train_loader = train_loader
        phase = 'joint'
        if sequential_loaders is not None:
            phase = ('external' if epoch < sequential_external_epochs
                     else 'ndpi')
            epoch_train_loader = sequential_loaders[phase]
        site_acc_start = len(STATE['site_acc'])
        lam = (dc['lam_max'] if dc.get('constant_lam', False)
               else lam_at(epoch / max(cfg.num_epochs - 1, 1), dc['lam_max']))
        ep_loss = ep_dom = ep_same_label = 0.0
        ep_external_seen = 0
        ep_external_seen_by_source = defaultdict(int)
        ep_external_effective_by_source = defaultdict(float)
        ep_train_seen_by_source = defaultdict(int)
        ep_appearance_seen_by_source = defaultdict(int)
        acc_s, acc_p, n_seen = [], [], 0
        optimizer.zero_grad()

        external_iter = iter(external_loader) if external_active else None
        external_source_iters = (
            {source: iter(loader)
             for source, loader in external_source_loaders.items()}
            if external_active and external_source_loaders else None)
        external_label_iters = (
            {label: iter(loader) for label, loader in external_label_loaders.items()}
            if external_active and external_label_loaders else None)
        # Exact-count alternative to Bernoulli attachment.  This permits a
        # controlled 372/50/50 experiment, or full auxiliary-cohort coverage,
        # while retaining the dual-loader optimizer cadence.  When necessary,
        # more than one external record can be attached to an NDPI step.
        exact_external_step_ids = None
        exact_external_sources_by_step = None
        if external_active and external_steps_per_epoch is not None:
            exact_external_step_ids = set(external_rng.choice(
                len(epoch_train_loader), size=external_steps_per_epoch,
                replace=False).tolist())
        elif external_active and external_steps_by_source is not None:
            total_quota = sum(external_steps_by_source.values())
            n_primary_steps = len(epoch_train_loader)
            if total_quota <= n_primary_steps:
                # Preserve the historical schedule exactly for existing
                # experiments whose quota fits within one external per step.
                chosen_steps = sorted(external_rng.choice(
                    n_primary_steps, size=total_quota,
                    replace=False).tolist())
            else:
                # Use shuffled complete passes over the NDPI steps so every
                # step receives floor(Q/N) or ceil(Q/N) external records.
                # Unlike sampling with replacement this avoids accidental
                # clustering while retaining a deterministic random schedule.
                chosen_steps = []
                remaining = total_quota
                while remaining:
                    step_pass = external_rng.permutation(n_primary_steps)
                    take = min(remaining, n_primary_steps)
                    chosen_steps.extend(step_pass[:take].tolist())
                    remaining -= take
                chosen_steps.sort()
            events = []
            for source in sorted(external_steps_by_source):
                quota = external_steps_by_source[source]
                events.extend(((index + 0.5) / quota, source, index)
                              for index in range(quota))
            events.sort()
            exact_external_sources_by_step = defaultdict(list)
            for step, event in zip(chosen_steps, events):
                exact_external_sources_by_step[step].append(event[1])

        for i, (features, label, slide_id, _) in enumerate(epoch_train_loader):
            features = features.squeeze(0).to(device)
            label = label[0].to(device)
            sid = str(slide_id[0] if isinstance(slide_id, (list, tuple))
                      else slide_id)
            src = str(_src_of.get(sid, ''))
            ep_train_seen_by_source[src] += 1
            dom = dom_of.get(sid, -1)
            raw_has_dom = dom >= 0
            # ``external_site_only`` originally guarded the dual-loader side
            # branch.  Natural mixed-loader training sends every source through
            # this main branch, so apply the same contract here as well: NDPI
            # retains appearance regression while SVS/KFB contribute task and
            # conditional-source losses only.
            has_dom = (raw_has_dom and not site_only_adv
                       and not (external_site_only and src != 'ndpi'))
            if has_dom:
                ep_appearance_seen_by_source[src] += 1
            site_id = site_of.get(norm_key(sid)) if n_sites else None
            # few-shot TCGA train 可能没有内部 60 维外观缓存，但仍有明确的
            # source/site 标签。站点对抗必须与外观回归解耦，否则这些样本会
            # 悄悄跳过 site adversary。
            sample_site_adv_w = _site_adv_for(label)
            has_site_adv = sample_site_adv_w > 0 and site_id is not None
            has_any_adv = has_dom or has_site_adv

            if alternating_adv:
                _disc_step(features, label, sid, dom, site_id,
                           has_dom, has_site_adv)
                for _, value in disc_named:
                    value.requires_grad_(False)

            main_result = model.forward_dann(
                features, label=label,
                domain=max(dom, 0), lam=lam * grl_w_of.get(src, 1.0),
                instance_eval=True, n_patch=dc['n_patch'],
                use_patch=dc['use_patch'] and has_any_adv,
                use_slide=dc['use_slide'] and has_any_adv,
                reg_target=(torch.from_numpy(
                    STATE['colour_vec'][norm_key(sid)])
                    if mode in ('appearregfull',) and has_dom
                    else None),
                site=site_id,
                site_adv_w=sample_site_adv_w, use_site_bias=use_site_bias,
                conditional_site_adv=conditional_site_adv,
                site_class_ids=site_class_ids,
                site_slide_only=site_slide_only,
                use_domain=has_dom,
                return_repr=same_label_align_w > 0,
                domain_lam=(lam * grl_w_of.get(src, 1.0)
                            * appearance_grl_w_of.get(src, 1.0)),
                site_lam=lam * grl_w_of.get(src, 1.0))
            (wsi_prob, _, inst_loss, dom_s, dom_p,
             a_s, a_p) = main_result[:7]
            main_repr = main_result[7] if same_label_align_w > 0 else None

            label_s = label * cfg.label_smooth + (1 - cfg.label_smooth) / 2
            bag_loss = criterion(torch.clamp(wsi_prob, 1e-3, 1 - 1e-3),
                                 label_s)
            inst_scale = inst_w_of.get(src, 1.0)
            task_loss = (cfg.bag_weight * bag_loss
                         + cfg.inst_weight * inst_loss * inst_scale)
            if label.item() == 1.0:
                task_loss = task_loss * _pos_weight_for(src)
            if task_w_of:
                task_loss = task_loss * task_w_of.get(
                    str(_src_of.get(sid, '')), 1.0)

            n_lv = int(dc['use_patch']) + int(dc['use_slide'])
            dom_loss = ((dom_s + dom_p) / max(n_lv, 1) if has_any_adv
                        else torch.tensor(0.0, device=device))
            if adv_w_of:
                dom_loss = dom_loss * adv_w_of.get(
                    str(_src_of.get(sid, '')), 1.0)
            # 无外观目标的 TCGA 记录仍贡献 site loss；但其单类 domain-head
            # "acc=1" 没有意义，不能混入外观回归诊断。
            if not has_dom:
                a_s = a_p = float('nan')
            src = str(_src_of.get(sid, ''))
            loss = (optim_grad_w[src]
                    * (task_loss + dc['dom_weight'] * dom_loss)
                    / optim_accum_steps)
            loss.backward(retain_graph=same_label_align_w > 0)
            if alternating_adv:
                for _, value in disc_named:
                    value.requires_grad_(True)

            # Attach external slides to NDPI steps and accumulate gradients
            # over cfg.accum_steps NDPI slides per optimizer update.
            use_external_this_step = (
                external_active and (
                    True if external_ratio_per_step else (
                        i in exact_external_sources_by_step
                        if exact_external_sources_by_step is not None else (
                            i in exact_external_step_ids
                            if exact_external_step_ids is not None
                            else external_rng.random_sample()
                            < external_step_prob))))
            if use_external_this_step:
                ext_batches = []
                if external_ratio_per_step:
                    for selected_source in sorted(external_source_iters):
                        try:
                            source_batch = next(
                                external_source_iters[selected_source])
                        except StopIteration:
                            external_source_iters[selected_source] = iter(
                                external_source_loaders[selected_source])
                            source_batch = next(
                                external_source_iters[selected_source])
                        source_mass = (
                            len(external_source_loaders[selected_source].dataset)
                            / len(epoch_train_loader))
                        ext_batches.append((source_batch, source_mass))
                elif external_source_iters is not None:
                    source_names = sorted(external_source_iters)
                    selected_sources = (
                        exact_external_sources_by_step[i]
                        if exact_external_sources_by_step is not None
                        else [source_names[
                            ep_external_seen % len(source_names)]])
                    for selected_source in selected_sources:
                        try:
                            ext_batch = next(
                                external_source_iters[selected_source])
                        except StopIteration:
                            external_source_iters[selected_source] = iter(
                                external_source_loaders[selected_source])
                            ext_batch = next(
                                external_source_iters[selected_source])
                        ext_batches.append((ext_batch, 1.0))
                elif external_label_iters is not None:
                    class_label = int(label.item())
                    try:
                        ext_batch = next(external_label_iters[class_label])
                    except StopIteration:
                        external_label_iters[class_label] = iter(
                            external_label_loaders[class_label])
                        ext_batch = next(external_label_iters[class_label])
                    ext_batches.append((ext_batch, 1.0))
                else:
                    try:
                        ext_batch = next(external_iter)
                    except StopIteration:
                        external_iter = iter(external_loader)
                        ext_batch = next(external_iter)
                    ext_batches.append((ext_batch, 1.0))
                for ext_batch, ext_mass_scale in ext_batches:
                    ext_features, ext_label, ext_slide_id, _ = ext_batch
                    ep_external_seen += 1
                    ext_features = ext_features.squeeze(0).to(device)
                    ext_label = ext_label[0].to(device)
                    ext_sid = str(ext_slide_id[0] if isinstance(
                        ext_slide_id, (list, tuple)) else ext_slide_id)
                    ext_src = str(_src_of.get(ext_sid, ''))
                    ep_external_seen_by_source[ext_src] += 1
                    ep_external_effective_by_source[ext_src] += ext_mass_scale
                    if external_adain:
                        ext_features = external_adain_to_ndpi(
                            ext_features, features)
                    ext_task_scale = task_w_of.get(ext_src, 1.0)
                    ext_inst_scale = inst_w_of.get(ext_src, 1.0)
                    ext_dom = dom_of.get(ext_sid, -1)
                    ext_adv_scale = adv_w_of.get(ext_src, 1.0)
                    ext_has_dom = (ext_adv_scale > 0 and ext_dom >= 0
                                   and not site_only_adv
                                   and not external_site_only)
                    ext_site_id = (site_of.get(norm_key(ext_sid))
                                   if n_sites else None)
                    ext_site_adv_w = (_site_adv_for(ext_label)
                                      if ext_adv_scale > 0 else 0.0)
                    ext_has_site = ext_site_adv_w > 0 and ext_site_id is not None
                    ext_has_any_adv = ext_has_dom or ext_has_site

                    ext_result = model.forward_dann(
                        ext_features, label=ext_label,
                        domain=max(ext_dom, 0),
                        lam=lam * grl_w_of.get(ext_src, 1.0),
                        instance_eval=(ext_task_scale > 0
                                       and ext_inst_scale > 0),
                        n_patch=dc['n_patch'],
                        use_patch=dc['use_patch'] and ext_has_any_adv,
                        use_slide=dc['use_slide'] and ext_has_any_adv,
                        reg_target=(torch.from_numpy(
                            STATE['colour_vec'][norm_key(ext_sid)])
                            if mode in ('appearregfull',) and ext_has_dom else None),
                        site=ext_site_id, site_adv_w=ext_site_adv_w,
                        use_site_bias=use_site_bias,
                        conditional_site_adv=conditional_site_adv,
                        site_class_ids=site_class_ids,
                        site_slide_only=site_slide_only,
                        use_domain=ext_has_dom,
                        return_repr=same_label_align_w > 0,
                        domain_lam=(lam * grl_w_of.get(ext_src, 1.0)
                                    * appearance_grl_w_of.get(ext_src, 1.0)),
                        site_lam=lam * grl_w_of.get(ext_src, 1.0))
                    (ext_prob, _, ext_inst, ext_ds, ext_dp,
                     ext_as, ext_ap) = ext_result[:7]
                    ext_repr = (ext_result[7]
                                if same_label_align_w > 0 else None)

                    ext_label_s = (ext_label * cfg.label_smooth
                                   + (1 - cfg.label_smooth) / 2)
                    ext_bag = criterion(
                        torch.clamp(ext_prob, 1e-3, 1 - 1e-3), ext_label_s)
                    ext_task = (cfg.bag_weight * ext_bag
                                + cfg.inst_weight * ext_inst
                                * ext_inst_scale)
                    if ext_label.item() == 1.0:
                        ext_task = ext_task * _pos_weight_for(ext_src)
                    ext_task = ext_task * ext_task_scale
                    ext_n_lv = int(dc['use_patch']) + int(dc['use_slide'])
                    ext_dom_loss = (
                        (ext_ds + ext_dp) / max(ext_n_lv, 1)
                        if ext_has_any_adv else torch.tensor(0.0, device=device))
                    ext_dom_loss = ext_dom_loss * ext_adv_scale
                    ext_loss = ext_task + dc['dom_weight'] * ext_dom_loss
                    if same_label_align_w > 0:
                        same_label_loss = same_label_align_w * (
                            1.0 - F.cosine_similarity(
                                main_repr, ext_repr, dim=1).mean())
                        ext_loss = ext_loss + same_label_loss
                        ep_same_label += (
                            float(same_label_loss.item()) * ext_mass_scale)
                    (ext_loss * ext_mass_scale
                     / optim_accum_steps).backward()
                    ep_loss += float(ext_task.item()) * ext_mass_scale
                    ep_dom += float(ext_dom_loss.item()) * ext_mass_scale
                    if ext_has_dom and not math.isnan(ext_as):
                        acc_s.append(ext_as)
                    if ext_has_dom and not math.isnan(ext_ap):
                        acc_p.append(ext_ap)

            if ((i + 1) % optim_accum_steps == 0
                    or (i + 1) == len(epoch_train_loader)):
                optimizer.step()
                optimizer.zero_grad()

            ep_loss += task_loss.item()
            ep_dom += float(dom_loss.item())
            if not math.isnan(a_s):
                acc_s.append(a_s)
            if not math.isnan(a_p):
                acc_p.append(a_p)
            n_seen += 1

        history['train_loss'].append(ep_loss / max(n_seen, 1))
        history['dom_loss'].append(ep_dom / max(n_seen, 1))
        history['dom_acc_slide'].append(float(np.mean(acc_s)) if acc_s else float('nan'))
        history['dom_acc_patch'].append(float(np.mean(acc_p)) if acc_p else float('nan'))
        epoch_site_acc = STATE['site_acc'][site_acc_start:]
        history['site_acc_slide'].append(
            float(np.mean(epoch_site_acc)) if epoch_site_acc else float('nan'))
        if same_label_align_w > 0:
            history['same_label_align_loss'].append(
                ep_same_label / max(n_seen, 1))
        history['lam'].append(lam)
        history.setdefault('external_seen', []).append(ep_external_seen)
        for source_name in ('ndpi', 'svs', 'kfb'):
            history.setdefault(
                f'train_seen_{source_name}', []).append(
                    int(ep_train_seen_by_source.get(source_name, 0)))
            history.setdefault(
                f'appearance_seen_{source_name}', []).append(
                    int(ep_appearance_seen_by_source.get(source_name, 0)))
        for source_name in (sorted(external_source_names) if dual_loader else []):
            history.setdefault(
                f'external_seen_{source_name}', []).append(
                    int(ep_external_seen_by_source.get(source_name, 0)))
            history.setdefault(
                f'external_effective_seen_{source_name}', []).append(
                    float(ep_external_effective_by_source.get(source_name, 0.0)))
        history.setdefault('phase', []).append(phase)

        if (sequential_loaders is not None
                and epoch + 1 == sequential_external_epochs):
            stage1_dir = os.path.join(
                os.path.dirname(ckpt_dir), 'ckpts_stage1_external')
            os.makedirs(stage1_dir, exist_ok=True)
            stage1_path = os.path.join(
                stage1_dir, f'fold_{fold_idx}.pth')
            torch.save(model.state_dict(), stage1_path)
            tqdm.write(f"    [{tag}] external pretrain checkpoint saved → "
                       f"{stage1_path}")

        if (epoch + 1) % 4 == 0:
            vl, va = evaluate(epoch + 1)
            model.train()
            tqdm.write(f"    [{tag}] ep{epoch+1:>3d}  λ={lam:.3f}  "
                       f"task={history['train_loss'][-1]:.4f}  "
                       f"dom={history['dom_loss'][-1]:.4f}  "
                       f"域acc(slide/patch)="
                       f"{history['dom_acc_slide'][-1]:.3f}/"
                       f"{history['dom_acc_patch'][-1]:.3f}  "
                       f"val_auc={va:.4f}  val_loss={vl:.4f}"
                       + (f"  站点acc={history['site_acc_slide'][-1]:.3f}"
                          f"(chance={0.5 if conditional_site_adv else 1.0/max(n_sites,1):.3f})"
                          if n_sites and epoch_site_acc else ""))
            if ckpt_select == 'last':
                torch.save(model.state_dict(), ckpt_path)
            elif ckpt_select == 'best_auc':
                if best_key is None or va >= best_key:
                    best_key = va
                    torch.save(model.state_dict(), ckpt_path)
            elif ckpt_select == 'best_loss':
                if best_key is None or vl <= best_key:
                    best_key = vl
                    torch.save(model.state_dict(), ckpt_path)
            snapshot_epochs = {int(value) for value in
                               dc.get('snapshot_epochs', [])}
            if (epoch + 1) in snapshot_epochs:
                snapshot_dir = os.path.join(os.path.dirname(ckpt_dir),
                                            f'ckpts_ep{epoch + 1}')
                os.makedirs(snapshot_dir, exist_ok=True)
                snapshot_path = os.path.join(
                    snapshot_dir, f'fold_{fold_idx}.pth')
                torch.save(model.state_dict(), snapshot_path)
                tqdm.write(f"    [{tag}] snapshot saved → {snapshot_path}")
        R.save_fold_history(history, os.path.dirname(ckpt_dir), fold_idx)

    # `last` 的语义必须是“最终 epoch”，不能依赖每 4 epoch 一次的验证节奏。
    # 旧实现把保存放在验证分支里，导致 1–3 epoch 的短程适配日志声称保留
    # checkpoint，实际目录却为空。这里无条件落盘最终状态；best 模式若总
    # epoch 少于首个验证点，也至少用最终状态兜底，避免交付缺失。
    if ckpt_select == 'last':
        torch.save(model.state_dict(), ckpt_path)
    elif not os.path.exists(ckpt_path):
        evaluate(cfg.num_epochs)
        torch.save(model.state_dict(), ckpt_path)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    val_labels, val_probs = R.predict(model, val_records, device)

    # 记录该折的对抗诊断（域 acc 越接近 1/K 说明对齐越成功）
    # Appearance regression is summarized by loss and relative residuals.
    is_reg = STATE['domain_mode'] in ('appearregfull',)
    chance = float('nan') if is_reg else 1.0 / max(len(dom_names), 1)
    STATE['log'].append(dict(
        fold=fold_idx, n_domains=len(dom_names), domains=dom_names,
        chance=None if is_reg else chance, lam_final=history['lam'][-1] if history['lam'] else 0.0,
        dom_acc_slide_final=history['dom_acc_slide'][-1] if history['dom_acc_slide'] else None,
        dom_acc_patch_final=history['dom_acc_patch'][-1] if history['dom_acc_patch'] else None,
        dom_loss_final=history['dom_loss'][-1] if history['dom_loss'] else None,
        site_acc_slide_final=(history['site_acc_slide'][-1]
                              if history['site_acc_slide'] else None),
        scanner_proj_alpha=scanner_proj_alpha,
        scanner_proj_rank=(scanner_projection['rank']
                           if scanner_projection is not None else 0),
    ))
    if site_only_adv:
        site_chance = 0.5 if conditional_site_adv else 1.0 / max(n_sites, 1)
        print(f"    [{tag}] 条件站点对抗诊断：site acc(slide)="
              f"{history['site_acc_slide'][-1]:.3f}  随机水平={site_chance:.3f}"
              f"   （越接近随机 = 来源信息越少）")
    elif is_reg:
        print(f"    [{tag}] 对抗诊断（回归模式）：域acc(slide)="
              f"{history['dom_acc_slide'][-1]:.3f}  域acc(patch)="
              f"{history['dom_acc_patch'][-1]:.3f}  dom损失="
              f"{history['dom_loss'][-1]:.4f}")
        print(f"           域acc = 1 − 相对残差；判别器只会预测均值时 ≈ 0，"
              f"完美回归出颜色时 ≈ 1。")
        print(f"           ⚠️ 低域acc 必须对照 λ=0 才能解读："
              f"λ=0 也低 → 判别器本身无能，实验读不出结论。")
    else:
        print(f"    [{tag}] 对抗诊断：域 acc(slide)="
              f"{history['dom_acc_slide'][-1]:.3f}  域 acc(patch)="
              f"{history['dom_acc_patch'][-1]:.3f}  随机水平={chance:.3f}"
              f"   （越接近随机 = 对齐越成功）")

    return model, ckpt_path, history, val_labels, val_probs


def install(only_folds=0):
    """只替换 run_ndpi_cv 模块对象上的名字，磁盘文件分毫未动。"""
    R.train_one_fold = train_one_fold_dann
    print("  [inject] train_one_fold → train_one_fold_dann (CDA-CLAM)")
    if only_folds and only_folds > 0:
        base = R.make_source_stratified_folds

        def truncated(records, n_folds, seed):
            folds = base(records, n_folds, seed)
            print(f"  [inject] 只跑前 {only_folds} 折（共 {len(folds)} 折）")
            return folds[:only_folds]
        R.make_source_stratified_folds = truncated


# ==============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description='CDA-CLAM: 跨域对齐 + 对抗训练的 TCGA zero-shot')
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--out_dir', default='results_zeroshot_dann')
    ap.add_argument('--name', default=None)
    ap.add_argument('--root', default='filt', choices=['filt', 'orig'])
    ap.add_argument('--root_dir', default=None)
    ap.add_argument('--extra_tcga_dir', action='append', default=[],
                    help='Additional TCGA feature directories for evaluation. May be supplied repeatedly.')
    ap.add_argument('--no_aug', action='store_true')
    ap.add_argument('--input_dim', type=int, default=None)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=1e-4,
                    help='主模型 AdamW 学习率；显式写入实验配置，便于复现短程 warm adaptation')
    ap.add_argument('--ckpt_select', default='last',
                    choices=['last', 'best_auc', 'best_loss'])
    ap.add_argument('--keep_ckpt', action='store_true')
    # 对抗相关
    ap.add_argument('--lam_max', type=float, default=1.0,
                    help='GRL 强度上限；0 = 关闭对抗（应复现基线）')
    ap.add_argument('--dom_weight', type=float, default=1.0,
                    help='域损失在总损失里的权重')
    ap.add_argument('--n_patch', type=int, default=256,
                    help='patch 级判别器每张片采样多少 patch')
    ap.add_argument('--no_patch_level', action='store_true',
                    help='关闭 local-level patch-wise 对齐')
    ap.add_argument('--no_slide_level', action='store_true',
                    help='关闭 global-level slide-wise 对齐')
    ap.add_argument('--disc_hidden', type=int, default=128)
    ap.add_argument('--enc_depth', type=int, default=1,
                    help='Encoder depth. Values above 1 add residual MLP blocks.')
    ap.add_argument('--gelu', action='store_true',
                    help='Use GELU activation. The default is ReLU.')
    ap.add_argument('--bottleneck', type=int, default=0,
                    help='Bottleneck dimension. Zero disables the bottleneck.')
    ap.add_argument('--domain_mode', default='appearregfull',
                    choices=['source', 'appearregfull'],
                    help='Source classification or appearance regression.')
    ap.add_argument('--site_adv_w', type=float, default=0.0,
                    help='Weight of the source-classification adversary.')
    ap.add_argument('--conditional_site_adv', action='store_true',
                    help='Align sources separately within each class.')
    ap.add_argument('--site_adv_pos_mult', type=float, default=1.0,
                    help='条件站点对抗正类分支（NDPI+ vs SVS）的权重倍数')
    ap.add_argument('--site_adv_neg_mult', type=float, default=1.0,
                    help='条件站点对抗负类分支（NDPI- vs KFB）的权重倍数')
    ap.add_argument('--site_only_adv', action='store_true',
                    help='只使用条件站点判别头，不叠加 domain/外观头')
    ap.add_argument('--external_site_only', action='store_true',
                    help='Apply appearance regression to NDPI samples and site adversaries to external samples.')
    ap.add_argument('--site_slide_only', action='store_true',
                    help='Apply the site adversary to slide representations.')
    ap.add_argument('--same_label_align_w', type=float, default=0.0,
                    help='Weight of cosine alignment between attention-pooled representations with the same label.')
    ap.add_argument('--external_adain', action='store_true',
                    help='Match external patch-feature statistics to a shuffled NDPI reference slide.')
    ap.add_argument('--scanner_proj_alpha', type=float, default=0.0,
                    help='训练折内估计同标签 SVS/KFB scanner 子空间，并在输入'
                         'GigaPath 特征上部分投影；0=恒等，1=完全去除')
    ap.add_argument('--scanner_proj_path', default=None,
                    help='复用已由同一训练折估计的固定投影 artifact，避免重复'
                         '扫描特征；只改变 alpha，不重新估计方向')
    ap.add_argument('--sequential_external_epochs', type=int, default=0,
                    help='>0 时在同一模型、optimizer 和 loss 下先用 SVS+KFB '
                         '训练指定 epoch，再只用 NDPI 完成剩余 epoch；保存阶段一'
                         '和最终 checkpoint')
    ap.add_argument('--alternating_adv', action='store_true',
                    help='判别器和主模型用独立 optimizer 交替更新')
    ap.add_argument('--disc_lr', type=float, default=1e-4)
    ap.add_argument('--disc_pretrain_epochs', type=int, default=0)
    ap.add_argument('--init_allow_missing_heads', action='store_true')
    ap.add_argument('--degen_site_w', type=float, default=1.0,
                    help='Task-loss weight for sources containing a single class. Default: 1.')
    ap.add_argument('--degen_instance_w', type=float, default=1.0,
                    help='Instance-loss weight for sources containing a single class. Default: 1.')
    ap.add_argument('--degen_adv_w', type=float, default=1.0,
                    help='Domain-loss weight for sources containing a single class. Default: 1.')
    ap.add_argument('--degen_grl_w', type=float, default=1.0,
                    help='Scale GRL gradients to the encoder for single-class sources. Discriminator gradients retain their weight.')
    ap.add_argument('--svs_task_w', type=float, default=None,
                    help='SVS 分类任务损失独立权重；默认沿用 degen_site_w')
    ap.add_argument('--kfb_task_w', type=float, default=None,
                    help='KFB 分类任务损失独立权重；默认沿用 degen_site_w')
    ap.add_argument('--svs_adv_w', type=float, default=None,
                    help='SVS 对抗损失独立权重；默认沿用 degen_adv_w')
    ap.add_argument('--kfb_adv_w', type=float, default=None,
                    help='KFB 对抗损失独立权重；默认沿用 degen_adv_w')
    ap.add_argument('--svs_pos_weight', type=float, default=None,
                    help='SVS 全阳性来源的独立 positive weight；默认沿用全局值')
    ap.add_argument('--source_balanced_optim', action='store_true',
                    help='混合池中按 ndpi 占比自动增大 gradient accumulation，'
                         '并重标定 ndpi 梯度，使每 epoch 的 Adam step 数和主来源'
                         '梯度量与 ndpi-only 基线近似一致；默认关闭以复现旧实验')
    ap.add_argument('--dual_loader', action='store_true',
                    help='单阶段双 loader：NDPI 保持 baseline 的 shuffle/accum/Adam '
                         '节奏，每个 NDPI step 从合并的 SVS/KFB 外部池按样本比例采样 '
                         '样本。避免把增加数据造成的优化轨迹变化误认为域对齐效果')
    ap.add_argument('--external_sources', nargs='+', choices=['svs', 'kfb'],
                    default=None,
                    help='双 loader 实际启用的 PUMC 外部来源；默认两者都用')
    ap.add_argument('--external_source_balanced', action='store_true',
                    help='双 loader 在启用的外部来源间等频轮转；所有来源仍使用'
                         '相同 HER2 task loss，避免大来源仅凭文件数主导优化')
    ap.add_argument('--external_ratio_per_step', action='store_true',
                    help='每个NDPI step从每个外部来源各取1条，但将该来源整项loss '
                         '乘以来源文件数/NDPI文件数；稳定配对且保持数据集比例')
    ap.add_argument('--external_step_prob', type=float, default=1.0,
                    help='每个 NDPI step 附加外部样本的概率；source-only 消融'
                         '用它保持该来源在完整池中的原始曝光频率')
    ap.add_argument('--external_steps_per_epoch', type=int, default=None,
                    help='双loader每轮精确附加的外部样本数；指定后替代Bernoulli '
                         '--external_step_prob，用于严格控制来源暴露计数')
    ap.add_argument('--external_steps_by_source', type=json.loads, default=None,
                    help='JSON来源精确配额，例如 '
                         '{"kfb":50,"svs":50,"tcga":58}')
    ap.add_argument('--init_ckpt_dir', default=None,
                    help='可选 warm-start checkpoint 目录；按 fold_N.pth 加载，'
                         '随后继续当前训练协议并写出新的 checkpoint/history')
    ap.add_argument('--constant_lam', action='store_true',
                    help='每个 epoch 都使用 lam_max；用于短程 warm-start 适配，'
                         '避免单 epoch 时默认 ramp 从 λ=0 开始而实际不更新 encoder')
    ap.add_argument('--site_bias_fixed', action='store_true',
                    help='站点截距不训练，直接固定为 logit(站内流行率)（按 '
                         'label_smooth 截断到 [0.05,0.95]，即 ±2.944）。'
                         '实测默认设置下可学截距只到 ±0.1，设计的机制没启动')
    ap.add_argument('--site_bias_lr_mult', type=float, default=1.0,
                    help='站点截距单独一组参数、学习率乘这个倍数且不加 '
                         'weight decay。默认 1.0')
    ap.add_argument('--site_intercept', action='store_true',
                    help='训练时给 logit 加一个每站点可学标量（固定效应），'
                         '推理时不加。让站点流行率由截距解释，特征分支只需'
                         '解释站点内标签变异，从而不与站点判别器拔河')
    ap.add_argument('--only_folds', type=int, default=0,
                    help='>0 时只跑前 N 折（快速看方向用）')
    return ap.parse_args()




def main():
    args = parse_args()
    root = args.root_dir or (R.ROOT_FILT if args.root == 'filt' else R.ROOT_ORIG)
    use_aug = not args.no_aug

    _TOKENS = ['resnet50', 'macenko']
    root_tag = next((t for t in _TOKENS if t in root),
                    args.root)
    lv = ('patch' if not args.no_patch_level else '') + \
         ('+slide' if not args.no_slide_level else '')
    name = args.name or (f"dann_tcga_{root_tag}_{'aug' if use_aug else 'noaug'}"
                         f"_{args.domain_mode}"
                         + f"_lam{args.lam_max:g}_dw{args.dom_weight:g}"
                         f"_{lv.strip('+') or 'none'}"
                         + (f"_sadv{args.site_adv_w:g}"
                            if args.site_adv_w > 0 else "")
                         + ("_sb" if args.site_intercept else "")
                         + (f"_s{args.seed}" if args.seed != 42 else "")
                         + ("_sbfix" if args.site_bias_fixed else "")
                         + (f"_sblr{args.site_bias_lr_mult:g}"
                            if args.site_bias_lr_mult != 1.0 else "")
                         + (f"_dgw{args.degen_site_w:g}"
                            if args.degen_site_w != 1.0 else "")
                         + (f"_diw{args.degen_instance_w:g}"
                            if args.degen_instance_w != 1.0 else "")
                         + (f"_d{args.enc_depth}" if args.enc_depth > 1 else "")
                         + ("_gelu" if args.gelu else "")
                         + (f"_b{args.bottleneck}" if args.bottleneck else "")
                         + (f"_ep{args.epochs}" if args.epochs != 16 else "")
                         + (f"_f{args.only_folds}" if args.only_folds else "")
                         + f"_{args.ckpt_select}")

    cv_dirs = Z.internal_train_dirs(root, use_aug)
    test_dirs = Z.tcga_test_dirs(root)
    test_dirs += R.collect_dirs(*args.extra_tcga_dir)
    leaked = [d for d in cv_dirs if 'tcga' in os.path.normpath(d).lower()]
    if leaked:
        raise RuntimeError(f"训练池里出现 TCGA 目录，不是 zero-shot：{leaked}")

    input_dim = args.input_dim or Z.detect_input_dim(cv_dirs)

    STATE['domain_mode'] = args.domain_mode
    STATE['colour_vec'], STATE['reg_dim'] = None, 0
    if args.domain_mode == 'appearregfull':
        vec, dim = build_appearance_vectors_full()
        STATE['colour_vec'], STATE['reg_dim'] = vec, dim
        print(f"  Appearance targets: {dim} dimensions, {len(vec)} slides")

    STATE['cfg_dann'] = dict(
        lam_max=args.lam_max, dom_weight=args.dom_weight, main_lr=args.lr,
        enc_depth=args.enc_depth, legacy_act=not args.gelu,
        bottleneck=args.bottleneck,
        site_adv_w=args.site_adv_w, site_intercept=args.site_intercept,
        conditional_site_adv=args.conditional_site_adv,
        site_adv_pos_mult=args.site_adv_pos_mult,
        site_adv_neg_mult=args.site_adv_neg_mult,
        site_only_adv=args.site_only_adv,
        external_site_only=args.external_site_only,
        site_slide_only=args.site_slide_only,
        same_label_align_w=args.same_label_align_w,
        scanner_proj_alpha=args.scanner_proj_alpha,
        scanner_proj_path=args.scanner_proj_path,
        sequential_external_epochs=args.sequential_external_epochs,
        alternating_adv=args.alternating_adv,
        disc_lr=args.disc_lr,
        disc_pretrain_epochs=args.disc_pretrain_epochs,
        init_allow_missing_heads=args.init_allow_missing_heads,
        site_bias_fixed=args.site_bias_fixed,
        site_bias_lr_mult=args.site_bias_lr_mult,
        degen_site_w=args.degen_site_w,
        degen_instance_w=args.degen_instance_w,
        degen_adv_w=args.degen_adv_w,
        degen_grl_w=args.degen_grl_w,
        svs_task_w=args.svs_task_w, kfb_task_w=args.kfb_task_w,
        svs_adv_w=args.svs_adv_w, kfb_adv_w=args.kfb_adv_w,
        svs_pos_weight=args.svs_pos_weight,
        source_balanced_optim=args.source_balanced_optim,
        dual_loader=args.dual_loader,
        external_sources=args.external_sources,
        external_source_balanced=args.external_source_balanced,
        external_ratio_per_step=args.external_ratio_per_step,
        external_step_prob=args.external_step_prob,
        external_steps_per_epoch=args.external_steps_per_epoch,
        external_steps_by_source=args.external_steps_by_source,
        init_ckpt_dir=args.init_ckpt_dir,
        constant_lam=args.constant_lam,
        n_patch=args.n_patch, disc_hidden=args.disc_hidden,
        use_patch=not args.no_patch_level, use_slide=not args.no_slide_level)
    if args.external_adain:
        STATE['cfg_dann']['external_adain'] = True

    print(f"\n{'='*100}")
    print(f"  CDA-CLAM  —  跨域对齐 + 对抗训练（TCGA zero-shot）")
    print(f"{'='*100}")
    print(f"  root        : {root}   input_dim={input_dim}")
    print(f"  Domain mode : {args.domain_mode}")
    print(f"  λ_max       : {args.lam_max}   dom_weight={args.dom_weight}")
    print(f"  对齐层级     : patch={not args.no_patch_level} "
          f"(n_patch={args.n_patch})   slide={not args.no_slide_level}")
    print(f"  epochs      : {args.epochs}  folds={args.folds}  seed={args.seed}")
    print(f"  exp name    : {name}")
    print(f"  [check] 训练池不含 TCGA ✓")
    if args.lam_max == 0:
        print(f"  [note] λ_max=0 → 对抗关闭，本跑用于自检是否复现基线")

    cfg = ExperimentConfig(
        name=name, cv_pool_dirs=cv_dirs, test_dirs=test_dirs,
        out_dir=args.out_dir, input_dim=input_dim, n_folds=args.folds,
        num_epochs=args.epochs, seed=args.seed, lr=args.lr,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    install(args.only_folds)
    print(f"  Start       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary = R.run(cfg, name, args.ckpt_select, args.keep_ckpt,
                    train_desc=f'ndpi + svs + kfb ({root_tag}) + 对抗域对齐',
                    test_desc='TCGA (zero-shot)')
    R.print_final_table(summary)

    fp = os.path.join(args.out_dir, name, 'dann_info.json')
    with open(fp, 'w') as f:
        json.dump(dict(cfg_dann=STATE['cfg_dann'], root=root,
                       domain_mode=args.domain_mode,
                       fold_log=STATE['log']), f, indent=2, ensure_ascii=False)
    print(f"\n  → {fp}")
    print(f"  End         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
