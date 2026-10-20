"""Binary classification metrics and validation-based threshold selection.

Reports ROC-AUC with DeLong confidence intervals, trapezoidal PR-AUC, BCE,
and threshold-based metrics including sensitivity, specificity, and F1.
"""

import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve, roc_auc_score

from final_train import delong_auc_ci


# ==============================================================
# 阈值
# ==============================================================
def youden_threshold(labels, probs) -> float:
    """取 Youden J = TPR - FPR 最大处的阈值。"""
    if len(set(labels)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(labels, probs)
    return float(thr[int(np.argmax(tpr - fpr))])


# ==============================================================
# 指标
# ==============================================================
def bce_loss(labels, probs, eps: float = 1e-3) -> float:
    """
    纯 BCE 均值（不加 label smoothing、不加 pos_weight）。

    训练时用的 loss 带 label_smooth / pos_weight / bag+inst 加权，
    数值不适合当作泛化指标；这里用标准 BCE 以便跨折、跨实验可比。
    probs 会被 clamp 到 [eps, 1-eps]，与训练里的 clamp 保持一致。
    """
    labels = np.asarray(labels, dtype=float)
    probs  = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    if len(labels) == 0:
        return float('nan')
    return float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """
    给定阈值下的全套二分类指标。

    阈值无关：auc / auc_ci / pr_auc（PR 曲线梯形积分）
    阈值相关：sensitivity(recall/TPR) / specificity(TNR) / ppv(precision)
             / npv / f1 / acc / 混淆矩阵

    注意 PPV、NPV 依赖正类 prevalence，故一并返回 n_pos / n_neg。
    """
    labels = np.asarray(labels, dtype=int)
    probs  = np.asarray(probs,  dtype=float)
    preds  = (probs > threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    def _safe(num, den):
        return float(num) / den if den > 0 else float('nan')

    if len(set(labels.tolist())) > 1:
        auc_val, ci_lo, ci_hi = delong_auc_ci(labels, probs)
        precision, recall, _ = precision_recall_curve(labels, probs)
        pr_auc = float(auc(recall, precision))
        baseline = float((labels == 1).mean())   # 随机分类器的 PR-AUC
    else:
        auc_val = ci_lo = ci_hi = pr_auc = baseline = float('nan')

    return {
        'threshold'    : float(threshold),
        'auc'          : auc_val,
        'auc_ci_lo'    : ci_lo,
        'auc_ci_hi'    : ci_hi,
        'pr_auc'       : pr_auc,
        'pr_auc_base'  : baseline,
        'sensitivity'  : _safe(tp, tp + fn),
        'specificity'  : _safe(tn, tn + fp),
        'ppv'          : _safe(tp, tp + fp),
        'npv'          : _safe(tn, tn + fn),
        'f1'           : _safe(2 * tp, 2 * tp + fp + fn),
        'acc'          : _safe(tp + tn, tp + tn + fp + fn),
        'loss'         : bce_loss(labels, probs),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_pos': int((labels == 1).sum()),
        'n_neg': int((labels == 0).sum()),
    }


def fmt(v, width: int = 0, prec: int = 4) -> str:
    """nan → 'n/a'，避免把无定义的指标印成看着像数值的 nan。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"{'n/a':>{width}s}" if width else 'n/a'
    return f"{v:{width}.{prec}f}" if width else f"{v:.{prec}f}"


def is_single_class(m: dict) -> bool:
    return m['n_pos'] == 0 or m['n_neg'] == 0


def format_metrics(m: dict) -> str:
    """把 binary_metrics 结果压成一行。"""
    note = '  <single-class group: AUC/PR-AUC undefined>' if is_single_class(m) else ''
    return (f"AUC={fmt(m['auc'])} [{fmt(m['auc_ci_lo'])},{fmt(m['auc_ci_hi'])}]  "
            f"PR-AUC={fmt(m['pr_auc'])}(base={fmt(m['pr_auc_base'], prec=3)})  "
            f"Se={fmt(m['sensitivity'])}  Sp={fmt(m['specificity'])}  "
            f"PPV={fmt(m['ppv'])}  NPV={fmt(m['npv'])}  F1={fmt(m['f1'])}  "
            f"Acc={m['acc']*100:.1f}%  loss={m['loss']:.4f}  "
            f"[thr={m['threshold']:.3f}  TP/FP/TN/FN={m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}  "
            f"+{m['n_pos']}/-{m['n_neg']}]{note}")


def report_metrics(labels, probs, tag: str, val_thr: float = None,
                   indent: str = '    ') -> dict:
    """
    统一打印：
      @0.50    固定阈值 0.5
      @Youden  在这份数据自身上选的最优阈值（乐观，仅作上界参考）
      @val-thr 用验证集选出的阈值（test 集上的诚实评估，推荐看这行）
    """
    out = {}
    out['at_0.5'] = binary_metrics(labels, probs, 0.5)
    print(f"{indent}{tag:22s} @0.50    {format_metrics(out['at_0.5'])}")

    out['at_youden'] = binary_metrics(labels, probs, youden_threshold(labels, probs))
    print(f"{indent}{tag:22s} @Youden  {format_metrics(out['at_youden'])}")

    if val_thr is not None:
        out['at_val_thr'] = binary_metrics(labels, probs, val_thr)
        print(f"{indent}{tag:22s} @val-thr {format_metrics(out['at_val_thr'])}")
    return out


# ==============================================================
# 导出成扁平 dict，方便写 csv
# ==============================================================
FLAT_KEYS = ['auc', 'auc_ci_lo', 'auc_ci_hi', 'pr_auc', 'sensitivity',
             'specificity', 'ppv', 'npv', 'f1', 'acc', 'loss', 'threshold',
             'tp', 'fp', 'tn', 'fn', 'n_pos', 'n_neg']


def flatten(m: dict, prefix: str = '') -> dict:
    return {f'{prefix}{k}': m[k] for k in FLAT_KEYS if k in m}
