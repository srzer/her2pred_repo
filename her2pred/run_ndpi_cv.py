"""Five-fold NDPI cross-validation and SVS/KFB evaluation.

Uses validation predictions to select a Youden threshold and reports metrics
for each external source and the combined evaluation set.
"""

import os
import sys

# --gpu 必须在 import torch 之前生效
for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
        break

import gc
import json
import csv
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from final_train import (
    ExperimentConfig, set_seed, build_model, scan_files, FileListDataset,
    make_source_stratified_folds, _plot_roc, plot_combined_roc,
)
from metrics_utils import (
    binary_metrics, report_metrics, youden_threshold, bce_loss,
    format_metrics, flatten, FLAT_KEYS, fmt, is_single_class,
)
from learning_curve_io import save_fold_history, consolidate_histories
from tcga_standard_b import DEFAULT_MANIFEST, apply_to_records, validate_manifest
from dataset_paths import pt_leaf_dirs


# ==============================================================
# 数据目录
# ==============================================================
ROOT_FILT = './slide_datasets_filtered'
ROOT_ORIG = './slide_datasets'


def p(*parts):
    return os.path.join(*parts)


def collect_dirs(*candidates):
    return [d for d in candidates if os.path.exists(d)]


def ndpi_train_dirs(root: str, use_aug: bool):
    """ndpi 全部并进 CV 池；兼容公开扁平布局和旧 split 布局。"""
    dirs = pt_leaf_dirs(p(root, 'ndpi_data'))
    if use_aug:
        dirs += pt_leaf_dirs(p(root, 'ndpi_data_aug'))
    return dirs


def external_test_dirs(root: str):
    """svs + kfb 的全部数据当外部测试集；test 永远不用 aug。"""
    dirs = []
    for s in ('svs', 'kfb'):
        dirs += pt_leaf_dirs(p(root, f'{s}_data'))
    return dirs


# ==============================================================
# 推理
# ==============================================================
def predict(models, records, device, batch_desc=None):
    """
    对 records 跑（单模型或多模型 ensemble）推理。
    返回 labels, probs —— 顺序与 records 一致，便于按 source / is_aug 切片。
    """
    if not records:
        return [], []
    if not isinstance(models, (list, tuple)):
        models = [models]
    loader = DataLoader(FileListDataset(records), batch_size=1, shuffle=False)
    labels, probs = [], []
    with torch.no_grad():
        it = tqdm(loader, desc=batch_desc, leave=False) if batch_desc else loader
        for features, label, _, _ in it:
            features = features.squeeze(0).to(device)
            ps = [m(features, instance_eval=False)[0].item() for m in models]
            labels.append(int(label.item()))
            probs.append(float(np.mean(ps)))
    return labels, probs


def subset(records, labels, probs, key, value):
    """按 record 的某个字段筛选出对应的 labels/probs。"""
    idx = [i for i, r in enumerate(records) if r[key] == value]
    return [labels[i] for i in idx], [probs[i] for i in idx]


# ==============================================================
# 单折训练与验证指标记录
# ==============================================================
def train_one_fold(fold_idx, train_records, val_records, cfg, ckpt_dir,
                   ckpt_select='last'):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(FileListDataset(train_records),
                              batch_size=1, shuffle=True, num_workers=0)

    model     = build_model(cfg, device)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    history = {'train_loss': [], 'val_loss': [], 'val_auc': [],
               'val_pos_prob': [], 'val_neg_prob': [], 'val_epochs': []}
    ckpt_path = os.path.join(ckpt_dir, f'fold_{fold_idx}.pth')
    best_key  = None

    tag = f"Fold {fold_idx}/{cfg.n_folds}"
    print(f"\n  [{tag}]  train={len(train_records)} files  val={len(val_records)} files")

    def evaluate(epoch):
        # Validation is read-only.  In particular, do not leave Dropout active
        # or let DataLoader/model RNG consumption alter the following training
        # epoch.  This matters when comparing validation sets of different
        # sizes (for example NDPI-only versus NDPI+SVS+KFB).
        was_training = model.training
        cuda_devices = ([torch.cuda.current_device()]
                        if device.type == 'cuda' else [])
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            labels, probs = predict(model, val_records, device)
        model.train(was_training)
        vl   = bce_loss(labels, probs)
        m    = binary_metrics(labels, probs, 0.5)
        pos  = [pr for l, pr in zip(labels, probs) if l == 1]
        neg  = [pr for l, pr in zip(labels, probs) if l == 0]
        history['val_loss'].append(vl)
        history['val_auc'].append(m['auc'])
        history['val_pos_prob'].append(float(np.mean(pos)) if pos else 0.0)
        history['val_neg_prob'].append(float(np.mean(neg)) if neg else 0.0)
        history['val_epochs'].append(epoch)
        return vl, m['auc']

    # epoch-0 baseline（未训练的随机初始化模型）
    model.eval()
    evaluate(0)
    save_fold_history(history, os.path.dirname(ckpt_dir), fold_idx)

    for epoch in tqdm(range(cfg.num_epochs), desc=tag, leave=False):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for i, (features, label, _, _) in enumerate(train_loader):
            features = features.squeeze(0).to(device)
            label    = label[0].to(device)

            wsi_prob, _, inst_loss = model(features, label=label, instance_eval=True)
            label_s  = label * cfg.label_smooth + (1 - cfg.label_smooth) / 2
            bag_loss = criterion(torch.clamp(wsi_prob, 1e-3, 1 - 1e-3), label_s)
            loss     = (cfg.bag_weight * bag_loss + cfg.inst_weight * inst_loss) / cfg.accum_steps
            if label.item() == 1.0:
                loss = loss * cfg.pos_weight_val
            loss.backward()

            if (i + 1) % cfg.accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.accum_steps

        history['train_loss'].append(epoch_loss / len(train_loader))

        if (epoch + 1) % 4 == 0:
            vl, va = evaluate(epoch + 1)
            model.train()
            if ckpt_select == 'last':
                # 与 final_train.py 现有行为一致：每次评估都覆盖存盘 → 取最后一个
                torch.save(model.state_dict(), ckpt_path)
            elif ckpt_select == 'best_auc':
                if best_key is None or va >= best_key:
                    best_key = va
                    torch.save(model.state_dict(), ckpt_path)
                    tqdm.write(f"    [{tag}] ep{epoch+1}  val_auc={va:.4f}  val_loss={vl:.4f} → saved")
            elif ckpt_select == 'best_loss':
                if best_key is None or vl <= best_key:
                    best_key = vl
                    torch.save(model.state_dict(), ckpt_path)
                    tqdm.write(f"    [{tag}] ep{epoch+1}  val_auc={va:.4f}  val_loss={vl:.4f} → saved")
        # 原始逐 epoch 轨迹增量落盘；中断时最多损失当前 epoch。
        save_fold_history(history, os.path.dirname(ckpt_dir), fold_idx)

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 选定 checkpoint 上的最终 val 预测
    val_labels, val_probs = predict(model, val_records, device)

    return model, ckpt_path, history, val_labels, val_probs


# ==============================================================
# 绘图
# ==============================================================
def plot_curves(history, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(range(1, len(history['train_loss']) + 1), history['train_loss'])
    axes[0].set_title('Train loss'); axes[0].set_xlabel('epoch')

    axes[1].plot(history['val_epochs'], history['val_loss'], 'o-', color='crimson')
    axes[1].set_title('Valid loss (BCE)'); axes[1].set_xlabel('epoch')

    axes[2].plot(history['val_epochs'], history['val_auc'], 'bo-')
    axes[2].set_title('Valid AUC'); axes[2].set_xlabel('epoch'); axes[2].set_ylim(0, 1)

    axes[3].plot(history['val_epochs'], history['val_pos_prob'], 'r-', label='pos')
    axes[3].plot(history['val_epochs'], history['val_neg_prob'], 'g-', label='neg')
    axes[3].axhline(0.5, color='gray', linestyle='--')
    axes[3].set_title('Predicted prob'); axes[3].legend()

    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()


def plot_loss_overlay(histories, save_path):
    """5 折的 valid loss 画在一张图上，方便看过拟合。"""
    plt.figure(figsize=(7, 5))
    for k, h in enumerate(histories, start=1):
        plt.plot(h['val_epochs'], h['val_loss'], marker='o', ms=3,
                 label=f'Fold {k}')
    mean_curve = np.mean([h['val_loss'] for h in histories], axis=0)
    plt.plot(histories[0]['val_epochs'], mean_curve, 'k-', lw=2.5, label='Mean')
    plt.xlabel('epoch'); plt.ylabel('Valid loss (BCE)')
    plt.title('Valid loss across folds'); plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()


# ==============================================================
# 主流程
# ==============================================================
def run(cfg: ExperimentConfig, exp_name: str, ckpt_select: str,
        keep_ckpt: bool, train_desc: str = 'ndpi',
        test_desc: str = 'svs + kfb',
        standard_b_manifest: str = DEFAULT_MANIFEST,
        standard_b_expected: dict = None,
        skip_test: bool = False) -> dict:
    exp_dir  = os.path.join(cfg.out_dir, exp_name)
    ckpt_dir = os.path.join(exp_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*100}")
    print(f"  Experiment  : {exp_name}")
    print(f"  Train (CV)  : {train_desc}  {cfg.n_folds}-fold")
    print(f"  Test        : "
          f"{'disabled (internal-only selection)' if skip_test else test_desc + ' (external evaluation)'}")
    print(f"  ckpt select : {ckpt_select}")
    print(f"  out_dir     : {exp_dir}")
    print(f"{'='*100}")

    set_seed(cfg.seed)

    # ── 1. 扫描数据 ───────────────────────────────────────────
    print("\n  Scanning CV pool (ndpi) ...")
    cv_pool = scan_files(cfg.cv_pool_dirs)
    if skip_test:
        print("  External test disabled; writing PUMCH OOF outputs only.")
        test_recs = []
    else:
        print("  Scanning test set (svs + kfb) ...")
        test_recs = scan_files(cfg.test_dirs)

    # 标准 B 的唯一实现：所有 TCGA 记录按临床 NGS 推导的 estimated
    # ERBB2 CN >= 2.75 重标；无有限 CN 的病例严格排除，绝不回退 GISTIC。
    # Select full TCGA evaluation for zero-shot and held-out cases for few-shot.
    has_tcga_train = any(
        str(r.get('slide_id', '')).upper().startswith('TCGA-')
        for r in cv_pool)
    manifest_cv_cohort = 'train' if has_tcga_train else 'full'
    manifest_test_cohort = 'heldout' if has_tcga_train else 'full'
    if skip_test and not has_tcga_train:
        # NDPI-only runs use the internal cohort.
        manifest_counts = None
        cv_audit = {"tcga_candidates": 0, "tcga_kept": 0,
                    "tcga_dropped_no_cn_or_wrong_cohort": 0,
                    "tcga_duplicate_feature_records_dropped": 0}
        test_audit = dict(cv_audit)
        print("  TCGA manifest access disabled for internal-only run.")
    else:
        manifest_counts = validate_manifest(standard_b_manifest,
                                            expected=standard_b_expected)
        cv_pool, cv_audit = apply_to_records(
            cv_pool, path=standard_b_manifest, cohort=manifest_cv_cohort)
        test_recs, test_audit = apply_to_records(
            test_recs, path=standard_b_manifest, cohort=manifest_test_cohort)
    if cv_audit['tcga_candidates'] or test_audit['tcga_candidates']:
        print(f"  [standard B] CV: {cv_audit}; test: {test_audit}")

    # Check the resolved train/test records for overlapping slides.
    cv_keys = {(str(r['source']), str(r['slide_id'])) for r in cv_pool}
    test_keys = {(str(r['source']), str(r['slide_id'])) for r in test_recs}
    overlap = sorted(cv_keys & test_keys)
    if overlap:
        raise RuntimeError(
            f"train/test leakage after manifest filtering: {len(overlap)} "
            f"source-slide keys; first={overlap[:5]}")

    if not cv_pool:
        raise RuntimeError("CV pool is empty — check ndpi data dirs.")
    if not test_recs and not skip_test:
        raise RuntimeError("Test set is empty — check svs/kfb data dirs.")

    test_by_src = defaultdict(lambda: [0, 0])
    for r in test_recs:
        test_by_src[r['source']][r['label']] += 1
    print(f"\n  Test files: {len(test_recs)}")
    single_class_srcs = []
    for s, (neg, pos) in sorted(test_by_src.items()):
        flag = ''
        if pos == 0 or neg == 0:
            flag = '   <-- 单一类别，该 source 内部无法计算 AUC / PR-AUC'
            single_class_srcs.append(s)
        print(f"    {s:6s}  pos={pos:4d}  neg={neg:4d}{flag}")

    if single_class_srcs:
        print(f"\n  [!] {', '.join(single_class_srcs)} 只含单一类别标签。")
        print(f"      → 这些 source 内部的 AUC/PR-AUC 无定义，只能报 Se（全正）或 Sp（全负）。")
        print(f"      → 合并后的 svs+kfb AUC 虽然可以算，但正类全部来自一种扫描格式、")
        print(f"        负类全部来自另一种，AUC 里混入了「区分扫描格式」的信号，")
        print(f"        不能等同于纯粹的 HER2 判别能力。解读时务必注意。")

    # ── 2. 5 折训练 ───────────────────────────────────────────
    folds = make_source_stratified_folds(cv_pool, cfg.n_folds, cfg.seed)

    models, ckpt_paths, histories = [], [], []
    fold_rows, roc_data = [], []
    oof_records, oof_labels, oof_probs = [], [], []

    for fold_idx, (train_recs, val_recs) in enumerate(folds, start=1):
        model, ckpt_path, history, val_labels, val_probs = train_one_fold(
            fold_idx, train_recs, val_recs, cfg, ckpt_dir, ckpt_select)

        models.append(model)
        ckpt_paths.append(ckpt_path)
        histories.append(history)

        oof_records += val_recs
        oof_labels  += val_labels
        oof_probs   += val_probs

        # 该折的 valid 指标（orig-only：去掉 aug 副本，一张切片只算一次）。
        # 混合池的 KFB/SVS 分别是单阴性/单阳性来源，若把三种格式直接合并
        # 选阈值，格式与标签的共线性会污染 operating point。因此保留 mixed
        # valid 作为诊断，但正式内部 valid 与阈值只使用 NDPI original OOF。
        vl_o, vp_o = subset(val_recs, val_labels, val_probs, 'is_aug', False)
        ndpi_idx = [i for i, rec in enumerate(val_recs)
                    if not rec['is_aug'] and rec['source'] == 'ndpi']
        vl_ndpi = [val_labels[i] for i in ndpi_idx]
        vp_ndpi = [val_probs[i] for i in ndpi_idx]
        m_all  = binary_metrics(val_labels, val_probs, 0.5)
        m_orig = binary_metrics(vl_o, vp_o, 0.5) if vl_o else m_all
        m_ndpi = binary_metrics(vl_ndpi, vp_ndpi, 0.5) if vl_ndpi else m_orig

        print(f"    [Fold {fold_idx}] valid_loss={m_ndpi['loss']:.4f}  "
              f"valid_auc={m_ndpi['auc']:.4f}   "
              f"(NDPI orig-only, n={len(vl_ndpi)})")
        if len(vl_o) != len(vl_ndpi):
            print(f"    [Fold {fold_idx}] mixed-valid_loss={m_orig['loss']:.4f}  "
                  f"mixed-valid_auc={m_orig['auc']:.4f}   "
                  f"(all sources orig-only, diagnostic, n={len(vl_o)})")
        print(f"    [Fold {fold_idx}] valid_loss={m_all['loss']:.4f}  "
              f"valid_auc={m_all['auc']:.4f}   (incl. aug, n={len(val_labels)})")

        fold_test = {}
        if not skip_test:
            # 单折模型在外部 test 上的表现
            t_labels, t_probs = predict(model, test_recs, device,
                                        batch_desc=f'Fold {fold_idx} → test')
            for src in sorted(test_by_src):
                sl, sp = subset(test_recs, t_labels, t_probs, 'source', src)
                fold_test[src] = binary_metrics(sl, sp, 0.5)
            fold_test['all'] = binary_metrics(t_labels, t_probs, 0.5)
            print(f"    [Fold {fold_idx}] test AUC  "
                  + '  '.join(f"{s}={fmt(fold_test[s]['auc'])}" for s in sorted(fold_test)))
            print(f"    [Fold {fold_idx}] test mean prob  "
                  + '  '.join(f"{s}={np.mean([p for r, p in zip(test_recs, t_probs) if r['source'] == s]):.4f}"
                              for s in sorted(test_by_src)))

        fold_rows.append({
            'fold'           : fold_idx,
            'valid_loss'     : m_ndpi['loss'],
            'valid_auc'      : m_ndpi['auc'],
            'valid_loss_mixed_orig': m_orig['loss'],
            'valid_auc_mixed_orig': m_orig['auc'],
            'valid_loss_aug' : m_all['loss'],
            'valid_auc_aug'  : m_all['auc'],
            'valid_n'        : len(vl_ndpi),
            'valid_n_mixed_orig': len(vl_o),
            **{f'test_{s}_auc'   : fold_test[s]['auc']    for s in fold_test},
            **{f'test_{s}_prauc' : fold_test[s]['pr_auc'] for s in fold_test},
        })

        roc_data.append({'labels': vl_ndpi or vl_o or val_labels,
                         'probs' : vp_ndpi or vp_o or val_probs,
                         'name'  : f'Val fold {fold_idx}'})
        plot_curves(history, os.path.join(exp_dir, f'curves_fold{fold_idx}.png'))
        _plot_roc(vl_ndpi or vl_o or val_labels,
                  vp_ndpi or vp_o or val_probs,
                  os.path.join(exp_dir, f'auc_valid_fold{fold_idx}.png'),
                  f'Valid fold {fold_idx}')

    plot_loss_overlay(histories, os.path.join(exp_dir, 'valid_loss_folds.png'))
    consolidate_histories(histories, exp_dir)

    # ── 3. OOF valid 汇总 ─────────────────────────────────────
    print(f"\n{'-'*100}")
    print("  VALIDATION (OOF, threshold from original NDPI slides)")
    print(f"{'-'*100}")

    ol_o, op_o = subset(oof_records, oof_labels, oof_probs, 'is_aug', False)
    if not ol_o:
        ol_o, op_o = oof_labels, oof_probs
    ndpi_oof_idx = [i for i, rec in enumerate(oof_records)
                    if not rec['is_aug'] and rec['source'] == 'ndpi']
    ndpi_ol = [oof_labels[i] for i in ndpi_oof_idx]
    ndpi_op = [oof_probs[i] for i in ndpi_oof_idx]
    if not ndpi_ol:
        ndpi_ol, ndpi_op = ol_o, op_o

    valid_metrics_ndpi = report_metrics(
        ndpi_ol, ndpi_op, 'OOF valid (NDPI orig; threshold source)')
    valid_metrics     = report_metrics(ol_o, op_o, 'OOF valid (mixed orig; diagnostic)')
    valid_metrics_aug = report_metrics(oof_labels, oof_probs, 'OOF valid (+aug)')

    fold_losses = [r['valid_loss'] for r in fold_rows]
    fold_aucs   = [r['valid_auc']  for r in fold_rows]
    print(f"\n    Per-fold valid loss : "
          + '  '.join(f"F{r['fold']}={r['valid_loss']:.4f}" for r in fold_rows))
    print(f"    Per-fold valid AUC  : "
          + '  '.join(f"F{r['fold']}={r['valid_auc']:.4f}" for r in fold_rows))
    print(f"    valid loss  mean±std = {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print(f"    valid AUC   mean±std = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"    pooled NDPI OOF valid loss = "
          f"{valid_metrics_ndpi['at_0.5']['loss']:.4f}")

    val_thr = youden_threshold(ndpi_ol, ndpi_op)
    print(f"    Youden threshold picked on NDPI original OOF valid = {val_thr:.4f}")

    _plot_roc(ndpi_ol, ndpi_op, os.path.join(exp_dir, 'auc_valid_oof.png'),
              'Valid (NDPI OOF)')
    roc_data.append({'labels': ndpi_ol, 'probs': ndpi_op,
                     'name': 'Valid NDPI OOF'})

    # ── 4. 外部 test（5 折模型 ensemble），按 source 分组 ──────
    print(f"\n{'-'*100}")
    print("  TEST — 5-fold ensemble on external data")
    print(f"{'-'*100}")

    test_metrics = {}
    t_labels, t_probs = [], []
    if not skip_test:
        t_labels, t_probs = predict(models, test_recs, device, batch_desc='Ensemble → test')
        for src in sorted(test_by_src) + ['all']:
            if src == 'all':
                sl, sp = t_labels, t_probs
            else:
                sl, sp = subset(test_recs, t_labels, t_probs, 'source', src)
            if not sl:
                continue
            print()
            test_metrics[src] = report_metrics(sl, sp, f'test[{src}]', val_thr=val_thr)
            _plot_roc(sl, sp, os.path.join(exp_dir, f'auc_test_{src}.png'), f'Test {src}')
            roc_data.append({'labels': sl, 'probs': sp, 'name': f'Test {src}'})

    plot_combined_roc(roc_data, os.path.join(exp_dir, 'auc_combined.png'))

    # ── 4b. 逐切片预测落盘 ────────────────────────────────────
    # 有了它才能事后换标签定义/换子集重新算指标，不必重训。
    with open(os.path.join(exp_dir, 'predictions_test.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide_id', 'source', 'root', 'label', 'prob_ensemble', 'path'])
        for rec, lab, pr in zip(test_recs, t_labels, t_probs):
            w.writerow([rec['slide_id'], rec['source'], rec['root'],
                        lab, f'{pr:.6f}', rec['path']])

    with open(os.path.join(exp_dir, 'predictions_valid.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slide_id', 'source', 'root', 'is_aug', 'label', 'prob_oof'])
        for rec, lab, pr in zip(oof_records, oof_labels, oof_probs):
            w.writerow([rec['slide_id'], rec['source'], rec['root'],
                        int(rec['is_aug']), lab, f'{pr:.6f}'])
    print(f"\n  逐切片预测已写入 predictions_test.csv / predictions_valid.csv")

    # ── 5. 落盘 ───────────────────────────────────────────────
    with open(os.path.join(exp_dir, 'folds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader(); w.writerows(fold_rows)

    metric_rows = []
    for split_name, mm in [('valid_ndpi_oof_orig', valid_metrics_ndpi),
                           ('valid_oof_orig_mixed_diagnostic', valid_metrics),
                           ('valid_oof_withaug', valid_metrics_aug)]:
        for thr_name, m in mm.items():
            metric_rows.append({'split': split_name, 'thr_mode': thr_name,
                                **flatten(m)})
    for src, mm in test_metrics.items():
        for thr_name, m in mm.items():
            metric_rows.append({'split': f'test_{src}', 'thr_mode': thr_name,
                                **flatten(m)})

    with open(os.path.join(exp_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'thr_mode'] + FLAT_KEYS)
        w.writeheader(); w.writerows(metric_rows)

    summary = {
        'exp_name'        : exp_name,
        'timestamp'       : datetime.now().isoformat(),
        'ckpt_select'     : ckpt_select,
        'n_folds'         : cfg.n_folds,
        'num_epochs'      : cfg.num_epochs,
        'seed'            : cfg.seed,
        'learning_rate'   : cfg.lr,
        'weight_decay'    : cfg.weight_decay,
        'accum_steps'     : cfg.accum_steps,
        'label_smooth'    : cfg.label_smooth,
        'bag_weight'      : cfg.bag_weight,
        'inst_weight'     : cfg.inst_weight,
        'cv_pool_dirs'    : cfg.cv_pool_dirs,
        'test_dirs'       : cfg.test_dirs,
        'standard_b_manifest': standard_b_manifest,
        'standard_b_manifest_counts': manifest_counts,
        'standard_b_cv_cohort': manifest_cv_cohort,
        'standard_b_test_cohort': manifest_test_cohort,
        'standard_b_cv_audit': cv_audit,
        'standard_b_test_audit': test_audit,
        'n_cv_files'      : len(cv_pool),
        'n_test_files'    : len(test_recs),
        'internal_only_no_test': bool(skip_test),
        'valid_thr'       : val_thr,
        'valid_threshold_source': 'original non-augmented NDPI OOF; Youden J',
        'valid_loss_mean' : float(np.mean(fold_losses)),
        'valid_loss_std'  : float(np.std(fold_losses)),
        'valid_auc_mean'  : float(np.mean(fold_aucs)),
        'valid_auc_std'   : float(np.std(fold_aucs)),
        'valid_ndpi_oof'  : {k: v for k, v in valid_metrics_ndpi.items()},
        'valid_oof'       : {k: v for k, v in valid_metrics.items()},
        'valid_oof_withaug': {k: v for k, v in valid_metrics_aug.items()},
        'test'            : test_metrics,
        'folds'           : fold_rows,
    }
    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # ── 6. 清理 ───────────────────────────────────────────────
    del models
    torch.cuda.empty_cache(); gc.collect()

    if not keep_ckpt:
        removed = 0
        for cp in ckpt_paths:
            if os.path.exists(cp):
                os.remove(cp); removed += 1
        if os.path.isdir(ckpt_dir) and not os.listdir(ckpt_dir):
            os.rmdir(ckpt_dir)
        print(f"\n  Removed {removed} checkpoint(s).")
    else:
        print(f"\n  Checkpoints kept in {ckpt_dir}")

    print(f"  Results written to {exp_dir}")
    return summary


# ==============================================================
# 最终结果表
# ==============================================================
def print_final_table(summary: dict):
    print(f"\n{'='*100}")
    print(f"  FINAL REPORT — {summary['exp_name']}")
    print(f"{'='*100}")

    print(f"\n  Valid (ndpi, 5-fold out-of-fold)")
    print(f"    valid loss (BCE) : {summary['valid_loss_mean']:.4f} "
          f"± {summary['valid_loss_std']:.4f}   (per-fold mean ± std)")
    valid_primary = summary.get('valid_ndpi_oof', summary['valid_oof'])
    print(f"    valid loss (pooled NDPI OOF) : "
          f"{valid_primary['at_0.5']['loss']:.4f}")

    hdr = (f"    {'split':16s} {'n(+/-)':>10s} {'thr':>6s} {'AUC':>8s} {'PR-AUC':>8s} "
           f"{'Se':>7s} {'Sp':>7s} {'PPV':>7s} {'NPV':>7s} {'F1':>7s} {'loss':>7s}")

    def row(name, m):
        n = f"{m['n_pos']}/{m['n_neg']}"
        return (f"    {name:16s} {n:>10s} {m['threshold']:6.3f} "
                f"{fmt(m['auc'], 8)} {fmt(m['pr_auc'], 8)} "
                f"{fmt(m['sensitivity'], 7)} {fmt(m['specificity'], 7)} {fmt(m['ppv'], 7)} "
                f"{fmt(m['npv'], 7)} {fmt(m['f1'], 7)} {m['loss']:7.4f}")

    print(f"\n  @ threshold 0.50")
    print(hdr)
    print(row('valid (NDPI OOF)', valid_primary['at_0.5']))
    for src, mm in summary['test'].items():
        print(row(f'test [{src}]', mm['at_0.5']))

    print(f"\n  @ threshold {summary['valid_thr']:.4f}  "
          f"(Youden, picked on original NDPI OOF valid)")
    print(hdr)
    print(row('valid (NDPI OOF)', valid_primary['at_youden']))
    for src, mm in summary['test'].items():
        if 'at_val_thr' in mm:
            print(row(f'test [{src}]', mm['at_val_thr']))

    sc = [s for s, mm in summary['test'].items() if is_single_class(mm['at_0.5'])]
    if sc:
        print(f"\n  注意：{', '.join(sc)} 为单一类别 source，其行内 AUC/PR-AUC 无定义（n/a），")
        print(f"       全正的 source 只有 Se 有意义，全负的 source 只有 Sp 有意义。")
        print(f"       test [all] 行是 svs+kfb 合并后的结果，可以算出 AUC，但正/负类与")
        print(f"       扫描格式完全共线，该 AUC 上界性地高估了真实的 HER2 判别能力。")
    print()


# ==============================================================
# CLI
# ==============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description='ndpi 5-fold CV training, external test on svs + kfb')
    ap.add_argument('--gpu', default=None, help='CUDA device id (sets CUDA_VISIBLE_DEVICES)')
    ap.add_argument('--out_dir', default='results_ndpi_cv', help='输出根目录')
    ap.add_argument('--name', default=None, help='实验名（默认按配置自动生成）')
    ap.add_argument('--root', default='filt', choices=['filt', 'orig'],
                    help='用 slide_datasets_filtered 还是 slide_datasets (默认 filt)')
    ap.add_argument('--root_dir', default=None,
                    help='Feature dataset root. Overrides --root.')
    ap.add_argument('--no_aug', action='store_true', help='训练不使用 aug 数据')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_select', default='last',
                    choices=['last', 'best_auc', 'best_loss'],
                    help="'last' 与 final_train.py 现有行为一致（固定 epoch 预算，"
                         "不在 val 上做模型选择，valid 指标无乐观偏差）")
    ap.add_argument('--keep_ckpt', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root_dir or (ROOT_FILT if args.root == 'filt' else ROOT_ORIG)
    use_aug = not args.no_aug

    root_tag = args.root
    name = args.name or (f"ndpi5fold_{root_tag}"
                         f"_{'aug' if use_aug else 'noaug'}"
                         f"_{args.ckpt_select}")

    cfg = ExperimentConfig(
        name         = name,
        cv_pool_dirs = ndpi_train_dirs(root, use_aug),
        test_dirs    = external_test_dirs(root),
        out_dir      = args.out_dir,
        n_folds      = args.folds,
        num_epochs   = args.epochs,
        seed         = args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n  CV dirs   : {cfg.cv_pool_dirs}")
    print(f"  Test dirs : {cfg.test_dirs}")
    print(f"  Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary = run(cfg, name, args.ckpt_select, args.keep_ckpt)
    print_final_table(summary)
    print(f"  End       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
