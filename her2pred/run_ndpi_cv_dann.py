"""Five-fold NDPI cross-validation with adversarial training.

Combines the cross-validation pipeline in run_ndpi_cv with the adversarial
models in run_zeroshot_dann. Supports internal-only, zero-shot, and few-shot
training. Use scripts/train_final.py for the experiment configurations.
"""

import os
import sys
import random

for _i, _a in enumerate(sys.argv):
    if _a == '--gpu' and _i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[_i + 1]
        break

import json
import argparse
import glob
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ndpi_cv as R                                             # noqa: E402
import run_zeroshot_dann as D                                       # noqa: E402
import run_ndpi_cv_adjusted as A                                    # noqa: E402
from final_train import ExperimentConfig                            # noqa: E402


def install(only_folds, group_by_case, fold_ids=None,
            external_all_folds=False, max_kfb_records=None,
            tcga_all_folds=False):
    """只替换 run_ndpi_cv 模块对象上的名字，磁盘文件分毫未动。"""
    R.train_one_fold = D.train_one_fold_dann
    print("  [inject] train_one_fold → run_zeroshot_dann.train_one_fold_dann")

    raw_base = (A.make_case_grouped_folds if group_by_case
                else R.make_source_stratified_folds)
    if group_by_case:
        print("  [inject] 折切分 → 病人（标本号）级分组")

    if external_all_folds and tcga_all_folds:
        raise ValueError('--external_all_folds and --tcga_all_folds are mutually exclusive')

    if external_all_folds:
        def base(records, n_folds, seed):
            # CV estimates the NDPI model.  KFB/SVS are a fixed auxiliary
            # domain pool (as is frozen TCGA-train in few-shot), so withholding
            # 20% of them wastes data and creates a format-confounded valid set.
            folds = raw_base(records, n_folds, seed)
            external = [r for r in records
                        if str(r.get('source', '')) != 'ndpi']
            if max_kfb_records is not None:
                # Keep original/augmented versions of a slide together.  A
                # record-level sample could otherwise select two views of one
                # slide while omitting another and would not be comparable to
                # the naturally paired SVS pool.
                non_kfb = [r for r in external
                           if str(r.get('source', '')) != 'kfb']
                kfb_by_slide = {}
                for rec in external:
                    if str(rec.get('source', '')) == 'kfb':
                        kfb_by_slide.setdefault(str(rec['slide_id']), []).append(rec)
                slide_ids = sorted(kfb_by_slide)
                random.Random(seed + 1729).shuffle(slide_ids)
                selected_kfb = []
                for slide_id in slide_ids:
                    group = sorted(kfb_by_slide[slide_id],
                                   key=lambda r: str(r['path']))
                    if len(selected_kfb) + len(group) <= max_kfb_records:
                        selected_kfb.extend(group)
                    if len(selected_kfb) == max_kfb_records:
                        break
                if len(selected_kfb) != max_kfb_records:
                    raise ValueError(
                        f'cannot form an exact paired KFB subset of '
                        f'{max_kfb_records} records; got {len(selected_kfb)}')
                external = non_kfb + selected_kfb
                print(f"  [inject] KFB 固定配对下采样："
                      f"{len(kfb_by_slide)} slides → "
                      f"{len({r['slide_id'] for r in selected_kfb})} slides / "
                      f"{len(selected_kfb)} records (seed={seed + 1729})")
            rebuilt = []
            for train, valid in folds:
                train_ndpi = [r for r in train
                              if str(r.get('source', '')) == 'ndpi']
                valid_ndpi = [r for r in valid
                              if str(r.get('source', '')) == 'ndpi']
                rebuilt.append((train_ndpi + external, valid_ndpi))
            print(f"  [inject] CV 只切 NDPI；每折固定加入全部外部记录 "
                  f"{len(external)} 个，valid 仅 NDPI")
            for index, (train, valid) in enumerate(rebuilt, 1):
                print(f"             Fold {index}: train={len(train)} "
                      f"(NDPI={sum(str(r.get('source','')) == 'ndpi' for r in train)}, "
                      f"external={len(external)}), valid-NDPI={len(valid)}")
            return rebuilt
    elif tcga_all_folds:
        def base(records, n_folds, seed):
            """Split NDPI/SVS/KFB normally, but train on all fixed TCGA few-shot cases."""
            folds = raw_base(records, n_folds, seed)
            tcga = [r for r in records
                    if str(r.get('source', '')) == 'tcga']
            if not tcga:
                raise ValueError('--tcga_all_folds requires TCGA records in the pool')
            rebuilt = []
            for train, valid in folds:
                train_non_tcga = [r for r in train
                                  if str(r.get('source', '')) != 'tcga']
                valid_non_tcga = [r for r in valid
                                  if str(r.get('source', '')) != 'tcga']
                rebuilt.append((train_non_tcga + tcga, valid_non_tcga))
            print(f"  [inject] 固定 TCGA few-shot：每折全部加入 {len(tcga)} 个；"
                  "NDPI/SVS/KFB 保持 source-stratified CV")
            return rebuilt
    else:
        base = raw_base

    if fold_ids:
        wanted = list(fold_ids)
        if len(set(wanted)) != len(wanted):
            raise ValueError('--fold_ids must not contain duplicates')

        def selected(records, n_folds, seed):
            folds = base(records, n_folds, seed)
            if any(i < 1 or i > len(folds) for i in wanted):
                raise ValueError(f'--fold_ids must be within 1..{len(folds)}')
            print(f"  [inject] 只跑指定折 {wanted}（完整切分共 {len(folds)} 折）")
            return [folds[i - 1] for i in wanted]

        # R.run enumerates the selected list from one. Translate that local
        # index back to the original fold id so checkpoint names remain unique
        # and independently trained workers can be merged without ambiguity.
        train_base = D.train_one_fold_dann
        def train_selected(local_idx, *args, **kwargs):
            return train_base(wanted[local_idx - 1], *args, **kwargs)

        R.make_source_stratified_folds = selected
        R.train_one_fold = train_selected
    elif only_folds and only_folds > 0:
        def truncated(records, n_folds, seed):
            folds = base(records, n_folds, seed)
            print(f"  [inject] 只跑前 {only_folds} 折（共 {len(folds)} 折，"
                  f"用于快速扫 λ）")
            return folds[:only_folds]
        R.make_source_stratified_folds = truncated
    elif group_by_case or external_all_folds or tcga_all_folds:
        R.make_source_stratified_folds = base


def parse_args():
    ap = argparse.ArgumentParser(
        description='ndpi 内部 5 折 + 颜色对抗回归')
    ap.add_argument('--gpu', default=None)
    ap.add_argument('--root', choices=['filt', 'orig'], default='filt')
    ap.add_argument('--root_dir', default=None)
    ap.add_argument('--no_aug', action='store_true')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--only_folds', type=int, default=0,
                    help='>0 时只跑前 N 折（快速扫 λ 用）')
    ap.add_argument('--fold_ids', type=int, nargs='+', default=None,
                    help='只跑指定的原始折编号，例如 --fold_ids 2 4；用于多 GPU 安全并行')
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_select', default='last',
                    choices=['last', 'best_auc', 'best_loss'])
    ap.add_argument('--keep_ckpt', action='store_true')
    ap.add_argument('--snapshot_epochs', type=int, nargs='*', default=[],
                    help='Additionally retain per-fold checkpoints at these epochs')
    ap.add_argument('--out_dir', default='results_ndpi_dann')
    ap.add_argument('--name', default=None)
    ap.add_argument('--group_by_case', action='store_true')
    ap.add_argument('--test_tcga', action='store_true',
                    help='测试集换成 TCGA（ndpi 单独训练 → TCGA 测试）。'
                         '训练池仍只有 ndpi —— svs 全阳/kfb 全阴，混进训练池'
                         '会引入第二个混杂，故本模式排除它们。')
    ap.add_argument('--internal_only_no_test', action='store_true',
                    help='只训练并写出 PUMCH OOF；不扫描或预测任何外部测试集。'
                         '用于完全 validation-only 的超参数选择。')
    # 对抗
    ap.add_argument('--domain_mode', default='appearregfull',
                    choices=['source', 'appearregfull'],
                    help='Source classification or appearance regression.')
    ap.add_argument('--lam_max', type=float, default=1.0)
    ap.add_argument('--dom_weight', type=float, default=1.0)
    ap.add_argument('--n_patch', type=int, default=256)
    ap.add_argument('--disc_hidden', type=int, default=128)
    ap.add_argument('--enc_depth', type=int, default=1,
                    help='Encoder depth. Values above 1 add residual MLP blocks.')
    ap.add_argument('--gelu', action='store_true',
                    help='Use GELU activation. The default is ReLU.')
    ap.add_argument('--bottleneck', type=int, default=0,
                    help='Bottleneck dimension. Zero disables the bottleneck.')
    ap.add_argument('--no_patch_level', action='store_true')
    ap.add_argument('--no_slide_level', action='store_true')
    ap.add_argument('--site_adv_w', type=float, default=0.0,
                    help='Weight of the source-classification adversary.')
    ap.add_argument('--conditional_site_adv', action='store_true',
                    help='Align sources separately within each class.')
    ap.add_argument('--site_adv_pos_mult', type=float, default=1.0,
                    help='条件站点对抗正类分支（NDPI+ vs SVS）的权重倍数')
    ap.add_argument('--site_adv_neg_mult', type=float, default=1.0,
                    help='条件站点对抗负类分支（NDPI- vs KFB）的权重倍数')
    ap.add_argument('--site_only_adv', action='store_true')
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
                    help='复用同一训练折预先估计的固定投影 artifact')
    ap.add_argument('--sequential_external_epochs', type=int, default=0,
                    help='先用 SVS+KFB 训练的 epoch 数；剩余 epoch 只用 NDPI，'
                         '两段共享模型、optimizer 和损失定义')
    ap.add_argument('--alternating_adv', action='store_true')
    ap.add_argument('--disc_lr', type=float, default=1e-4)
    ap.add_argument('--disc_pretrain_epochs', type=int, default=0)
    ap.add_argument('--init_allow_missing_heads', action='store_true')
    ap.add_argument('--site_intercept', action='store_true',
                    help='训练时 logit 加每站点可学标量（固定效应），推理时不加')
    ap.add_argument('--site_bias_fixed', action='store_true',
                    help='把站点截距固定为平滑后的站内流行率 logit')
    ap.add_argument('--site_bias_lr_mult', type=float, default=1.0,
                    help='站点截距的独立学习率倍数；默认 1 保持历史行为')
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
    ap.add_argument('--tcga_adv_w', type=float, default=None,
                    help='Adversarial-loss weight for TCGA few-shot samples.')
    ap.add_argument('--svs_pos_weight', type=float, default=None,
                    help='SVS 全阳性来源的独立 positive weight；默认沿用全局值')
    ap.add_argument('--tcga_pos_weight', type=float, default=None,
                    help='few-shot TCGA 阳性分类损失权重；平衡采样时通常设为1')
    ap.add_argument('--svs_appearance_grl_w', type=float, default=None,
                    help='只缩放SVS外观对抗回到encoder的GRL梯度；不缩放条件来源对抗')
    ap.add_argument('--kfb_appearance_grl_w', type=float, default=None,
                    help='只缩放KFB外观对抗回到encoder的GRL梯度；不缩放条件来源对抗')
    ap.add_argument('--source_balanced_optim', action='store_true',
                    help='混合池按 ndpi 比例对齐 accumulation 和主来源梯度，'
                         '使每 epoch 的优化步数近似 ndpi-only 基线')
    ap.add_argument('--dual_loader', action='store_true',
                    help='一阶段双 loader：NDPI 主 loader 保持 baseline 更新节奏；'
                         'SVS/KFB 由独立、按实际切片数采样的 pooled loader '
                         '旁路加入同一次反传')
    ap.add_argument('--external_sources', nargs='+', choices=['svs', 'kfb'],
                    default=None,
                    help='双 loader 实际启用的外部来源；默认同时使用 SVS/KFB')
    ap.add_argument('--external_source_balanced', action='store_true',
                    help='双 loader 在启用的外部来源间等频轮转，同时保留各来源'
                         '完全相同的 HER2 task loss')
    ap.add_argument('--external_ratio_per_step', action='store_true',
                    help='每个NDPI step从每个外部来源各取1条，并按来源/NDPI '
                         '记录数比例缩放该来源整项loss')
    ap.add_argument('--external_step_prob', type=float, default=1.0,
                    help='每个 NDPI step 附加外部样本的概率')
    ap.add_argument('--external_steps_per_epoch', type=int, default=None,
                    help='双loader每轮精确附加的外部样本数；指定后替代随机step概率')
    ap.add_argument('--external_steps_by_source', type=json.loads, default=None,
                    help='JSON来源精确配额；配额总数超过NDPI step数时会在一个'
                         'step附加多条外部样本，仍保持每轮Adam更新次数不变')
    ap.add_argument('--external_all_folds', action='store_true',
                    help='Use NDPI cross-validation and include external training samples in every fold.')
    ap.add_argument('--tcga_all_folds', action='store_true',
                    help='仅将固定 few-shot TCGA 训练子集加入每一折；NDPI/SVS/KFB仍正常分折')
    ap.add_argument('--natural_mixed_loader', action='store_true',
                    help='与 --external_all_folds 同用：不用 dual loader，'
                         '而是将该折 NDPI 和全部 SVS/KFB 放进一个普通 shuffled '
                         'loader，每个文件每 epoch 恰好遍历一次')
    ap.add_argument('--max_kfb_records', type=int, default=None,
                    help='仅与 --external_all_folds 同用：按 slide_id 保留原始/增强配对，'
                         '将每折固定 KFB 辅助池确定性下采样到指定记录数')
    ap.add_argument('--init_ckpt_dir', default=None,
                    help='按原始 fold 编号从该目录加载 fold_N.pth 后继续训练')
    ap.add_argument('--constant_lam', action='store_true',
                    help='短程 warm-start 时固定使用 lam_max，不走从零开始的 ramp')
    ap.add_argument('--mixed_pool', action='store_true',
                    help='Train on the NDPI, SVS, and KFB feature pool.')
    ap.add_argument('--fewshot_cn275', action='store_true',
                    help='训练池使用 ndpi+svs+kfb（含增强）并加入 '
                         'manifest-defined tcga_data/train；外部测试使用 '
                         'tcga_data/heldout。用于当前 CN>=2.75 adaptation '
                         '协议；必须与 --mixed_pool 同用。')
    ap.add_argument('--fewshot_tcga_dir', action='append', default=[],
                    help='重采样 few-shot 的额外 TCGA 候选特征目录；可重复指定。'
                         '最终 train/heldout 成员仍由 standard-B manifest 严格过滤。')
    ap.add_argument('--extra_tcga_dir', action='append', default=[],
                    help='Additional TCGA feature directories for evaluation. May be supplied repeatedly.')
    ap.add_argument('--standard_b_manifest', default=None,
                    help='Versioned TCGA label/inclusion manifest; default keeps canonical standard B')
    ap.add_argument('--standard_b_expected_counts', default=None,
                    help='Optional JSON count contract for the selected manifest')
    args = ap.parse_args()
    if args.only_folds and args.fold_ids:
        ap.error('--only_folds and --fold_ids are mutually exclusive')
    return args


def main():
    args = parse_args()
    root = args.root_dir or (R.ROOT_FILT if args.root == 'filt' else R.ROOT_ORIG)
    use_aug = not args.no_aug
    # Include the encoder or stain method in generated experiment names.
    _TOKENS = ['resnet50', 'macenko']
    root_tag = next((t for t in _TOKENS if t in root), args.root)

    name = args.name or (f"ndpi_dann_{root_tag}_{args.domain_mode}"
                         f"_lam{args.lam_max:g}_dw{args.dom_weight:g}"
                         f"_ep{args.epochs}"
                         + (f"_d{args.enc_depth}" if args.enc_depth > 1 else "")
                         + (f"_b{args.bottleneck}" if args.bottleneck else "")
                         + ("_gelu" if args.gelu else "")
                         + (f"_sadv{args.site_adv_w:g}"
                            if args.site_adv_w > 0 else "")
                         + ("_csadv" if args.conditional_site_adv else "")
                         + ("_sb" if args.site_intercept else "")
                         + ("_sbfix" if args.site_bias_fixed else "")
                         + (f"_sblr{args.site_bias_lr_mult:g}"
                            if args.site_bias_lr_mult != 1.0 else "")
                         + (f"_dgw{args.degen_site_w:g}"
                            if args.degen_site_w != 1.0 else "")
                         + (f"_diw{args.degen_instance_w:g}"
                            if args.degen_instance_w != 1.0 else "")
                         + ("_mix" if args.mixed_pool else "")
                         + ("_fewshotcn275" if args.fewshot_cn275 else "")
                         + (f"_f{args.only_folds}" if args.only_folds else "")
                         + (f"_case" if args.group_by_case else "")
                         + (f"_tcga" if args.test_tcga else "")
                         + f"_{args.ckpt_select}")

    D.STATE['domain_mode'] = args.domain_mode
    D.STATE['colour_vec'], D.STATE['reg_dim'] = None, 0
    if args.domain_mode == 'appearregfull':
        vec, dim = D.build_appearance_vectors_full()
        D.STATE['colour_vec'], D.STATE['reg_dim'] = vec, dim
        print(f"  Appearance targets: {dim} dimensions, {len(vec)} slides")

    D.STATE['cfg_dann'] = dict(
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
        tcga_adv_w=args.tcga_adv_w,
        svs_pos_weight=args.svs_pos_weight,
        tcga_pos_weight=args.tcga_pos_weight,
        svs_appearance_grl_w=args.svs_appearance_grl_w,
        kfb_appearance_grl_w=args.kfb_appearance_grl_w,
        source_balanced_optim=args.source_balanced_optim,
        dual_loader=args.dual_loader,
        external_sources=args.external_sources,
        external_source_balanced=args.external_source_balanced,
        external_ratio_per_step=args.external_ratio_per_step,
        external_step_prob=args.external_step_prob,
        external_steps_per_epoch=args.external_steps_per_epoch,
        external_steps_by_source=args.external_steps_by_source,
        natural_mixed_loader=args.natural_mixed_loader,
        init_ckpt_dir=args.init_ckpt_dir,
        constant_lam=args.constant_lam,
        snapshot_epochs=args.snapshot_epochs,
        n_patch=args.n_patch, disc_hidden=args.disc_hidden,
        use_patch=not args.no_patch_level, use_slide=not args.no_slide_level)
    if args.external_adain:
        D.STATE['cfg_dann']['external_adain'] = True
    # Omit the false/default key so new baseline folds remain configuration-
    # identical to already-running folds launched before this opt-in protocol
    # was added.  True is always recorded for the new formal protocol.
    if args.external_all_folds:
        D.STATE['cfg_dann']['external_all_folds'] = True
    D.STATE['log'] = []

    print(f"{'='*100}")
    print(f"  ndpi 内部 5 折 + 颜色对抗  —  「内部为何学太满」")
    print(f"{'='*100}")
    print(f"  root        : {root}")
    print(f"  域定义       : {args.domain_mode}")
    print(f"  λ_max       : {args.lam_max}   dom_weight={args.dom_weight}")
    print(f"  epochs      : {args.epochs}   folds={args.folds}"
          + (f"（只跑前 {args.only_folds} 折）" if args.only_folds else ""))
    print(f"  exp name    : {name}")
    print(f"  对照（已有）  : 内部 5 折 valid AUC = 1.0000  (valid loss 0.0699)")
    print(f"  ⚠️ 内部颜色与标签近乎共线 → 内部 AUC 下降本身不能证明去掉的是捷径")

    if args.fewshot_cn275 and not args.mixed_pool:
        raise ValueError('--fewshot_cn275 必须与 --mixed_pool 同用')

    if args.internal_only_no_test:
        if args.fewshot_cn275 or args.test_tcga:
            raise ValueError('--internal_only_no_test excludes TCGA train/test modes')
        test_dirs = []
        test_desc = 'disabled (internal-only hyperparameter selection)'
    elif args.fewshot_cn275:
        from tcga_split_dirs import heldout_test_dirs
        test_dirs = heldout_test_dirs(root)
        test_dirs += R.collect_dirs(*args.extra_tcga_dir)
        test_dirs = list(dict.fromkeys(test_dirs))
        test_desc = 'TCGA manifest-defined held-out set'
    elif args.test_tcga:
        import run_zeroshot_tcga as Z
        test_dirs = Z.tcga_test_dirs(root)
        test_dirs += R.collect_dirs(*args.extra_tcga_dir)
        test_dirs = list(dict.fromkeys(test_dirs))
        test_desc = 'TCGA (zero-shot；训练池不含任何 TCGA)'
    else:
        test_dirs = R.external_test_dirs(root)
        test_desc = 'svs + kfb（格式共线，仅参考）'
    if args.mixed_pool:
        # 主协议全量池 = ndpi + svs + kfb。与 run_zeroshot_dann 用的是同一个
        # 目录清单函数，保证两条协议的池定义一致。
        import run_zeroshot_tcga as _Z
        pool_dirs = _Z.internal_train_dirs(root, use_aug)
        if args.fewshot_cn275:
            from tcga_split_dirs import adaptation_train_dirs
            pool_dirs += adaptation_train_dirs(root)
            pool_dirs += R.collect_dirs(*args.fewshot_tcga_dir)
            pool_dirs = list(dict.fromkeys(pool_dirs))
    else:
        pool_dirs = R.ndpi_train_dirs(root, use_aug)

    # Feature roots are not all 1536-dimensional GigaPath tensors.  The
    # original entry point relied on ExperimentConfig's default, which makes
    # PLIP (512) and ResNet50 (2048) fail only after training begins.  Infer
    # the dimension from the actual selected pool for every encoder.
    input_dim = None
    for directory in pool_dirs:
        for feature_path in sorted(glob.glob(os.path.join(directory, '*.pt'))):
            try:
                feature = torch.load(feature_path, map_location='cpu')['features']
                input_dim = int(feature.shape[-1])
                break
            except Exception:
                continue
        if input_dim is not None:
            break
    if input_dim is None:
        raise RuntimeError(f'cannot infer feature dimension from {pool_dirs}')
    tcga_pool = [d for d in pool_dirs
                 if 'tcga' in os.path.normpath(d).lower()]
    added_leak = [d for d in tcga_pool
                  if 'added' in os.path.normpath(d).lower()]
    if added_leak and not args.fewshot_tcga_dir:
        raise RuntimeError(f"训练池混入 TCGA added 测试集：{added_leak}")
    if tcga_pool and not args.fewshot_cn275:
        raise RuntimeError(f"训练池意外混入 TCGA：{tcga_pool}")
    if args.fewshot_cn275 and not tcga_pool:
        raise RuntimeError('few-shot 训练池缺少 tcga_data/train')

    cfg = ExperimentConfig(
        name=name,
        cv_pool_dirs=pool_dirs,
        test_dirs=test_dirs,
        out_dir=args.out_dir, n_folds=args.folds,
        num_epochs=args.epochs, seed=args.seed,
        lr=args.lr, weight_decay=args.weight_decay, input_dim=input_dim,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    if args.natural_mixed_loader and (
            not args.external_all_folds or not args.mixed_pool
            or args.dual_loader or args.sequential_external_epochs > 0):
        raise ValueError(
            '--natural_mixed_loader requires --mixed_pool and '
            '--external_all_folds, and excludes dual/sequential loaders')
    if args.external_all_folds and not (
            args.mixed_pool and (args.dual_loader
                                 or args.sequential_external_epochs > 0
                                 or args.natural_mixed_loader)):
        raise ValueError(
            '--external_all_folds requires --mixed_pool and either '
            '--dual_loader, --natural_mixed_loader, or '
            '--sequential_external_epochs > 0')
    if args.max_kfb_records is not None:
        if not args.external_all_folds or args.max_kfb_records <= 0:
            raise ValueError('--max_kfb_records requires --external_all_folds and a positive value')
        D.STATE['cfg_dann']['max_kfb_records'] = args.max_kfb_records
    if args.tcga_all_folds and not args.fewshot_cn275:
        raise ValueError('--tcga_all_folds requires --fewshot_cn275')
    install(args.only_folds, args.group_by_case, args.fold_ids,
            args.external_all_folds, args.max_kfb_records,
            args.tcga_all_folds)
    print(f"  Start       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    expected_counts = (json.loads(args.standard_b_expected_counts)
                       if args.standard_b_expected_counts else None)
    run_kwargs = {}
    if args.standard_b_manifest:
        run_kwargs['standard_b_manifest'] = args.standard_b_manifest
        run_kwargs['standard_b_expected'] = expected_counts
    summary = R.run(cfg, name, args.ckpt_select, args.keep_ckpt,
                    train_desc=(f'{"ndpi+svs+kfb" if args.mixed_pool else "ndpi"}'
                                + ('+TCGA train (CN>=2.75)'
                                   if args.fewshot_cn275 else '')
                                + f' ({root_tag}) + 颜色对抗'),
                    test_desc=test_desc,
                    skip_test=args.internal_only_no_test, **run_kwargs)
    R.print_final_table(summary)

    fp = os.path.join(args.out_dir, name, 'dann_info.json')
    with open(fp, 'w') as f:
        json.dump(dict(cfg_dann=D.STATE['cfg_dann'],
                       domain_mode=args.domain_mode, root=root,
                       only_folds=args.only_folds,
                       appearance_cache_info=D.STATE.get(
                           'appearance_cache_info'),
                       fold_log=D.STATE['log']), f, indent=2, ensure_ascii=False)
    print(f"\n  → {fp}")
    print(f"  End         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
