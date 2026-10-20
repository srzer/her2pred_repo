"""Train on internal NDPI/SVS/KFB features and evaluate on TCGA.

Uses five-fold cross-validation and averages fold probabilities for TCGA
inference. Validation predictions provide the classification threshold.
"""

import os
import sys
import argparse
from datetime import datetime

# run_ndpi_cv 顶部会扫 --gpu 并设置 CUDA_VISIBLE_DEVICES，
# 必须在它 import torch 之前，所以这里直接 import 它即可。
from run_ndpi_cv import (
    ROOT_FILT, ROOT_ORIG, p, collect_dirs, run, print_final_table,
)
from final_train import ExperimentConfig
from dataset_paths import pt_leaf_dirs


# ==============================================================
# 数据目录
# ==============================================================
INTERNAL_SOURCES = ['ndpi', 'svs', 'kfb']      # 明确排除 tcga


def internal_train_dirs(root: str, use_aug: bool):
    """训练池：ndpi + svs + kfb 全部并进 CV 池（支持两种布局）。"""
    dirs = []
    for s in INTERNAL_SOURCES:
        dirs += pt_leaf_dirs(p(root, f'{s}_data'))
        if use_aug:
            dirs += pt_leaf_dirs(p(root, f'{s}_data_aug'))
    return dirs


def tcga_test_dirs(root: str):
    """
    TCGA 测试集：全部 TCGA 数据，train 那批也算进来
    —— zero-shot 下它同样没被训练过。test 永远不用 aug。
    """
    return collect_dirs(
        p(root, 'tcga_data', 'train'),
        p(root, 'tcga_data', 'heldout'),
    )


def detect_input_dim(dirs) -> int:
    """
    从第一个能读到的 .pt 里读出特征维度。

    --root_dir 可以指向任意特征集（gigapath 1536 / resnet50 2048 /
    CLAM 式 1024），而 ExperimentConfig 的默认 input_dim 是 1536 —— 
    不探测的话，换了特征集就会在 CLAM 第一层 LayerNorm 上炸。
    """
    import glob
    import torch
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.pt'))):
            try:
                return int(torch.load(path, map_location='cpu')['features'].shape[1])
            except Exception:
                continue
    raise RuntimeError(f"Cannot read any .pt to detect feature dim under: {dirs}")


# ==============================================================
# CLI
# ==============================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description='TCGA zero-shot: train on ndpi+svs+kfb only, test on TCGA')
    ap.add_argument('--gpu', default=None, help='CUDA device id')
    ap.add_argument('--out_dir', default='results_zeroshot_tcga')
    ap.add_argument('--name', default=None)
    ap.add_argument('--root', default='filt', choices=['filt', 'orig'])
    ap.add_argument('--root_dir', default=None,
                    help='Feature dataset root. Overrides --root.')
    ap.add_argument('--no_aug', action='store_true')
    ap.add_argument('--input_dim', type=int, default=None,
                    help='特征维度（默认从数据里自动读出）')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--ckpt_select', default='last',
                    choices=['last', 'best_auc', 'best_loss'])
    ap.add_argument('--keep_ckpt', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    root = args.root_dir or (ROOT_FILT if args.root == 'filt' else ROOT_ORIG)
    use_aug = not args.no_aug

    root_tag = args.root
    name = args.name or (f"zeroshot_tcga_{root_tag}"
                         f"_{'aug' if use_aug else 'noaug'}"
                         f"_{args.ckpt_select}")

    cv_dirs   = internal_train_dirs(root, use_aug)
    test_dirs = tcga_test_dirs(root)

    # 安全检查：训练池里绝对不能混进 tcga
    leaked = [d for d in cv_dirs if 'tcga' in os.path.normpath(d).lower()]
    if leaked:
        raise RuntimeError(f"训练池里出现了 TCGA 目录，不是 zero-shot：{leaked}")

    input_dim = args.input_dim or detect_input_dim(cv_dirs)

    cfg = ExperimentConfig(
        name         = name,
        cv_pool_dirs = cv_dirs,
        test_dirs    = test_dirs,
        out_dir      = args.out_dir,
        input_dim    = input_dim,
        n_folds      = args.folds,
        num_epochs   = args.epochs,
        seed         = args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n  input_dim : {input_dim}")
    print(f"  CV dirs   ({len(cv_dirs)}):")
    for d in cv_dirs:
        print(f"      {d}")
    print(f"  Test dirs ({len(test_dirs)}):")
    for d in test_dirs:
        print(f"      {d}")
    print(f"  [check] 训练池不含 TCGA ✓")
    print(f"  Start     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    summary = run(cfg, name, args.ckpt_select, args.keep_ckpt,
                  train_desc='ndpi + svs + kfb (no TCGA)',
                  test_desc='TCGA (zero-shot)')
    print_final_table(summary)
    print(f"  End       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
