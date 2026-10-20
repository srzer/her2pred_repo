"""Feature dataset paths and collection discovery.

Builds directory names from encoder, stain-normalization method, slide format,
and augmentation settings. Resolves flat and train/valid PT collections.
"""

import os


ENCODERS = ('gigapath', 'resnet50', 'plip')
STAINS = ('none', 'macenko')


def pt_leaf_dirs(path: str):
    """Resolve PT collections stored as <path>/*.pt or <path>/{train,valid}/*.pt.

    Each collection uses one layout. Mixed layouts raise RuntimeError.
    """
    if not os.path.isdir(path):
        return []
    direct = any(
        name.endswith('.pt') and os.path.isfile(os.path.join(path, name))
        for name in os.listdir(path)
    )
    split = [
        os.path.join(path, name)
        for name in ('train', 'valid')
        if os.path.isdir(os.path.join(path, name))
    ]
    if direct and split:
        raise RuntimeError(
            f'ambiguous PT layout under {path}: both flat files and '
            'train/valid directories are present'
        )
    if direct:
        return [path]
    return split


def dataset_root(encoder: str = 'resnet50', stain: str = 'none',
                 filtered: bool = False) -> str:
    """返回该组合对应的特征数据集根目录（相对当前项目目录）。"""
    if encoder not in ENCODERS:
        raise ValueError(f"unknown encoder {encoder!r}, choose from {ENCODERS}")
    if stain not in STAINS:
        raise ValueError(f"unknown stain {stain!r}, choose from {STAINS}")

    # 项目原有基线，沿用历史目录名
    if encoder == 'gigapath' and stain == 'none':
        return './slide_datasets_filtered' if filtered else './slide_datasets'

    base = f'./slide_datasets_{encoder}' + ('' if stain == 'none' else f'_{stain}')
    return base + '_filtered' if filtered else base


def source_dir(encoder: str, stain: str, fmt: str,
               augment: bool = False, filtered: bool = False) -> str:
    """再往下一层：<root>/<fmt>_data[_aug]"""
    suffix = '_aug' if augment else ''
    return os.path.join(dataset_root(encoder, stain, filtered),
                        f'{fmt}_data{suffix}')


def describe(encoder: str, stain: str) -> str:
    """给实验名/日志用的简短标签。"""
    return f"{encoder}" + ('' if stain == 'none' else f"+{stain}")


if __name__ == '__main__':
    print(f"{'encoder':10s} {'stain':9s} {'未过滤':38s} tumor-filtered")
    print('-' * 100)
    for e in ENCODERS:
        for s in ('none', 'macenko'):
            print(f"{e:10s} {s:9s} {dataset_root(e, s, False):38s} "
                  f"{dataset_root(e, s, True)}")
