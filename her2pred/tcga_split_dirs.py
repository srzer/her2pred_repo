"""Directory helpers for the current manifest-built TCGA training view."""

from run_ndpi_cv import collect_dirs, p


def adaptation_train_dirs(root: str):
    return collect_dirs(p(root, "tcga_data", "train"))


def heldout_test_dirs(root: str):
    return collect_dirs(p(root, "tcga_data", "heldout"))
