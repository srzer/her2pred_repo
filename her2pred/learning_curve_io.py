"""Save per-epoch training histories as JSON and long-form CSV files."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path


def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_fold_history(history: dict, exp_dir: str, fold: int) -> None:
    """Persist the current fold after every epoch/evaluation (safe to overwrite)."""
    out = Path(exp_dir) / "learning_history"
    out.mkdir(parents=True, exist_ok=True)
    clean = {key: [_clean(x) for x in values] for key, values in history.items()}
    _atomic_json(out / f"fold_{fold}.json", {
        "schema_version": 1, "fold": fold, "history": clean,
    })

    train_keys = [key for key, values in clean.items()
                  if key not in {"val_epochs", "val_loss", "val_auc",
                                 "val_pos_prob", "val_neg_prob"}
                  and isinstance(values, list)]
    n_train = max((len(clean[key]) for key in train_keys), default=0)
    train_rows = []
    for index in range(n_train):
        row = {"fold": fold, "epoch": index + 1}
        row.update({key: clean[key][index] if index < len(clean[key]) else None
                    for key in train_keys})
        train_rows.append(row)
    _atomic_csv(out / f"fold_{fold}_train.csv",
                ["fold", "epoch"] + train_keys, train_rows)

    val_keys = [key for key in ("val_loss", "val_auc", "val_pos_prob",
                                "val_neg_prob") if key in clean]
    val_epochs = clean.get("val_epochs", [])
    val_rows = []
    for index, epoch in enumerate(val_epochs):
        row = {"fold": fold, "epoch": epoch}
        row.update({key: clean[key][index] if index < len(clean[key]) else None
                    for key in val_keys})
        val_rows.append(row)
    _atomic_csv(out / f"fold_{fold}_valid.csv",
                ["fold", "epoch"] + val_keys, val_rows)


def consolidate_histories(histories: list[dict], exp_dir: str) -> None:
    """Write one experiment-level JSON in addition to fold-level live traces."""
    payload = {"schema_version": 1, "folds": [
        {"fold": index, "history": {
            key: [_clean(x) for x in values] for key, values in history.items()
        }} for index, history in enumerate(histories, 1)
    ]}
    _atomic_json(Path(exp_dir) / "learning_curves.json", payload)
