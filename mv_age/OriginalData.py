"""
Helpers for working with the original Feats_LessCategories.npz dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MASTER_SEED = 42


def balanced_subset_indices(labels, n_subset: int, seed: int):
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    n_classes = len(uniq)

    if n_subset < n_classes:
        chosen_classes = rng.choice(uniq, size=n_subset, replace=False)
        idx = []
        for cls in chosen_classes:
            cls_idx = np.where(labels == cls)[0]
            idx.append(int(rng.choice(cls_idx, size=1)[0]))
        return np.array(sorted(idx), dtype=np.int64)

    per_class = n_subset // n_classes
    min_count = min(int(np.sum(labels == cls)) for cls in uniq)
    per_class = min(per_class, min_count)

    idx = []
    for cls in uniq:
        cls_idx = np.where(labels == cls)[0]
        take = rng.choice(cls_idx, size=per_class, replace=False)
        idx.extend(take.tolist())
    rng.shuffle(idx)
    return np.array(idx, dtype=np.int64)


def choose_initial_labels(labels, n_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    labeled = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        take = min(int(n_per_class), len(cls_idx))
        if take == 0:
            continue
        labeled.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return np.array(sorted(set(labeled)), dtype=np.int64)


def load_original_npz(
    *,
    npz_path: str | Path,
    n_subset: int | None = 20000,
    balance_subset_by_label: bool = True,
    seed: int = MASTER_SEED,
):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Could not find NPZ file: {npz_path}")

    saved = np.load(npz_path)
    enc = saved["encoding_values_PC"].astype(np.float32)
    lab = saved["category_labels"]
    nut = saved["nutrient_np"].astype(np.float32)

    if n_subset is not None and n_subset < len(lab):
        if balance_subset_by_label:
            idx = balanced_subset_indices(lab, n_subset=n_subset, seed=seed)
        else:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(lab), size=n_subset, replace=False)
        enc = enc[idx]
        lab = lab[idx]
        nut = nut[idx]

    return enc, lab, nut


def export_original_dataset_csvs(
    *,
    npz_path: str | Path,
    output_dir: str | Path,
    n_subset: int | None = 20000,
    balance_subset_by_label: bool = True,
    initial_labels_per_class: int = 4,
    seed: int = MASTER_SEED,
) -> dict[str, object]:
    enc, lab, nut = load_original_npz(
        npz_path=npz_path,
        n_subset=n_subset,
        balance_subset_by_label=balance_subset_by_label,
        seed=seed,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    item_ids = [f"item_{i:05d}" for i in range(len(lab))]
    initial_idx = choose_initial_labels(lab, n_per_class=initial_labels_per_class, seed=seed)
    initial_mask = np.zeros(len(lab), dtype=bool)
    initial_mask[initial_idx] = True

    enc_columns = [f"enc_{i:04d}" for i in range(enc.shape[1])]
    nut_columns = [f"nutrient_{i:04d}" for i in range(nut.shape[1])]

    enc_df = pd.DataFrame(enc, columns=enc_columns)
    enc_df.insert(0, "item_id", item_ids)

    nut_df = pd.DataFrame(nut, columns=nut_columns)
    nut_df.insert(0, "item_id", item_ids)

    labels_as_string = pd.Series(lab).astype(str)
    label_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "item_description": [f"Original precomputed dataset row {i}" for i in range(len(lab))],
            "true_label": labels_as_string,
            "label": labels_as_string.where(initial_mask, other=pd.NA),
        }
    )

    enc_path = output_dir / "encoding_values_pc.csv"
    labels_path = output_dir / "category_labels.csv"
    nutrients_path = output_dir / "nutrient_values.csv"

    enc_df.to_csv(enc_path, index=False)
    label_df.to_csv(labels_path, index=False)
    nut_df.to_csv(nutrients_path, index=False)

    return {
        "output_dir": str(output_dir.resolve()),
        "n_rows": int(len(lab)),
        "n_encoding_columns": int(enc.shape[1]),
        "n_nutrient_columns": int(nut.shape[1]),
        "n_classes": int(len(np.unique(lab))),
        "initial_labels_per_class": int(initial_labels_per_class),
        "encoding_csv": str(enc_path.resolve()),
        "labels_csv": str(labels_path.resolve()),
        "nutrients_csv": str(nutrients_path.resolve()),
    }
