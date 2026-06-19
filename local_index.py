"""Local-only index helpers: discover .npy rows and load cached shape/label maps.

No Hugging Face dependency. Training uses:
  data/imagenet100_samples/{row:07d}.npy          float32 CHW [0,1]  (train split)
  data/imagenet100_val_samples/{row:07d}.npy      float32 CHW [0,1]  (HF validation split)
  data/imagenet100_shapes.json                    {version, shapes: {row: [H,W]}}
  data/imagenet100_true_labels.json               {version, labels: {row: idx}}
  data/imagenet100_val_shapes.json                validation shapes
  data/imagenet100_val_true_labels.json           validation labels
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from perturb_mirror.constants import IMAGENET100_REPO_ID


def dataset_version(split: str, num_rows: int) -> str:
    """Stable version id for a prepared split (matches imagenet100_dataset_version)."""
    import hashlib

    base = f"{IMAGENET100_REPO_ID}:{split}:{int(num_rows)}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]


def discover_npy_rows(samples_dir: Path) -> list[int]:
    """Return sorted row ids for every {row:07d}.npy under samples_dir."""
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"samples dir not found: {samples_dir}")
    rows = sorted(int(p.stem) for p in samples_dir.glob("???????.npy"))
    if not rows:
        raise FileNotFoundError(f"no .npy files in {samples_dir}")
    return rows


def load_shape_index(
    cache_path: Path,
    rows: Sequence[int],
    *,
    allow_missing: bool = False,
) -> dict[int, tuple[int, int]]:
    """Load {row: (H, W)} from the consolidated shape cache."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"shape cache not found: {cache_path}")
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    all_shapes = {int(k): (int(v[0]), int(v[1])) for k, v in data.get("shapes", {}).items()}
    out: dict[int, tuple[int, int]] = {}
    missing: list[int] = []
    for row in rows:
        r = int(row)
        if r in all_shapes:
            out[r] = all_shapes[r]
        else:
            missing.append(r)
    if missing and not allow_missing:
        raise KeyError(
            f"shape cache missing {len(missing)} rows (e.g. {missing[:5]}); "
            f"rebuild with train_generator.py or fill {cache_path}"
        )
    return out


def load_label_index(
    cache_path: Path,
    rows: Sequence[int],
    *,
    allow_missing: bool = False,
) -> dict[int, int]:
    """Load {row: true_label_index} from the consolidated label cache."""
    if not cache_path.is_file():
        raise FileNotFoundError(f"label cache not found: {cache_path}")
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    all_labels = {int(k): int(v) for k, v in data.get("labels", {}).items()}
    out: dict[int, int] = {}
    missing: list[int] = []
    for row in rows:
        r = int(row)
        if r in all_labels:
            out[r] = all_labels[r]
        else:
            missing.append(r)
    if missing and not allow_missing:
        raise KeyError(
            f"label cache missing {len(missing)} rows (e.g. {missing[:5]}); "
            f"rebuild with train_generator.py or fill {cache_path}"
        )
    return out


def cache_version(cache_path: Path) -> str:
    """Return the version string stored in a cache file (empty if absent)."""
    if not cache_path.is_file():
        return ""
    try:
        return str(json.loads(cache_path.read_text(encoding="utf-8")).get("version", ""))
    except Exception:
        return ""


class NpyDataset(Dataset):
    """Load pre-decoded float32 CHW [0,1] tensors and cached true-label indices."""

    def __init__(
        self,
        rows: list[int],
        samples_dir: Path,
        labels: dict[int, int],
    ) -> None:
        self.rows = [int(r) for r in rows]
        self.samples_dir = samples_dir
        self.labels = labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.rows[idx]
        npy_path = self.samples_dir / f"{row:07d}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(f"missing .npy for row {row}: {npy_path}")
        arr = np.load(npy_path)
        if arr.ndim != 3 or arr.shape[0] != 3:
            raise ValueError(f"row {row}: expected CHW float32, got shape {arr.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
        label = int(self.labels[row])
        return tensor, label


def _write_json_cache(path: Path, version: str, key: str, mapping: dict[int, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, key: {str(k): v for k, v in mapping.items()}}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def build_shape_index_from_npy(
    samples_dir: Path,
    rows: Sequence[int],
    cache_path: Path,
    version: str,
    *,
    log_every: int = 5000,
) -> dict[int, tuple[int, int]]:
    """Build {row: (H, W)} from .npy files and write consolidated cache."""
    shapes: dict[int, tuple[int, int]] = {}
    rows_list = [int(r) for r in rows]
    for i, row in enumerate(rows_list, start=1):
        npy_path = samples_dir / f"{row:07d}.npy"
        if not npy_path.is_file():
            continue
        arr = np.load(npy_path, mmap_mode="r")
        if arr.ndim != 3:
            raise ValueError(f"row {row}: expected CHW npy, got shape {arr.shape}")
        shapes[row] = (int(arr.shape[1]), int(arr.shape[2]))
        if log_every > 0 and i % log_every == 0:
            print(f"  shapes [{i}/{len(rows_list)}]")
    _write_json_cache(cache_path, version, "shapes", {k: list(v) for k, v in shapes.items()})
    print(f"shape index: wrote {cache_path} ({len(shapes)} rows)")
    return shapes


def build_label_index_from_json(
    samples_dir: Path,
    rows: Sequence[int],
    cache_path: Path,
    version: str,
    *,
    log_every: int = 5000,
) -> dict[int, int]:
    """Build {row: true_label_index} from per-row inference .json and write cache."""
    labels: dict[int, int] = {}
    rows_list = [int(r) for r in rows]
    missing = 0
    for i, row in enumerate(rows_list, start=1):
        json_path = samples_dir / f"{row:07d}.json"
        if not json_path.is_file():
            missing += 1
            continue
        try:
            rec = json.loads(json_path.read_text(encoding="utf-8"))
            labels[row] = int(np.argmax(rec["logits"]))
        except Exception:
            missing += 1
            continue
        if log_every > 0 and i % log_every == 0:
            print(f"  labels [{i}/{len(rows_list)}]")
    if missing:
        print(f"label index: skipped {missing} rows (no .json or bad file)")
    _write_json_cache(cache_path, version, "labels", labels)
    print(f"label index: wrote {cache_path} ({len(labels)} rows)")
    return labels
