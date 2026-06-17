"""
Read/query the local per-image cache built by scripts/build_challenge_cache.py.

Each file: data/imagenet100_samples/{row:07d}.json
  {
    "row": int,
    "image_id": str,
    "clean_image_b64": str,
    "logits": [float, ...]   # 1000 raw EfficientNet values
  }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch

from paths import IMAGENET100_SAMPLES_DIR

CACHE_DIR = IMAGENET100_SAMPLES_DIR


def _row_path(row: int, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / f"{row:07d}.json"


def exists(row: int, cache_dir: Path = CACHE_DIR) -> bool:
    return _row_path(row, cache_dir).exists()


def load_row(row: int, cache_dir: Path = CACHE_DIR) -> dict:
    """Load a single cached row by its dataset index."""
    return json.loads(_row_path(row, cache_dir).read_text(encoding="utf-8"))


def logits_tensor(row: int, cache_dir: Path = CACHE_DIR) -> torch.Tensor:
    """Return the cached logits as a float32 tensor of shape (1000,)."""
    record = load_row(row, cache_dir)
    return torch.tensor(record["logits"], dtype=torch.float32)


def iter_cache(cache_dir: Path = CACHE_DIR) -> Iterator[dict]:
    """Yield every cached record in row-number order."""
    for path in sorted(cache_dir.glob("???????.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def cached_rows(cache_dir: Path = CACHE_DIR) -> list[int]:
    """Return sorted list of all row indices that have been cached."""
    return sorted(
        int(p.stem) for p in cache_dir.glob("???????.json")
    )


def cache_stats(cache_dir: Path = CACHE_DIR) -> dict:
    """Quick statistics: total files, unique argmax labels."""
    from collections import Counter
    from perturb_mirror.model import LABELS, normalize_prediction_label

    total = 0
    labels: Counter = Counter()
    for record in iter_cache(cache_dir):
        idx = int(max(range(len(record["logits"])),
                      key=lambda i: record["logits"][i]))
        label = normalize_prediction_label(LABELS[idx]) if idx < len(LABELS) else str(idx)
        labels[label] += 1
        total += 1

    return {
        "total": total,
        "unique_labels": len(labels),
        "top10": labels.most_common(10),
    }
