"""Cache of clean-image true-label indices (argmax of precomputed logits).

The validator derives `true_label` from the clean image. We already ran clean
inference once (scripts/run_inference.py -> data/imagenet100_samples/{row:07d}.json
containing {"row", "logits"}). Reusing the argmax of those logits lets training
skip a full EfficientNetV2-L forward on the clean image every step.

The clean pipeline that produced the json (convert RGB -> JPEG q=95 -> PREPROCESS)
is identical to the one train_generator.py uses for clean images, so the cached
argmax equals model_utils.true_label_indices(clean) exactly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _read_consolidated(cache_path: Path, version: str) -> dict[int, int]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if data.get("version") != version:
        return {}
    return {int(k): int(v) for k, v in data.get("labels", {}).items()}


def _write_consolidated(cache_path: Path, version: str, labels: dict[int, int]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": version, "labels": {str(k): int(v) for k, v in labels.items()}}
    cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _argmax_from_row_json(path: Path) -> int | None:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        return int(np.argmax(rec["logits"]))
    except Exception:
        return None


def load_true_label_index(
    rows: Sequence[int],
    samples_dir: Path,
    cache_path: Path,
    version: str,
    log_every: int = 20000,
) -> dict[int, int]:
    """Return {row: true_label_index} for the requested rows.

    Pulls from a consolidated cache first, then fills misses by reading the
    per-row inference json. Rows with no available logits are simply absent from
    the result (the caller falls back to on-the-fly classifier inference).
    """
    cache = _read_consolidated(cache_path, version)
    want = [int(r) for r in rows]
    missing = [r for r in want if r not in cache]

    if missing:
        print(
            f"label cache: filling {len(missing)} labels from per-row json "
            f"(cached={len(cache)})"
        )
        filled = 0
        for i, row in enumerate(missing, start=1):
            idx = _argmax_from_row_json(samples_dir / f"{row:07d}.json")
            if idx is not None:
                cache[row] = idx
                filled += 1
            if log_every > 0 and i % log_every == 0:
                print(f"  [{i}/{len(missing)}] labels read")
        if filled:
            _write_consolidated(cache_path, version, cache)
            print(f"label cache: wrote {cache_path} ({len(cache)} labels)")

    return {r: cache[r] for r in want if r in cache}
