"""Resolution bucketing for batched generator training.

The generator is fully convolutional, so its perturbation has the same (H, W)
as its input. Images can therefore be batched **only** when they share an exact
(H, W). This module:

  1. Builds a cached row -> (H, W) shape index (JPEG q=95 preserves dimensions,
     so the size equals the native PIL size used by decode_image_b64).
  2. Groups dataset positions by (H, W) and yields fixed-size batches per bucket
     via `BucketBatchSampler`, with an optional per-bucket cap by pixel budget.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Sampler


def build_shape_index(
    dataset: Any,
    rows: Sequence[int],
    cache_path: Path,
    version: str,
    log_every: int = 5000,
) -> dict[int, tuple[int, int]]:
    """Return {row: (H, W)} for the given rows, caching results to disk.

    JPEG q=95 encode/decode (the validator path) does not change resolution, so
    the (H, W) is read directly from the decoded PIL image size.
    """
    cache: dict[int, tuple[int, int]] = {}
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("version") == version:
                cache = {int(k): tuple(v) for k, v in data.get("shapes", {}).items()}
        except Exception:
            cache = {}

    missing = [int(r) for r in rows if int(r) not in cache]
    if missing:
        print(f"shape index: computing {len(missing)} missing shapes (cached={len(cache)})")
        for i, row in enumerate(missing, start=1):
            width, height = dataset[int(row)]["image"].size  # PIL size is (W, H)
            cache[int(row)] = (int(height), int(width))
            if log_every > 0 and i % log_every == 0:
                print(f"  [{i}/{len(missing)}] shapes computed")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": version, "shapes": {str(k): list(v) for k, v in cache.items()}}
        cache_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        print(f"shape index: wrote {cache_path}")

    return {int(r): cache[int(r)] for r in rows}


def bucket_collate(batch):
    """Stack a same-shape bucket into a batch.

    Accepts either plain CHW tensors -> (B, C, H, W), or (tensor, label) tuples
    -> ((B, C, H, W), (B,) long labels). Labels of -1 mean "not cached".
    """
    if isinstance(batch[0], tuple):
        tensors = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([int(item[1]) for item in batch], dtype=torch.long)
        return tensors, labels
    return torch.stack(batch, dim=0)


class BucketBatchSampler(Sampler[list[int]]):
    """Yield batches of dataset positions that all share the same (H, W).

    Args:
        shapes_by_position: shapes_by_position[p] = (H, W) for dataset position p.
        batch_size: max images per batch.
        max_pixels: optional cap; per-bucket batch is reduced so B*H*W <= max_pixels.
        shuffle: shuffle within buckets and across batches each epoch.
        drop_last: drop the final short batch of each bucket.
        seed: base RNG seed (combined with epoch via set_epoch).
    """

    def __init__(
        self,
        shapes_by_position: Sequence[tuple[int, int]],
        batch_size: int,
        max_pixels: int | None = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.max_pixels = int(max_pixels) if max_pixels else None
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

        self.buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, hw in enumerate(shapes_by_position):
            self.buckets[(int(hw[0]), int(hw[1]))].append(position)

        self._batches: list[list[int]] = self._make_batches()

    def _bucket_batch_size(self, hw: tuple[int, int]) -> int:
        if self.max_pixels is None:
            return self.batch_size
        h, w = hw
        cap = max(1, self.max_pixels // max(1, h * w))
        return max(1, min(self.batch_size, cap))

    def _make_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for hw, positions in self.buckets.items():
            pos = list(positions)
            if self.shuffle:
                rng.shuffle(pos)
            bs = self._bucket_batch_size(hw)
            for start in range(0, len(pos), bs):
                chunk = pos[start : start + bs]
                if self.drop_last and len(chunk) < bs:
                    continue
                batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._batches = self._make_batches()

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)

    def describe(self) -> str:
        sizes = sorted((len(v) for v in self.buckets.values()), reverse=True)
        top = sizes[:5]
        return (
            f"buckets={len(self.buckets)} images={sum(sizes)} "
            f"batches={len(self._batches)} largest_buckets={top}"
        )
