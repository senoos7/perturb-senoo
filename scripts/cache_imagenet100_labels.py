#!/usr/bin/env python3
"""Precompute EfficientNetV2-L true labels for all ImageNet-100 train rows.

Validators use the model prediction as true_label (not the dataset ClassLabel).
This script caches those labels for local training and evaluation aligned with
production scoring.

Output (default data/imagenet100/):
  labels.jsonl  one JSON object per row
  meta.json     run metadata

Example:
  .venv/bin/python scripts/cache_imagenet100_labels.py
  .venv/bin/python scripts/cache_imagenet100_labels.py --batch-size 64 --device cuda
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perturbnet.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT
from perturbnet.image_io import decode_image_b64
from perturbnet.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100
from perturbnet.imagenet100_labels import (
    append_label_rows,
    default_meta,
    manifest_path,
    meta_path,
    load_label_manifest,
    write_meta,
)
from perturbnet.model import (
    LABELS,
    load_efficientnet_v2_l,
    normalize_prediction_label,
)


def _example_to_chw(example: dict[str, Any]) -> torch.Tensor:
    """Match validator challenge path: PIL -> JPEG(q=95) -> decode_image_b64."""
    image = example.get("image")
    if image is None:
        raise ValueError("dataset row has no image")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return decode_image_b64(image_b64)


def _predict_batch_indices(model: torch.nn.Module, batch_chw: list[torch.Tensor], device: torch.device) -> list[int]:
    from perturbnet.model import _preprocess_for_efficientnet_v2_l

    if not batch_chw:
        return []
    batch_bchw = torch.cat([_preprocess_for_efficientnet_v2_l(image.unsqueeze(0)) for image in batch_chw], dim=0).to(
        device
    )
    with torch.no_grad():
        logits = model(batch_bchw)
        return [int(x) for x in logits.argmax(dim=1).tolist()]


def _index_to_true_label(index: int) -> str:
    if 0 <= index < len(LABELS):
        return normalize_prediction_label(LABELS[index])
    return normalize_prediction_label(str(index))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache EfficientNetV2-L labels for ImageNet-100 train split")
    parser.add_argument("--output-dir", type=Path, default=manifest_path().parent)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    out_manifest = args.output_dir / manifest_path().name
    out_meta = args.output_dir / meta_path().name

    print(f"Loading dataset repo={IMAGENET100_REPO_ID} split={IMAGENET100_SPLIT}")
    dataset = load_imagenet100(repo_id=IMAGENET100_REPO_ID, split=IMAGENET100_SPLIT)
    total_rows = int(dataset.num_rows)
    version = imagenet100_dataset_version(dataset=dataset, repo_id=IMAGENET100_REPO_ID, split=IMAGENET100_SPLIT)
    label_names = list(dataset.features["label"].names)

    done = load_label_manifest(out_manifest)
    end_row = total_rows if args.max_rows <= 0 else min(total_rows, args.start_row + args.max_rows)

    device = torch.device(args.device)
    print(f"Loading {IMAGENET100_REPO_ID} model on {device}")
    model = load_efficientnet_v2_l(device)

    write_meta(
        out_meta,
        {
            **default_meta(dataset_rows=total_rows, dataset_version=version),
            "cached_rows": len(done),
            "resume_from_row": args.start_row,
        },
    )

    pending = sum(1 for row_idx in range(args.start_row, end_row) if row_idx not in done)
    if pending <= 0:
        print(f"Nothing to do: cached={len(done)} total={total_rows}")
        return 0

    print(
        f"Caching labels rows={args.start_row}..{end_row - 1} "
        f"(pending={pending}, cached={len(done)}, total={total_rows}, batch={args.batch_size})"
    )

    t0 = time.time()
    processed = 0
    batch_rows: list[int] = []
    batch_tensors: list[torch.Tensor] = []

    def flush_batch() -> None:
        nonlocal processed
        if not batch_rows:
            return
        pred_indices = _predict_batch_indices(model, batch_tensors, device)
        out: list[dict[str, Any]] = []
        for row_idx, example, pred_idx in zip(batch_rows, batch_examples, pred_indices, strict=True):
            dataset_label = int(example["label"])
            true_label = _index_to_true_label(pred_idx)
            out.append(
                {
                    "row": row_idx,
                    "true_label": true_label,
                    "true_label_index": int(pred_idx),
                    "dataset_label": dataset_label,
                    "dataset_label_name": label_names[dataset_label],
                    "prediction_matches_dataset": true_label == normalize_prediction_label(label_names[dataset_label]),
                }
            )
        append_label_rows(out_manifest, out)
        processed += len(out)
        batch_rows.clear()
        batch_tensors.clear()
        batch_examples.clear()

    batch_examples: list[dict[str, Any]] = []

    for row_idx in range(args.start_row, end_row):
        if row_idx in done:
            continue
        example = dataset[row_idx]
        batch_rows.append(row_idx)
        batch_examples.append(example)
        batch_tensors.append(_example_to_chw(example))
        if len(batch_rows) >= args.batch_size:
            flush_batch()
            if processed and processed % args.log_every == 0:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                print(f"  cached={len(done) + processed}/{total_rows} rate={rate:.1f} rows/s")

    flush_batch()

    write_meta(
        out_meta,
        {
            **default_meta(dataset_rows=total_rows, dataset_version=version),
            "cached_rows": len(load_label_manifest(out_manifest)),
            "completed": len(load_label_manifest(out_manifest)) >= total_rows,
        },
    )
    elapsed = time.time() - t0
    print(
        f"Done: wrote {processed} rows to {out_manifest} "
        f"({len(load_label_manifest(out_manifest))}/{total_rows} total cached, {elapsed:.1f}s)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted — rerun to resume from manifest.", file=sys.stderr)
        raise SystemExit(130) from None
