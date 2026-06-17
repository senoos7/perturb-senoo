#!/usr/bin/env python3
"""
Build a local per-image cache of (clean_image_b64, logits) pairs
that exactly match what the validator pipeline produces.

Inference is done ONE IMAGE AT A TIME — exactly as the validator does:
  predict_label(model, image_chw)  →  image_chw.unsqueeze(0)  →  PREPROCESS  →  model

This matters because images have different resolutions; batching would
require padding/resizing to a common size, which the validator never does.

Pipeline per row (mirrors neurons/validator.py dev branch):
  HF dataset[row]["image"]
      → convert("RGB")
      → JPEG quality=95 → bytes
      → base64                            ← clean_image_b64
      → decode_image_b64 → CHW float      (one image, native resolution)
      → image_chw.unsqueeze(0)            (batch size = 1)
      → PREPROCESS → model forward        ← logits (1000-dim float32)

Output: one JSON file per image
  data/imagenet100_samples/{row:07d}.json
  {
    "row": int,
    "image_id": str,
    "clean_image_b64": str,
    "logits": [float, ...]   # 1000 raw EfficientNet values, pre-softmax
  }

Files are gitignored — local only.

Usage (from my_work/):
  python scripts/build_challenge_cache.py --limit 1000
  python scripts/build_challenge_cache.py              # full ~126k
  python scripts/build_challenge_cache.py --resume     # skip existing files
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import IMAGENET100_SAMPLES_DIR
from perturb_mirror.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100
from perturb_mirror.model import (
    load_efficientnet_v2_l,
    logits_for_images,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build per-image validator-matched challenge cache")
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows to process (0 = full dataset, ~126k)")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows whose output file already exists")
    p.add_argument("--out-dir", type=Path, default=IMAGENET100_SAMPLES_DIR,
                   help=f"Output directory (default: {IMAGENET100_SAMPLES_DIR})")
    p.add_argument("--log-every", type=int, default=500,
                   help="Print progress every N rows")
    return p.parse_args()


def row_to_clean_b64(pil_image) -> str:
    """
    Mirrors _imagenet100_image_bytes():
      PIL Image → convert("RGB") → JPEG quality=95 → bytes → base64
    """
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clean_b64_to_chw(clean_b64: str, device: torch.device) -> torch.Tensor:
    """
    Mirrors decode_image_b64():
      base64 → JPEG bytes → PIL.open → RGB → float32 [0,1] CHW

    Returns a single CHW tensor at the image's NATIVE resolution.
    No resizing here — exactly like the validator.
    """
    raw = base64.b64decode(clean_b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device)


def infer_one(model: torch.nn.Module, image_chw: torch.Tensor) -> list[float]:
    """
    Mirrors predict_index() / predict_label() in perturbnet/model.py:
      image_chw.unsqueeze(0)  →  PREPROCESS  →  model  →  logits

    Batch size is exactly 1 — one image at a time, matching the validator.
    """
    with torch.no_grad():
        # unsqueeze(0): CHW → 1CHW  (batch of 1, native resolution)
        logits = logits_for_images(model, image_chw.unsqueeze(0))
    return logits.squeeze(0).cpu().tolist()   # → list of 1000 floats


def out_path(out_dir: Path, row: int) -> Path:
    return out_dir / f"{row:07d}.json"


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device  : {device}")

    print("loading ImageNet-100 dataset …")
    dataset = load_imagenet100()
    version = imagenet100_dataset_version(
        dataset=dataset,
        repo_id="clane9/imagenet-100",
        split="train",
    )
    total_rows = int(dataset.num_rows)
    print(f"dataset : {total_rows} rows  version={version}")

    print("loading EfficientNet-V2-L …")
    model = load_efficientnet_v2_l(device)

    # Determine which rows to process
    rows_to_process = list(range(total_rows))
    if args.resume:
        rows_to_process = [r for r in rows_to_process
                           if not out_path(args.out_dir, r).exists()]
        print(f"resume  : {total_rows - len(rows_to_process)} already done, "
              f"{len(rows_to_process)} remaining")
    if args.limit > 0:
        rows_to_process = rows_to_process[: args.limit]
    print(f"to write: {len(rows_to_process)} files")
    print("inference: 1 image at a time (variable resolution, mirrors validator)")

    t0 = time.time()
    written = 0
    errors = 0

    for i, row in enumerate(rows_to_process, start=1):
        try:
            # Step 1: PIL → JPEG q=95 → base64  (validator: _imagenet100_image_bytes)
            pil_img = dataset[row]["image"]
            clean_b64 = row_to_clean_b64(pil_img)

            # Step 2: base64 → CHW float at native resolution  (validator: decode_image_b64)
            image_chw = clean_b64_to_chw(clean_b64, device)

            # Step 3: CHW → unsqueeze(0) → PREPROCESS → model  (validator: predict_label)
            # Batch size = 1, one image at a time
            logits = infer_one(model, image_chw)

            image_id = f"hf-{version}-{row:07d}"
            record = {
                "row": row,
                "image_id": image_id,
                "clean_image_b64": clean_b64,
                "logits": logits,
            }
            out_path(args.out_dir, row).write_text(
                json.dumps(record, separators=(",", ":")),
                encoding="utf-8",
            )
            written += 1

        except Exception as exc:
            errors += 1
            print(f"  [row {row}] ERROR: {exc}", file=sys.stderr)
            continue

        if args.log_every > 0 and i % args.log_every == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(rows_to_process) - i) / rate if rate > 0 else 0
            print(
                f"  [{i:>7}/{len(rows_to_process)}]  "
                f"written={written}  errors={errors}  "
                f"rate={rate:.1f} rows/s  ETA={eta/60:.1f} min"
            )

    elapsed = time.time() - t0
    print(f"\ndone  written={written}  errors={errors}  "
          f"elapsed={elapsed/60:.1f} min  dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
