#!/usr/bin/env python3
"""
Step 1 of 2: Save clean_image_b64 for every ImageNet-100 train row.

Pipeline (CPU only, no model):
  HF dataset[row]["image"]
      → convert("RGB")
      → JPEG quality=95 → bytes
      → base64 string

Output: one file per image
  data/imagenet100_samples/{row:07d}.b64   (plain base64 text)

Files are gitignored — local only.

Step 2 (separate script) loads these files and runs EfficientNet inference
in batches, since after PREPROCESS all images become 480×480.

Usage (from my_work/):
  python scripts/save_images.py
  python scripts/save_images.py --workers 8
  python scripts/save_images.py --resume
  python scripts/save_images.py --limit 1000
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import IMAGENET100_SAMPLES_DIR, IMAGENET100_VAL_SAMPLES_DIR
from perturb_mirror.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT
from perturb_mirror.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100


def samples_dir_for_split(hf_split: str) -> Path:
    if hf_split == "train":
        return IMAGENET100_SAMPLES_DIR
    if hf_split in ("validation", "val"):
        return IMAGENET100_VAL_SAMPLES_DIR
    raise ValueError(f"unknown split: {hf_split}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 1: save clean_image_b64 files")
    p.add_argument(
        "--hf-split",
        choices=["train", "validation"],
        default="train",
        help="Hugging Face split to export (default: train)",
    )
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel threads for JPEG encode (default: 4)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max rows to process (0 = full split)")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows whose .b64 file already exists")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: data/imagenet100_*_samples by split)")
    p.add_argument("--log-every", type=int, default=1000)
    return p.parse_args()


def row_to_b64(pil_image) -> str:
    """PIL → RGB → JPEG quality=95 → base64  (mirrors _imagenet100_image_bytes)"""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_one(dataset, row: int, out_dir: Path) -> int:
    """Encode one row and write its .b64 file. Returns row on success."""
    pil_img = dataset[row]["image"]
    b64 = row_to_b64(pil_img)
    (out_dir / f"{row:07d}.b64").write_text(b64, encoding="utf-8")
    return row


def main() -> int:
    args = parse_args()
    hf_split = args.hf_split
    args.out_dir = args.out_dir or samples_dir_for_split(hf_split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading ImageNet-100 dataset …")
    dataset = load_imagenet100(repo_id=IMAGENET100_REPO_ID, split=hf_split)
    version = imagenet100_dataset_version(
        dataset=dataset,
        repo_id=IMAGENET100_REPO_ID,
        split=hf_split,
    )
    total_rows = int(dataset.num_rows)
    print(f"split   : {hf_split}")
    print(f"out_dir : {args.out_dir}")
    print(f"dataset : {total_rows} rows  version={version}")

    rows_to_process = list(range(total_rows))
    if args.resume:
        rows_to_process = [r for r in rows_to_process
                           if not (args.out_dir / f"{r:07d}.b64").exists()]
        print(f"resume  : {total_rows - len(rows_to_process)} already done, "
              f"{len(rows_to_process)} remaining")
    if args.limit > 0:
        rows_to_process = rows_to_process[: args.limit]

    print(f"to write: {len(rows_to_process)} files")
    print(f"workers : {args.workers} threads (CPU, JPEG encode)")

    t0 = time.time()
    written = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, dataset, row, args.out_dir): row
            for row in rows_to_process
        }
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
                written += 1
            except Exception as exc:
                errors += 1
                print(f"  [row {futures[future]}] ERROR: {exc}", file=sys.stderr)

            if args.log_every > 0 and i % args.log_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(rows_to_process) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>7}/{len(rows_to_process)}]  "
                    f"written={written}  errors={errors}  "
                    f"rate={rate:.0f} rows/s  ETA={eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    print(f"\ndone  written={written}  errors={errors}  "
          f"elapsed={elapsed/60:.1f} min")
    print(f"next  : python scripts/run_inference.py --hf-split {hf_split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
