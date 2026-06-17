#!/usr/bin/env python3
"""
Decode saved .b64 files to float32 CHW numpy arrays ([0, 1]).

Pipeline per row (matches decode_image_b64):
  read {row:07d}.b64
      → base64 decode → JPEG bytes
      → PIL.open().convert("RGB") → float32 / 255
      → CHW numpy
      → save {row:07d}.npy

Usage (from my_work/):
  python scripts/save_decoded_npy.py
  python scripts/save_decoded_npy.py --workers 8 --resume
  python scripts/save_decoded_npy.py --limit 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import IMAGENET100_SAMPLES_DIR, IMAGENET100_VAL_SAMPLES_DIR
from perturb_mirror.image_io import decode_image_b64_to_numpy


def samples_dir_for_split(hf_split: str) -> Path:
    if hf_split == "train":
        return IMAGENET100_SAMPLES_DIR
    if hf_split in ("validation", "val"):
        return IMAGENET100_VAL_SAMPLES_DIR
    raise ValueError(f"unknown split: {hf_split}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode .b64 files to float32 CHW .npy arrays")
    p.add_argument(
        "--hf-split",
        choices=["train", "validation"],
        default="train",
        help="Which prepared split directory to use (default: train)",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="Max files (0 = all)")
    p.add_argument("--resume", action="store_true", help="Skip rows whose .npy already exists")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override samples directory")
    p.add_argument("--log-every", type=int, default=1000)
    return p.parse_args()


def process_one(row: int, data_dir: Path) -> tuple[int, tuple[int, int, int]]:
    b64 = (data_dir / f"{row:07d}.b64").read_text(encoding="utf-8")
    chw = decode_image_b64_to_numpy(b64)
    np.save(data_dir / f"{row:07d}.npy", chw)
    return row, tuple(chw.shape)


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir or samples_dir_for_split(args.hf_split)

    all_b64 = sorted(args.data_dir.glob("???????.b64"))
    if not all_b64:
        print(f"ERROR: no .b64 files in {args.data_dir}")
        print(f"Run step 1 first:  python scripts/save_images.py --hf-split {args.hf_split}")
        return 1

    rows = [int(p.stem) for p in all_b64]
    if args.resume:
        rows = [r for r in rows if not (args.data_dir / f"{r:07d}.npy").exists()]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"split   : {args.hf_split}")
    print(f"data_dir: {args.data_dir}")
    print(f"to write: {len(rows)} .npy files")
    print(f"workers : {args.workers}")

    t0 = time.time()
    written = 0
    errors = 0
    sample_shape: tuple[int, int, int] | None = None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, row, args.data_dir): row for row in rows}
        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            try:
                _, shape = future.result()
                if sample_shape is None:
                    sample_shape = shape
                written += 1
            except Exception as exc:
                errors += 1
                print(f"  [row {row}] ERROR: {exc}", file=sys.stderr)

            if args.log_every > 0 and i % args.log_every == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(rows) - i) / rate if rate > 0 else 0
                print(
                    f"  [{i:>7}/{len(rows)}]  written={written}  errors={errors}  "
                    f"rate={rate:.0f} rows/s  ETA={eta/60:.1f} min"
                )

    elapsed = time.time() - t0
    print(f"\ndone  written={written}  errors={errors}  elapsed={elapsed/60:.1f} min")
    if sample_shape is not None:
        print(f"shape   : {sample_shape}  dtype=float32  range=[0, 1]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
