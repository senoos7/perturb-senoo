#!/usr/bin/env python3
"""
Prepare ImageNet-100 train or validation split (same pipeline as original train prep).

Steps:
  1. save_images.py      HF image -> JPEG q=95 -> {row:07d}.b64
  2. run_inference.py    .b64 -> EfficientNet logits -> {row:07d}.json
  3. save_decoded_npy.py .b64 -> float32 CHW [0,1] -> {row:07d}.npy
  4. build_indexes.py    .npy/.json -> consolidated shape + label caches

Usage (from my_work/):
  python scripts/prepare_dataset.py --split validation --workers 8
  python scripts/prepare_dataset.py --split train --resume
  python scripts/prepare_dataset.py --split validation --indexes-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

MY_WORK = Path(__file__).resolve().parent.parent
SCRIPTS = MY_WORK / "scripts"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full data prep for train or validation split")
    p.add_argument("--split", choices=["train", "validation"], required=True)
    p.add_argument("--workers", type=int, default=8, help="Threads for b64/npy encode")
    p.add_argument("--batch-size", type=int, default=32, help="EfficientNet inference batch size")
    p.add_argument("--inference-workers", type=int, default=4, help="DataLoader workers for inference")
    p.add_argument("--limit", type=int, default=0, help="Max rows (0 = full split)")
    p.add_argument("--resume", action="store_true", help="Skip files that already exist")
    p.add_argument("--indexes-only", action="store_true", help="Only rebuild shape/label caches")
    p.add_argument("--skip-indexes", action="store_true", help="Skip final build_indexes step")
    return p.parse_args()


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(MY_WORK))


def main() -> int:
    args = parse_args()
    py = sys.executable

    if not args.indexes_only:
        save_cmd = [
            py, str(SCRIPTS / "save_images.py"),
            "--hf-split", args.split,
            "--workers", str(args.workers),
        ]
        if args.resume:
            save_cmd.append("--resume")
        if args.limit > 0:
            save_cmd.extend(["--limit", str(args.limit)])
        run_step(f"step 1/4: save .b64 ({args.split})", save_cmd)

        infer_cmd = [
            py, str(SCRIPTS / "run_inference.py"),
            "--hf-split", args.split,
            "--batch-size", str(args.batch_size),
            "--workers", str(args.inference_workers),
        ]
        if args.resume:
            infer_cmd.append("--resume")
        run_step(f"step 2/4: inference -> .json ({args.split})", infer_cmd)

        npy_cmd = [
            py, str(SCRIPTS / "save_decoded_npy.py"),
            "--hf-split", args.split,
            "--workers", str(args.workers),
        ]
        if args.resume:
            npy_cmd.append("--resume")
        if args.limit > 0:
            npy_cmd.extend(["--limit", str(args.limit)])
        run_step(f"step 3/4: decode -> .npy ({args.split})", npy_cmd)

    if not args.skip_indexes:
        run_step(
            f"step 4/4: build indexes ({args.split})",
            [py, str(SCRIPTS / "build_indexes.py"), "--split", args.split],
        )

    print(f"\nall done — {args.split} split ready")
    if args.split == "validation":
        print("train with HF validation set:")
        print("  python train_generator_local.py --use-hf-val ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
