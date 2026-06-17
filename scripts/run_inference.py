#!/usr/bin/env python3
"""
Step 2 of 2: Run EfficientNet inference on saved .b64 files.

Exact pipeline per image:
  read {row:07d}.b64
      → base64 decode → JPEG bytes
      → PIL.open().convert("RGB") → float [0,1] CHW  (native resolution)
      → PREPROCESS per image individually             (→ 480×480)
      ↓
  [accumulate until batch_size]
      ↓
  model.forward(batch)  → logits [batch_size, 1000]
      ↓
  write {row:07d}.json  {"row": int, "logits": [float×1000]}

PREPROCESS is applied per image BEFORE batching because images have
different native resolutions. After PREPROCESS all are 480×480 → safe to batch.

Usage (from my_work/):
  python scripts/run_inference.py
  python scripts/run_inference.py --batch-size 64 --workers 8
  python scripts/run_inference.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from paths import IMAGENET100_SAMPLES_DIR, IMAGENET100_VAL_SAMPLES_DIR
from perturb_mirror.image_io import decode_image_b64_to_numpy
from perturb_mirror.model import PREPROCESS, load_efficientnet_v2_l


def samples_dir_for_split(hf_split: str) -> Path:
    if hf_split == "train":
        return IMAGENET100_SAMPLES_DIR
    if hf_split in ("validation", "val"):
        return IMAGENET100_VAL_SAMPLES_DIR
    raise ValueError(f"unknown split: {hf_split}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2: EfficientNet inference on .b64 files")
    p.add_argument(
        "--hf-split",
        choices=["train", "validation"],
        default="train",
        help="Which prepared split directory to use (default: train)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4,
                   help="DataLoader workers for parallel decode+PREPROCESS")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows whose .json file already exists")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Override samples directory")
    p.add_argument("--log-every", type=int, default=100,
                   help="Print progress every N batches")
    return p.parse_args()


# ── Dataset ───────────────────────────────────────────────────────────────────

class B64Dataset(Dataset):
    """
    Reads .b64 files and returns (row, preprocessed_tensor).

    Per-image pipeline:
      read .b64 → base64 decode → JPEG bytes
      → PIL.open → RGB → float [0,1] CHW  (native resolution)
      → PREPROCESS (resize+crop to 480×480)
    """

    def __init__(self, rows: list[int], data_dir: Path) -> None:
        self.rows = rows
        self.data_dir = data_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]

        # Step 1: read saved base64 string
        b64 = (self.data_dir / f"{row:07d}.b64").read_text(encoding="utf-8")

        # Step 2: decode → PIL RGB → float [0,1] CHW (native resolution)
        tensor_chw = torch.from_numpy(decode_image_b64_to_numpy(b64)).contiguous()

        # Step 3: PREPROCESS per image (handles variable resolution → 480×480)
        tensor_480 = PREPROCESS(tensor_chw.unsqueeze(0)).squeeze(0)

        return row, tensor_480


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir or samples_dir_for_split(args.hf_split)

    # Collect all .b64 files
    all_b64 = sorted(args.data_dir.glob("???????.b64"))
    if not all_b64:
        print(f"ERROR: no .b64 files found in {args.data_dir}")
        print(f"Run step 1 first:  python scripts/save_images.py --hf-split {args.hf_split}")
        return 1

    rows_all = [int(p.stem) for p in all_b64]

    if args.resume:
        rows_todo = [r for r in rows_all
                     if not (args.data_dir / f"{r:07d}.json").exists()]
        print(f"resume  : {len(rows_all) - len(rows_todo)} already done, "
              f"{len(rows_todo)} remaining")
    else:
        rows_todo = rows_all

    print(f"split   : {args.hf_split}")
    print(f"data_dir: {args.data_dir}")

    print(f"images  : {len(rows_todo)}")
    print(f"batch   : {args.batch_size}")
    print(f"workers : {args.workers}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device  : {device}")

    print("loading EfficientNet-V2-L …")
    model = load_efficientnet_v2_l(device)

    dataset = B64Dataset(rows_todo, args.data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    t0 = time.time()
    written = 0
    errors = 0

    for batch_idx, (rows_batch, tensors_batch) in enumerate(loader, start=1):
        # tensors_batch: [B, 3, 480, 480] — already preprocessed
        tensors_batch = tensors_batch.to(device)

        with torch.no_grad():
            logits_batch = model(tensors_batch)   # [B, 1000]

        # Write one .json per image
        for i, row in enumerate(rows_batch.tolist()):
            try:
                record = {
                    "row": row,
                    "logits": logits_batch[i].cpu().tolist(),
                }
                (args.data_dir / f"{row:07d}.json").write_text(
                    json.dumps(record, separators=(",", ":")),
                    encoding="utf-8",
                )
                written += 1
            except Exception as exc:
                errors += 1
                print(f"  [row {row}] write ERROR: {exc}", file=sys.stderr)

        if args.log_every > 0 and batch_idx % args.log_every == 0:
            elapsed = time.time() - t0
            imgs_done = batch_idx * args.batch_size
            rate = imgs_done / elapsed
            eta = (len(rows_todo) - imgs_done) / rate if rate > 0 else 0
            print(
                f"  [batch {batch_idx:>5}]  "
                f"written={written}  errors={errors}  "
                f"rate={rate:.0f} imgs/s  ETA={eta/60:.1f} min"
            )

    elapsed = time.time() - t0
    print(f"\ndone  written={written}  errors={errors}  "
          f"elapsed={elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
