#!/usr/bin/env python3
"""Export ImageNet-100 rows for offline testing (validator challenge source)."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import torch

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from challenge_io import imagenet100_row_to_b64, infer_true_label, load_imagenet100_dataset
from paths import IMAGENET100_SAMPLES_DIR
from perturb_mirror.model import load_efficientnet_v2_l


def main() -> int:
    parser = argparse.ArgumentParser(description="Export ImageNet-100 challenge images for local testing")
    parser.add_argument("--row", type=int, default=0, help="Dataset row index (default 0)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=IMAGENET100_SAMPLES_DIR,
        help="Directory for exported JPEG + metadata",
    )
    parser.add_argument("--infer", action="store_true", help="Run EfficientNet and print true_label")
    args = parser.parse_args()

    dataset = load_imagenet100_dataset()
    row = int(args.row)
    if row < 0 or row >= int(dataset.num_rows):
        print(f"row must be in [0, {int(dataset.num_rows) - 1}]", file=sys.stderr)
        return 2

    image_id, clean_b64 = imagenet100_row_to_b64(dataset, row)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    jpeg_path = out_dir / f"{image_id}.jpg"
    jpeg_path.write_bytes(base64.b64decode(clean_b64))

    print(f"row                 : {row}")
    print(f"image_id            : {image_id}")
    print(f"saved               : {jpeg_path}")
    print(f"dataset rows        : {int(dataset.num_rows)}")

    if args.infer:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_efficientnet_v2_l(device)
        true_label = infer_true_label(clean_b64, model, device)
        print(f"true_label          : {true_label!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
