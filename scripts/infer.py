#!/usr/bin/env python3
"""Run EfficientNet-V2-L inference exactly like the Perturb validator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from challenge_io import file_to_b64, imagenet100_row_to_b64, infer_true_label, load_imagenet100_dataset
from paths import OUTPUTS
from perturb_mirror.image_io import decode_image_b64, decode_image_b64_to_numpy
from perturb_mirror.model import (
    PREPROCESS,
    _preprocess_for_efficientnet_v2_l,
    load_efficientnet_v2_l,
    predict_label,
)


def _save_preprocessed_png(preprocessed_bchw: torch.Tensor, out_path: Path) -> None:
    mean = torch.tensor(PREPROCESS.mean, dtype=preprocessed_bchw.dtype, device=preprocessed_bchw.device)
    std = torch.tensor(PREPROCESS.std, dtype=preprocessed_bchw.dtype, device=preprocessed_bchw.device)
    vis = preprocessed_bchw[0].detach()
    vis = (vis * std.view(3, 1, 1) + mean.view(3, 1, 1)).clamp(0.0, 1.0).cpu()
    arr = (vis.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(out_path, format="PNG")


def infer_from_b64(image_b64: str, stem: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)

    image = decode_image_b64(image_b64).to(device)

    decoded_path = OUTPUTS / f"{stem}_decoded.npy"
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(decoded_path, decode_image_b64_to_numpy(image_b64))
    print(f"saved decoded       : {decoded_path}")

    print(f"decoded tensor      : shape={tuple(image.shape)} dtype={image.dtype} "
          f"min={image.min():.4f} max={image.max():.4f}")

    preprocessed = _preprocess_for_efficientnet_v2_l(image.unsqueeze(0))
    print(f"preprocessed tensor : shape={tuple(preprocessed.shape)} dtype={preprocessed.dtype} "
          f"min={preprocessed.min():.4f} max={preprocessed.max():.4f}")
    print(f"PREPROCESS object   : {PREPROCESS}")

    out_path = OUTPUTS / f"{stem}_preprocessed.png"
    _save_preprocessed_png(preprocessed, out_path)
    print(f"saved preprocessed  : {out_path}")

    predicted = predict_label(model, image)
    true_label = infer_true_label(image_b64, model, device)
    print(f"predicted label     : {predicted!r}")
    print(f"true_label          : {true_label!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validator-style EfficientNet inference")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("image_path", nargs="?", help="Local image path")
    source.add_argument("--imagenet100-row", type=int, help="ImageNet-100 train row index")
    args = parser.parse_args()

    if args.imagenet100_row is not None:
        dataset = load_imagenet100_dataset()
        row = int(args.imagenet100_row)
        image_id, image_b64 = imagenet100_row_to_b64(dataset, row)
        print(f"imagenet100 row     : {row}")
        print(f"image_id            : {image_id}")
        infer_from_b64(image_b64, image_id)
    else:
        path = Path(args.image_path)
        infer_from_b64(file_to_b64(path), path.stem)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
