"""Load challenge images the way Perturb validators do (ImageNet-100 era)."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch

from perturb_mirror.constants import IMAGENET100_REPO_ID, IMAGENET100_SPLIT
from perturb_mirror.imagenet100_bootstrap import imagenet100_dataset_version, load_imagenet100
from perturb_mirror.validator import derive_true_label


def file_to_b64(path: Path) -> str:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"image is empty: {path}")
    return base64.b64encode(raw).decode("utf-8")


def imagenet100_row_to_b64(dataset, row: int) -> tuple[str, str]:
    """Encode one dataset row like neurons/validator.py _imagenet100_image_bytes."""
    example = dataset[int(row)]
    image = example.get("image")
    if image is None:
        raise ValueError(f"ImageNet-100 row {row} has no image payload")

    version = imagenet100_dataset_version(
        dataset=dataset,
        repo_id=IMAGENET100_REPO_ID,
        split=IMAGENET100_SPLIT,
    )
    image_id = f"hf-{version}-{int(row):07d}"

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    raw = buffer.getvalue()
    if not raw:
        raise ValueError(f"ImageNet-100 row {row} encoded to empty JPEG")
    return image_id, base64.b64encode(raw).decode("utf-8")


def infer_true_label(clean_b64: str, model: torch.nn.Module, device: torch.device) -> str:
    return derive_true_label(clean_b64, model, device)


def load_imagenet100_dataset():
    return load_imagenet100()
