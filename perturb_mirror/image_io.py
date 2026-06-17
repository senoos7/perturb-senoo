"""Mirror of perturbnet/image_io.py (verbatim)."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image


def decode_image_b64_to_numpy(image_b64: str) -> np.ndarray:
    """PIL RGB float32 CHW in [0, 1] — same values as decode_image_b64()."""
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1)).copy()


def decode_image_b64(image_b64: str) -> torch.Tensor:
    return torch.from_numpy(decode_image_b64_to_numpy(image_b64)).contiguous()


def encode_image_b64(image_chw: torch.Tensor) -> str:
    clipped = image_chw.detach().cpu().clamp(0.0, 1.0)
    arr = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
