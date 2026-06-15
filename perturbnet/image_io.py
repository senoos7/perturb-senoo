from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def decode_image_b64(image_b64: str) -> torch.Tensor:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def sanitize_task_id_for_filename(task_id: str) -> str:
    safe = re.sub(r"[^\w.\-]+", "_", task_id.strip())
    return (safe[:200] or "unknown").strip("_") or "unknown"


def challenge_image_b64_to_jpeg_bytes(image_b64: str, *, jpeg_quality: int = 95) -> bytes:
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality)
    return buffer.getvalue()


def save_challenge_image_b64(image_b64: str, output_path: Path, *, jpeg_quality: int = 95) -> Path:
    """Decode validator challenge base64 and write a viewable JPEG."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(challenge_image_b64_to_jpeg_bytes(image_b64, jpeg_quality=jpeg_quality))
    return output_path


def encode_image_b64(image_chw: torch.Tensor) -> str:
    clipped = quantize_chw_to_8bit(image_chw)
    arr = (clipped.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def quantize_chw_to_8bit(image_chw: torch.Tensor) -> torch.Tensor:
    """Snap tensor to 8-bit pixel grid (matches validator PNG round-trip)."""
    clipped = image_chw.detach().cpu().clamp(0.0, 1.0)
    arr = (clipped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()

