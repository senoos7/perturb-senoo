from __future__ import annotations

import base64
import io
from typing import Any


def imagenet_row_to_clean_b64(example: dict[str, Any], *, jpeg_quality: int = 95) -> str:
    """Match validator challenge encoding: PIL RGB -> JPEG -> base64."""
    image = example.get("image")
    if image is None:
        raise ValueError("dataset row has no image")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=jpeg_quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
