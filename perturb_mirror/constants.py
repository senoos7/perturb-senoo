"""Mirror of perturbnet/constants.py (scoring + ImageNet-100 challenge source)."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


SUBNET_NAMESPACE = "perturb"
MODEL_NAME = "EfficientNetV2-L"

# Challenge dataset: full ImageNet-100 train split from Hugging Face.
IMAGENET100_REPO_ID = "clane9/imagenet-100"
IMAGENET100_SPLIT = "train"

TIMEOUT_SECONDS = _env_int("PERTURB_TIMEOUT_SECONDS", 20)
MIN_LINF_DELTA = _env_float("PERTURB_MIN_LINF_DELTA", 0.003)
MAX_LINF_DELTA = _env_float("PERTURB_MAX_LINF_DELTA", 0.03)
MIN_SSIM = _env_float("PERTURB_MIN_SSIM", 0.98)
MIN_PSNR_DB = _env_float("PERTURB_MIN_PSNR_DB", 38.0)
LINF_COMPONENT_WEIGHT = _env_float("PERTURB_LINF_COMPONENT_WEIGHT", 0.7)
RMSE_COMPONENT_WEIGHT = _env_float("PERTURB_RMSE_COMPONENT_WEIGHT", 0.3)

SPEED_WEIGHT = _env_float("PERTURB_SPEED_WEIGHT", 0)
PERTURBATION_WEIGHT = _env_float("PERTURB_PERTURBATION_WEIGHT", 1)
