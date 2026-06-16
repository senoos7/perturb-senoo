from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64
from perturbnet.model import load_efficientnet_v2_l, normalize_prediction_label, predict_label


@dataclass
class ScoreResult:
    score: float
    reason: str
    model_prediction: str = ""
    response_time_ms: int = 0
    norm: float = 0.0
    rmse: float = 0.0
    epsilon: float = 0.0
    ssim: float = 0.0
    psnr_db: float = 0.0


def compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + 1e-12)).mean().item())


def compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def verify_and_score(
    *,
    model: torch.nn.Module,
    device: torch.device,
    clean_image_b64: str,
    perturbed_image_b64: str,
    true_label: str,
    epsilon: float,
    response_time_ms: int,
    timeout_seconds: int | None = None,
) -> ScoreResult:
    """Mirror of neurons/validator.py verify_and_score (Linf path)."""
    timeout_s = int(timeout_seconds if timeout_seconds is not None else C.TIMEOUT_SECONDS)
    try:
        x_clean = decode_image_b64(clean_image_b64).to(device)
        x_adv = decode_image_b64(perturbed_image_b64).to(device)
    except Exception as exc:
        return ScoreResult(score=0.0, reason=f"decode_failed:{exc}", response_time_ms=response_time_ms)

    if x_adv.shape != x_clean.shape:
        return ScoreResult(score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms)
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return ScoreResult(score=0.0, reason="value_out_of_range", response_time_ms=response_time_ms)

    try:
        prediction = predict_label(model, x_adv)
    except Exception as exc:
        return ScoreResult(score=0.0, reason=f"model_inference_failed:{exc}", response_time_ms=response_time_ms)

    norm = float((x_adv - x_clean).abs().max().item())
    if norm < C.MIN_LINF_DELTA:
        return ScoreResult(
            score=0.0,
            reason="below_min_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            epsilon=epsilon,
        )

    effective_max_delta = min(float(epsilon), C.MAX_LINF_DELTA)
    if norm > effective_max_delta:
        return ScoreResult(
            score=0.0,
            reason="above_max_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=epsilon,
        )

    normalized_prediction = normalize_prediction_label(prediction)
    if normalized_prediction == normalize_prediction_label(true_label):
        return ScoreResult(
            score=0.0,
            reason="label_match_with_original",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=epsilon,
        )

    rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())
    ssim = compute_ssim(x_clean, x_adv)
    if ssim < C.MIN_SSIM:
        return ScoreResult(
            score=0.0,
            reason="below_min_ssim",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=epsilon,
            ssim=ssim,
        )

    psnr_db = compute_psnr_db(x_clean, x_adv)
    if C.MIN_PSNR_DB > 0.0 and psnr_db < C.MIN_PSNR_DB:
        return ScoreResult(
            score=0.0,
            reason="below_min_psnr_db",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=epsilon,
            ssim=ssim,
            psnr_db=psnr_db,
        )

    denom = max(1e-12, effective_max_delta - C.MIN_LINF_DELTA)
    linf_ratio = min(max((norm - C.MIN_LINF_DELTA) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_weight = max(1e-12, C.LINF_COMPONENT_WEIGHT + C.RMSE_COMPONENT_WEIGHT)
    perturbation_score = (
        (C.LINF_COMPONENT_WEIGHT * linf_score) + (C.RMSE_COMPONENT_WEIGHT * rmse_score)
    ) / total_weight
    time_ratio = response_time_ms / (timeout_s * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)
    score = C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score

    return ScoreResult(
        score=float(score),
        reason="success",
        model_prediction=normalized_prediction,
        response_time_ms=response_time_ms,
        norm=norm,
        rmse=rmse,
        epsilon=epsilon,
        ssim=ssim,
        psnr_db=psnr_db,
    )


def load_scoring_model(device: torch.device) -> torch.nn.Module:
    return load_efficientnet_v2_l(device)
