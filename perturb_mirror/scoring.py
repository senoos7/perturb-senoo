"""Mirror of PerturbValidator.verify_and_score() and related helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturb_mirror import constants as C
from perturb_mirror.image_io import decode_image_b64
from perturb_mirror.model import normalize_prediction_label, predict_label


@dataclass
class ChallengeSpec:
    """Same fields as neurons/validator.py ChallengeSpec."""

    task_id: str
    model_name: str
    clean_image_b64: str
    true_label: str
    epsilon: float
    norm_type: str
    timeout_seconds: int


@dataclass
class ScoringConfig:
    """Mirrors self.config.perturb fields used during verify_and_score()."""

    min_linf_delta: float = C.MIN_LINF_DELTA
    max_linf_delta: float = C.MAX_LINF_DELTA
    min_ssim: float = C.MIN_SSIM
    min_psnr_db: float = C.MIN_PSNR_DB
    linf_component_weight: float = C.LINF_COMPONENT_WEIGHT
    rmse_component_weight: float = C.RMSE_COMPONENT_WEIGHT


@dataclass
class EvaluationResult:
    """Same fields as neurons/validator.py EvaluationResult."""

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
    """Same SSIM as neurons/validator.py _compute_ssim."""
    if x_clean.ndim != 3 or x_adv.ndim != 3:
        return 0.0
    if x_clean.shape != x_adv.shape:
        return 0.0
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
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    """Same PSNR as neurons/validator.py _compute_psnr_db."""
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def verify_and_score(
    model: torch.nn.Module,
    device: torch.device,
    challenge: ChallengeSpec,
    perturbed_image_b64: str,
    response_time_ms: int,
    config: ScoringConfig | None = None,
) -> EvaluationResult:
    """
    Local mirror of PerturbValidator.verify_and_score().
    """
    cfg = config or ScoringConfig()

    try:
        x_clean = decode_image_b64(challenge.clean_image_b64).to(device)
        x_adv = decode_image_b64(perturbed_image_b64).to(device)
    except Exception as exc:
        return EvaluationResult(score=0.0, reason=f"decode_failed:{exc}", response_time_ms=response_time_ms)

    if x_adv.shape != x_clean.shape:
        return EvaluationResult(score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms)
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return EvaluationResult(score=0.0, reason="value_out_of_range", response_time_ms=response_time_ms)

    prediction = ""
    try:
        prediction = predict_label(model, x_adv)
    except Exception as exc:
        return EvaluationResult(
            score=0.0,
            reason=f"model_inference_failed:{exc}",
            response_time_ms=response_time_ms,
        )

    if challenge.norm_type == "Linf":
        norm = (x_adv - x_clean).abs().max().item()
    elif challenge.norm_type == "L2":
        norm = float((x_adv - x_clean).norm(2).item())
    else:
        norm = float((x_adv - x_clean).ne(0).sum().item())

    if norm < cfg.min_linf_delta:
        return EvaluationResult(
            score=0.0,
            reason="below_min_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            epsilon=float(challenge.epsilon),
        )

    effective_max_delta = min(float(challenge.epsilon), float(cfg.max_linf_delta))
    if norm > effective_max_delta:
        return EvaluationResult(
            score=0.0,
            reason="above_max_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=float(challenge.epsilon),
        )

    normalized_prediction = normalize_prediction_label(prediction)
    if normalized_prediction == challenge.true_label:
        return EvaluationResult(
            score=0.0,
            reason="label_match_with_original",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=float(challenge.epsilon),
        )

    rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())

    min_ssim = float(cfg.min_ssim)
    ssim = compute_ssim(x_clean=x_clean, x_adv=x_adv)
    if ssim < min_ssim:
        return EvaluationResult(
            score=0.0,
            reason="below_min_ssim",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(rmse),
            epsilon=float(challenge.epsilon),
            ssim=float(ssim),
        )

    min_psnr_db = float(cfg.min_psnr_db)
    psnr_db = compute_psnr_db(x_clean=x_clean, x_adv=x_adv)
    if min_psnr_db > 0.0 and psnr_db < min_psnr_db:
        return EvaluationResult(
            score=0.0,
            reason="below_min_psnr_db",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=float(norm),
            rmse=float(rmse),
            epsilon=float(challenge.epsilon),
            ssim=float(ssim),
            psnr_db=float(psnr_db),
        )

    denom = max(1e-12, effective_max_delta - float(cfg.min_linf_delta))
    linf_ratio = (norm - float(cfg.min_linf_delta)) / denom
    linf_ratio = min(max(linf_ratio, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2

    rmse_ratio = rmse / max(1e-12, effective_max_delta)
    rmse_ratio = min(max(rmse_ratio, 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2

    linf_weight = float(cfg.linf_component_weight)
    rmse_weight = float(cfg.rmse_component_weight)
    total_weight = max(1e-12, linf_weight + rmse_weight)
    perturbation_score = ((linf_weight * linf_score) + (rmse_weight * rmse_score)) / total_weight

    time_ratio = response_time_ms / (challenge.timeout_seconds * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)

    score = C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score
    return EvaluationResult(
        score=float(score),
        reason="success",
        model_prediction=normalized_prediction,
        response_time_ms=response_time_ms,
        norm=float(norm),
        rmse=float(rmse),
        epsilon=float(challenge.epsilon),
        ssim=float(ssim),
        psnr_db=float(psnr_db),
    )
