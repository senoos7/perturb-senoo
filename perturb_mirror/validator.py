"""Validator challenge build + miner response scoring (mirrors neurons/validator.py)."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from perturb_mirror import constants as C
from perturb_mirror.image_io import decode_image_b64, encode_image_b64
from perturb_mirror.model import (
    load_efficientnet_v2_l,
    logits_for_images,
    normalize_prediction_label,
    predict_index,
    predict_label,
    resolve_target_index,
)
from perturb_mirror.scoring import ChallengeSpec, EvaluationResult, ScoringConfig, verify_and_score


def sample_epsilon(seed: int) -> float:
    """Same as PerturbValidator._sample_epsilon(): deterministic in [0.06, 0.2]."""
    return 0.06 + (seed % 1400) / 10000.0


def derive_true_label(clean_image_b64: str, model: torch.nn.Module, device: torch.device) -> str:
    """Same label derivation as generate_challenge() after model inference."""
    image = decode_image_b64(clean_image_b64).to(device)
    predicted = predict_label(model, image)
    return normalize_prediction_label(predicted)


def build_challenge_spec(
    clean_image_b64: str,
    model: torch.nn.Module,
    device: torch.device,
    epsilon: float,
    task_id: str,
    timeout_seconds: int = C.TIMEOUT_SECONDS,
    norm_type: str = "Linf",
) -> ChallengeSpec:
    """Build a challenge the way the validator does (true_label from EfficientNet, not dataset)."""
    return ChallengeSpec(
        task_id=task_id,
        model_name=C.MODEL_NAME,
        clean_image_b64=clean_image_b64,
        true_label=derive_true_label(clean_image_b64, model, device),
        epsilon=float(epsilon),
        norm_type=norm_type,
        timeout_seconds=int(timeout_seconds),
    )


def baseline_miner_forward(
    clean_image_b64: str,
    true_label: str,
    epsilon: float,
    min_delta: float,
    model: torch.nn.Module,
    device: torch.device,
    norm_type: str = "Linf",
) -> tuple[str, int]:
    """
    Same PGD attack as neurons/miner.py PerturbMiner.forward() for Linf challenges.
    Returns (perturbed_image_b64, elapsed_ms).
    """
    if norm_type != "Linf":
        return clean_image_b64, 0

    clean = decode_image_b64(clean_image_b64).to(device)
    target_index = resolve_target_index(true_label)
    if target_index is None:
        return clean_image_b64, 0

    steps = 10
    step_size = max(float(epsilon) / 4.0, 1.0 / 255.0)
    adv = clean.clone().detach()
    best = adv.clone()
    best_delta = 0.0

    started = time.perf_counter()
    for _ in range(steps):
        adv.requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))
        loss = F.cross_entropy(logits, torch.tensor([target_index], device=device))
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + step_size * grad.sign()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

        pred = predict_index(model=model, image_chw=adv)
        delta = float((adv - clean).abs().max().item())
        if delta > best_delta:
            best = adv.clone()
            best_delta = delta
        if pred != target_index and delta >= float(min_delta):
            best = adv.clone()
            break

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return encode_image_b64(best), elapsed_ms


def score_miner_response(
    challenge: ChallengeSpec,
    perturbed_image_b64: str | None,
    model: torch.nn.Module,
    device: torch.device,
    status_code: int = 200,
    process_time_seconds: float | None = None,
    config: ScoringConfig | None = None,
) -> EvaluationResult:
    """
    Score a miner response the way the validator loop does:
    - response_time_ms from dendrite process_time (or timeout fallback)
    - zero score on missing/error responses before verify_and_score()
    """
    response_time_ms = int((process_time_seconds or challenge.timeout_seconds) * 1000)

    if status_code != 200 or not perturbed_image_b64:
        return EvaluationResult(
            score=0.0,
            reason="response_missing_or_status_error",
            model_prediction="unavailable",
            response_time_ms=response_time_ms,
        )

    return verify_and_score(
        model=model,
        device=device,
        challenge=challenge,
        perturbed_image_b64=perturbed_image_b64,
        response_time_ms=response_time_ms,
        config=config,
    )
