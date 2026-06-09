"""
SMA-MI-FGSM: Sparse Momentum Attack with Momentum Iterative FGSM.

Untargeted L∞ white-box attack for the Perturb subnet (EfficientNetV2-L).

Acronym breakdown
-----------------
S  — Sparse: saliency-guided pixel flips and masked gradients (top-5% pixels)
M  — Momentum: MI-FGSM velocity accumulation (Dong et al., 2018)
A  — Attack pipeline with random restarts
MI — Momentum Iterative Fast Gradient Sign Method
FGSM — projected L∞ gradient-sign steps

Post-processing (subnet-specific)
---------------------------------
- Binary shrink: minimize L∞ while keeping misclassification
- Delta prune: zero low-magnitude spatial changes to reduce RMSE

Selection criterion matches validator scoring priority:
  1. attack success (label flip, norm >= min_delta)
  2. lowest L∞
  3. lowest RMSE (30% of perturbation score)

Future algorithms (MAGI, FMAA, etc.) should live in separate modules, e.g.:
  perturbnet/magi.py, perturbnet/fmaa.py
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from perturbnet.model import logits_for_images, predict_index

ALGORITHM_NAME = "SMA-MI-FGSM"

# --- tunables (env) ---------------------------------------------------------
_DEFAULT_STEPS = int(os.getenv("PERTURB_ATTACK_STEPS", "60"))
_DEFAULT_RESTARTS = int(os.getenv("PERTURB_ATTACK_RESTARTS", "3"))
_DEFAULT_MOMENTUM = float(os.getenv("PERTURB_ATTACK_MOMENTUM", "0.9"))
_MAX_LINF_DELTA = float(os.getenv("PERTURB_MAX_LINF_DELTA", "0.03"))
_SHRINK_ITERATIONS = int(os.getenv("PERTURB_ATTACK_SHRINK_ITERS", "16"))
_SPARSE_TOPK = int(os.getenv("PERTURB_SPARSE_TOPK", "512"))
_ONE_STEP = 1.0 / 255.0  # smallest 8-bit pixel step; practical norm floor


# --- metrics ----------------------------------------------------------------
def _effective_epsilon(epsilon: float) -> float:
    return min(float(epsilon), _MAX_LINF_DELTA)


def _linf_delta(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


def _is_success(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_index: int,
    min_delta: float,
) -> bool:
    if _linf_delta(clean, adv) < min_delta:
        return False
    return predict_index(model=model, image_chw=adv) != target_index


def _attack_quality(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_index: int,
    min_delta: float,
) -> tuple[float, float] | None:
    if not _is_success(model, clean, adv, target_index, min_delta):
        return None
    return _linf_delta(clean, adv), _rmse(clean, adv)


def _better_quality(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a < b  # lexicographic: (L∞, RMSE)


# --- S: sparse saliency helpers ---------------------------------------------
def _saliency_order(clean: torch.Tensor, model: torch.nn.Module, target_index: int) -> torch.Tensor:
    """Gradient magnitude on clean image; high values = most influential pixels."""
    probe = clean.clone().detach().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=probe.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([target_index], device=clean.device))
    grad = torch.autograd.grad(loss, probe)[0]
    return grad.abs().flatten().argsort(descending=True)


def _try_sparse_pixel_attack(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
) -> torch.Tensor | None:
    """
    S component: single-pixel ±1/255 search on top-K salient locations.
    Best-case RMSE when one pixel flip is enough to fool the model.
    """
    order = _saliency_order(clean, model, target_index)
    channels, height, width = clean.shape
    topk = min(_SPARSE_TOPK, order.numel())
    best: torch.Tensor | None = None
    best_quality: tuple[float, float] | None = None

    for rank in range(topk):
        flat_idx = int(order[rank].item())
        channel = flat_idx // (height * width)
        spatial = flat_idx % (height * width)
        row = spatial // width
        col = spatial % width
        for direction in (-1.0, 1.0):
            candidate = clean.clone()
            candidate[channel, row, col] = (clean[channel, row, col] + direction * _ONE_STEP).clamp(0.0, 1.0)
            quality = _attack_quality(model, clean, candidate, target_index, min_delta)
            if _better_quality(quality, best_quality):
                best = candidate
                best_quality = quality
                if quality is not None and quality[0] <= _ONE_STEP + 1e-6:
                    return best
    return best


def _build_saliency_mask(clean: torch.Tensor, model: torch.nn.Module, target_index: int) -> torch.Tensor:
    """Keep gradients only on the top 5% most salient pixels."""
    order = _saliency_order(clean, model, target_index)
    mask = torch.zeros(clean.numel(), device=clean.device)
    keep = max(1, int(clean.numel() * 0.05))
    mask[order[:keep]] = 1.0
    return mask.view_as(clean)


# --- MI-FGSM core -----------------------------------------------------------
def _mi_fgsm_untargeted(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    steps: int,
    step_size: float,
    momentum: float,
    noise_scale: float,
    saliency_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    MI-FGSM: maximize CE loss on true class (untargeted) with momentum updates.

        v_t = μ · v_{t-1} + normalize(∇_x CE(f(x), y))
        x_{t+1} = Π_{L∞,ε}( x_t + α · sign(v_t) )
    """
    adv = clean.clone().detach()
    if noise_scale > 0.0:
        adv = adv + torch.empty_like(adv).uniform_(-noise_scale, noise_scale)
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

    velocity = torch.zeros_like(clean)
    target = torch.tensor([target_index], device=clean.device)

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = logits_for_images(model=model, image_bchw=adv.unsqueeze(0))
        loss = F.cross_entropy(logits, target)
        grad = torch.autograd.grad(loss, adv)[0]
        if saliency_mask is not None:
            grad = grad * saliency_mask
        grad_mean = grad.abs().mean().item()
        if grad_mean > 1e-12:
            grad = grad / grad_mean
        velocity = momentum * velocity + grad
        adv = adv.detach() + step_size * velocity.sign()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

        if _is_success(model, clean, adv, target_index, min_delta=0.0):
            break

    return adv.detach()


# --- binary shrink (minimize L∞) --------------------------------------------
def _shrink_perturbation(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
) -> torch.Tensor:
    """
    Scale delta by α ∈ [0, 1] via binary search to find minimum L∞
    that still misclassifies.
    """
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv

    delta = adv - clean
    lo, hi = 0.0, 1.0
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)

    for _ in range(_SHRINK_ITERATIONS):
        mid = (lo + hi) / 2.0
        candidate = (clean + mid * delta).clamp(0.0, 1.0)
        linf = _linf_delta(clean, candidate)
        if linf < min_delta:
            lo = mid
            continue
        if predict_index(model=model, image_chw=candidate) != target_index:
            hi = mid
            quality = (linf, _rmse(clean, candidate))
            if _better_quality(quality, best_quality):
                best = candidate.clone()
                best_quality = quality
        else:
            lo = mid

    return best


# --- delta prune (minimize RMSE) --------------------------------------------
def _prune_perturbation(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
) -> torch.Tensor:
    """
    Zero perturbation on spatial locations with smallest |delta|, chunk by chunk.
    Keeps misclassification while spreading less noise → lower RMSE.
    """
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv

    delta = adv - clean
    height, width = delta.shape[1], delta.shape[2]
    spatial = delta.abs().amax(dim=0).flatten()
    order = spatial.argsort()
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)
    chunk = max(1, (height * width) // 64)

    for zero_count in range(chunk, height * width, chunk):
        keep = torch.ones(height * width, device=clean.device)
        keep[order[:zero_count]] = 0.0
        keep = keep.view(height, width)
        candidate = (clean + delta * keep.unsqueeze(0)).clamp(0.0, 1.0)
        quality = _attack_quality(model, clean, candidate, target_index, min_delta)
        if _better_quality(quality, best_quality):
            best = candidate
            best_quality = quality
        elif zero_count > chunk:
            break

    return best


# --- candidate selection ----------------------------------------------------
def _consider_candidate(
    *,
    clean: torch.Tensor,
    candidate: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    best: torch.Tensor,
    best_quality: tuple[float, float] | None,
    best_pred: int,
) -> tuple[torch.Tensor, tuple[float, float] | None, int]:
    quality = _attack_quality(model, clean, candidate, target_index, min_delta)
    if _better_quality(quality, best_quality):
        return candidate, quality, predict_index(model=model, image_chw=candidate)
    if best_quality is None:
        linf = _linf_delta(clean, candidate)
        if linf > _linf_delta(clean, best):
            return candidate, best_quality, predict_index(model=model, image_chw=candidate)
    return best, best_quality, best_pred


# --- public entry point -----------------------------------------------------
def run_sma_mi_fgsm(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, int]:
    """
    Full SMA-MI-FGSM pipeline:
      1. sparse ±1/255 saliency search
      2. saliency-masked MI-FGSM (+ random restarts)
      3. binary shrink (L∞)
      4. delta prune (RMSE)
      5. pick best (L∞, RMSE) success
    """
    effective_eps = _effective_epsilon(epsilon)
    step_size = max(effective_eps / 10.0, _ONE_STEP)
    steps = max(10, _DEFAULT_STEPS)
    restarts = max(1, _DEFAULT_RESTARTS)

    best = clean.clone()
    best_quality: tuple[float, float] | None = None
    best_pred = target_index

    sparse = _try_sparse_pixel_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
    )
    if sparse is not None:
        best, best_quality, best_pred = _consider_candidate(
            clean=clean,
            candidate=sparse,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            best=best,
            best_quality=best_quality,
            best_pred=best_pred,
        )

    saliency_mask = _build_saliency_mask(clean, model, target_index)

    for restart in range(restarts):
        noise_scale = effective_eps if restart > 0 else 0.0
        candidate = _mi_fgsm_untargeted(
            clean=clean,
            model=model,
            target_index=target_index,
            epsilon=effective_eps,
            steps=steps,
            step_size=step_size,
            momentum=_DEFAULT_MOMENTUM,
            noise_scale=noise_scale,
            saliency_mask=saliency_mask if restart == 0 else None,
        )
        candidate = _shrink_perturbation(
            clean=clean,
            adv=candidate,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
        )
        candidate = _prune_perturbation(
            clean=clean,
            adv=candidate,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
        )
        best, best_quality, best_pred = _consider_candidate(
            clean=clean,
            candidate=candidate,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            best=best,
            best_quality=best_quality,
            best_pred=best_pred,
        )

    return best.to(device), _linf_delta(clean, best), best_pred
