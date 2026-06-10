"""
SMA-MI-FGSM: Sparse Momentum Attack with Momentum Iterative FGSM.

Untargeted L∞ white-box attack for the Perturb subnet (EfficientNetV2-L).

GPU profile
-----------
Set PERTURB_GPU_PROFILE=on on cloud GPU servers (e.g. RTX 5090).
Defaults then use higher sparse search / restart budget tuned for ~15s timeout.

Time budget
-----------
Set PERTURB_ATTACK_BUDGET_MS (default 80% of PERTURB_TIMEOUT_SECONDS on GPU).
The attack stops early when the budget is exhausted or when a successful
perturbation reaches the 8-bit L∞ floor (1/255).

Acronym breakdown
-----------------
S  — Sparse: saliency-guided pixel flips and masked gradients
M  — Momentum: MI-FGSM velocity accumulation (Dong et al., 2018)
A  — Attack pipeline with random restarts
MI — Momentum Iterative Fast Gradient Sign Method
FGSM — projected L∞ gradient-sign steps

Post-processing
---------------
- Binary shrink: minimize L∞ while keeping misclassification
- Delta prune: zero low-magnitude spatial changes to reduce RMSE
- 8-bit quantize snap: match validator PNG round-trip
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import logits_for_images, predict_index

ALGORITHM_NAME = "SMA-MI-FGSM"
_ONE_STEP = 1.0 / 255.0
_OPTIMAL_LINF = _ONE_STEP + 1e-6


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _gpu_profile_enabled() -> bool:
    mode = os.getenv("PERTURB_GPU_PROFILE", "auto").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return True
    if mode in {"0", "false", "no", "off"}:
        return False
    return torch.cuda.is_available()


def _profile_int(name: str, cpu_default: int, gpu_default: int) -> int:
    raw = os.getenv(name)
    if raw is not None and raw.strip():
        return int(raw)
    return gpu_default if _gpu_profile_enabled() else cpu_default


def _profile_float(name: str, cpu_default: float, gpu_default: float) -> float:
    raw = os.getenv(name)
    if raw is not None and raw.strip():
        return float(raw)
    return gpu_default if _gpu_profile_enabled() else cpu_default


def _attack_budget_ms() -> int | None:
    raw = os.getenv("PERTURB_ATTACK_BUDGET_MS")
    if raw is not None and raw.strip():
        value = int(raw)
        return None if value <= 0 else value
    if not _gpu_profile_enabled():
        return None
    timeout_s = int(os.getenv("PERTURB_TIMEOUT_SECONDS", "15"))
    return int(timeout_s * 1000 * 0.8)


# --- tunables (env); GPU profile uses higher search budget ------------------
_DEFAULT_STEPS = _profile_int("PERTURB_ATTACK_STEPS", cpu_default=40, gpu_default=80)
_DEFAULT_RESTARTS = _profile_int("PERTURB_ATTACK_RESTARTS", cpu_default=2, gpu_default=5)
_DEFAULT_MOMENTUM = _profile_float("PERTURB_ATTACK_MOMENTUM", cpu_default=0.9, gpu_default=0.9)
_MAX_LINF_DELTA = float(os.getenv("PERTURB_MAX_LINF_DELTA", "0.03"))
_SHRINK_ITERATIONS = _profile_int("PERTURB_ATTACK_SHRINK_ITERS", cpu_default=16, gpu_default=24)
_SPARSE_TOPK = _profile_int("PERTURB_SPARSE_TOPK", cpu_default=512, gpu_default=2048)
_SALIENCY_FRACTION = _profile_float("PERTURB_SALIENCY_FRACTION", cpu_default=0.05, gpu_default=0.01)
_PRUNE_CHUNKS = _profile_int("PERTURB_PRUNE_CHUNKS", cpu_default=64, gpu_default=256)
_QUANTIZE_FINAL = _env_bool("PERTURB_QUANTIZE_FINAL", default=True)
_EARLY_STOP_OPTIMAL = _env_bool("PERTURB_ATTACK_EARLY_STOP", default=True)
_ATTACK_BUDGET_MS = _attack_budget_ms()


@dataclass
class _AttackBudget:
    deadline: float | None
    early_stop_optimal: bool

    @classmethod
    def from_env(cls) -> _AttackBudget:
        ms = _ATTACK_BUDGET_MS
        deadline = time.perf_counter() + ms / 1000.0 if ms is not None else None
        return cls(deadline=deadline, early_stop_optimal=_EARLY_STOP_OPTIMAL)

    def exceeded(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def should_stop(self, quality: tuple[float, float] | None) -> bool:
        if self.exceeded():
            return True
        if self.early_stop_optimal and _is_optimal_quality(quality):
            return True
        return False


# --- metrics ----------------------------------------------------------------
def _effective_epsilon(epsilon: float) -> float:
    return min(float(epsilon), _MAX_LINF_DELTA)


def _linf_delta(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


def _is_optimal_quality(quality: tuple[float, float] | None) -> bool:
    if quality is None:
        return False
    return quality[0] <= _OPTIMAL_LINF


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


def _snap_8bit(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
) -> torch.Tensor:
    if not _QUANTIZE_FINAL:
        return adv
    snapped = quantize_chw_to_8bit(adv).to(clean.device)
    if _is_success(model, clean, snapped, target_index, min_delta):
        return snapped
    return adv


# --- S: sparse saliency helpers ---------------------------------------------
def _saliency_order(clean: torch.Tensor, model: torch.nn.Module, target_index: int) -> torch.Tensor:
    probe = clean.clone().detach().requires_grad_(True)
    logits = logits_for_images(model=model, image_bchw=probe.unsqueeze(0))
    loss = F.cross_entropy(logits, torch.tensor([target_index], device=clean.device))
    grad = torch.autograd.grad(loss, probe)[0]
    return grad.abs().flatten().argsort(descending=True)


def _build_saliency_mask_from_order(order: torch.Tensor, shape: torch.Size, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(order.numel(), device=device)
    fraction = max(1e-6, min(1.0, _SALIENCY_FRACTION))
    keep = max(1, int(order.numel() * fraction))
    mask[order[:keep]] = 1.0
    return mask.view(*shape)


def _try_sparse_pixel_attack(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    order: torch.Tensor,
    budget: _AttackBudget,
) -> torch.Tensor | None:
    """Single-pixel ±1/255 search on top-K salient locations (lowest RMSE path)."""
    channels, height, width = clean.shape
    topk = min(_SPARSE_TOPK, order.numel())
    best: torch.Tensor | None = None
    best_quality: tuple[float, float] | None = None

    for rank in range(topk):
        if budget.exceeded():
            break
        flat_idx = int(order[rank].item())
        channel = flat_idx // (height * width)
        spatial = flat_idx % (height * width)
        row = spatial // width
        col = spatial % width
        for direction in (-1.0, 1.0):
            candidate = clean.clone()
            candidate[channel, row, col] = (clean[channel, row, col] + direction * _ONE_STEP).clamp(0.0, 1.0)
            candidate = _snap_8bit(
                clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
            )
            quality = _attack_quality(model, clean, candidate, target_index, min_delta)
            if _better_quality(quality, best_quality):
                best = candidate
                best_quality = quality
                if _is_optimal_quality(quality):
                    return best
    return best


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
    min_delta: float,
    budget: _AttackBudget,
) -> torch.Tensor:
    adv = clean.clone().detach()
    if noise_scale > 0.0:
        adv = adv + torch.empty_like(adv).uniform_(-noise_scale, noise_scale)
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

    velocity = torch.zeros_like(clean)
    target = torch.tensor([target_index], device=clean.device)

    for _ in range(steps):
        if budget.exceeded():
            break
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

        if _is_success(model, clean, adv, target_index, min_delta=min_delta):
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
    budget: _AttackBudget,
) -> torch.Tensor:
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv

    delta = adv - clean
    lo, hi = 0.0, 1.0
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)

    for _ in range(_SHRINK_ITERATIONS):
        if budget.exceeded():
            break
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
                if _is_optimal_quality(best_quality):
                    break
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
    budget: _AttackBudget,
) -> torch.Tensor:
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv

    delta = adv - clean
    height, width = delta.shape[1], delta.shape[2]
    spatial = delta.abs().amax(dim=0).flatten()
    order = spatial.argsort()
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)
    chunk = max(1, (height * width) // _PRUNE_CHUNKS)

    if _is_optimal_quality(best_quality):
        return best

    for zero_count in range(chunk, height * width, chunk):
        if budget.exceeded():
            break
        keep = torch.ones(height * width, device=clean.device)
        keep[order[:zero_count]] = 0.0
        keep = keep.view(height, width)
        candidate = (clean + delta * keep.unsqueeze(0)).clamp(0.0, 1.0)
        quality = _attack_quality(model, clean, candidate, target_index, min_delta)
        if _better_quality(quality, best_quality):
            best = candidate
            best_quality = quality
            if _is_optimal_quality(best_quality):
                break
        elif zero_count > chunk:
            break

    return best


def _refine_candidate(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    budget: _AttackBudget,
) -> torch.Tensor:
    """Shrink → prune → shrink → 8-bit snap (RMSE-focused post-processing)."""
    if budget.exceeded():
        return adv
    candidate = _shrink_perturbation(
        clean=clean, adv=adv, model=model, target_index=target_index, min_delta=min_delta, budget=budget
    )
    if budget.should_stop(_attack_quality(model, clean, candidate, target_index, min_delta)):
        return _snap_8bit(
            clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
        )
    candidate = _prune_perturbation(
        clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta, budget=budget
    )
    if budget.should_stop(_attack_quality(model, clean, candidate, target_index, min_delta)):
        return _snap_8bit(
            clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
        )
    candidate = _shrink_perturbation(
        clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta, budget=budget
    )
    return _snap_8bit(
        clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
    )


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


def _finalize(
    *,
    clean: torch.Tensor,
    best: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    best = _snap_8bit(
        clean=clean, adv=best, model=model, target_index=target_index, min_delta=min_delta
    )
    return best.to(device), _linf_delta(clean, best), _rmse(clean, best), predict_index(model=model, image_chw=best)


# --- public entry point -----------------------------------------------------
def run_sma_mi_fgsm(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    """
    Full SMA-MI-FGSM pipeline. Returns (adv, linf, rmse, pred_idx).

    Stops early when PERTURB_ATTACK_BUDGET_MS is exhausted or when a successful
    candidate reaches the 8-bit L∞ floor (1/255).
    """
    budget = _AttackBudget.from_env()
    effective_eps = _effective_epsilon(epsilon)
    step_size = max(effective_eps / 10.0, _ONE_STEP)
    steps = max(10, _DEFAULT_STEPS)
    restarts = max(1, _DEFAULT_RESTARTS)

    best = clean.clone()
    best_quality: tuple[float, float] | None = None
    best_pred = target_index

    saliency_order = _saliency_order(clean, model, target_index)
    saliency_mask = _build_saliency_mask_from_order(saliency_order, clean.shape, clean.device)

    sparse = _try_sparse_pixel_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        order=saliency_order,
        budget=budget,
    )
    if sparse is not None:
        sparse = _refine_candidate(
            clean=clean,
            adv=sparse,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            budget=budget,
        )
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
        if budget.should_stop(best_quality):
            return _finalize(
                clean=clean,
                best=best,
                model=model,
                target_index=target_index,
                min_delta=min_delta,
                device=device,
            )

    for restart in range(restarts):
        if budget.should_stop(best_quality):
            break
        noise_scale = effective_eps if restart > 0 else 0.0
        raw = _mi_fgsm_untargeted(
            clean=clean,
            model=model,
            target_index=target_index,
            epsilon=effective_eps,
            steps=steps,
            step_size=step_size,
            momentum=_DEFAULT_MOMENTUM,
            noise_scale=noise_scale,
            saliency_mask=saliency_mask if restart == 0 else None,
            min_delta=min_delta,
            budget=budget,
        )
        if budget.exceeded():
            break
        candidate = _refine_candidate(
            clean=clean,
            adv=raw,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            budget=budget,
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
        if budget.should_stop(best_quality):
            break

    return _finalize(
        clean=clean,
        best=best,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        device=device,
    )


def runtime_profile_summary() -> str:
    gpu = _gpu_profile_enabled()
    budget = "unlimited" if _ATTACK_BUDGET_MS is None else f"{_ATTACK_BUDGET_MS}ms"
    return (
        f"gpu_profile={gpu} budget={budget} early_stop={_EARLY_STOP_OPTIMAL} "
        f"steps={_DEFAULT_STEPS} restarts={_DEFAULT_RESTARTS} "
        f"sparse_topk={_SPARSE_TOPK} saliency_frac={_SALIENCY_FRACTION} "
        f"shrink_iters={_SHRINK_ITERATIONS} prune_chunks={_PRUNE_CHUNKS} quantize={_QUANTIZE_FINAL}"
    )
