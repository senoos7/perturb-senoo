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

import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import logits_for_images, predict_index

logger = logging.getLogger(__name__)
ALGORITHM_NAME = "SMA-MI-FGSM"
_ONE_STEP = 1.0 / 255.0
_OPTIMAL_LINF = _ONE_STEP + 1e-6
_PROGRESS_INTERVAL_PCT = 25


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


def _attack_log_mode() -> str:
    raw = os.getenv("PERTURB_ATTACK_LOG", "on").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"verbose", "debug", "v"}:
        return "verbose"
    return "on"


def _format_fields(**fields: object) -> str:
    parts: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.6f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


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


@dataclass
class _AttackContext:
    budget: _AttackBudget
    started_at: float
    log_mode: str = field(default_factory=_attack_log_mode)
    _progress_marks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> _AttackContext:
        return cls(budget=_AttackBudget.from_env(), started_at=time.perf_counter())

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    def remaining_ms(self) -> int | None:
        if self.budget.deadline is None:
            return None
        return max(0, int((self.budget.deadline - time.perf_counter()) * 1000))

    def exceeded(self) -> bool:
        return self.budget.exceeded()

    def should_stop(self, quality: tuple[float, float] | None) -> bool:
        return self.budget.should_stop(quality)

    def _timing_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {"elapsed_ms": self.elapsed_ms()}
        remaining = self.remaining_ms()
        if remaining is not None:
            fields["remaining_ms"] = remaining
        return fields

    def phase(self, name: str, **fields: object) -> None:
        if self.log_mode == "off":
            return
        payload = _format_fields(**self._timing_fields(), **fields)
        logger.info("[SMA-MI-FGSM] phase=%s %s", name, payload)

    def event(self, message: str, **fields: object) -> None:
        if self.log_mode == "off":
            return
        payload = _format_fields(**self._timing_fields(), **fields)
        logger.info("[SMA-MI-FGSM] %s %s", message, payload)

    def progress(self, phase: str, current: int, total: int, **fields: object) -> None:
        if self.log_mode == "off" or total <= 0:
            return
        pct = int(100 * current / total)
        interval = 10 if self.log_mode == "verbose" else _PROGRESS_INTERVAL_PCT
        mark = (pct // interval) * interval
        if mark <= 0 or mark >= 100:
            return
        last = self._progress_marks.get(phase, -1)
        if mark <= last:
            return
        self._progress_marks[phase] = mark
        payload = _format_fields(
            **self._timing_fields(),
            progress_pct=mark,
            current=current,
            total=total,
            **fields,
        )
        logger.info("[SMA-MI-FGSM] phase=%s %s", phase, payload)

    def quality_fields(self, quality: tuple[float, float] | None) -> dict[str, object]:
        if quality is None:
            return {"success": False}
        return {"success": True, "linf": quality[0], "rmse": quality[1]}


def _stop_reason(ctx: _AttackContext, quality: tuple[float, float] | None) -> str:
    if _is_optimal_quality(quality):
        return "optimal_linf"
    if ctx.exceeded():
        return "time_budget"
    return "complete"


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
    ctx: _AttackContext,
) -> tuple[torch.Tensor | None, str]:
    """Single-pixel ±1/255 search on top-K salient locations (lowest RMSE path)."""
    _, height, width = clean.shape
    topk = min(_SPARSE_TOPK, order.numel())
    best: torch.Tensor | None = None
    best_quality: tuple[float, float] | None = None
    stop_reason = "exhausted"

    ctx.phase("sparse_start", topk=topk, pixels=f"{height}x{width}")

    for rank in range(topk):
        if ctx.exceeded():
            stop_reason = "time_budget"
            break
        ctx.progress("sparse", rank + 1, topk, **ctx.quality_fields(best_quality))
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
                ctx.event("sparse_improved", rank=rank + 1, **ctx.quality_fields(best_quality))
                if _is_optimal_quality(quality):
                    stop_reason = "optimal_linf"
                    ctx.event("sparse_optimal_hit", rank=rank + 1, **ctx.quality_fields(best_quality))
                    return best, stop_reason

    if best is None:
        ctx.phase("sparse_done", result="miss")
    else:
        ctx.phase("sparse_done", result="hit", **ctx.quality_fields(best_quality), stop=stop_reason)
    return best, stop_reason


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
    ctx: _AttackContext,
    restart_idx: int,
) -> tuple[torch.Tensor, str]:
    adv = clean.clone().detach()
    masked = saliency_mask is not None
    if noise_scale > 0.0:
        adv = adv + torch.empty_like(adv).uniform_(-noise_scale, noise_scale)
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

    ctx.phase(
        "mi_fgsm_start",
        restart=restart_idx,
        steps=steps,
        masked=masked,
        noise=noise_scale > 0.0,
    )

    velocity = torch.zeros_like(clean)
    target = torch.tensor([target_index], device=clean.device)
    stop_reason = "max_steps"

    for step_idx in range(steps):
        if ctx.exceeded():
            stop_reason = "time_budget"
            break
        ctx.progress("mi_fgsm", step_idx + 1, steps, restart=restart_idx)
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
            stop_reason = "success"
            quality = _attack_quality(model, clean, adv, target_index, min_delta)
            ctx.event(
                "mi_fgsm_success",
                restart=restart_idx,
                step=step_idx + 1,
                **ctx.quality_fields(quality),
            )
            break

    if stop_reason == "max_steps":
        linf = _linf_delta(clean, adv)
        ctx.phase(
            "mi_fgsm_done",
            restart=restart_idx,
            result="no_success",
            linf=linf,
            stop=stop_reason,
        )
    return adv.detach(), stop_reason


# --- binary shrink (minimize L∞) --------------------------------------------
def _shrink_perturbation(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    ctx: _AttackContext,
    label: str,
) -> torch.Tensor:
    if not _is_success(model, clean, adv, target_index, min_delta):
        ctx.phase(f"{label}_skip", reason="not_success")
        return adv

    delta = adv - clean
    lo, hi = 0.0, 1.0
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)
    ctx.phase(f"{label}_start", iters=_SHRINK_ITERATIONS, **ctx.quality_fields(best_quality))

    for iter_idx in range(_SHRINK_ITERATIONS):
        if ctx.exceeded():
            break
        ctx.progress(label, iter_idx + 1, _SHRINK_ITERATIONS, **ctx.quality_fields(best_quality))
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
                    ctx.event(f"{label}_optimal", **ctx.quality_fields(best_quality))
                    break
        else:
            lo = mid

    ctx.phase(f"{label}_done", **ctx.quality_fields(best_quality))
    return best


# --- delta prune (minimize RMSE) --------------------------------------------
def _prune_perturbation(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    ctx: _AttackContext,
    label: str,
) -> torch.Tensor:
    if not _is_success(model, clean, adv, target_index, min_delta):
        ctx.phase(f"{label}_skip", reason="not_success")
        return adv

    delta = adv - clean
    height, width = delta.shape[1], delta.shape[2]
    spatial = delta.abs().amax(dim=0).flatten()
    order = spatial.argsort()
    best = adv
    best_quality = _attack_quality(model, clean, adv, target_index, min_delta)
    chunk = max(1, (height * width) // _PRUNE_CHUNKS)
    total_steps = max(1, len(range(chunk, height * width, chunk)))

    if _is_optimal_quality(best_quality):
        ctx.phase(f"{label}_skip", reason="already_optimal", **ctx.quality_fields(best_quality))
        return best

    ctx.phase(f"{label}_start", chunks=_PRUNE_CHUNKS, chunk_pixels=chunk, **ctx.quality_fields(best_quality))
    step_idx = 0

    for zero_count in range(chunk, height * width, chunk):
        if ctx.exceeded():
            break
        step_idx += 1
        ctx.progress(label, step_idx, total_steps, **ctx.quality_fields(best_quality))
        keep = torch.ones(height * width, device=clean.device)
        keep[order[:zero_count]] = 0.0
        keep = keep.view(height, width)
        candidate = (clean + delta * keep.unsqueeze(0)).clamp(0.0, 1.0)
        quality = _attack_quality(model, clean, candidate, target_index, min_delta)
        if _better_quality(quality, best_quality):
            best = candidate
            best_quality = quality
            ctx.event(f"{label}_improved", zero_pixels=zero_count, **ctx.quality_fields(best_quality))
            if _is_optimal_quality(best_quality):
                ctx.event(f"{label}_optimal", **ctx.quality_fields(best_quality))
                break
        elif zero_count > chunk:
            break

    ctx.phase(f"{label}_done", **ctx.quality_fields(best_quality))
    return best


def _refine_candidate(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    ctx: _AttackContext,
    label: str,
) -> torch.Tensor:
    """Shrink → prune → shrink → 8-bit snap (RMSE-focused post-processing)."""
    ctx.phase(f"{label}_refine_start")
    if ctx.exceeded():
        ctx.event(f"{label}_refine_skip", reason="time_budget")
        return adv
    candidate = _shrink_perturbation(
        clean=clean,
        adv=adv,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        ctx=ctx,
        label=f"{label}_shrink1",
    )
    quality = _attack_quality(model, clean, candidate, target_index, min_delta)
    if ctx.should_stop(quality):
        ctx.event(f"{label}_refine_early_stop", stage="shrink1", **ctx.quality_fields(quality))
        return _snap_8bit(
            clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
        )
    candidate = _prune_perturbation(
        clean=clean,
        adv=candidate,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        ctx=ctx,
        label=f"{label}_prune",
    )
    quality = _attack_quality(model, clean, candidate, target_index, min_delta)
    if ctx.should_stop(quality):
        ctx.event(f"{label}_refine_early_stop", stage="prune", **ctx.quality_fields(quality))
        return _snap_8bit(
            clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
        )
    candidate = _shrink_perturbation(
        clean=clean,
        adv=candidate,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        ctx=ctx,
        label=f"{label}_shrink2",
    )
    candidate = _snap_8bit(
        clean=clean, adv=candidate, model=model, target_index=target_index, min_delta=min_delta
    )
    ctx.phase(
        f"{label}_refine_done",
        **ctx.quality_fields(_attack_quality(model, clean, candidate, target_index, min_delta)),
    )
    return candidate


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
    ctx: _AttackContext,
    stop_reason: str,
) -> tuple[torch.Tensor, float, float, int]:
    best = _snap_8bit(
        clean=clean, adv=best, model=model, target_index=target_index, min_delta=min_delta
    )
    linf = _linf_delta(clean, best)
    rmse = _rmse(clean, best)
    pred_idx = predict_index(model=model, image_chw=best)
    ctx.event(
        "done",
        stop=stop_reason,
        linf=linf,
        rmse=rmse,
        pred_idx=pred_idx,
        target_idx=target_index,
        misclassified=pred_idx != target_index,
    )
    return best.to(device), linf, rmse, pred_idx


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
    ctx = _AttackContext.from_env()
    effective_eps = _effective_epsilon(epsilon)
    step_size = max(effective_eps / 10.0, _ONE_STEP)
    steps = max(10, _DEFAULT_STEPS)
    restarts = max(1, _DEFAULT_RESTARTS)
    stop_reason = "complete"

    ctx.event(
        "start",
        image_shape=tuple(clean.shape),
        target_idx=target_index,
        epsilon=epsilon,
        effective_epsilon=effective_eps,
        min_delta=min_delta,
        steps=steps,
        restarts=restarts,
        budget_ms=_ATTACK_BUDGET_MS if _ATTACK_BUDGET_MS is not None else "unlimited",
        log_mode=ctx.log_mode,
    )

    best = clean.clone()
    best_quality: tuple[float, float] | None = None
    best_pred = target_index

    ctx.phase("saliency_start")
    saliency_order = _saliency_order(clean, model, target_index)
    saliency_mask = _build_saliency_mask_from_order(saliency_order, clean.shape, clean.device)
    ctx.phase("saliency_done", saliency_pixels=saliency_order.numel())

    sparse, _sparse_stop = _try_sparse_pixel_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        order=saliency_order,
        ctx=ctx,
    )
    if sparse is not None:
        sparse = _refine_candidate(
            clean=clean,
            adv=sparse,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            ctx=ctx,
            label="sparse",
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
        ctx.event("best_updated", source="sparse", pred_idx=best_pred, **ctx.quality_fields(best_quality))
        if ctx.should_stop(best_quality):
            stop_reason = _stop_reason(ctx, best_quality)
            ctx.event("early_stop", reason=stop_reason, stage="sparse", **ctx.quality_fields(best_quality))
            return _finalize(
                clean=clean,
                best=best,
                model=model,
                target_index=target_index,
                min_delta=min_delta,
                device=device,
                ctx=ctx,
                stop_reason=stop_reason,
            )

    for restart in range(restarts):
        if ctx.should_stop(best_quality):
            stop_reason = _stop_reason(ctx, best_quality)
            ctx.event("early_stop", reason=stop_reason, stage=f"before_restart_{restart}")
            break
        noise_scale = effective_eps if restart > 0 else 0.0
        raw, mi_stop = _mi_fgsm_untargeted(
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
            ctx=ctx,
            restart_idx=restart,
        )
        if ctx.exceeded():
            stop_reason = "time_budget"
            ctx.event("early_stop", reason=stop_reason, stage=f"restart_{restart}_mi_fgsm", mi_stop=mi_stop)
            break
        candidate = _refine_candidate(
            clean=clean,
            adv=raw,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            ctx=ctx,
            label=f"restart_{restart}",
        )
        prev_quality = best_quality
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
        if best_quality != prev_quality:
            ctx.event(
                "best_updated",
                source=f"restart_{restart}",
                pred_idx=best_pred,
                mi_stop=mi_stop,
                **ctx.quality_fields(best_quality),
            )
        else:
            ctx.phase(f"restart_{restart}_no_improve", mi_stop=mi_stop, **ctx.quality_fields(best_quality))
        if ctx.should_stop(best_quality):
            stop_reason = _stop_reason(ctx, best_quality)
            ctx.event("early_stop", reason=stop_reason, stage=f"restart_{restart}", **ctx.quality_fields(best_quality))
            break

    if stop_reason == "complete" and ctx.exceeded():
        stop_reason = "time_budget"

    return _finalize(
        clean=clean,
        best=best,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        device=device,
        ctx=ctx,
        stop_reason=stop_reason,
    )


def runtime_profile_summary() -> str:
    gpu = _gpu_profile_enabled()
    budget = "unlimited" if _ATTACK_BUDGET_MS is None else f"{_ATTACK_BUDGET_MS}ms"
    return (
        f"gpu_profile={gpu} budget={budget} early_stop={_EARLY_STOP_OPTIMAL} "
        f"attack_log={_attack_log_mode()} "
        f"steps={_DEFAULT_STEPS} restarts={_DEFAULT_RESTARTS} "
        f"sparse_topk={_SPARSE_TOPK} saliency_frac={_SALIENCY_FRACTION} "
        f"shrink_iters={_SHRINK_ITERATIONS} prune_chunks={_PRUNE_CHUNKS} quantize={_QUANTIZE_FINAL}"
    )
