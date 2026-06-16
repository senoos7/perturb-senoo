"""
MAGI: Momentum Integrated Gradients attack.

Untargeted L∞ adversarial attack using Integrated Gradients (IG) as the
gradient signal, combined with Momentum I-FGSM update steps.

Algorithm (from Sundararajan et al., 2017 + Dong et al., 2018):

  IG(x) = (x - x0) * (1/m) * Σᵢ₌₁ᵐ ∇_x L(x0 + (i/m)(x - x0), y)
  g_t   = μ·g_{t-1} + IG(x_t) / ||IG(x_t)||₁
  x_{t+1} = Π_{ε}(x_t + α·sign(g_t))

where x0 is the zero-baseline (black image) and Π_{ε} is L∞ projection.

Key advantages over single-point gradient (SMA-MI-FGSM):
  - IG smooths gradient noise by averaging over the path baseline → x
  - Identifies model-agnostic critical regions (better cross-model transfer)
  - More stable updates; less gradient cancellation on texture-rich regions

IG steps are batched into a single model forward pass (all m interpolation
points at once), so compute cost ≈ m × single-point gradient per attack step.

GPU profile
-----------
Set PERTURB_GPU_PROFILE=on on cloud GPU servers.
Tunables: PERTURB_MAGI_IG_STEPS, PERTURB_ATTACK_STEPS, PERTURB_ATTACK_RESTARTS.

Time budget
-----------
PERTURB_ATTACK_BUDGET_MS caps total wall-clock time. Defaults to 80% of
PERTURB_TIMEOUT_SECONDS on GPU, unlimited on CPU.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from perturbnet.attacks.losses import dlr_untargeted_objective
from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import logits_for_images, predict_index

logger = logging.getLogger(__name__)
ALGORITHM_NAME = "MAGI"
ALGORITHM_KEY = "magi"

_ONE_STEP = 1.0 / 255.0
_OPTIMAL_LINF = _ONE_STEP + 1e-6
_PROGRESS_INTERVAL_PCT = 25


# ---------------------------------------------------------------------------
# Env helpers (duplicated from sma_mi_fgsm to keep modules self-contained)
# ---------------------------------------------------------------------------

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


def _attack_loss_mode() -> str:
    raw = os.getenv("PERTURB_ATTACK_LOSS", "ce").strip().lower()
    if raw in {"dlr", "difference_of_logits", "difference-of-logits"}:
        return "dlr"
    return "ce"


def _attack_log_mode() -> str:
    raw = os.getenv("PERTURB_ATTACK_LOG", "on").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"verbose", "debug", "v"}:
        return "verbose"
    return "on"


# ---------------------------------------------------------------------------
# Tunables (evaluated at import time; consistent with PERTURB_* env vars)
# ---------------------------------------------------------------------------
# IG steps per attack step: higher = smoother gradient, higher compute cost.
# Batched into one forward pass, so GPU can handle 20–30 cheaply.
_IG_STEPS = _profile_int("PERTURB_MAGI_IG_STEPS", cpu_default=8, gpu_default=20)
_DEFAULT_STEPS = _profile_int("PERTURB_ATTACK_STEPS", cpu_default=30, gpu_default=80)
_DEFAULT_RESTARTS = _profile_int("PERTURB_ATTACK_RESTARTS", cpu_default=1, gpu_default=3)
_DEFAULT_MOMENTUM = _profile_float("PERTURB_ATTACK_MOMENTUM", cpu_default=0.9, gpu_default=0.9)
_MAX_LINF_DELTA = float(os.getenv("PERTURB_MAX_LINF_DELTA", "0.03"))
_SHRINK_ITERATIONS = _profile_int("PERTURB_ATTACK_SHRINK_ITERS", cpu_default=12, gpu_default=20)
_PRUNE_CHUNKS = _profile_int("PERTURB_PRUNE_CHUNKS", cpu_default=64, gpu_default=256)
_QUANTIZE_FINAL = _env_bool("PERTURB_QUANTIZE_FINAL", default=True)
_EARLY_STOP_OPTIMAL = _env_bool("PERTURB_ATTACK_EARLY_STOP", default=True)
_ATTACK_BUDGET_MS = _attack_budget_ms()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def _untargeted_loss(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    device = logits.device
    if _attack_loss_mode() == "ce":
        target = torch.tensor([target_index], device=device)
        return F.cross_entropy(logits, target)
    return -dlr_untargeted_objective(logits, target_index)


# ---------------------------------------------------------------------------
# Logging / budget context
# ---------------------------------------------------------------------------

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
class _Budget:
    deadline: float | None
    early_stop_optimal: bool

    @classmethod
    def from_env(cls) -> _Budget:
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
class _Ctx:
    budget: _Budget
    started_at: float
    log_mode: str = field(default_factory=_attack_log_mode)
    _progress_marks: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> _Ctx:
        return cls(budget=_Budget.from_env(), started_at=time.perf_counter())

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

    def _timing(self) -> dict[str, object]:
        d: dict[str, object] = {"elapsed_ms": self.elapsed_ms()}
        r = self.remaining_ms()
        if r is not None:
            d["remaining_ms"] = r
        return d

    def phase(self, name: str, **kw: object) -> None:
        if self.log_mode == "off":
            return
        logger.info("[MAGI] phase=%s %s", name, _format_fields(**self._timing(), **kw))

    def event(self, msg: str, **kw: object) -> None:
        if self.log_mode == "off":
            return
        logger.info("[MAGI] %s %s", msg, _format_fields(**self._timing(), **kw))

    def progress(self, phase: str, current: int, total: int, **kw: object) -> None:
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
        logger.info(
            "[MAGI] phase=%s %s",
            phase,
            _format_fields(**self._timing(), progress_pct=mark, current=current, total=total, **kw),
        )

    def qf(self, quality: tuple[float, float] | None) -> dict[str, object]:
        if quality is None:
            return {"success": False}
        return {"success": True, "linf": quality[0], "rmse": quality[1]}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _effective_epsilon(epsilon: float) -> float:
    return min(float(epsilon), _MAX_LINF_DELTA)


def _linf(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


def _is_optimal_quality(quality: tuple[float, float] | None) -> bool:
    return quality is not None and quality[0] <= _OPTIMAL_LINF


def _is_success(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_index: int,
    min_delta: float,
) -> bool:
    if _linf(clean, adv) < min_delta:
        return False
    return predict_index(model=model, image_chw=adv) != target_index


def _quality(
    model: torch.nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    target_index: int,
    min_delta: float,
) -> tuple[float, float] | None:
    if not _is_success(model, clean, adv, target_index, min_delta):
        return None
    return _linf(clean, adv), _rmse(clean, adv)


def _better(a: tuple[float, float] | None, b: tuple[float, float] | None) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a < b  # lexicographic: (L∞, RMSE)


# ---------------------------------------------------------------------------
# Integrated Gradients (batched — all m interpolation points in one fwd pass)
# ---------------------------------------------------------------------------

def _integrated_gradients(
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
) -> torch.Tensor:
    """
    Compute IG from zero baseline to current adv, batching all interpolation
    points into a single model forward pass.

    Returns IG tensor same shape as adv (C, H, W).
    """
    steps = _IG_STEPS
    # Build interpolation batch: α ∈ {1/m, 2/m, …, 1}  shape (m, C, H, W)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=adv.device)
    # adv serves as both path endpoint and zero-baseline reference (x0 = 0)
    path_delta = adv.detach()  # x - x0, x0=0 ⟹ delta = adv

    batch = torch.stack([alpha * path_delta for alpha in alphas])  # (m, C, H, W)
    batch = batch.requires_grad_(True)

    # Single batched forward pass through EfficientNetV2-L
    logits_batch = logits_for_images(model=model, image_bchw=batch)  # (m, 1000)

    # Sum CE/DLR loss across all m points, then backprop once
    device = logits_batch.device
    if _attack_loss_mode() == "ce":
        targets = torch.full((steps,), target_index, dtype=torch.long, device=device)
        total_loss = F.cross_entropy(logits_batch, targets, reduction="sum")
    else:
        total_loss = sum(
            -dlr_untargeted_objective(logits_batch[i : i + 1], target_index)
            for i in range(steps)
        )

    grad_batch = torch.autograd.grad(total_loss, batch)[0]  # (m, C, H, W)
    grad_mean = grad_batch.mean(dim=0)  # average over m steps → (C, H, W)

    # IG = (x - x0) * mean_grad  (x0=0 so x - x0 = adv)
    return (path_delta * grad_mean).detach()


# ---------------------------------------------------------------------------
# Core MAGI loop (one restart)
# ---------------------------------------------------------------------------

def _magi_loop(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    steps: int,
    momentum: float,
    noise_scale: float,
    ctx: _Ctx,
    restart_idx: int,
) -> tuple[torch.Tensor, str]:
    step_size = max(epsilon / max(steps, 1), _ONE_STEP)

    adv = clean.clone().detach()
    if noise_scale > 0.0:
        adv = adv + torch.empty_like(adv).uniform_(-noise_scale, noise_scale)
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

    ctx.phase(
        f"restart_{restart_idx}_start",
        steps=steps,
        noise=noise_scale > 0.0,
        loss=_attack_loss_mode(),
    )

    velocity = torch.zeros_like(clean)
    best_adv = adv.clone()
    best_quality: tuple[float, float] | None = None
    stop_reason = "steps_exhausted"

    for step in range(steps):
        if ctx.exceeded():
            stop_reason = "time_budget"
            ctx.event(
                "early_stop",
                reason=stop_reason,
                stage=f"restart_{restart_idx}_step_{step}",
                **ctx.qf(best_quality),
            )
            break

        # IG from zero baseline to current adv (batched single fwd pass)
        ig = _integrated_gradients(adv, model, target_index)

        # Normalize by L1 norm (stabilizes update magnitude across images)
        l1 = ig.abs().sum().clamp_min(1e-12)
        ig_norm = ig / l1

        # Momentum accumulation
        velocity = momentum * velocity + ig_norm

        # L∞-projected sign step
        adv = (adv + step_size * velocity.sign()).detach()
        adv = torch.max(torch.min(adv, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

        q = _quality(model, clean, adv, target_index, min_delta)
        if _better(q, best_quality):
            best_adv = adv.clone()
            best_quality = q
            ctx.event(
                "magi_improved",
                restart=restart_idx,
                step=step + 1,
                **ctx.qf(best_quality),
            )
            if _is_optimal_quality(best_quality):
                stop_reason = "optimal_linf"
                break

        ctx.progress(f"restart_{restart_idx}_magi", step + 1, steps, **ctx.qf(best_quality))

    ctx.phase(
        f"restart_{restart_idx}_done",
        stop=stop_reason,
        **ctx.qf(best_quality),
    )
    return best_adv, stop_reason


# ---------------------------------------------------------------------------
# Post-processing: binary shrink + delta prune
# ---------------------------------------------------------------------------

def _binary_shrink(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    iterations: int,
    ctx: _Ctx,
    label: str,
) -> torch.Tensor:
    """Bisect between clean and adv to minimize L∞ while keeping flip."""
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv
    ctx.phase(f"{label}_start", iters=iterations, **ctx.qf(_quality(model, clean, adv, target_index, min_delta)))
    lo, hi = clean.clone(), adv.clone()
    for i in range(iterations):
        if ctx.exceeded():
            break
        mid = (lo + hi) / 2.0
        if _is_success(model, clean, mid, target_index, min_delta):
            hi = mid
        else:
            lo = mid
        ctx.progress(label, i + 1, iterations, **ctx.qf(_quality(model, clean, hi, target_index, min_delta)))
    ctx.phase(f"{label}_done", **ctx.qf(_quality(model, clean, hi, target_index, min_delta)))
    return hi


def _delta_prune(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    n_chunks: int,
    ctx: _Ctx,
    label: str,
) -> torch.Tensor:
    """Zero low-magnitude delta chunks to reduce RMSE without losing flip."""
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv

    delta = adv - clean
    flat_delta = delta.flatten()
    n_chunks = max(1, min(n_chunks, flat_delta.numel()))
    chunk_size = max(1, flat_delta.numel() // n_chunks)

    # Rank chunks by ascending mean |delta| — try zeroing weakest first
    chunk_ranks = sorted(
        range(n_chunks),
        key=lambda c: flat_delta[c * chunk_size : (c + 1) * chunk_size].abs().mean().item(),
    )

    ctx.phase(
        f"{label}_start",
        chunks=n_chunks,
        chunk_pixels=chunk_size,
        **ctx.qf(_quality(model, clean, adv, target_index, min_delta)),
    )

    current = adv.clone()
    for c in chunk_ranks:
        if ctx.exceeded():
            break
        start, end = c * chunk_size, min((c + 1) * chunk_size, flat_delta.numel())
        candidate_flat = (current - clean).flatten().clone()
        candidate_flat[start:end] = 0.0
        candidate = (clean + candidate_flat.view_as(delta)).clamp(0.0, 1.0)
        if _is_success(model, clean, candidate, target_index, min_delta):
            current = candidate

    ctx.phase(f"{label}_done", **ctx.qf(_quality(model, clean, current, target_index, min_delta)))
    return current


def _refine(
    *,
    clean: torch.Tensor,
    adv: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    ctx: _Ctx,
    label: str,
) -> torch.Tensor:
    """shrink → prune → shrink → quantize-snap."""
    if not _is_success(model, clean, adv, target_index, min_delta):
        return adv
    ctx.phase(f"{label}_refine_start")
    adv = _binary_shrink(
        clean=clean, adv=adv, model=model, target_index=target_index,
        min_delta=min_delta, iterations=_SHRINK_ITERATIONS, ctx=ctx, label=f"{label}_shrink1",
    )
    adv = _delta_prune(
        clean=clean, adv=adv, model=model, target_index=target_index,
        min_delta=min_delta, n_chunks=_PRUNE_CHUNKS, ctx=ctx, label=f"{label}_prune",
    )
    adv = _binary_shrink(
        clean=clean, adv=adv, model=model, target_index=target_index,
        min_delta=min_delta, iterations=_SHRINK_ITERATIONS, ctx=ctx, label=f"{label}_shrink2",
    )
    if _QUANTIZE_FINAL:
        snapped = quantize_chw_to_8bit(adv).to(clean.device)
        if _is_success(model, clean, snapped, target_index, min_delta):
            adv = snapped
    ctx.phase(f"{label}_refine_done", **ctx.qf(_quality(model, clean, adv, target_index, min_delta)))
    return adv


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------

def _finalize(
    *,
    clean: torch.Tensor,
    best: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    device: torch.device,
    ctx: _Ctx,
    stop_reason: str,
) -> tuple[torch.Tensor, float, float, int]:
    if _QUANTIZE_FINAL:
        snapped = quantize_chw_to_8bit(best).to(clean.device)
        if _is_success(model, clean, snapped, target_index, min_delta):
            best = snapped
    linf_val = _linf(clean, best)
    rmse_val = _rmse(clean, best)
    pred_idx = predict_index(model=model, image_chw=best)
    ctx.event(
        "done",
        stop=stop_reason,
        linf=linf_val,
        rmse=rmse_val,
        pred_idx=pred_idx,
        target_idx=target_index,
        misclassified=pred_idx != target_index,
    )
    return best.to(device), linf_val, rmse_val, pred_idx


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_magi(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    """
    MAGI: Momentum Integrated Gradients attack.
    Returns (adv, linf, rmse, pred_idx).
    """
    ctx = _Ctx.from_env()
    effective_eps = _effective_epsilon(epsilon)
    steps = max(5, _DEFAULT_STEPS)
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
        ig_steps=_IG_STEPS,
        budget_ms=_ATTACK_BUDGET_MS if _ATTACK_BUDGET_MS is not None else "unlimited",
        attack_loss=_attack_loss_mode(),
        log_mode=ctx.log_mode,
    )

    best = clean.clone()
    best_quality: tuple[float, float] | None = None
    best_pred = target_index

    for restart in range(restarts):
        if ctx.should_stop(best_quality):
            stop_reason = "optimal_linf" if _is_optimal_quality(best_quality) else "time_budget"
            ctx.event("early_stop", reason=stop_reason, stage=f"before_restart_{restart}")
            break

        # First restart: no noise; subsequent restarts: random init within ε
        noise_scale = 0.0 if restart == 0 else effective_eps * 0.5

        candidate, mi_stop = _magi_loop(
            clean=clean,
            model=model,
            target_index=target_index,
            epsilon=effective_eps,
            min_delta=min_delta,
            steps=steps,
            momentum=_DEFAULT_MOMENTUM,
            noise_scale=noise_scale,
            ctx=ctx,
            restart_idx=restart,
        )

        # Refine only if flip achieved and budget remains
        if _is_success(model, clean, candidate, target_index, min_delta) and not ctx.exceeded():
            candidate = _refine(
                clean=clean,
                adv=candidate,
                model=model,
                target_index=target_index,
                min_delta=min_delta,
                ctx=ctx,
                label=f"restart_{restart}",
            )

        q = _quality(model, clean, candidate, target_index, min_delta)
        if _better(q, best_quality):
            best = candidate.clone()
            best_quality = q
            best_pred = predict_index(model=model, image_chw=best)
            ctx.event(
                "best_updated",
                source=f"restart_{restart}",
                pred_idx=best_pred,
                mi_stop=mi_stop,
                **ctx.qf(best_quality),
            )
        elif best_quality is None:
            # No flip yet — keep candidate with largest perturbation as fallback
            if _linf(clean, candidate) > _linf(clean, best):
                best = candidate.clone()
                best_pred = predict_index(model=model, image_chw=best)

        if ctx.should_stop(best_quality):
            stop_reason = "optimal_linf" if _is_optimal_quality(best_quality) else "time_budget"
            ctx.event("early_stop", reason=stop_reason, stage=f"after_restart_{restart}")
            break

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
    return (
        f"gpu_profile={gpu} "
        f"budget={_ATTACK_BUDGET_MS}ms "
        f"ig_steps={_IG_STEPS} "
        f"steps={_DEFAULT_STEPS} "
        f"restarts={_DEFAULT_RESTARTS} "
        f"attack_loss={_attack_loss_mode()} "
        f"shrink_iters={_SHRINK_ITERATIONS} "
        f"prune_chunks={_PRUNE_CHUNKS} "
        f"quantize={_QUANTIZE_FINAL}"
    )
