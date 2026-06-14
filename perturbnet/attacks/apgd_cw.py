"""
APGD-CW: Auto-PGD (DLR) discovery + Carlini-Wagner minimum-distortion shrink.

Untargeted L∞ white-box attack for the Perturb subnet (EfficientNetV2-L).

Theory
------
Phase 1 — Auto-PGD with the DLR loss (Croce & Hein, ICML 2020,
          "Reliable Evaluation of Adversarial Robustness ...", arXiv:2003.01690):
          budget-aware adaptive step size, momentum blend, restart-from-best.
          The Difference-of-Logits-Ratio loss is shift/scale invariant and
          immune to the gradient masking that stalls cross-entropy PGD.
          Goal: find ANY misclassifying delta quickly and reliably.

Phase 2 — Carlini & Wagner minimum-distortion search (Carlini & Wagner,
          IEEE S&P 2017, arXiv:1608.04644): minimize ||delta||_inf while
          keeping misclassification. We realise the C&W objective via a binary
          search over the L∞ radius, using Phase-1's solution as a warm start
          and APGD-DLR as the inner solver. This drives L∞ down toward the
          subnet's min_delta (the 1/255 floor), which is what the validator
          score rewards quadratically.

Phase 2.5 — SMA-style shrink + prune refine (reuse sma_mi_fgsm helpers):
          scale the perturbation direction to minimize L∞, then zero low-magnitude
          pixels to cut RMSE and restore SSIM. Uses all budget remaining after
          Phase 2 until the global deadline.

Phase 3 — project to full resolution, optional full-res refine, 8-bit snap +
          gate re-verification (match validator PNG round-trip).

Budget
------
Set PERTURB_ATTACK_BUDGET_MS (default 13000 on GPU). Phase 1 discovery and
Phase 2 binary shrink are capped by APGD_CW_PHASE1_FRAC / APGD_CW_PHASE2_FRAC;
Phase 2.5 refine consumes the remainder until the deadline.

Switch on with PERTURB_ATTACK_ALGORITHM=apgd_cw.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from perturbnet.attacks import sma_mi_fgsm as sma
from perturbnet.attacks.losses import dlr_untargeted_objective
from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import logits_for_images, predict_index

logger = logging.getLogger(__name__)

ALGORITHM_NAME = "APGD-CW"
ALGORITHM_KEY = "apgd_cw"

_ONE_STEP = 1.0 / 255.0
_OPTIMAL_LINF = _ONE_STEP + 1e-6


# --- env helpers ------------------------------------------------------------
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
    # Default: leave ~2s safety margin under the validator timeout.
    timeout_s = int(os.getenv("PERTURB_TIMEOUT_SECONDS", "15"))
    return max(1000, int(timeout_s * 1000) - 2000)


def _attack_max_dim() -> int:
    raw = os.getenv("PERTURB_ATTACK_MAX_DIM")
    if raw is not None and raw.strip():
        return int(raw)
    return 512 if _gpu_profile_enabled() else 0


def _attack_log_mode() -> str:
    raw = os.getenv("PERTURB_ATTACK_LOG", "on").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"verbose", "debug", "v"}:
        return "verbose"
    return "on"


# --- tunables (env) ---------------------------------------------------------
_MAX_LINF_DELTA = float(os.getenv("PERTURB_MAX_LINF_DELTA", "0.03"))
_PHASE1_FRAC = float(os.getenv("APGD_CW_PHASE1_FRAC", "0.25"))
_PHASE2_FRAC = float(os.getenv("APGD_CW_PHASE2_FRAC", "0.20"))
_MOMENTUM = float(os.getenv("APGD_CW_MOMENTUM", "0.75"))
_INIT_STEP_MULT = float(os.getenv("APGD_CW_INIT_STEP", "2.0"))
_CHECKPOINT_RHO = float(os.getenv("APGD_CW_CHECKPOINT_RHO", "0.75"))
_BINARY_LEVELS = _profile_int("APGD_CW_BINARY_LEVELS", cpu_default=8, gpu_default=12)
_PHASE1_STEPS = _profile_int("APGD_CW_PHASE1_STEPS", cpu_default=40, gpu_default=100)
_STEPS_PER_LEVEL = _profile_int("APGD_CW_STEPS_PER_LEVEL", cpu_default=12, gpu_default=20)
_QUANTIZE_FINAL = _env_bool("PERTURB_QUANTIZE_FINAL", default=True)
_EARLY_STOP_OPTIMAL = _env_bool("PERTURB_ATTACK_EARLY_STOP", default=True)
_ATTACK_BUDGET_MS = _attack_budget_ms()
_ATTACK_MAX_DIM = _attack_max_dim()


# --- metrics ----------------------------------------------------------------
def _effective_epsilon(epsilon: float) -> float:
    return min(float(epsilon), _MAX_LINF_DELTA)


def _linf_delta(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((adv - clean) ** 2)).item())


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


# --- timing -----------------------------------------------------------------
@dataclass
class _Clock:
    started_at: float
    deadline: float | None
    log_mode: str = field(default_factory=_attack_log_mode)

    @classmethod
    def create(cls) -> _Clock:
        started_at = time.perf_counter()
        deadline = None
        if _ATTACK_BUDGET_MS is not None:
            deadline = started_at + _ATTACK_BUDGET_MS / 1000.0
        return cls(started_at=started_at, deadline=deadline)

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    def remaining_ms(self) -> int | None:
        if self.deadline is None:
            return None
        return max(0, int((self.deadline - time.perf_counter()) * 1000))

    def exceeded(self) -> bool:
        return self.deadline is not None and time.perf_counter() >= self.deadline

    def _timing(self) -> dict[str, object]:
        fields: dict[str, object] = {"elapsed_ms": self.elapsed_ms()}
        remaining = self.remaining_ms()
        if remaining is not None:
            fields["remaining_ms"] = remaining
        return fields

    def event(self, message: str, **fields: object) -> None:
        if self.log_mode == "off":
            return
        logger.info("[APGD-CW] %s %s", message, _format_fields(**self._timing(), **fields))

    def phase(self, name: str, **fields: object) -> None:
        if self.log_mode == "off":
            return
        logger.info("[APGD-CW] phase=%s %s", name, _format_fields(**self._timing(), **fields))


@dataclass
class _RefineContext:
    """Minimal _AttackContext stand-in for SMA shrink/prune helpers."""

    clock: _Clock
    _progress_marks: dict[str, int] = field(default_factory=dict)

    def exceeded(self) -> bool:
        return self.clock.exceeded()

    def should_stop(self, quality: tuple[float, float] | None) -> bool:
        if self.clock.exceeded():
            return True
        if _EARLY_STOP_OPTIMAL and quality is not None and quality[0] <= _OPTIMAL_LINF:
            return True
        return False

    def _timing_fields(self) -> dict[str, object]:
        return self.clock._timing()

    def phase(self, name: str, **fields: object) -> None:
        self.clock.phase(name, **fields)

    def event(self, message: str, **fields: object) -> None:
        self.clock.event(message, **fields)

    def progress(self, phase: str, current: int, total: int, **fields: object) -> None:
        if self.clock.log_mode == "off" or total <= 0:
            return
        pct = int(100 * current / total)
        interval = 10 if self.clock.log_mode == "verbose" else 10
        mark = (pct // interval) * interval
        if mark <= 0 or mark >= 100:
            return
        last = self._progress_marks.get(phase, -1)
        if mark <= last:
            return
        self._progress_marks[phase] = mark
        self.clock.phase(
            phase,
            progress_pct=mark,
            current=current,
            total=total,
            **fields,
        )

    def quality_fields(self, quality: tuple[float, float] | None) -> dict[str, object]:
        if quality is None:
            return {"success": False}
        return {"success": True, "linf": quality[0], "rmse": quality[1]}


# --- work scale (downscale core loops, upsample delta) ----------------------
@dataclass
class _WorkScale:
    clean_full: torch.Tensor
    clean_work: torch.Tensor

    @property
    def uses_work(self) -> bool:
        return self.clean_work.shape != self.clean_full.shape

    @property
    def full_pixels(self) -> int:
        _, height, width = self.clean_full.shape
        return height * width

    @classmethod
    def create(cls, clean: torch.Tensor, max_dim: int) -> _WorkScale:
        if max_dim <= 0:
            return cls(clean_full=clean, clean_work=clean)
        _, height, width = clean.shape
        long_edge = max(height, width)
        if long_edge <= max_dim:
            return cls(clean_full=clean, clean_work=clean)
        scale = max_dim / float(long_edge)
        work_h = max(1, int(round(height * scale)))
        work_w = max(1, int(round(width * scale)))
        clean_work = F.interpolate(
            clean.unsqueeze(0), size=(work_h, work_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        return cls(clean_full=clean, clean_work=clean_work)

    def project_adv_to_full(self, adv_work: torch.Tensor, epsilon: float) -> torch.Tensor:
        if not self.uses_work:
            return adv_work.clamp(0.0, 1.0)
        delta_work = adv_work - self.clean_work
        _, full_h, full_w = self.clean_full.shape
        delta_full = F.interpolate(
            delta_work.unsqueeze(0), size=(full_h, full_w), mode="bilinear", align_corners=False
        ).squeeze(0)
        adv_full = self.clean_full + delta_full
        return torch.max(
            torch.min(adv_full, self.clean_full + epsilon),
            self.clean_full - epsilon,
        ).clamp(0.0, 1.0)


# --- DLR loss (shared with SMA-MI-FGSM) -------------------------------------
_dlr_ascent_objective = dlr_untargeted_objective


def _apgd_checkpoints(n_iter: int) -> set[int]:
    """Checkpoint iterations per APGD schedule (p0=0, p1=0.22, shrinking)."""
    points = [0.0, 0.22]
    while True:
        nxt = points[-1] + max(points[-1] - points[-2] - 0.03, 0.06)
        if nxt >= 1.0:
            break
        points.append(nxt)
    return {int(round(p * n_iter)) for p in points if 0 < p < 1}


@dataclass
class _ApgdOutcome:
    adv: torch.Tensor
    success: bool
    linf: float
    rmse: float
    pred_index: int


def _apgd_maximize_dlr(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    x_init: torch.Tensor,
    n_iter: int,
    deadline: float | None,
) -> _ApgdOutcome:
    """Auto-PGD ascent on the DLR loss inside the L∞ ball of radius `epsilon`.

    Tracks the lowest-L∞ misclassifying iterate; if none misclassify, returns
    the iterate with the highest DLR (closest to flipping).
    """
    def project(t: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.min(t, clean + epsilon), clean - epsilon).clamp(0.0, 1.0)

    def forward(t: torch.Tensor) -> torch.Tensor:
        return logits_for_images(model=model, image_bchw=t.unsqueeze(0))

    step = max(_INIT_STEP_MULT * epsilon, _ONE_STEP)
    checkpoints = _apgd_checkpoints(n_iter)

    x_prev = project(x_init.detach()).clone()
    x_curr = x_prev.clone()

    best_dlr_adv = x_curr.clone()
    best_dlr_val = float("-inf")
    best_succ_adv: torch.Tensor | None = None
    best_succ_linf = float("inf")
    best_succ_rmse = float("inf")
    best_succ_pred = target_index

    improved_since_ckpt = 0
    last_ckpt_best = float("-inf")
    step_reduced_last = False

    def evaluate(t: torch.Tensor) -> tuple[float, int, torch.Tensor]:
        probe = t.detach().requires_grad_(True)
        logits = forward(probe)
        obj = _dlr_ascent_objective(logits, target_index)
        grad = torch.autograd.grad(obj, probe)[0]
        pred = int(logits.detach().argmax(dim=1).item())
        return float(obj.detach().item()), pred, grad

    obj_val, pred_idx, grad = evaluate(x_curr)

    def consider(t: torch.Tensor, obj_value: float, pred: int) -> None:
        nonlocal best_dlr_adv, best_dlr_val
        nonlocal best_succ_adv, best_succ_linf, best_succ_rmse, best_succ_pred
        if obj_value > best_dlr_val:
            best_dlr_val = obj_value
            best_dlr_adv = t.detach().clone()
        if pred != target_index:
            linf = _linf_delta(clean, t)
            if linf >= min_delta:
                rmse = _rmse(clean, t)
                if (linf, rmse) < (best_succ_linf, best_succ_rmse):
                    best_succ_adv = t.detach().clone()
                    best_succ_linf = linf
                    best_succ_rmse = rmse
                    best_succ_pred = pred

    consider(x_curr, obj_val, pred_idx)

    for k in range(1, n_iter + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            break

        z = project(x_curr + step * grad.sign())
        if k == 1:
            x_next = z
        else:
            x_next = project(
                x_curr
                + _MOMENTUM * (z - x_curr)
                + (1.0 - _MOMENTUM) * (x_curr - x_prev)
            )

        obj_next, pred_next, grad_next = evaluate(x_next)
        consider(x_next, obj_next, pred_next)
        if obj_next > obj_val:
            improved_since_ckpt += 1

        x_prev = x_curr
        x_curr = x_next
        obj_val = obj_next
        pred_idx = pred_next
        grad = grad_next

        # Early stop: already at the 8-bit optimal floor with a success.
        if _EARLY_STOP_OPTIMAL and best_succ_adv is not None and best_succ_linf <= _OPTIMAL_LINF:
            break

        if k in checkpoints:
            period = max(1, k - _last_checkpoint(checkpoints, k))
            cond1 = improved_since_ckpt < _CHECKPOINT_RHO * period
            cond2 = (not step_reduced_last) and (best_dlr_val <= last_ckpt_best)
            if cond1 or cond2:
                step *= 0.5
                step_reduced_last = True
                # Restart from the best point found so far.
                x_curr = best_dlr_adv.clone()
                x_prev = x_curr.clone()
                obj_val, pred_idx, grad = evaluate(x_curr)
            else:
                step_reduced_last = False
            improved_since_ckpt = 0
            last_ckpt_best = best_dlr_val

    if best_succ_adv is not None:
        return _ApgdOutcome(
            adv=best_succ_adv,
            success=True,
            linf=best_succ_linf,
            rmse=best_succ_rmse,
            pred_index=best_succ_pred,
        )
    return _ApgdOutcome(
        adv=best_dlr_adv,
        success=False,
        linf=_linf_delta(clean, best_dlr_adv),
        rmse=_rmse(clean, best_dlr_adv),
        pred_index=pred_idx,
    )


def _last_checkpoint(checkpoints: set[int], current: int) -> int:
    earlier = [c for c in checkpoints if c < current]
    return max(earlier) if earlier else 0


# --- C&W binary-epsilon shrink ----------------------------------------------
def _cw_binary_shrink(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    init: _ApgdOutcome,
    deadline: float | None,
    clock: _Clock,
) -> _ApgdOutcome:
    """Minimise L∞ via binary search over the radius, warm-starting from `init`."""
    best = init
    lo = min_delta
    hi = min(init.linf, _MAX_LINF_DELTA)

    for level in range(1, _BINARY_LEVELS + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            clock.phase("shrink_stop", reason="time_budget", level=level)
            break
        if hi - lo < _ONE_STEP:
            clock.phase("shrink_stop", reason="precision_floor", level=level, lo=lo, hi=hi)
            break

        eps_try = 0.5 * (lo + hi)
        # Warm start: project current best into the smaller ball.
        warm = torch.max(
            torch.min(best.adv, clean + eps_try), clean - eps_try
        ).clamp(0.0, 1.0)

        if deadline is not None:
            level_deadline = min(
                deadline,
                time.perf_counter() + (deadline - time.perf_counter()) / (_BINARY_LEVELS - level + 1),
            )
        else:
            level_deadline = None

        outcome = _apgd_maximize_dlr(
            clean=clean,
            model=model,
            target_index=target_index,
            epsilon=eps_try,
            min_delta=min_delta,
            x_init=warm,
            n_iter=_STEPS_PER_LEVEL,
            deadline=level_deadline,
        )

        if outcome.success:
            if (outcome.linf, outcome.rmse) < (best.linf, best.rmse):
                best = outcome
            hi = min(eps_try, outcome.linf)
            clock.phase(
                "shrink_level", level=level, eps_try=eps_try, result="success",
                linf=outcome.linf, rmse=outcome.rmse, lo=lo, hi=hi,
            )
        else:
            lo = eps_try
            clock.phase(
                "shrink_level", level=level, eps_try=eps_try, result="fail", lo=lo, hi=hi,
            )

        if best.linf <= _OPTIMAL_LINF and best.rmse <= 0.0:
            break

    return best


# --- snap -------------------------------------------------------------------
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
    if _linf_delta(clean, snapped) >= min_delta and predict_index(model=model, image_chw=snapped) != target_index:
        return snapped
    return adv


def _outcome_from_adv(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    adv: torch.Tensor,
    fallback: _ApgdOutcome,
) -> _ApgdOutcome:
    pred = predict_index(model=model, image_chw=adv)
    linf = _linf_delta(clean, adv)
    rmse = _rmse(clean, adv)
    if pred != target_index and linf >= min_delta:
        return _ApgdOutcome(adv=adv, success=True, linf=linf, rmse=rmse, pred_index=pred)
    return fallback


def _refine_outcome(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    min_delta: float,
    outcome: _ApgdOutcome,
    clock: _Clock,
    label: str,
) -> _ApgdOutcome:
    if not outcome.success:
        clock.phase(f"{label}_skip", reason="no_success")
        return outcome
    if clock.exceeded():
        clock.phase(f"{label}_skip", reason="time_budget")
        return outcome

    clock.phase(f"{label}_start", linf=outcome.linf, rmse=outcome.rmse)
    refine_ctx = _RefineContext(clock=clock)
    refined = sma._refine_candidate(
        clean=clean,
        adv=outcome.adv,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        ctx=refine_ctx,  # type: ignore[arg-type]
        label=label,
    )
    refined_outcome = _outcome_from_adv(
        clean=clean,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        adv=refined,
        fallback=outcome,
    )
    clock.phase(
        f"{label}_done",
        linf=refined_outcome.linf,
        rmse=refined_outcome.rmse,
        pred_idx=refined_outcome.pred_index,
        kept=refined_outcome is not outcome,
    )
    return refined_outcome


# --- public runner ----------------------------------------------------------
def run_apgd_cw(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    clock = _Clock.create()
    effective_eps = _effective_epsilon(epsilon)

    work = _WorkScale.create(clean, _ATTACK_MAX_DIM)
    clean_work = work.clean_work

    total_deadline = clock.deadline
    if total_deadline is not None:
        phase1_deadline = clock.started_at + (total_deadline - clock.started_at) * _PHASE1_FRAC
        phase2_deadline = clock.started_at + (total_deadline - clock.started_at) * (_PHASE1_FRAC + _PHASE2_FRAC)
    else:
        phase1_deadline = phase2_deadline = None

    clock.event(
        "start",
        image_shape=tuple(clean.shape),
        work_shape=tuple(clean_work.shape),
        uses_work=work.uses_work,
        full_pixels=work.full_pixels,
        attack_max_dim=_ATTACK_MAX_DIM or "full",
        target_idx=target_index,
        epsilon=epsilon,
        effective_epsilon=effective_eps,
        min_delta=min_delta,
        steps=_PHASE1_STEPS,
        restarts=_BINARY_LEVELS,
        budget_ms=_ATTACK_BUDGET_MS if _ATTACK_BUDGET_MS is not None else "unlimited",
        phase1_frac=_PHASE1_FRAC,
        phase2_frac=_PHASE2_FRAC,
        log_mode=clock.log_mode,
    )

    # PHASE 1: APGD-DLR discovery at full epsilon.
    clock.phase("apgd_discovery_start", epsilon=effective_eps, steps=_PHASE1_STEPS)
    phase1 = _apgd_maximize_dlr(
        clean=clean_work,
        model=model,
        target_index=target_index,
        epsilon=effective_eps,
        min_delta=min_delta,
        x_init=clean_work.clone(),
        n_iter=_PHASE1_STEPS,
        deadline=phase1_deadline,
    )
    clock.phase(
        "apgd_discovery_done",
        success=phase1.success,
        linf=phase1.linf,
        rmse=phase1.rmse,
        pred_idx=phase1.pred_index,
    )

    # PHASE 2: C&W minimum-distortion shrink (only if discovery succeeded).
    if phase1.success:
        best = _cw_binary_shrink(
            clean=clean_work,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            init=phase1,
            deadline=phase2_deadline,
            clock=clock,
        )
    else:
        clock.phase("shrink_skip", reason="no_discovery")
        best = phase1

    # PHASE 2.5: shrink + prune refine on work resolution (uses remaining budget).
    best = _refine_outcome(
        clean=clean_work,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
        outcome=best,
        clock=clock,
        label="work_refine",
    )

    # PHASE 3: project to full resolution, optional full-res refine, snap, verify.
    adv_full = work.project_adv_to_full(best.adv, effective_eps)
    full_quality = sma._attack_quality(model, clean, adv_full, target_index, min_delta)
    if work.uses_work and full_quality is None:
        clock.event("full_res_miss_after_project")

    if (
        full_quality is not None
        and work.full_pixels <= sma._FULL_REFINE_MAX_PIXELS
        and not clock.exceeded()
    ):
        adv_full = sma._refine_candidate(
            clean=clean,
            adv=adv_full,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
            ctx=_RefineContext(clock=clock),  # type: ignore[arg-type]
            label="full_refine",
        )
    elif full_quality is not None:
        adv_full = _snap_8bit(
            clean=clean,
            adv=adv_full,
            model=model,
            target_index=target_index,
            min_delta=min_delta,
        )

    snapped = _snap_8bit(
        clean=clean,
        adv=adv_full,
        model=model,
        target_index=target_index,
        min_delta=min_delta,
    )

    final_linf = _linf_delta(clean, snapped)
    final_rmse = _rmse(clean, snapped)
    final_pred = predict_index(model=model, image_chw=snapped)
    misclassified = final_pred != target_index

    if clock.exceeded() and not misclassified:
        stop = "time_budget"
    elif misclassified and final_linf <= _OPTIMAL_LINF:
        stop = "optimal_linf"
    elif misclassified:
        stop = "complete"
    else:
        stop = "no_success"

    clock.event(
        "done",
        stop=stop,
        linf=final_linf,
        rmse=final_rmse,
        pred_idx=final_pred,
        target_idx=target_index,
        misclassified=misclassified,
    )

    return snapped.to(device), final_linf, final_rmse, final_pred


def runtime_profile_summary() -> str:
    gpu = _gpu_profile_enabled()
    budget = "unlimited" if _ATTACK_BUDGET_MS is None else f"{_ATTACK_BUDGET_MS}ms"
    max_dim = _ATTACK_MAX_DIM if _ATTACK_MAX_DIM > 0 else "full"
    return (
        f"gpu_profile={gpu} budget={budget} early_stop={_EARLY_STOP_OPTIMAL} "
        f"attack_log={_attack_log_mode()} attack_max_dim={max_dim} "
        f"phase1_frac={_PHASE1_FRAC} phase2_frac={_PHASE2_FRAC} "
        f"phase1_steps={_PHASE1_STEPS} binary_levels={_BINARY_LEVELS} "
        f"steps_per_level={_STEPS_PER_LEVEL} momentum={_MOMENTUM} "
        f"init_step_mult={_INIT_STEP_MULT} quantize={_QUANTIZE_FINAL}"
    )
