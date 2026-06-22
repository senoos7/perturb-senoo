"""OPS budget attack — minimum-norm L-inf search tuned for <16s on A100."""

from __future__ import annotations

import os
import threading
from dataclasses import replace

import torch

from ops_attack import BudgetConfig, build_ops_budget_attack
from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import predict_index

ALGORITHM_NAME = "OPS-Budget"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _budget_config_from_env() -> BudgetConfig:
    cfg = BudgetConfig()
    return replace(
        cfg,
        max_level=_env_int("OPS_MAX_LEVEL", cfg.max_level),
        inner_iters=_env_int("OPS_INNER_ITERS", cfg.inner_iters),
        restarts=_env_int("OPS_RESTARTS", cfg.restarts),
        first_level_restarts=_env_int("OPS_FIRST_LEVEL_RESTARTS", cfg.first_level_restarts),
        sparse=not _env_bool("OPS_NO_SPARSE", not cfg.sparse),
        sparse_steps=_env_int("OPS_SPARSE_STEPS", cfg.sparse_steps),
        sparse_restarts=_env_int("OPS_SPARSE_RESTARTS", cfg.sparse_restarts),
        polish_rounds=_env_int("OPS_POLISH_ROUNDS", cfg.polish_rounds),
        restart_batch_size=_env_int("OPS_RESTART_BATCH_SIZE", cfg.restart_batch_size),
        probe_frac=_env_float("OPS_PROBE_FRAC", cfg.probe_frac),
        fgsm_init_slots=_env_int("OPS_FGMSM_INIT_SLOTS", cfg.fgsm_init_slots),
        sign_random_slots=_env_int("OPS_SIGN_RANDOM_SLOTS", cfg.sign_random_slots),
        plateau_patience=_env_int("OPS_PLATEAU_PATIENCE", cfg.plateau_patience),
        ce_loss_slots=_env_int("OPS_CE_LOSS_SLOTS", cfg.ce_loss_slots),
    )


_lock = threading.Lock()
_cached: dict[tuple[str, int], object] = {}


def _get_attack(device: torch.device):
    cfg = _budget_config_from_env()
    key = (
        str(device),
        cfg.max_level,
        cfg.inner_iters,
        cfg.restarts,
        cfg.first_level_restarts,
        cfg.sparse,
        cfg.sparse_steps,
        cfg.sparse_restarts,
        cfg.polish_rounds,
        cfg.restart_batch_size,
        cfg.probe_frac,
        cfg.fgsm_init_slots,
        cfg.sign_random_slots,
        cfg.plateau_patience,
        cfg.ce_loss_slots,
    )
    with _lock:
        attack = _cached.get(key)
        if attack is None:
            attack = build_ops_budget_attack(device, budget=cfg)
            _cached[key] = attack
        return attack


def _linf_delta(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).pow(2).mean().sqrt().item())


def run_ops_budget(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    """Run the OPS minimum-norm budget attack. Returns (adv, linf, rmse, pred_idx)."""
    del epsilon, min_delta  # budget attack searches levels internally

    attack = _get_attack(device)
    clean_bchw = clean.unsqueeze(0) if clean.dim() == 3 else clean
    label = torch.tensor([target_index], device=device, dtype=torch.long)

    delta = attack(clean_bchw, label)
    adv = torch.clamp(clean_bchw + delta, 0.0, 1.0).squeeze(0)
    if _env_bool("PERTURB_QUANTIZE_FINAL", True):
        adv = quantize_chw_to_8bit(adv).to(device)

    linf = _linf_delta(clean, adv)
    rmse = _rmse(clean, adv)
    pred_index = predict_index(model, adv)
    return adv.detach(), linf, rmse, pred_index


def runtime_profile_summary() -> str:
    cfg = _budget_config_from_env()
    return (
        f"inner_iters={cfg.inner_iters} restarts={cfg.restarts} "
        f"first_level_restarts={cfg.first_level_restarts} "
        f"sparse={cfg.sparse} sparse_steps={cfg.sparse_steps} "
        f"sparse_restarts={cfg.sparse_restarts} polish_rounds={cfg.polish_rounds} "
        f"restart_batch_size={cfg.restart_batch_size} probe_frac={cfg.probe_frac} "
        f"fgsm_init_slots={cfg.fgsm_init_slots} sign_random_slots={cfg.sign_random_slots} "
        f"plateau_patience={cfg.plateau_patience} ce_loss_slots={cfg.ce_loss_slots}"
    )
