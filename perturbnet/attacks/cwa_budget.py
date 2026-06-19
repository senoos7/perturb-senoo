"""CWA ensemble + budget-minimizing attack — miner adapter.

Mirrors the structure of ops_budget.py.  Configure via environment variables:
  CWA_CWA_EPOCHS          outer CWA iterations        (default 6)
  CWA_MAX_LEVEL           max L-inf level in 1/255     (default 7)
  CWA_INNER_ITERS         PGD iters per level          (default 10)
  CWA_RESTARTS            PGD restarts per level       (default 2)
  CWA_FIRST_LEVEL_RESTARTS extra restarts at level 1   (default 6)
  CWA_SPARSE              enable sparse refinement     (default true)
  CWA_SPARSE_STEPS        PGD steps in sparse phase    (default 8)
  CWA_SPARSE_RESTARTS     restarts in sparse phase     (default 1)
  CWA_POLISH_ROUNDS       backward-elimination passes  (default 1)
  CWA_RESTART_BATCH_SIZE  parallel GPU restarts        (default 8)
"""

from __future__ import annotations

import os
import threading
from dataclasses import replace

import torch

from cwa_attack import CWABudgetConfig, build_cwa_budget_attack
from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import predict_index

ALGORITHM_NAME = "CWA-Budget"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cwa_config_from_env() -> CWABudgetConfig:
    cfg = CWABudgetConfig()
    return replace(
        cfg,
        cwa_epochs=_env_int("CWA_CWA_EPOCHS", cfg.cwa_epochs),
        max_level=_env_int("CWA_MAX_LEVEL", cfg.max_level),
        inner_iters=_env_int("CWA_INNER_ITERS", cfg.inner_iters),
        restarts=_env_int("CWA_RESTARTS", cfg.restarts),
        first_level_restarts=_env_int("CWA_FIRST_LEVEL_RESTARTS", cfg.first_level_restarts),
        sparse=not _env_bool("CWA_NO_SPARSE", not cfg.sparse),
        sparse_steps=_env_int("CWA_SPARSE_STEPS", cfg.sparse_steps),
        sparse_restarts=_env_int("CWA_SPARSE_RESTARTS", cfg.sparse_restarts),
        polish_rounds=_env_int("CWA_POLISH_ROUNDS", cfg.polish_rounds),
        restart_batch_size=_env_int("CWA_RESTART_BATCH_SIZE", cfg.restart_batch_size),
    )


_lock = threading.Lock()
_cached: dict[tuple, object] = {}


def _get_attack(device: torch.device):
    cfg = _cwa_config_from_env()
    key = (
        str(device),
        cfg.cwa_epochs,
        cfg.max_level,
        cfg.inner_iters,
        cfg.restarts,
        cfg.first_level_restarts,
        cfg.sparse,
        cfg.sparse_steps,
        cfg.sparse_restarts,
        cfg.polish_rounds,
        cfg.restart_batch_size,
    )
    with _lock:
        attack = _cached.get(key)
        if attack is None:
            attack = build_cwa_budget_attack(device, cfg=cfg)
            _cached[key] = attack
        return attack


def _linf_delta(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).abs().max().item())


def _rmse(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return float((adv - clean).pow(2).mean().sqrt().item())


def run_cwa_budget(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    """Run the CWA+budget attack. Returns (adv, linf, rmse, pred_idx)."""
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
    cfg = _cwa_config_from_env()
    return (
        f"cwa_epochs={cfg.cwa_epochs} inner_iters={cfg.inner_iters} "
        f"restarts={cfg.restarts} first_level_restarts={cfg.first_level_restarts} "
        f"sparse={cfg.sparse} sparse_steps={cfg.sparse_steps} "
        f"sparse_restarts={cfg.sparse_restarts} polish_rounds={cfg.polish_rounds} "
        f"restart_batch_size={cfg.restart_batch_size}"
    )
