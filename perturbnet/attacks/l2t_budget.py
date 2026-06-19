"""L2T+budget attack — miner adapter.

Configure via environment variables:
  L2T_L2T_EPOCHS           outer policy-learning iterations  (default 4)
  L2T_NUM_SCALE            augmented views per PGD step      (default 3)
  L2T_OPS_NUM              ops chained per view              (default 2)
  L2T_AUG_LR              policy gradient learning rate     (default 0.01)
  L2T_MAX_LEVEL            max L-inf level in 1/255          (default 7)
  L2T_INNER_ITERS          PGD steps per level               (default 10)
  L2T_RESTARTS             PGD restarts per level            (default 2)
  L2T_FIRST_LEVEL_RESTARTS extra restarts at level 1         (default 6)
  L2T_SPARSE               enable sparse refinement          (default true)
  L2T_SPARSE_STEPS         steps in sparse phase             (default 8)
  L2T_SPARSE_RESTARTS      restarts in sparse phase          (default 1)
  L2T_POLISH_ROUNDS        backward-elimination passes       (default 1)
  L2T_RESTART_BATCH_SIZE   parallel GPU restarts             (default 8)
"""

from __future__ import annotations

import os
import threading
from dataclasses import replace

import torch

from l2t_attack import L2TBudgetConfig, build_l2t_budget_attack
from perturbnet.image_io import quantize_chw_to_8bit
from perturbnet.model import predict_index

ALGORITHM_NAME = "L2T-Budget"


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


def _l2t_config_from_env() -> L2TBudgetConfig:
    cfg = L2TBudgetConfig()
    return replace(
        cfg,
        l2t_epochs=_env_int("L2T_L2T_EPOCHS", cfg.l2t_epochs),
        num_scale=_env_int("L2T_NUM_SCALE", cfg.num_scale),
        ops_num=_env_int("L2T_OPS_NUM", cfg.ops_num),
        aug_lr=float(os.getenv("L2T_AUG_LR", str(cfg.aug_lr))),
        max_level=_env_int("L2T_MAX_LEVEL", cfg.max_level),
        inner_iters=_env_int("L2T_INNER_ITERS", cfg.inner_iters),
        restarts=_env_int("L2T_RESTARTS", cfg.restarts),
        first_level_restarts=_env_int("L2T_FIRST_LEVEL_RESTARTS", cfg.first_level_restarts),
        sparse=not _env_bool("L2T_NO_SPARSE", not cfg.sparse),
        sparse_steps=_env_int("L2T_SPARSE_STEPS", cfg.sparse_steps),
        sparse_restarts=_env_int("L2T_SPARSE_RESTARTS", cfg.sparse_restarts),
        polish_rounds=_env_int("L2T_POLISH_ROUNDS", cfg.polish_rounds),
        restart_batch_size=_env_int("L2T_RESTART_BATCH_SIZE", cfg.restart_batch_size),
    )


_lock = threading.Lock()
_cached: dict[tuple, object] = {}


def _get_attack(device: torch.device):
    cfg = _l2t_config_from_env()
    key = (
        str(device),
        cfg.l2t_epochs,
        cfg.num_scale,
        cfg.ops_num,
        cfg.aug_lr,
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
            attack = build_l2t_budget_attack(device, cfg=cfg)
            _cached[key] = attack
        return attack


def run_l2t_budget(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    """Run the L2T+budget attack. Returns (adv, linf, rmse, pred_idx)."""
    del epsilon, min_delta

    attack = _get_attack(device)
    clean_bchw = clean.unsqueeze(0) if clean.dim() == 3 else clean
    label = torch.tensor([target_index], device=device, dtype=torch.long)

    delta = attack(clean_bchw, label)
    adv = torch.clamp(clean_bchw + delta, 0.0, 1.0).squeeze(0)
    if _env_bool("PERTURB_QUANTIZE_FINAL", True):
        adv = quantize_chw_to_8bit(adv).to(device)

    linf = float((adv - clean).abs().max().item())
    rmse = float((adv - clean).pow(2).mean().sqrt().item())
    pred_index = predict_index(model, adv)
    return adv.detach(), linf, rmse, pred_index


def runtime_profile_summary() -> str:
    cfg = _l2t_config_from_env()
    return (
        f"l2t_epochs={cfg.l2t_epochs} num_scale={cfg.num_scale} ops_num={cfg.ops_num} "
        f"aug_lr={cfg.aug_lr} inner_iters={cfg.inner_iters} "
        f"restarts={cfg.restarts} first_level_restarts={cfg.first_level_restarts} "
        f"sparse={cfg.sparse} sparse_steps={cfg.sparse_steps} "
        f"sparse_restarts={cfg.sparse_restarts} polish_rounds={cfg.polish_rounds}"
    )
