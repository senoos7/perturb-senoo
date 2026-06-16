"""
Attack algorithm registry and dispatcher.

Select via environment variable:
  PERTURB_ATTACK_ALGORITHM=sma_mi_fgsm   (default)
  PERTURB_ATTACK_ALGORITHM=apgd_cw      (APGD-DLR + C&W shrink)
  PERTURB_ATTACK_ALGORITHM=magi          (stub, not yet implemented)

Add a new algorithm:
  1. Create perturbnet/attacks/<name>.py with run_<name>() and runtime_profile_summary()
  2. Register it in _REGISTRY below
  3. Set PERTURB_ATTACK_ALGORITHM=<name> in scripts/miner.env
"""
from __future__ import annotations

import os
from typing import NamedTuple

import torch

from perturbnet.attacks.base import AttackResult, AttackRunner, ProfileSummaryFn
from perturbnet.attacks import apgd_cw as apgd_cw_module
from perturbnet.attacks import magi as magi_module
from perturbnet.attacks import sma_mi_fgsm as sma_mi_fgsm_module

DEFAULT_ATTACK_ALGORITHM = "sma_mi_fgsm"


class _RegisteredAttack(NamedTuple):
    key: str
    title: str
    runner: AttackRunner
    profile_summary: ProfileSummaryFn


def _normalize_algorithm_key(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_")
    aliases = {
        "sma": "sma_mi_fgsm",
        "smi_fgsm": "sma_mi_fgsm",
        "default": "sma_mi_fgsm",
        "apgd": "apgd_cw",
        "cw": "apgd_cw",
        "apgd_dlr_cw": "apgd_cw",
        "momentum_ig": "magi",
        "ig": "magi",
    }
    return aliases.get(key, key)


_REGISTRY: dict[str, _RegisteredAttack] = {
    "sma_mi_fgsm": _RegisteredAttack(
        key="sma_mi_fgsm",
        title=sma_mi_fgsm_module.ALGORITHM_NAME,
        runner=sma_mi_fgsm_module.run_sma_mi_fgsm,
        profile_summary=sma_mi_fgsm_module.runtime_profile_summary,
    ),
    "apgd_cw": _RegisteredAttack(
        key="apgd_cw",
        title=apgd_cw_module.ALGORITHM_NAME,
        runner=apgd_cw_module.run_apgd_cw,
        profile_summary=apgd_cw_module.runtime_profile_summary,
    ),
    "magi": _RegisteredAttack(
        key="magi",
        title=magi_module.ALGORITHM_NAME,
        runner=magi_module.run_magi,
        profile_summary=magi_module.runtime_profile_summary,
    ),
}


def list_attack_algorithms() -> list[str]:
    return sorted(_REGISTRY.keys())


def resolve_attack_algorithm(raw: str | None = None) -> _RegisteredAttack:
    key = _normalize_algorithm_key(raw or os.getenv("PERTURB_ATTACK_ALGORITHM", DEFAULT_ATTACK_ALGORITHM))
    attack = _REGISTRY.get(key)
    if attack is None:
        known = ", ".join(list_attack_algorithms())
        raise ValueError(
            f"Unknown PERTURB_ATTACK_ALGORITHM={key!r}. Known algorithms: {known}"
        )
    return attack


def get_attack_algorithm_key() -> str:
    return resolve_attack_algorithm().key


def get_attack_algorithm_name() -> str:
    return resolve_attack_algorithm().title


def runtime_profile_summary(algorithm: str | None = None) -> str:
    attack = resolve_attack_algorithm(algorithm)
    return f"algorithm={attack.key} title={attack.title} {attack.profile_summary()}"


def run_attack(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
    algorithm: str | None = None,
) -> AttackResult:
    """Run the configured (or explicitly requested) attack algorithm."""
    attack = resolve_attack_algorithm(algorithm)
    adv, linf, rmse, pred_index = attack.runner(
        clean=clean,
        model=model,
        target_index=target_index,
        epsilon=epsilon,
        min_delta=min_delta,
        device=device,
    )
    return AttackResult(
        adv=adv,
        linf=linf,
        rmse=rmse,
        pred_index=pred_index,
        algorithm_key=attack.key,
        algorithm_name=attack.title,
    )
