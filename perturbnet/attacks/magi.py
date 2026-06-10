"""
MAGI: Momentum Integrated Gradients attack (placeholder).

Implement run_magi() here, then register in perturbnet/attacks/__init__.py.
"""
from __future__ import annotations

import torch

ALGORITHM_NAME = "MAGI"
ALGORITHM_KEY = "magi"


def run_magi(
    *,
    clean: torch.Tensor,
    model: torch.nn.Module,
    target_index: int,
    epsilon: float,
    min_delta: float,
    device: torch.device,
) -> tuple[torch.Tensor, float, float, int]:
    raise NotImplementedError(
        f"{ALGORITHM_NAME} is not implemented yet. "
        f"Set PERTURB_ATTACK_ALGORITHM=sma_mi_fgsm or implement perturbnet/attacks/magi.py"
    )


def runtime_profile_summary() -> str:
    return f"algorithm={ALGORITHM_KEY} status=not_implemented"
