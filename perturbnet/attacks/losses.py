"""Shared adversarial loss helpers for Perturb attack algorithms."""
from __future__ import annotations

import torch


def dlr_untargeted_objective(logits: torch.Tensor, target_index: int) -> torch.Tensor:
    """Difference-of-Logits-Ratio for untargeted attacks.

    DLR(x, y) = -(z_y - max_{i!=y} z_i) / (z_pi1 - z_pi3)

    Higher value => target class y is weaker vs rivals. Shift/scale invariant.
    Use with gradient ascent (or minimize ``-dlr_untargeted_objective``).
    """
    z = logits.squeeze(0)
    z_sorted, _ = z.sort(descending=True)
    z_y = z[target_index]
    masked = z.clone()
    masked[target_index] = float("-inf")
    z_max_other = masked.max()
    denom = (z_sorted[0] - z_sorted[2]).clamp_min(1e-12)
    return -(z_y - z_max_other) / denom
