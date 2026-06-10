from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import torch


@dataclass(frozen=True)
class AttackResult:
    adv: torch.Tensor
    linf: float
    rmse: float
    pred_index: int
    algorithm_key: str
    algorithm_name: str


class AttackRunner(Protocol):
    def __call__(
        self,
        *,
        clean: torch.Tensor,
        model: torch.nn.Module,
        target_index: int,
        epsilon: float,
        min_delta: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, float, float, int]: ...


ProfileSummaryFn = Callable[[], str]
