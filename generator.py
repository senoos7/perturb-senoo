"""Perturbation generator architectures."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with a group count that divides `channels` (batch/res independent)."""
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ResBlock(nn.Module):
    """Pre-activation residual block. `dilation` widens the receptive field cheaply."""

    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = _group_norm(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


class Down(nn.Module):
    """Strided downsample (x2) followed by a residual block."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)
        self.norm = _group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.res = ResBlock(c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.act(self.norm(self.down(x))))


class Up(nn.Module):
    """Upsample to the skip's exact size, concat, fuse, residual block.

    Size-matched interpolation keeps the network valid for arbitrary H, W
    (resolution buckets need not be divisible by the downsample factor).
    """

    def __init__(self, c_in: int, c_skip: int, c_out: int) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(c_in + c_skip, c_out, 3, padding=1)
        self.norm = _group_norm(c_out)
        self.act = nn.SiLU(inplace=True)
        self.res = ResBlock(c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        return self.res(self.act(self.norm(self.fuse(x))))


class Generator(nn.Module):
    """
    Multi-scale U-Net perturbation generator.

    Encoder downsamples 3x (large receptive field), a dilated bottleneck widens
    context further, and a skip-connected decoder restores full resolution.
    Output: Tanh x max_linf  ->  L-inf <= max_linf guaranteed by architecture.

    The head is zero-initialized so the initial perturbation is ~0 (adv ~ clean),
    which gives a stable optimisation start before the flip term takes over.
    """

    def __init__(self, max_linf: float = 0.03, base: int = 48) -> None:
        super().__init__()
        self.max_linf = float(max_linf)
        b = int(base)

        # Stem (full resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            _group_norm(b),
            nn.SiLU(inplace=True),
            ResBlock(b),
        )

        # Encoder
        self.down1 = Down(b, b * 2)        # H/2
        self.down2 = Down(b * 2, b * 4)    # H/4
        self.down3 = Down(b * 4, b * 4)    # H/8

        # Bottleneck — dilations enlarge the receptive field over global structure
        self.mid = nn.Sequential(
            ResBlock(b * 4, dilation=2),
            ResBlock(b * 4, dilation=4),
            ResBlock(b * 4, dilation=8),
        )

        # Decoder (skip connections from stem / down1 / down2)
        self.up3 = Up(b * 4, b * 4, b * 4)  # -> H/4
        self.up2 = Up(b * 4, b * 2, b * 2)  # -> H/2
        self.up1 = Up(b * 2, b, b)          # -> H

        self.head = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)       # b,    H
        s1 = self.down1(s0)     # 2b,   H/2
        s2 = self.down2(s1)     # 4b,   H/4
        s3 = self.down3(s2)     # 4b,   H/8

        m = self.mid(s3)        # 4b,   H/8

        d3 = self.up3(m, s2)    # 4b,   H/4
        d2 = self.up2(d3, s1)   # 2b,   H/2
        d1 = self.up1(d2, s0)   # b,    H

        return self.tanh(self.head(d1)) * self.max_linf


def load_generator_checkpoint(
    generator: Generator,
    path: Path | str,
    *,
    strict: bool = True,
) -> dict:
    """Load weights from a training checkpoint or raw state_dict.

    Accepts files saved by train_generator*.py:
      {"epoch", "max_linf", "generator_state", "val", ...}
    or a bare state_dict mapping param names -> tensors.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "generator_state" in payload:
        state = payload["generator_state"]
        meta = {k: payload[k] for k in ("epoch", "max_linf", "val") if k in payload}
    elif isinstance(payload, dict):
        state = payload
        meta = {}
    else:
        raise ValueError(f"unsupported checkpoint format: {path}")

    incompatible = generator.load_state_dict(state, strict=strict)
    if not strict:
        print(
            f"checkpoint load (strict=False): "
            f"missing={len(incompatible.missing_keys)} "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )
    meta["path"] = str(path)
    return meta
