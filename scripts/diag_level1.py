#!/usr/bin/env python3
"""Diagnostic: how hard is a 1/255 flip for a specific image?

For a given row we run margin-PGD constrained to L-inf = 1/255 with several
configurations (zeros init, many random restarts, longer schedules) and report
the QUANTIZED logit margin each achieves. This isolates whether a passing 1/255
solution exists at all, and how fragile it is -- the question the benchmark's
running-mean log cannot answer.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from model_utils import _ssim_per_image, quantize_ste  # noqa: E402
from ops_attack import EfficientNetV2LSurrogate  # noqa: E402
from paths import IMAGENET100_VAL_SAMPLES_DIR  # noqa: E402
from perturb_mirror.image_io import decode_image_b64, encode_image_b64  # noqa: E402
from perturb_mirror.model import resolve_target_index  # noqa: E402
from perturb_mirror.validator import build_challenge_spec  # noqa: E402

UNIT = 1.0 / 255.0


def load_clean_b64(samples_dir: Path, row: int) -> str:
    import base64
    import io

    import numpy as np
    from PIL import Image

    arr = np.load(samples_dir / f"{row:07d}.npy")
    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).clamp(0.0, 1.0)
    hwc = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def eval_q(model, data1, label1, delta1):
    adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
    dq = adv_q - data1
    linf = float(dq.abs().max().item())
    mse = float(dq.pow(2).mean().item())
    ssim = float(_ssim_per_image(data1, adv_q)[0].item())
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    logits = model(adv_q)[0]
    idx = int(label1.item())
    other = logits.clone()
    other[idx] = float("-inf")
    margin = float((other.max() - logits[idx]).item())
    return linf, margin, ssim, psnr


def cw_grad(model, data1, label1, delta):
    d = delta.detach().requires_grad_(True)
    logits = model(torch.clamp(data1 + d, 0.0, 1.0))[0]
    idx = int(label1.item())
    other = logits.clone()
    other[idx] = float("-inf")
    loss = other.max() - logits[idx]
    return torch.autograd.grad(loss, d)[0].detach()


def pgd_level1(model, data1, label1, init, iters, alpha_frac, decay=1.0):
    eps = UNIT
    alpha = max(eps * alpha_frac, UNIT)
    delta = init.clone()
    delta = torch.min(torch.max(delta, -data1), 1.0 - data1)
    momentum = torch.zeros_like(data1)
    best = delta.clone()
    best_m = -1e9
    for _ in range(iters):
        g = cw_grad(model, data1, label1, delta)
        momentum = decay * momentum + g / (g.abs().mean() + 1e-12)
        delta = delta + alpha * momentum.sign()
        delta = delta.clamp(-eps, eps)
        delta = torch.min(torch.max(delta, -data1), 1.0 - data1)
        _, m, _, _ = eval_q(model, data1, label1, delta)
        if m > best_m:
            best_m = m
            best = delta.detach().clone()
    return best, best_m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--long-iters", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    surrogate = EfficientNetV2LSurrogate(device)
    model = surrogate
    raw = surrogate.model

    clean_b64 = load_clean_b64(IMAGENET100_VAL_SAMPLES_DIR, args.row)
    challenge = build_challenge_spec(
        clean_image_b64=clean_b64, model=raw, device=device,
        epsilon=0.1, task_id=f"diag-{args.row}",
    )
    target_index = resolve_target_index(challenge.true_label)
    data1 = decode_image_b64(clean_b64).to(device).unsqueeze(0)
    label1 = torch.tensor([target_index], device=device)
    print(f"row {args.row}  true_label={challenge.true_label}  shape={tuple(data1.shape)}")
    print(f"accept_margin gate = 1.0 ; flip_margin (prune) = 2.0")
    print("-" * 60)

    torch.manual_seed(args.seed + args.row)

    # 1) zeros init, standard schedule
    d, m = pgd_level1(model, data1, label1, torch.zeros_like(data1), args.iters, 0.5)
    linf, mm, ssim, psnr = eval_q(model, data1, label1, d)
    print(f"zeros-init  iters={args.iters:<4} -> margin={mm:+.4f}  ssim={ssim:.4f}  psnr={psnr:.1f}")

    # 2) zeros init, long schedule, smaller alpha
    d, m = pgd_level1(model, data1, label1, torch.zeros_like(data1), args.long_iters, 0.25)
    linf, mm, ssim, psnr = eval_q(model, data1, label1, d)
    print(f"zeros-init  iters={args.long_iters:<4} a=.25 -> margin={mm:+.4f}  ssim={ssim:.4f}  psnr={psnr:.1f}")

    # 3) random restarts, standard schedule
    best_over = -1e9
    margins = []
    for r in range(args.restarts):
        init = torch.empty_like(data1).uniform_(-UNIT, UNIT)
        d, m = pgd_level1(model, data1, label1, init, args.iters, 0.5)
        _, mm, _, _ = eval_q(model, data1, label1, d)
        margins.append(mm)
        best_over = max(best_over, mm)
    margins.sort(reverse=True)
    n_pass = sum(1 for x in margins if x >= 1.0)
    print(f"rand x{args.restarts:<3} iters={args.iters:<4} -> best={best_over:+.4f}  "
          f">=1.0: {n_pass}/{args.restarts}  top5={[f'{x:+.3f}' for x in margins[:5]]}")
    print("-" * 60)
    verdict = "1/255 FEASIBLE" if best_over >= 1.0 else "1/255 INFEASIBLE at margin>=1.0"
    print(f"VERDICT: {verdict}  (best quantized margin at 1/255 = {best_over:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
