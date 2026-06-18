"""Wrapper around the OPS attack (OPS/TransferAttack) pinned to EfficientNetV2-L.

OPS (Operator-Perturbation-based Stochastic optimization) is designed for
*transferable* attacks: it averages gradients over random image operators and
neighbor perturbations so a perturbation crafted on a surrogate transfers to
unknown black-box models.

For subnet26 the target model is known and fixed (EfficientNetV2-L), so there is
nothing to transfer to. This wrapper points OPS directly at EfficientNetV2-L
behind the validator's *exact* preprocess pipeline, turning OPS into a white-box
attack that overfits to EfficientNetV2-L instead of optimising for transfer.

Two knobs control the transfer-vs-overfit trade-off (see build_ops_attack):
  * num_sample_operator / num_sample_neighbor > 0  -> full OPS (transfer-robust)
  * either set to 0                                -> sampling off == MI-FGSM,
                                                      pure white-box overfit.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from model_utils import _ssim_per_image, apply_validator_preprocess, quantize_ste
from perturb_mirror.constants import (
    MAX_LINF_DELTA,
    MIN_LINF_DELTA,
    MIN_PSNR_DB,
    MIN_SSIM,
)
from perturb_mirror.model import load_efficientnet_v2_l

# OPS repo lives at <project_root>/OPS/TransferAttack (sibling of my_work).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_OPS_PKG_ROOT = _PROJECT_ROOT / "OPS" / "TransferAttack"

# Largest L-inf budget that still satisfies the validator's max-delta gate AFTER
# uint8 PNG rounding: a float delta d encodes to round(d * 255) / 255, so the
# biggest safe integer step is floor(MAX_LINF_DELTA * 255) / 255 (= 7/255 for
# the default 0.03 cap). Using exactly MAX_LINF_DELTA rounds up to 8/255 > 0.03.
QUANT_SAFE_MAX_LINF = math.floor(MAX_LINF_DELTA * 255.0) / 255.0


def _import_ops_class():
    """Import the upstream OPS class without modifying the OPS repo."""
    if not _OPS_PKG_ROOT.is_dir():
        raise FileNotFoundError(
            f"OPS repo not found at {_OPS_PKG_ROOT}. Expected OPS/TransferAttack "
            "to exist at the project root."
        )
    if str(_OPS_PKG_ROOT) not in sys.path:
        sys.path.insert(0, str(_OPS_PKG_ROOT))
    from transferattack.input_transformation.ops import OPS  # noqa: E402

    return OPS


class EfficientNetV2LSurrogate(nn.Module):
    """Validator-exact classifier head for OPS.

    OPS calls ``self.model(x)`` on images in pixel space ``[0, 1]``. The Perturb
    validator feeds the perturbed image through ``WEIGHTS.transforms()`` (resize
    to 480 + center crop + ImageNet normalize) before EfficientNetV2-L, so we
    fold that exact preprocess in here. This makes OPS gradients overfit to the
    same model+pipeline that scores the submission.

    Note: normalization is part of PREPROCESS, so unlike OPS's default
    ``wrap_model`` we do NOT add a separate Normalize layer (no double-norm).
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.model = load_efficientnet_v2_l(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(apply_validator_preprocess(x))


def build_ops_attack(
    device: torch.device,
    *,
    epsilon: float = QUANT_SAFE_MAX_LINF,
    num_iter: int = 10,
    num_sample_neighbor: int = 10,
    num_sample_operator: int = 20,
    beta: float = 2.0,
    decay: float = 1.0,
    targeted: bool = False,
    random_start: bool = False,
    surrogate: EfficientNetV2LSurrogate | None = None,
):
    """Construct an OPS attack instance bound to EfficientNetV2-L.

    Args:
        device: torch device for the model and tensors.
        epsilon: L-inf budget. Defaults to QUANT_SAFE_MAX_LINF (7/255) so that
            after uint8 PNG rounding the perturbation still satisfies the
            validator's max-delta gate (effective cap min(challenge_eps, 0.03)).
        num_iter: OPS iterations (alpha = epsilon / num_iter).
        num_sample_neighbor: neighbor perturbation samples per step.
        num_sample_operator: random-operator samples per neighbor.
            Set either *_sample_* to 0 for pure white-box MI-FGSM (overfit).
        beta, decay: OPS neighbor radius scale and momentum decay.
        targeted: targeted attack (expects [gt, target] labels) if True.
        random_start: random delta init.
        surrogate: reuse an existing surrogate (avoids reloading weights).

    Returns:
        An OPS instance whose ``__call__(data, label)`` returns delta.
        The bound classifier head is available as ``attack.model``.
    """
    OPS = _import_ops_class()
    head = surrogate if surrogate is not None else EfficientNetV2LSurrogate(device)

    class OPSEfficientNetV2L(OPS):
        """OPS pinned to the validator-exact EfficientNetV2-L head."""

        def load_model(self, model_name):  # noqa: ARG002 - name ignored on purpose
            return head

    attack = OPSEfficientNetV2L(
        model_name="efficientnet_v2_l",  # ignored by overridden load_model
        epsilon=float(epsilon),
        beta=float(beta),
        num_iter=int(num_iter),
        num_sample_neighbor=int(num_sample_neighbor),
        num_sample_operator=int(num_sample_operator),
        decay=float(decay),
        targeted=bool(targeted),
        random_start=bool(random_start),
        norm="linfty",
        loss="crossentropy",
        device=device,
    )
    return attack


def forwards_per_image(
    num_iter: int,
    num_sample_neighbor: int,
    num_sample_operator: int,
) -> int:
    """Estimate classifier forward passes per image (for cost reporting)."""
    if num_sample_neighbor * num_sample_operator > 0:
        per_step = num_sample_neighbor * num_sample_operator + 1
    else:
        per_step = 1
    return int(num_iter) * per_step


# ─── Center-crop mask ──────────────────────────────────────────────────────────


def _center_crop_mask(
    H: int, W: int, device: torch.device, crop_size: int = 480
) -> torch.Tensor:
    """Return a (1, 1, H, W) float32 binary mask covering the center-crop window.

    EfficientNetV2-L preprocessing: torchvision Resize(480) resizes the shorter
    side to 480 via int-truncation, then CenterCrop(480) takes the center square.
    Pixels *outside* this window are invisible to the model (gradient == 0 through
    the crop op), so zeroing their perturbation never hurts classification but
    always lowers RMSE — a free score gain.

    For non-square images the wasted area is 10–34% of total pixels (measured on
    the benchmark dataset which stores images at ~160px shortest side).
    """
    if H == W:
        return torch.ones(1, 1, H, W, device=device)

    # torchvision Resize(crop_size) with a single int:
    #   shorter side → crop_size; other → int(other * crop_size / shorter)
    if H <= W:
        new_H, new_W = crop_size, int(W * crop_size / H)
    else:
        new_H, new_W = int(H * crop_size / W), crop_size

    crop_top = (new_H - crop_size) // 2
    crop_left = (new_W - crop_size) // 2

    scale_y = new_H / H
    scale_x = new_W / W
    # Conservative (inclusive) mapping back to original pixel coordinates
    y0 = max(0, int(math.floor(crop_top / scale_y)))
    y1 = min(H, int(math.ceil((crop_top + crop_size) / scale_y)))
    x0 = max(0, int(math.floor(crop_left / scale_x)))
    x1 = min(W, int(math.ceil((crop_left + crop_size) / scale_x)))

    m = torch.zeros(1, 1, H, W, device=device)
    m[:, :, y0:y1, x0:x1] = 1.0
    return m


# ─── Budget-minimizing (L-inf / SSIM-aware) objective ──────────────────────────
#
# Vanilla OPS only maximises cross-entropy: it flips the label but spends the
# whole L-inf budget, which tanks SSIM/PSNR and wins zero validator reward.
#
# The validator reward (gates passed, SPEED_WEIGHT == 0) is dominated by the
# SMALLEST L-inf just above the 0.003 floor that still flips, with SSIM >= 0.98
# and PSNR >= 38 dB. So we want a *minimum-norm* adversarial example, not a
# max-confidence one.
#
# OPSBudgetMin keeps OPS's (optionally operator/neighbor-sampled, model-overfit)
# gradient as the flip direction, and adds a differentiable objective that:
#   * shrinks L-inf (top-k STE) and RMSE once the sample already flips,
#   * holds L-inf above the min-delta floor (avoids `below_min_delta`),
#   * lifts SSIM and PSNR over their gates (hinge penalties),
# all evaluated on the uint8-quantized image so the in-loop gates match the
# validator byte-for-byte. The smallest passing iterate is returned per image.


@dataclass
class BudgetConfig:
    """Highest-score (minimum-norm) objective for the validator reward.

    The reward (gates passed, SPEED_WEIGHT == 0) is
        0.7 * (1 - (linf - 0.003)/(eff_max - 0.003))**2  +  0.3 * (1 - rmse/eff_max)**2
    with eff_max == 0.03. After uint8 rounding L-inf is an integer number of
    1/255 steps, so there are only 7 feasible L-inf levels (1/255 .. 7/255). The
    score is maximised by the SMALLEST level that flips (L-inf term) and the
    FEWEST changed pixels at that level (RMSE term). Strategy:

      1. Ascending-scan the smallest feasible level in {1..max_level}/255,
         running a strong PGD (many iters + momentum + restarts) at each level.
         Level 1 gets `first_level_restarts` extra random restarts because it is
         0.7-weighted and worth ~0.2 over level 2, so even hard images flip at
         1/255 rather than slipping to 2/255.
      2. Sparse L0 refinement at the chosen level: binary-search the minimum
         support (re-optimised per k, with restarts) that keeps the flip, then a
         greedy backward-elimination polish drops any remaining redundant pixels.
         Both cut RMSE, which (at fixed 1/255) is sqrt(k/N)/255.

    A logit ``flip_margin`` buffer makes solutions survive the validator's own
    (independent, slightly non-deterministic) forward pass.
    """

    max_level: int = 7          # max L-inf in 1/255 units (floor(0.03*255) == 7)
    inner_iters: int = 18       # PGD steps per level  (15 -> 18: stronger convergence)
    restarts: int = 2           # PGD restarts per level (>=1; >1 adds random init)
    first_level_restarts: int = 12  # extra restarts at level 1 (8 -> 12: more exploration, all in one rbs=12 batch)
    alpha_frac: float = 0.5     # PGD step size as a fraction of eps (>= 1/255)
    decay: float = 1.0          # momentum decay
    min_delta: float = MIN_LINF_DELTA
    min_ssim: float = MIN_SSIM
    min_psnr_db: float = MIN_PSNR_DB
    eff_max_eps: float = MAX_LINF_DELTA
    flip_margin: float = 1.5    # logit margin required when PRUNING (2.0 -> 1.5: allows sparse on marginal flips)
    accept_margin: float = 0.5  # logit margin to accept an L-inf level (1.0 -> 0.5: accept thin 1/255 flips)
    sparse: bool = True
    sparse_steps: int = 8       # PGD steps per support size in the k binary search
    sparse_restarts: int = 1    # restarts of the support-restricted PGD
    polish_rounds: int = 1      # backward-elimination passes after the k search
    restart_batch_size: int = 12  # restarts run in parallel on GPU (match first_level_restarts=12)


def _make_budget_forward(cfg: BudgetConfig):
    """Build the highest-score (level-search + sparse) forward() for the config."""

    unit = 1.0 / 255.0

    def _eval(model, data1, label1, delta1):
        """Validator-exact metrics for one sample on the uint8-quantized image.

        Returns (gates_ok, linf, margin) where gates_ok covers delta/SSIM/PSNR
        bounds and ``margin`` is the (best_other - true) logit gap (>0 == flip).
        """
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        dq = adv_q - data1
        linf = float(dq.abs().max().item())
        mse = float(dq.pow(2).mean().item())
        ssim = float(_ssim_per_image(data1, adv_q)[0].item())
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        logits = model(adv_q)[0]
        idx = int(label1.item())
        true_logit = logits[idx]
        other = logits.clone()
        other[idx] = float("-inf")
        margin = float((other.max() - true_logit).item())
        gates = (
            (linf >= cfg.min_delta)
            and (linf <= cfg.eff_max_eps)
            and (ssim >= cfg.min_ssim)
            and (psnr >= cfg.min_psnr_db)
        )
        return gates, linf, margin

    def budget_forward(self, data: torch.Tensor, label: torch.Tensor, **kwargs):
        device = self.device
        data = data.clone().detach().to(device)
        label = label.clone().detach().to(device)
        model = self.model

        def _grad(data1, label1, delta):
            """Gradient of the CW logit-margin loss (max_other - true) wrt delta.

            We attack the EXACT quantity the validator gates on. Unlike CE -- whose
            gradient saturates once the sample crosses the boundary -- the margin
            loss keeps widening the (best_other - true) gap, which buys the extra
            logit margin needed to flip hard images at the 1/255 floor.
            """
            d = delta.detach().requires_grad_(True)
            logits = model(torch.clamp(data1 + d, 0.0, 1.0))[0]
            idx = int(label1.item())
            true_logit = logits[idx]
            other = logits.clone()
            other[idx] = float("-inf")
            loss = other.max() - true_logit
            return torch.autograd.grad(loss, d)[0].detach()

        def _pgd_at_level(data1, label1, level, init=None, restarts=None, target=None, mask=None):
            """Strong margin-PGD constrained to L-inf = level/255.

            Restarts are processed in parallel sub-batches of cfg.restart_batch_size
            on the GPU. All restarts within a sub-batch share a single forward/backward
            pass per PGD iteration, fully utilizing available VRAM.

            Returns the best delta over all restarts, preferring gate-passing
            solutions and then the largest flip margin. ``target`` enables early-stop
            across sub-batches: once any restart passes gates with margin >= target,
            remaining sub-batches are skipped.

            ``mask`` (optional, [1,1,H,W]) zeros perturbation outside the model's
            center-crop window — those pixels are invisible to the classifier so
            zeroing them is a free RMSE reduction.
            """
            eps = level * unit
            alpha = max(eps * cfg.alpha_frac, unit)
            n_restarts = max(1, int(restarts if restarts is not None else cfg.restarts))
            rbs = max(1, cfg.restart_batch_size)
            idx = int(label1.item())

            # Build all restart initializations upfront
            inits = []
            for r in range(n_restarts):
                if r == 0 and init is not None:
                    d = init.detach().clamp(-eps, eps).clone()
                elif r == 0:
                    d = torch.zeros_like(data1)
                else:
                    d = torch.empty_like(data1).uniform_(-eps, eps)
                    if mask is not None:
                        d = d * mask  # zero outside-crop pixels in random restarts
                inits.append(torch.min(torch.max(d, -data1), 1.0 - data1))

            best_delta = None
            best_key = float("-inf")

            for batch_start in range(0, n_restarts, rbs):
                chunk = inits[batch_start : batch_start + rbs]
                bs = len(chunk)
                # [bs, C, H, W] — all restarts in this sub-batch run in parallel
                delta_b = torch.cat(chunk, dim=0)
                data_b = data1.expand(bs, -1, -1, -1).contiguous()
                momentum_b = torch.zeros_like(delta_b)

                for _ in range(int(cfg.inner_iters)):
                    d = delta_b.detach().requires_grad_(True)
                    logits = model(torch.clamp(data_b + d, 0.0, 1.0))  # [bs, C]
                    true_logits = logits[:, idx]
                    other = logits.clone()
                    other[:, idx] = float("-inf")
                    # Sum loss so grad[i] == d(margin_i)/d(delta[i]) for each restart
                    loss = (other.max(dim=1).values - true_logits).sum()
                    grads = torch.autograd.grad(loss, d)[0].detach()  # [bs, C, H, W]
                    # Normalize per restart independently (not across the batch)
                    g_norm = grads / (grads.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
                    momentum_b = cfg.decay * momentum_b + g_norm
                    delta_b = delta_b + alpha * momentum_b.sign()
                    delta_b = delta_b.clamp(-eps, eps)
                    delta_b = torch.min(torch.max(delta_b, -data_b), 1.0 - data_b)
                    if mask is not None:
                        delta_b = delta_b * mask  # enforce zero outside center crop

                # Evaluate each restart and track the best
                for r in range(bs):
                    d_r = delta_b[r : r + 1].detach()
                    gates, _, margin = _eval(model, data1, label1, d_r)
                    key = margin + (1000.0 if gates else 0.0)
                    if best_delta is None or key > best_key:
                        best_key = key
                        best_delta = d_r.clone()
                    if target is not None and gates and margin >= target:
                        return best_delta  # early-stop across sub-batches

            return best_delta

        def _sparse_pgd(data1, label1, eps, k, init, steps, restarts=1, mask=None):
            """Iterative hard-thresholding PGD: keep the top-k gradient-salient
            elements at +/-eps and re-optimise their sign pattern each step.

            Unlike prune-from-dense, the surviving pixels are RE-OPTIMISED, so a
            far smaller support can still rebuild the flip margin. L-inf stays at
            eps (== chosen level), so RMSE ~ sqrt(k/N)*eps shrinks with k.

            Restarts are processed in parallel sub-batches of cfg.restart_batch_size.
            Returns the best (delta, ok, margin) over all restarts.

            ``mask`` (optional, [1,1,H,W]) zeroes outside-crop pixels in random
            restarts; crop pixels are naturally preferred by top-k once momentum
            accumulates since outside pixels always have zero gradient.
            """
            numel = init.numel()  # C*H*W per image
            n_restarts = max(1, int(restarts))
            rbs = max(1, cfg.restart_batch_size)
            idx = int(label1.item())

            inits = []
            for r in range(n_restarts):
                if r == 0:
                    d = init.detach().clone()
                else:
                    d = torch.empty_like(data1).uniform_(-eps, eps)
                    if mask is not None:
                        d = d * mask  # zero outside-crop pixels in random restarts
                    d = torch.min(torch.max(d, -data1), 1.0 - data1)
                inits.append(d)

            best_delta = None
            best_key = float("-inf")

            for batch_start in range(0, n_restarts, rbs):
                chunk = inits[batch_start : batch_start + rbs]
                bs = len(chunk)
                delta_b = torch.cat(chunk, dim=0)          # [bs, C, H, W]
                data_b = data1.expand(bs, -1, -1, -1).contiguous()
                momentum_b = torch.zeros_like(delta_b)
                chw = numel

                for _ in range(int(steps)):
                    d = delta_b.detach().requires_grad_(True)
                    logits = model(torch.clamp(data_b + d, 0.0, 1.0))  # [bs, C]
                    true_logits = logits[:, idx]
                    other = logits.clone()
                    other[:, idx] = float("-inf")
                    loss = (other.max(dim=1).values - true_logits).sum()
                    grads = torch.autograd.grad(loss, d)[0].detach()   # [bs, C, H, W]
                    g_norm = grads / (grads.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
                    momentum_b = cfg.decay * momentum_b + g_norm
                    # Top-k support applied per restart independently
                    flat_m = momentum_b.abs().view(bs, -1)              # [bs, chw]
                    if k < chw:
                        topk_idx = torch.topk(flat_m, k, dim=1).indices
                        support = torch.zeros(bs, chw, dtype=torch.bool, device=delta_b.device)
                        support.scatter_(1, topk_idx, True)
                        support = support.view_as(momentum_b)
                    else:
                        support = torch.ones_like(momentum_b, dtype=torch.bool)
                    delta_b = support * (eps * momentum_b.sign())
                    delta_b = torch.min(torch.max(delta_b, -data_b), 1.0 - data_b)

                # Evaluate each restart; prefer passing + fewest active pixels + margin
                for r in range(bs):
                    d_r = delta_b[r : r + 1].detach()
                    gates, _, margin = _eval(model, data1, label1, d_r)
                    ok = gates and margin >= cfg.flip_margin
                    active = int((d_r.abs() > 0).sum().item())
                    key = (1.0e9 if ok else 0.0) - active + 1e-4 * margin
                    if best_delta is None or key > best_key:
                        best_key = key
                        best_delta = d_r.clone()

            best_delta = best_delta.detach()
            gates, _, margin = _eval(model, data1, label1, best_delta)
            return best_delta, (gates and margin >= cfg.flip_margin), margin

        def _polish(data1, label1, delta1, eps, mask=None):
            """Greedy backward elimination after the k search.

            The top-k support PGD returns keeps the k most salient pixels, but
            some are redundant once the others lock in the flip. Here we rank the
            active pixels by their current contribution to the margin
            (grad * delta; small/negative == least useful) and try to ZERO the
            least-useful ones in batches, re-verifying the gates + flip_margin
            after every removal. Removal-only keeps L-inf at eps, so every
            accepted drop strictly lowers RMSE. After each round we re-optimise
            the surviving support so later rounds can remove still more.
            """
            cur = delta1.detach().clone()
            gates, _, margin = _eval(model, data1, label1, cur)
            if not (gates and margin >= cfg.flip_margin):
                return cur  # nothing safe to trim
            for _ in range(int(cfg.polish_rounds)):
                g = _grad(data1, label1, cur)
                contrib = (g * cur).flatten()
                active = cur.abs().flatten() > 0
                n_active = int(active.sum().item())
                if n_active <= 1:
                    break
                # least-useful active pixels first; inactive pushed to the end.
                ranked = torch.where(
                    active, contrib, torch.full_like(contrib, float("inf"))
                )
                order = torch.argsort(ranked)[:n_active]
                ptr = 0
                batch = max(1, n_active // 2)
                removed_any = False
                while ptr < n_active and batch >= 1:
                    take = order[ptr : ptr + batch]
                    trial = cur.clone()
                    tf = trial.flatten()
                    tf[take] = 0.0
                    trial = tf.view_as(cur)
                    gates, _, margin = _eval(model, data1, label1, trial)
                    if gates and margin >= cfg.flip_margin:
                        cur = trial
                        ptr += batch
                        removed_any = True
                    else:
                        batch //= 2
                if not removed_any:
                    break
                k = int((cur.abs().flatten() > 0).sum().item())
                # Re-optimise the surviving support from the warm start only
                # (restarts=1): polish already has a good support, so random
                # reseeds here add downside risk without RMSE benefit.
                cur2, ok, _ = _sparse_pgd(
                    data1, label1, eps, k, cur, cfg.sparse_steps, restarts=1, mask=mask
                )
                if ok:
                    cur = cur2
            return cur.detach()

        def _sparse_one(data1, label1, delta1, eps, mask=None):
            """Binary-search the minimum number of perturbed elements (k) for
            which support-restricted PGD still passes every gate + flip_margin,
            then greedily polish away any remaining redundant pixels."""
            di = delta1[0]
            n_active = int(((di.abs() * 255.0).round() != 0).sum().item())
            if n_active <= 1:
                return delta1
            best = delta1.detach().clone()
            warm = delta1
            lo, hi = 1, n_active
            while lo <= hi:
                mid = (lo + hi) // 2
                d, ok, _ = _sparse_pgd(
                    data1, label1, eps, mid, warm, cfg.sparse_steps, cfg.sparse_restarts,
                    mask=mask,
                )
                if ok:
                    best, warm = d, d
                    hi = mid - 1
                else:
                    lo = mid + 1
            return _polish(data1, label1, best, eps, mask=mask)

        outs = []
        for i in range(int(data.shape[0])):
            data1 = data[i : i + 1]
            label1 = label[i : i + 1]

            # Compute center-crop mask: pixels outside the 480×480 crop window
            # are invisible to EfficientNetV2-L (gradient == 0 through the crop
            # op), so we zero them in every delta — a free RMSE reduction.
            H, W = data1.shape[2], data1.shape[3]
            crop_mask = _center_crop_mask(H, W, device)  # [1, 1, H, W]

            # Ascending, SCORE-AWARE level search. The L-inf term carries 0.7
            # weight, so a lower level almost always wins even if it costs all
            # sparsity (1/255 dense ~0.88 >> 2/255 sparse ~0.76). We therefore
            # accept the smallest level that flips with only `accept_margin`
            # (conservative-but-lower); the heavier `flip_margin` is reserved for
            # pruning, where thin margins are fragile. If the accepted level's
            # dense margin is below flip_margin, the sparse step simply returns
            # the dense delta (still 1/255 -- the big win), instead of falling
            # back to a higher level.
            chosen = None
            chosen_ok = False
            chosen_level = cfg.max_level
            warm = None
            for level in range(1, cfg.max_level + 1):
                # Level 1 is 0.7-weighted and worth ~0.2 over level 2, so spend
                # extra restarts there to reliably flip borderline-hard images.
                # Early-stop once a restart clears flip_margin so easy images
                # don't pay for the extra restarts (only hard ones do).
                n_restarts = cfg.first_level_restarts if level == 1 else cfg.restarts
                d = _pgd_at_level(
                    data1, label1, level,
                    init=warm, restarts=n_restarts, target=cfg.flip_margin,
                    mask=crop_mask,
                )
                warm = d
                gates, _, margin = _eval(model, data1, label1, d)
                if gates and margin >= cfg.accept_margin:
                    chosen, chosen_ok, chosen_level = d, True, level
                    break
            if chosen is None:
                chosen = warm  # best effort (nothing flipped within budget)

            if cfg.sparse and chosen_ok:
                chosen = _sparse_one(data1, label1, chosen, chosen_level * unit,
                                     mask=crop_mask)
            outs.append(chosen.detach())

        return torch.cat(outs, dim=0)

    return budget_forward


def build_ops_budget_attack(
    device: torch.device,
    *,
    num_sample_neighbor: int = 0,
    num_sample_operator: int = 0,
    beta: float = 2.0,
    decay: float = 1.0,
    budget: BudgetConfig | None = None,
    surrogate: EfficientNetV2LSurrogate | None = None,
):
    """OPS pinned to EfficientNetV2-L with a minimum-norm (L-inf/SSIM-aware) loop.

    Reuses OPS's averaged-gradient flip direction (operator/neighbor sampling is
    optional; default off == fast white-box overfit) but replaces the plain
    CE-ascent forward with a budget-minimizing objective that targets the
    validator reward directly.

    Returns an OPS instance whose ``__call__(data, label)`` returns the
    minimum-norm delta; ``attack.model`` is the validator-exact classifier head.
    """
    OPS = _import_ops_class()
    cfg = budget or BudgetConfig()
    head = surrogate if surrogate is not None else EfficientNetV2LSurrogate(device)
    budget_forward = _make_budget_forward(cfg)

    class OPSBudgetMinEffNet(OPS):
        """OPS + minimum-norm objective, bound to EfficientNetV2-L."""

        def load_model(self, model_name):  # noqa: ARG002 - name ignored on purpose
            return head

        def forward(self, data, label, **kwargs):
            return budget_forward(self, data, label, **kwargs)

    attack = OPSBudgetMinEffNet(
        model_name="efficientnet_v2_l",
        epsilon=float(cfg.max_level) / 255.0,
        beta=float(beta),
        num_iter=int(cfg.inner_iters),
        num_sample_neighbor=int(num_sample_neighbor),
        num_sample_operator=int(num_sample_operator),
        decay=float(decay),
        targeted=False,
        random_start=False,
        norm="linfty",
        loss="crossentropy",
        device=device,
    )
    return attack
