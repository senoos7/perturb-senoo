"""CWA ensemble attack adapted for Perturb/subnet26.

CWA: 'Rethinking Model Ensemble in Transfer-based Adversarial Attacks'
https://arxiv.org/abs/2303.09105

Design for this project:
  - Primary target: EfficientNetV2-L with validator-exact preprocessing (known white-box)
  - Surrogate models: ResNet50 + MobileNetV2 (fast, different architecture families)
  - CWA's outer momentum + inner per-model gradient exploration finds a consensus
    flip direction that is robust across surrogate feature spaces, which in practice
    discovers adversarial examples at lower L-inf than single-model MI-FGSM.
  - Budget refinement (from ops_attack): after CWA lands a flip, binary-search the
    minimum L-inf level + sparse L0 polish to maximise the validator reward.

Timing on A100 40GB:
  cwa_epochs=6, K=3 models  →  ~5-7s CWA
  budget sparse refinement   →  ~5-7s
  Total                      →  ≤16s
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms.functional as TF

from model_utils import _ssim_per_image, apply_validator_preprocess, quantize_ste
from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA, MIN_PSNR_DB, MIN_SSIM
from perturb_mirror.model import load_efficientnet_v2_l

QUANT_SAFE_MAX_LINF = math.floor(MAX_LINF_DELTA * 255.0) / 255.0  # 7/255


# ─── Surrogate wrappers ────────────────────────────────────────────────────────

class _ValidatorPreprocessSurrogate(nn.Module):
    """EfficientNetV2-L with the validator-exact 480px pipeline."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = load_efficientnet_v2_l(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(apply_validator_preprocess(x))


class _StandardPreprocessSurrogate(nn.Module):
    """Standard-ImageNet surrogate (224px, mean/std from torchvision defaults)."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        self.base = base_model
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_224 = TF.resize(x, [224], antialias=True)
        x_norm = (x_224 - self.mean.to(x.device)) / self.std.to(x.device)
        return self.base(x_norm)


def _load_surrogate_models(device: torch.device) -> list[nn.Module]:
    """Return [effnet_v2_l (validator-exact), resnet50, mobilenet_v2]."""
    effnet = _ValidatorPreprocessSurrogate(device)
    effnet.eval()

    resnet = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT).eval().to(device)
    for p in resnet.parameters():
        p.requires_grad_(False)

    mnet = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.DEFAULT).eval().to(device)
    for p in mnet.parameters():
        p.requires_grad_(False)

    return [
        effnet,
        _StandardPreprocessSurrogate(resnet).to(device),
        _StandardPreprocessSurrogate(mnet).to(device),
    ]


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class CWABudgetConfig:
    """Combined CWA + budget-minimizing objective.

    CWA phase finds a strong flip direction via ensemble consensus.
    Budget phase finds the minimum-norm version via L-inf level search + sparse PGD.
    """
    # --- CWA outer loop ---
    cwa_epochs: int = 6          # outer CWA iterations (trade-off: more = better flip, slower)
    epsilon: float = QUANT_SAFE_MAX_LINF  # max L-inf for CWA phase (7/255)
    alpha_frac: float = 0.5      # outer step size = epsilon * alpha_frac
    decay: float = 1.0           # outer momentum decay
    r_size: float = 16/255/15    # inner neighbourhood radius (original CWA default)
    inner_step_size: float = 250.0  # inner model step size (large: jumps to model's optimum)

    # --- Budget level search ---
    max_level: int = 7           # max L-inf level in 1/255 steps
    inner_iters: int = 10        # PGD iters per level
    restarts: int = 2            # restarts per level
    first_level_restarts: int = 6  # extra restarts at level 1 (highest reward)
    alpha_pgd_frac: float = 0.5  # PGD step size fraction within each level
    flip_margin: float = 2.0     # logit margin for accepting a level
    accept_margin: float = 1.0   # logit margin to accept (lower is more permissive)
    restart_batch_size: int = 8  # parallel GPU restarts

    # --- Sparse refinement ---
    sparse: bool = True
    sparse_steps: int = 8
    sparse_restarts: int = 1
    polish_rounds: int = 1

    # --- Gates (must match validator) ---
    min_delta: float = MIN_LINF_DELTA
    min_ssim: float = MIN_SSIM
    min_psnr_db: float = MIN_PSNR_DB
    eff_max_eps: float = MAX_LINF_DELTA


# ─── CWA + budget forward ─────────────────────────────────────────────────────

def _make_cwa_budget_forward(cfg: CWABudgetConfig, models: list[nn.Module]):
    """Build the forward() closure that implements CWA + budget minimization."""

    unit = 1.0 / 255.0
    K = len(models)
    # First model is always the validator-exact EfficientNetV2-L.
    primary = models[0]

    # ---- helpers ----

    def _eval(data1, label1, delta1):
        """Validator-exact metrics for one sample (no grad)."""
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        dq = adv_q - data1
        linf = float(dq.abs().max().item())
        mse  = float(dq.pow(2).mean().item())
        ssim = float(_ssim_per_image(data1, adv_q)[0].item())
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        logits = primary(adv_q)[0]
        idx = int(label1.item())
        true_logit = logits[idx]
        other = logits.clone(); other[idx] = float("-inf")
        margin = float((other.max() - true_logit).item())
        gates = (
            linf >= cfg.min_delta
            and linf <= cfg.eff_max_eps
            and ssim >= cfg.min_ssim
            and psnr >= cfg.min_psnr_db
        )
        return gates, linf, margin

    def _primary_grad(data1, label1, delta):
        """Margin-loss gradient on the primary model only."""
        d = delta.detach().requires_grad_(True)
        logits = primary(torch.clamp(data1 + d, 0.0, 1.0))[0]
        idx = int(label1.item())
        other = logits.clone(); other[idx] = float("-inf")
        loss = other.max() - logits[idx]
        return torch.autograd.grad(loss, d)[0].detach()

    # ---- CWA phase ----

    def _cwa_flip(data1, label1):
        """Run CWA outer loop; return the best delta found."""
        device = data1.device
        eps = cfg.epsilon
        alpha = eps * cfg.alpha_frac

        delta = torch.zeros_like(data1)
        outer_momentum = torch.zeros_like(data1)
        inner_momentum = torch.zeros_like(data1)

        best_delta = delta.clone()
        best_margin = float("-inf")

        for _ in range(cfg.cwa_epochs):
            original_delta = delta.clone().detach()

            # Outer gradient: ensemble mean of all K models
            ens_grad = torch.zeros_like(data1)
            for model in models:
                d = delta.detach().requires_grad_(True)
                x_adv = torch.clamp(data1 + d, 0.0, 1.0)
                logits = model(x_adv)
                idx = int(label1.item())
                other = logits[0].clone(); other[idx] = float("-inf")
                loss = other.max() - logits[0, idx]
                ens_grad = ens_grad + torch.autograd.grad(loss, d)[0].detach()
            ens_grad = ens_grad / K

            # Inner: move opposite to ensemble gradient (find neighbourhood adversary)
            inner_delta = original_delta.clone()
            # Step away from ensemble grad direction by r_size
            inner_delta = (inner_delta - cfg.r_size * ens_grad.sign()).clamp(-eps, eps)
            inner_delta = torch.min(torch.max(inner_delta, -data1), 1.0 - data1)

            # Inner per-model CWA step (one step each, large step size)
            inner_momentum = cfg.decay * inner_momentum
            for model in models:
                d = inner_delta.detach().requires_grad_(True)
                x_adv = torch.clamp(data1 + d, 0.0, 1.0)
                logits = model(x_adv)
                idx = int(label1.item())
                other = logits[0].clone(); other[idx] = float("-inf")
                loss = other.max() - logits[0, idx]
                g = torch.autograd.grad(loss, d)[0].detach()
                g_norm = g / (g.pow(2).sum().sqrt() + 1e-12)  # L2 normalise (CWA paper)
                inner_momentum = inner_momentum + g_norm
                inner_delta = (inner_delta + cfg.inner_step_size * inner_momentum).clamp(-eps, eps)
                inner_delta = torch.min(torch.max(inner_delta, -data1), 1.0 - data1)

            # Fake gradient: net movement from inner optimisation
            fake_grad = inner_delta - original_delta
            norm_p1 = fake_grad.abs().sum().clamp(min=1e-12)
            outer_momentum = cfg.decay * outer_momentum + fake_grad / norm_p1

            delta = (delta + alpha * outer_momentum.sign()).clamp(-eps, eps)
            delta = torch.min(torch.max(delta, -data1), 1.0 - data1)

            # Track the best flip seen so far
            _, _, margin = _eval(data1, label1, delta)
            if margin > best_margin:
                best_margin = margin
                best_delta = delta.clone()

        return best_delta

    # ---- Budget PGD at fixed L-inf level (with parallel restarts) ----

    def _pgd_at_level(data1, label1, level, init=None, restarts=None, target=None):
        eps   = level * unit
        alpha = max(eps * cfg.alpha_pgd_frac, unit)
        n_r   = max(1, int(restarts if restarts is not None else cfg.restarts))
        rbs   = max(1, cfg.restart_batch_size)
        idx   = int(label1.item())

        inits = []
        for r in range(n_r):
            if r == 0 and init is not None:
                d = init.detach().clamp(-eps, eps).clone()
            elif r == 0:
                d = torch.zeros_like(data1)
            else:
                d = torch.empty_like(data1).uniform_(-eps, eps)
            inits.append(torch.min(torch.max(d, -data1), 1.0 - data1))

        best_delta, best_key = None, float("-inf")

        for start in range(0, n_r, rbs):
            chunk = inits[start: start + rbs]
            bs = len(chunk)
            delta_b = torch.cat(chunk, dim=0)
            data_b  = data1.expand(bs, -1, -1, -1).contiguous()
            mom_b   = torch.zeros_like(delta_b)

            for _ in range(cfg.inner_iters):
                d = delta_b.detach().requires_grad_(True)
                logits = primary(torch.clamp(data_b + d, 0.0, 1.0))
                t_l = logits[:, idx]
                other = logits.clone(); other[:, idx] = float("-inf")
                loss = (other.max(dim=1).values - t_l).sum()
                g = torch.autograd.grad(loss, d)[0].detach()
                g_n = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
                mom_b = cfg.decay * mom_b + g_n
                delta_b = (delta_b + alpha * mom_b.sign()).clamp(-eps, eps)
                delta_b = torch.min(torch.max(delta_b, -data_b), 1.0 - data_b)

            for r in range(bs):
                d_r = delta_b[r:r+1].detach()
                gates, _, margin = _eval(data1, label1, d_r)
                key = margin + (1000.0 if gates else 0.0)
                if best_delta is None or key > best_key:
                    best_key, best_delta = key, d_r.clone()
                if target is not None and gates and margin >= target:
                    return best_delta

        return best_delta

    # ---- Sparse PGD (same as ops_attack) ----

    def _sparse_pgd(data1, label1, eps, k, init, steps, restarts=1):
        numel  = init.numel()
        n_r    = max(1, int(restarts))
        rbs    = max(1, cfg.restart_batch_size)
        idx    = int(label1.item())

        inits = []
        for r in range(n_r):
            d = init.detach().clone() if r == 0 else (
                torch.min(torch.max(
                    torch.empty_like(data1).uniform_(-eps, eps), -data1
                ), 1.0 - data1)
            )
            inits.append(d)

        best_delta, best_key = None, float("-inf")

        for start in range(0, n_r, rbs):
            chunk = inits[start: start + rbs]
            bs = len(chunk)
            delta_b = torch.cat(chunk, dim=0)
            data_b  = data1.expand(bs, -1, -1, -1).contiguous()
            mom_b   = torch.zeros_like(delta_b)

            for _ in range(steps):
                d = delta_b.detach().requires_grad_(True)
                logits = primary(torch.clamp(data_b + d, 0.0, 1.0))
                t_l = logits[:, idx]
                other = logits.clone(); other[:, idx] = float("-inf")
                loss = (other.max(dim=1).values - t_l).sum()
                g = torch.autograd.grad(loss, d)[0].detach()
                g_n = g / (g.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)
                mom_b = cfg.decay * mom_b + g_n

                flat_m = mom_b.abs().view(bs, -1)
                if k < numel:
                    topk_idx = torch.topk(flat_m, k, dim=1).indices
                    mask = torch.zeros(bs, numel, dtype=torch.bool, device=delta_b.device)
                    mask.scatter_(1, topk_idx, True)
                    mask = mask.view_as(mom_b)
                else:
                    mask = torch.ones_like(mom_b, dtype=torch.bool)

                delta_b = mask * (eps * mom_b.sign())
                delta_b = torch.min(torch.max(delta_b, -data_b), 1.0 - data_b)

            for r in range(bs):
                d_r = delta_b[r:r+1].detach()
                gates, _, margin = _eval(data1, label1, d_r)
                ok = gates and margin >= cfg.flip_margin
                active = int((d_r.abs() > 0).sum().item())
                key = (1e9 if ok else 0.0) - active + 1e-4 * margin
                if best_delta is None or key > best_key:
                    best_key, best_delta = key, d_r.clone()

        best_delta = best_delta.detach()
        gates, _, margin = _eval(data1, label1, best_delta)
        return best_delta, (gates and margin >= cfg.flip_margin), margin

    def _polish(data1, label1, delta1, eps):
        cur = delta1.detach().clone()
        gates, _, margin = _eval(data1, label1, cur)
        if not (gates and margin >= cfg.flip_margin):
            return cur
        for _ in range(cfg.polish_rounds):
            d = cur.detach().requires_grad_(True)
            logits = primary(torch.clamp(data1 + d, 0.0, 1.0))[0]
            idx = int(label1.item())
            other = logits.clone(); other[idx] = float("-inf")
            loss = other.max() - logits[idx]
            g = torch.autograd.grad(loss, d)[0].detach()
            contrib = (g * cur).flatten()
            active = cur.abs().flatten() > 0
            n_active = int(active.sum().item())
            if n_active <= 1:
                break
            ranked = torch.where(active, contrib, torch.full_like(contrib, float("inf")))
            order = torch.argsort(ranked)[:n_active]
            ptr, batch = 0, max(1, n_active // 2)
            removed_any = False
            while ptr < n_active and batch >= 1:
                take = order[ptr: ptr + batch]
                trial = cur.clone()
                tf = trial.flatten()
                tf[take] = 0.0
                trial = tf.view_as(cur)
                g2, _, m2 = _eval(data1, label1, trial)
                if g2 and m2 >= cfg.flip_margin:
                    cur = trial; ptr += batch; removed_any = True
                else:
                    batch //= 2
            if not removed_any:
                break
            k = int((cur.abs().flatten() > 0).sum().item())
            cur2, ok, _ = _sparse_pgd(data1, label1, eps, k, cur, cfg.sparse_steps, restarts=1)
            if ok:
                cur = cur2
        return cur.detach()

    def _sparse_one(data1, label1, delta1, eps):
        di = delta1[0]
        n_active = int(((di.abs() * 255.0).round() != 0).sum().item())
        if n_active <= 1:
            return delta1
        best = delta1.detach().clone()
        warm = delta1
        lo, hi = 1, n_active
        while lo <= hi:
            mid = (lo + hi) // 2
            d, ok, _ = _sparse_pgd(data1, label1, eps, mid, warm, cfg.sparse_steps, cfg.sparse_restarts)
            if ok:
                best, warm = d, d; hi = mid - 1
            else:
                lo = mid + 1
        return _polish(data1, label1, best, eps)

    # ---- Main forward ----

    def cwa_budget_forward(data: torch.Tensor, label: torch.Tensor):
        """Run CWA + budget for a batch of images (processes one at a time)."""
        device = data.device
        data   = data.clone().detach().to(device)
        label  = label.clone().detach().to(device)
        outs   = []

        for i in range(int(data.shape[0])):
            data1  = data[i:i+1]
            label1 = label[i:i+1]

            # Phase 1: CWA to find a strong flip direction (full epsilon)
            cwa_delta = _cwa_flip(data1, label1)

            # Phase 2: ascending L-inf level search using CWA delta as warm start.
            # At each level, CWA delta (clamped) is used as init for the first restart;
            # extra restarts add random initializations for harder images.
            chosen      = None
            chosen_ok   = False
            chosen_level = cfg.max_level
            warm = cwa_delta  # hand off CWA direction to budget PGD

            for level in range(1, cfg.max_level + 1):
                n_r = cfg.first_level_restarts if level == 1 else cfg.restarts
                d = _pgd_at_level(
                    data1, label1, level,
                    init=warm, restarts=n_r, target=cfg.flip_margin,
                )
                warm = d
                gates, _, margin = _eval(data1, label1, d)
                if gates and margin >= cfg.accept_margin:
                    chosen, chosen_ok, chosen_level = d, True, level
                    break
            if chosen is None:
                chosen = warm

            # Phase 3: sparse refinement to reduce RMSE at chosen level
            if cfg.sparse and chosen_ok:
                chosen = _sparse_one(data1, label1, chosen, chosen_level * unit)

            outs.append(chosen.detach())

        return torch.cat(outs, dim=0)

    return cwa_budget_forward


# ─── Public builder ───────────────────────────────────────────────────────────

def build_cwa_budget_attack(
    device: torch.device,
    *,
    cfg: CWABudgetConfig | None = None,
    preloaded_models: list[nn.Module] | None = None,
):
    """Build the CWA+budget attack.

    Args:
        device: torch device.
        cfg: CWABudgetConfig; uses defaults if None.
        preloaded_models: optional pre-loaded [effnetv2l, resnet50, mobilenet_v2]
            to avoid re-loading weights across calls.

    Returns:
        callable with signature (data: Tensor[B,C,H,W], label: Tensor[B]) -> delta Tensor
    """
    cfg = cfg or CWABudgetConfig()
    models = preloaded_models if preloaded_models is not None else _load_surrogate_models(device)
    for m in models:
        m.eval()
    return _make_cwa_budget_forward(cfg, models)
