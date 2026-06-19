"""L2T (Learning to Transform) attack adapted for Perturb/subnet26.

L2T: 'Learning to Transform Dynamically for Better Adversarial Transferability'
https://arxiv.org/abs/2405.14077  (CVPR 2024)

Core idea: instead of randomly sampling from a fixed augmentation set, L2T
maintains a learnable policy vector `aug_param` over the op library and updates
it via REINFORCE-style policy gradient at every PGD step. Over iterations it
concentrates probability mass on the transformations that maximise the adversarial
loss for THIS image, finding a more consistent flip direction.

Adaptation for Perturb (white-box, known EfficientNetV2-L target):
  - All ops are size-agnostic (no hard-coded 224px) so they compose correctly
    with apply_validator_preprocess (resize → 480px, mean=std=0.5).
  - Phase 1: L2T policy-learning loop (finds the best flip direction).
  - Phase 2: Budget level search initialised from the L2T delta (same as OPS).
  - Phase 3: Sparse L0 refinement + polish (same as OPS).

Timing budget on A100 40GB:
  l2t_epochs=4, num_scale=3  →  ~3-4s
  budget level search         →  ~9-10s  (inner_iters=10, reduced vs OPS)
  sparse refinement            →  ~2s
  Total                        →  ≤16s
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from model_utils import _ssim_per_image, apply_validator_preprocess, quantize_ste
from perturb_mirror.constants import MAX_LINF_DELTA, MIN_LINF_DELTA, MIN_PSNR_DB, MIN_SSIM
from perturb_mirror.model import load_efficientnet_v2_l

QUANT_SAFE_MAX_LINF = math.floor(MAX_LINF_DELTA * 255.0) / 255.0  # 7/255

_softmax = torch.nn.Softmax(dim=0)


# ─── Augmentation op library (size-agnostic) ──────────────────────────────────
# All ops take a [B, C, H, W] tensor in [0,1] and return same-shape tensor.
# We deliberately exclude ops that hard-code 224px (crop/ssm from the original).

def _vshift(x: torch.Tensor) -> torch.Tensor:
    step = torch.randint(0, x.shape[2], (1,)).item()
    return x.roll(int(step), dims=2)

def _hshift(x: torch.Tensor) -> torch.Tensor:
    step = torch.randint(0, x.shape[3], (1,)).item()
    return x.roll(int(step), dims=3)

def _make_dim(resize_rate: float):
    """Random resize-pad-resize (Diverse Input Method), size-agnostic."""
    import torch.nn.functional as F
    def _dim(x: torch.Tensor) -> torch.Tensor:
        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)
        lo, hi = min(img_size, img_resize), max(img_size, img_resize)
        if lo >= hi:
            return x
        rnd = int(torch.randint(lo, hi, (1,)).item())
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
        h_rem = img_resize - rnd
        pad_top = int(torch.randint(0, max(1, h_rem), (1,)).item())
        pad_bot = h_rem - pad_top
        pad_left = int(torch.randint(0, max(1, h_rem), (1,)).item())
        pad_right = h_rem - pad_left
        padded = F.pad(rescaled, [pad_left, pad_right, pad_top, pad_bot], value=0)
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return _dim

def _add_noise(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.empty_like(x).uniform_(-16/255, 16/255)).clamp(0.0, 1.0)

def _scale_half(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5

def _scale_third(x: torch.Tensor) -> torch.Tensor:
    return x * (1.0 / 3.0)

def _scale_quarter(x: torch.Tensor) -> torch.Tensor:
    return x * 0.25

# Build the op table once at module load (all lambdas/closures are picklable).
_OP_TABLE: list[tuple[str, object]] = [
    ("identity",   lambda x: x),
    ("vflip",      lambda x: x.flip(2)),
    ("hflip",      lambda x: x.flip(3)),
    ("vshift",     _vshift),
    ("hshift",     _hshift),
    ("rot30",      lambda x: TF.rotate(x, 30)),
    ("rot60",      lambda x: TF.rotate(x, 60)),
    ("rot90",      lambda x: x.rot90(1, (2, 3))),
    ("rot120",     lambda x: TF.rotate(x, 120)),
    ("rot150",     lambda x: TF.rotate(x, 150)),
    ("rot180",     lambda x: x.rot90(2, (2, 3))),
    ("rot_n30",    lambda x: TF.rotate(x, -30)),
    ("rot_n60",    lambda x: TF.rotate(x, -60)),
    ("rot_n90",    lambda x: x.rot90(3, (2, 3))),
    ("scale_half", _scale_half),
    ("scale_1/3",  _scale_third),
    ("scale_1/4",  _scale_quarter),
    ("noise",      _add_noise),
    ("dim_1.1",    _make_dim(1.1)),
    ("dim_1.2",    _make_dim(1.2)),
    ("dim_1.3",    _make_dim(1.3)),
]
_N_OPS = len(_OP_TABLE)
_OPS   = [fn for _, fn in _OP_TABLE]


def _select_ops(aug_param: torch.Tensor, ops_num: int) -> list[int]:
    """Sample `ops_num` op indices from the learned categorical distribution."""
    probs = _softmax(aug_param)
    return torch.multinomial(probs, ops_num, replacement=True).tolist()


def _trace_prob(aug_param: torch.Tensor, op_ids: list[int]) -> torch.Tensor:
    """Product of sampled op probabilities (for REINFORCE log-prob surrogate)."""
    probs = _softmax(aug_param)
    tp = probs[op_ids[0]]
    for idx in op_ids[1:]:
        tp = tp * probs[idx]
    return tp


def _apply_ops(x: torch.Tensor, op_ids: list[int]) -> torch.Tensor:
    """Chain ops in order."""
    for idx in op_ids:
        x = _OPS[idx](x)
    return x


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class L2TBudgetConfig:
    """L2T policy-learning phase + budget-minimizing level search + sparse."""

    # --- L2T phase ---
    l2t_epochs: int = 4          # outer PGD+policy iterations
    num_scale: int = 3           # augmented views sampled per PGD step
    ops_num: int = 2             # ops chained per augmented view
    aug_lr: float = 0.01         # policy-gradient learning rate for aug_param
    l2t_epsilon: float = QUANT_SAFE_MAX_LINF  # L-inf for L2T phase
    l2t_alpha_frac: float = 0.5  # step size = epsilon * alpha_frac
    decay: float = 1.0           # momentum decay

    # --- Budget level search ---
    max_level: int = 7
    inner_iters: int = 10        # PGD steps per level (slightly fewer than OPS to compensate L2T time)
    restarts: int = 2
    first_level_restarts: int = 6
    alpha_pgd_frac: float = 0.5
    flip_margin: float = 2.0
    accept_margin: float = 1.0
    restart_batch_size: int = 8

    # --- Sparse refinement ---
    sparse: bool = True
    sparse_steps: int = 8
    sparse_restarts: int = 1
    polish_rounds: int = 1

    # --- Gates ---
    min_delta: float = MIN_LINF_DELTA
    min_ssim: float = MIN_SSIM
    min_psnr_db: float = MIN_PSNR_DB
    eff_max_eps: float = MAX_LINF_DELTA


# ─── Surrogate ────────────────────────────────────────────────────────────────

class _EfficientNetV2LSurrogate(nn.Module):
    """Validator-exact surrogate: pixel [0,1] → apply_validator_preprocess → logits."""

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.model = load_efficientnet_v2_l(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(apply_validator_preprocess(x))


# ─── L2T + budget forward ─────────────────────────────────────────────────────

def _make_l2t_budget_forward(cfg: L2TBudgetConfig, primary: nn.Module):

    unit = 1.0 / 255.0

    # ---- helpers ----

    def _eval(data1, label1, delta1):
        adv_q = quantize_ste(torch.clamp(data1 + delta1, 0.0, 1.0))
        dq = adv_q - data1
        linf = float(dq.abs().max().item())
        mse  = float(dq.pow(2).mean().item())
        ssim = float(_ssim_per_image(data1, adv_q)[0].item())
        psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
        logits = primary(adv_q)[0]
        idx = int(label1.item())
        other = logits.clone(); other[idx] = float("-inf")
        margin = float((other.max() - logits[idx]).item())
        gates = (
            linf >= cfg.min_delta and linf <= cfg.eff_max_eps
            and ssim >= cfg.min_ssim and psnr >= cfg.min_psnr_db
        )
        return gates, linf, margin

    # ---- Phase 1: L2T policy learning ----

    def _l2t_phase(data1, label1):
        """Run L2T's REINFORCE-style augmentation policy update.

        At each epoch:
          1. For `num_scale` sampled policies: apply ops to x_adv, forward pass,
             record loss and op-probability.
          2. Update delta with the mean gradient (MI-FGSM style).
          3. Update aug_param: ∇_θ E[p(τ) * L] = Σ p(τ) * ∇_θ log p(τ) * L
             (REINFORCE without baseline; approximate with p(τ) * L directly
             since aug_param enters through softmax inside trace_prob).
        """
        device  = data1.device
        eps     = cfg.l2t_epsilon
        alpha   = max(eps * cfg.l2t_alpha_frac, unit)
        idx     = int(label1.item())

        aug_param = torch.zeros(_N_OPS, device=device, requires_grad=False)

        delta    = torch.zeros_like(data1)
        momentum = torch.zeros_like(data1)
        best_delta, best_margin = delta.clone(), float("-inf")

        for _ in range(cfg.l2t_epochs):
            # --- sample num_scale augmented views and compute gradients ---
            grad_accum = torch.zeros_like(data1)
            aug_losses: list[torch.Tensor] = []
            aug_probs:  list[torch.Tensor] = []

            aug_param_leaf = aug_param.detach().requires_grad_(True)

            for _ in range(cfg.num_scale):
                op_ids = _select_ops(aug_param_leaf, cfg.ops_num)
                aug_prob = _trace_prob(aug_param_leaf, op_ids)

                # Apply augmentation to the current adversarial input
                x_aug = _apply_ops((data1 + delta).clamp(0.0, 1.0).detach(), op_ids)

                d = delta.detach().requires_grad_(True)
                # Re-apply aug to x_aug + (d - delta) so gradient flows through d.
                # Simpler: gradient of loss on un-augmented adv w.r.t. d is sufficient
                # for the delta update; augmentation only diversifies the direction.
                x_in = _apply_ops((data1 + d).clamp(0.0, 1.0), op_ids)
                logits = primary(x_in)
                true_l = logits[0, idx]
                other  = logits[0].clone(); other[idx] = float("-inf")
                loss   = other.max() - true_l  # margin loss (no saturation)

                g = torch.autograd.grad(loss, d, retain_graph=False)[0].detach()
                grad_accum = grad_accum + g

                aug_losses.append(loss.detach().reshape(1))
                aug_probs.append(aug_prob)

            # --- update delta via mean gradient + momentum ---
            avg_grad = grad_accum / cfg.num_scale
            g_norm = avg_grad / (avg_grad.abs().mean() + 1e-12)
            momentum = cfg.decay * momentum + g_norm
            delta = (delta + alpha * momentum.sign()).clamp(-eps, eps)
            delta = torch.min(torch.max(delta, -data1), 1.0 - data1)

            # --- REINFORCE update for aug_param ---
            # Maximise E_{τ~π_θ}[p(τ) * loss] — helps ops that yield high loss
            aug_loss_tensor = torch.cat(aug_losses)  # [num_scale]
            aug_prob_tensor = torch.stack(aug_probs)  # [num_scale] (requires_grad via aug_param_leaf)
            aug_objective = (aug_prob_tensor * aug_loss_tensor).mean()
            aug_grad = torch.autograd.grad(aug_objective, aug_param_leaf)[0].detach()
            # Gradient ASCENT on aug_param (maximise attack loss)
            aug_param = (aug_param.detach() + cfg.aug_lr * aug_grad)

            # track best flip
            _, _, margin = _eval(data1, label1, delta)
            if margin > best_margin:
                best_margin, best_delta = margin, delta.clone()

        return best_delta

    # ---- Phase 2: Budget PGD level search ----

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
            chunk  = inits[start: start + rbs]
            bs     = len(chunk)
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

    # ---- Phase 3: Sparse refinement ----

    def _sparse_pgd(data1, label1, eps, k, init, steps, restarts=1):
        numel = init.numel()
        n_r   = max(1, int(restarts))
        rbs   = max(1, cfg.restart_batch_size)
        idx   = int(label1.item())

        inits = []
        for r in range(n_r):
            d = init.detach().clone() if r == 0 else torch.min(
                torch.max(torch.empty_like(data1).uniform_(-eps, eps), -data1), 1.0 - data1)
            inits.append(d)

        best_delta, best_key = None, float("-inf")

        for start in range(0, n_r, rbs):
            chunk  = inits[start: start + rbs]
            bs     = len(chunk)
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
        idx = int(label1.item())
        for _ in range(cfg.polish_rounds):
            d = cur.detach().requires_grad_(True)
            logits = primary(torch.clamp(data1 + d, 0.0, 1.0))[0]
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
                tf = trial.flatten(); tf[take] = 0.0; trial = tf.view_as(cur)
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

    def l2t_budget_forward(data: torch.Tensor, label: torch.Tensor):
        device = data.device
        data   = data.clone().detach().to(device)
        label  = label.clone().detach().to(device)
        outs   = []

        for i in range(int(data.shape[0])):
            data1  = data[i:i+1]
            label1 = label[i:i+1]

            # Phase 1: L2T policy learning → strong initial flip direction
            l2t_delta = _l2t_phase(data1, label1)

            # Phase 2: ascending L-inf level search, warm-started with L2T delta
            chosen, chosen_ok, chosen_level = None, False, cfg.max_level
            warm = l2t_delta

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

            # Phase 3: sparse refinement at chosen level
            if cfg.sparse and chosen_ok:
                chosen = _sparse_one(data1, label1, chosen, chosen_level * unit)

            outs.append(chosen.detach())

        return torch.cat(outs, dim=0)

    return l2t_budget_forward


# ─── Public builder ───────────────────────────────────────────────────────────

def build_l2t_budget_attack(
    device: torch.device,
    *,
    cfg: L2TBudgetConfig | None = None,
    surrogate: _EfficientNetV2LSurrogate | None = None,
):
    """Build the L2T+budget attack.

    Returns:
        callable (data: Tensor[B,C,H,W], label: Tensor[B]) -> delta Tensor
    """
    cfg = cfg or L2TBudgetConfig()
    primary = surrogate if surrogate is not None else _EfficientNetV2LSurrogate(device)
    primary.eval()
    return _make_l2t_budget_forward(cfg, primary)
