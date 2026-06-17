"""Shared train/validation loops for generator training scripts."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from generator import Generator
from model_utils import (
    cw_loss,
    eval_batch,
    forward_adv,
    psnr_loss_differentiable,
    reward_aligned_loss,
    ssim_loss_differentiable,
    true_label_indices,
)


def train_one_epoch(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    log_every: int,
    desc: str = "train",
) -> dict[str, float]:
    generator.train()
    if args.loss == "reward":
        totals: dict[str, float] = {
            "loss": 0.0, "flip": 0.0, "score": 0.0, "floor": 0.0,
            "ssim_h": 0.0, "psnr_h": 0.0, "flip_rate": 0.0,
            "pert_score": 0.0, "linf_mean": 0.0,
        }
    else:
        totals = {"loss": 0.0, "cw": 0.0, "ssim": 0.0, "psnr": 0.0}
    steps = 0

    channels_last = (not args.no_channels_last) and device.type == "cuda"
    grad_checkpoint = bool(getattr(args, "grad_checkpoint", False))
    checkpoint_segments = int(getattr(args, "checkpoint_segments", 4))

    pbar = tqdm(loader, desc=desc, dynamic_ncols=True, leave=True)
    for step, (clean, labels) in enumerate(pbar, start=1):
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if (labels < 0).any():
            computed = true_label_indices(classifier, clean)
            labels = torch.where(labels < 0, computed, labels)
        target_idx = labels

        logits, adv_quant = forward_adv(
            classifier, generator, clean,
            channels_last=channels_last,
            grad_checkpoint=grad_checkpoint,
            checkpoint_segments=checkpoint_segments,
        )

        if args.loss == "reward":
            loss, comps = reward_aligned_loss(
                logits=logits,
                target_indices=target_idx,
                clean_bchw=clean,
                adv_quant=adv_quant,
                cw_confidence=args.cw_confidence,
                floor_margin=args.floor_margin,
                linf_topk=args.linf_topk,
                w_flip=args.w_flip,
                w_score=args.w_score,
                w_floor=args.w_floor,
                w_ssim=args.w_ssim,
                w_psnr=args.w_psnr,
            )
        else:
            loss_cw = cw_loss(logits, target_idx)
            loss_ssim = ssim_loss_differentiable(clean, adv_quant)
            loss_psnr = psnr_loss_differentiable(clean, adv_quant)
            loss = loss_cw + args.ssim_weight * loss_ssim + args.psnr_weight * loss_psnr
            comps = {
                "loss": float(loss.item()), "cw": float(loss_cw.item()),
                "ssim": float(loss_ssim.item()), "psnr": float(loss_psnr.item()),
            }

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        for k in totals:
            totals[k] += comps.get(k, 0.0)
        steps += 1

        if log_every > 0 and step % log_every == 0:
            if args.loss == "reward":
                pbar.set_postfix(
                    {
                        "loss": f"{comps['loss']:.3f}",
                        "flip": f"{comps['flip']:.3f}",
                        "score": f"{comps['score']:.3f}",
                        "floor": f"{comps['floor']:.3f}",
                        "ssim_h": f"{comps['ssim_h']:.3f}",
                        "psnr_h": f"{comps['psnr_h']:.3f}",
                        "fr": f"{comps['flip_rate']:.3f}",
                        "ps": f"{comps['pert_score']:.3f}",
                        "linf": f"{comps['linf_mean']:.5f}",
                    },
                    refresh=False,
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{comps['loss']:.3f}",
                        "cw": f"{comps['cw']:.3f}",
                        "ssim": f"{comps['ssim']:.5f}",
                        "psnr": f"{comps['psnr']:.6f}",
                    },
                    refresh=False,
                )

    pbar.close()
    denom = max(steps, 1)
    return {k: v / denom for k, v in totals.items()}


@torch.no_grad()
def validate(
    generator: Generator,
    classifier: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epsilon: float,
) -> dict[str, float]:
    generator.eval()
    agg: dict[str, float] = {
        "pass_rate": 0.0, "flip_rate": 0.0, "ssim_mean": 0.0,
        "psnr_mean": 0.0, "score_mean": 0.0,
    }
    steps = 0

    pbar = tqdm(loader, desc="val", dynamic_ncols=True, leave=True)
    for clean, labels in pbar:
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if (labels < 0).any():
            computed = true_label_indices(classifier, clean)
            labels = torch.where(labels < 0, computed, labels)
        stats = eval_batch(
            model=classifier,
            generator=generator,
            clean_bchw=clean,
            true_indices=labels,
            epsilon=epsilon,
        )
        for k in agg:
            agg[k] += stats[k]
        steps += 1
        pbar.set_postfix(
            {
                "score": f"{agg['score_mean'] / steps:.4f}",
                "pass": f"{agg['pass_rate'] / steps:.3f}",
                "fr": f"{agg['flip_rate'] / steps:.3f}",
            },
            refresh=False,
        )
    pbar.close()

    denom = max(steps, 1)
    return {k: v / denom for k, v in agg.items()}
