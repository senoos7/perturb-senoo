#!/usr/bin/env python3
"""
Train a perturbation generator that matches the validator's full pipeline.

Key fixes vs naive implementation:
  1. Clean image: JPEG q=95 round-trip (matches validator ImageNet-100 path).
  2. Adversarial: uint8 STE before PREPROCESS (matches PNG miner submission).
  3. Loss: differentiable SSIM from scoring.py formula + MSE-PSNR surrogate.
  4. Epsilon: sampled per batch from validator range [0.06, 0.2], capped at MAX_LINF_DELTA.
  5. Validation: checks all validator gates (delta, SSIM, PSNR, label flip).

Usage (from my_work/):
  python train_generator.py --limit 512 --epochs 5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

MY_WORK = Path(__file__).resolve().parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from bucketing import BucketBatchSampler, bucket_collate, build_shape_index
from challenge_io import load_imagenet100_dataset
from generator import Generator, load_generator_checkpoint
from label_cache import load_true_label_index
from model_utils import load_frozen_classifier
from paths import DATA, IMAGENET100_SAMPLES_DIR, OUTPUTS
from perturb_mirror.image_io import decode_image_b64_to_numpy
from perturb_mirror.constants import (
    IMAGENET100_REPO_ID,
    IMAGENET100_SPLIT,
    MAX_LINF_DELTA,
    MIN_LINF_DELTA,
)
from perturb_mirror.imagenet100_bootstrap import imagenet100_dataset_version
from train_loop import train_one_epoch, validate


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ImageNet100Dataset(Dataset):
    """
    Returns (JPEG-decoded float tensor [0,1] CHW, true_label_index).
    Matches validator path: PIL → JPEG q=95 → decode_image_b64.

    The label is the cached clean-image argmax (-1 if not cached → the train
    loop falls back to an on-the-fly classifier forward for that image).
    """

    def __init__(
        self,
        hf_dataset,
        indices: list[int],
        labels: dict[int, int] | None = None,
        samples_dir: Path | None = None,
    ) -> None:
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.labels = labels or {}
        self.samples_dir = samples_dir

    def __len__(self) -> int:
        return len(self.indices)

    def _decode_clean(self, row: int) -> np.ndarray:
        # Fastest path: pre-decoded float32 CHW [0,1] .npy (no JPEG decode).
        # Then pre-saved JPEG q=95 .b64 (decode only). Both are byte-identical to
        # the validator path and to what produced the cached labels. Finally fall
        # back to re-encoding from the HF dataset.
        if self.samples_dir is not None:
            npy_path = self.samples_dir / f"{row:07d}.npy"
            if npy_path.exists():
                return np.load(npy_path)
            b64_path = self.samples_dir / f"{row:07d}.b64"
            if b64_path.exists():
                return decode_image_b64_to_numpy(b64_path.read_text(encoding="utf-8"))

        import io

        from PIL import Image

        pil_img = self.hf_dataset[row]["image"].convert("RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
        return np.transpose(arr, (2, 0, 1)).copy()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = int(self.indices[idx])
        tensor = torch.from_numpy(self._decode_clean(row)).contiguous()
        return tensor, int(self.labels.get(row, -1))


# ─── Args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train perturbation generator (validator-matched)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["reward", "legacy"], default="reward",
                        help="reward = maximise validator perturbation_score; "
                             "legacy = cw + ssim + psnr gate proxies")
    # legacy loss weights
    parser.add_argument("--ssim-weight", type=float, default=10.0,
                        help="[legacy] Weight on SSIM loss (keep adv perceptually close)")
    parser.add_argument("--psnr-weight", type=float, default=5.0,
                        help="[legacy] Weight on MSE/PSNR surrogate loss")
    # reward loss weights
    parser.add_argument("--cw-confidence", type=float, default=6.0,
                        help="[reward] CW margin (logit gap) for quantization-robust flips")
    parser.add_argument("--floor-margin", type=float, default=0.0005,
                        help="[reward] Keep L-inf this far above MIN_LINF_DELTA")
    parser.add_argument("--linf-topk", type=int, default=32,
                        help="[reward] Top-k abs deltas used for the L-inf STE gradient")
    parser.add_argument("--w-flip", type=float, default=1.0, help="[reward] flip term weight")
    parser.add_argument("--w-score", type=float, default=4.0, help="[reward] perturbation_score weight")
    parser.add_argument("--w-floor", type=float, default=80.0, help="[reward] floor guard weight")
    parser.add_argument("--w-ssim", type=float, default=60.0, help="[reward] SSIM gate hinge weight")
    parser.add_argument("--w-psnr", type=float, default=0.05, help="[reward] PSNR gate hinge weight")
    parser.add_argument("--max-linf", type=float, default=MAX_LINF_DELTA,
                        help=f"Generator L-inf cap (default={MAX_LINF_DELTA}, validator max)")
    parser.add_argument("--gen-base", type=int, default=48,
                        help="Generator base channel width (capacity; e.g. 32/48/64)")
    parser.add_argument("--limit", type=int, default=0, help="Max train rows (0=full split)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Max images per batch (same-resolution buckets)")
    parser.add_argument("--max-pixels", type=int, default=0,
                        help="Per-batch pixel cap B*H*W (0=auto: batch_size*240*240)")
    parser.add_argument("--drop-last", action="store_true",
                        help="Drop final short batch of each resolution bucket")
    parser.add_argument("--workers", type=int, default=8,
                        help="DataLoader workers for parallel JPEG decode")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="Batches prefetched per worker (workers>0 only)")
    parser.add_argument("--no-b64-cache", action="store_true",
                        help="Decode from HF dataset instead of pre-saved .npy/.b64 files")
    parser.add_argument("--shape-cache", type=Path, default=DATA / "imagenet100_shapes.json",
                        help="Cache file for row -> (H, W) index")
    parser.add_argument("--label-cache", type=Path, default=DATA / "imagenet100_true_labels.json",
                        help="Consolidated cache for row -> clean true-label index")
    parser.add_argument("--samples-dir", type=Path, default=IMAGENET100_SAMPLES_DIR,
                        help="Dir with per-row clean inference json (from run_inference.py)")
    parser.add_argument("--no-label-cache", action="store_true",
                        help="Disable cached clean labels (recompute every step)")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--val-epsilon", type=float, default=0.12,
                        help="Fixed epsilon for validation gates (validator samples 0.06-0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=OUTPUTS / "generator.pt")
    parser.add_argument("--load", type=Path, default=None,
                        help="Pre-trained checkpoint to fine-tune (generator_state from a prior run)")
    parser.add_argument("--resume", action="store_true",
                        help="With --load: continue best-score tracking from checkpoint val score")
    parser.add_argument("--log-every", type=int, default=3)
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run validation every N epochs (always on the last epoch)")
    # speed: GPU
    parser.add_argument("--no-channels-last", action="store_true",
                        help="Disable channels_last memory format for the classifier")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the classifier (480x480 fixed shape; may be "
                             "unavailable on Windows -> falls back gracefully)")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Gradient-checkpoint the classifier (saves VRAM, ~20-30%% slower)")
    parser.add_argument("--checkpoint-segments", type=int, default=4,
                        help="Number of checkpoint_sequential segments over model.features")
    return parser.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Classifier input is a fixed 480x480 after PREPROCESS -> autotune wins.
        torch.backends.cudnn.benchmark = True
    print(f"device              : {device}")
    print(f"max_linf (generator): {args.max_linf}  (validator MAX_LINF_DELTA={MAX_LINF_DELTA})")
    print(f"min_linf_delta      : {MIN_LINF_DELTA}  (validator gate)")
    print(f"val_epsilon         : {args.val_epsilon}")

    hf_dataset = load_imagenet100_dataset()
    indices = list(range(int(hf_dataset.num_rows)))
    random.shuffle(indices)
    if args.limit > 0:
        indices = indices[: args.limit]

    val_count = max(1, int(len(indices) * args.val_fraction))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    print(f"train rows          : {len(train_indices)}")
    print(f"val rows            : {len(val_indices)}")

    # Resolution bucketing: batch only images that share an exact (H, W).
    max_pixels = args.max_pixels if args.max_pixels > 0 else args.batch_size * 240 * 240
    print(f"batch_size          : {args.batch_size}")
    print(f"max_pixels (cap)    : {max_pixels}")

    version = imagenet100_dataset_version(
        dataset=hf_dataset, repo_id=IMAGENET100_REPO_ID, split=IMAGENET100_SPLIT
    )
    shape_index = build_shape_index(
        dataset=hf_dataset,
        rows=train_indices + val_indices,
        cache_path=args.shape_cache,
        version=version,
    )
    train_shapes = [shape_index[r] for r in train_indices]
    val_shapes = [shape_index[r] for r in val_indices]

    # Cached clean true-label indices (skip the per-step clean forward pass).
    if args.no_label_cache:
        labels_map: dict[int, int] = {}
        print("label cache         : disabled (recomputing clean labels every step)")
    else:
        labels_map = load_true_label_index(
            rows=train_indices + val_indices,
            samples_dir=args.samples_dir,
            cache_path=args.label_cache,
            version=version,
        )
        total = len(train_indices) + len(val_indices)
        print(f"label cache         : {len(labels_map)}/{total} rows cached "
              f"({total - len(labels_map)} fall back to live inference)")

    train_sampler = BucketBatchSampler(
        train_shapes,
        batch_size=args.batch_size,
        max_pixels=max_pixels,
        shuffle=True,
        drop_last=args.drop_last,
        seed=args.seed,
    )
    val_sampler = BucketBatchSampler(
        val_shapes,
        batch_size=args.batch_size,
        max_pixels=max_pixels,
        shuffle=False,
        drop_last=False,
        seed=args.seed,
    )
    print(f"train buckets       : {train_sampler.describe()}")
    print(f"val buckets         : {val_sampler.describe()}")

    decode_dir = None if args.no_b64_cache else args.samples_dir
    loader_kwargs = dict(
        num_workers=args.workers,
        collate_fn=bucket_collate,
        pin_memory=(device.type == "cuda"),
    )
    if args.workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)

    train_loader = DataLoader(
        ImageNet100Dataset(hf_dataset, train_indices, labels_map, decode_dir),
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ImageNet100Dataset(hf_dataset, val_indices, labels_map, decode_dir),
        batch_sampler=val_sampler,
        **loader_kwargs,
    )

    classifier = load_frozen_classifier(device)
    if device.type == "cuda" and not args.no_channels_last:
        classifier = classifier.to(memory_format=torch.channels_last)
        print("channels_last       : enabled (classifier)")
    if args.compile:
        try:
            classifier = torch.compile(classifier)
            print("torch.compile       : enabled (classifier)")
        except Exception as exc:
            print(f"torch.compile       : unavailable ({exc}) -> running eager")
    print(f"dtype               : float32")
    print(f"workers/prefetch    : {args.workers}/{args.prefetch_factor}")
    print(f"fast decode (npy/b64): {'disabled' if args.no_b64_cache else 'enabled'}")

    generator = Generator(max_linf=args.max_linf, base=args.gen_base).to(device)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"generator           : base={args.gen_base}  params={n_params/1e6:.2f}M")

    best_score = -1.0
    if args.load is not None:
        ckpt = load_generator_checkpoint(generator, args.load)
        print(f"loaded checkpoint   : {ckpt['path']}")
        if "epoch" in ckpt:
            print(f"  saved epoch       : {ckpt['epoch']}")
        if "max_linf" in ckpt:
            print(f"  saved max_linf    : {ckpt['max_linf']}")
        if "val" in ckpt and isinstance(ckpt["val"], dict):
            v = ckpt["val"]
            print(
                f"  saved val         : score={v.get('score_mean', 0):.4f}  "
                f"pass={v.get('pass_rate', 0):.3f}  flip={v.get('flip_rate', 0):.3f}"
            )
            if args.resume:
                best_score = float(v.get("score_mean", -1.0))
                print(f"  resume best_score : {best_score:.4f}")

    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    args.save.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            generator=generator,
            classifier=classifier,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            log_every=args.log_every,
            desc=f"epoch {epoch}/{args.epochs}",
        )
        if args.loss == "reward":
            print(
                f"train  loss={train_stats['loss']:.4f} | "
                f"flip={train_stats['flip']:.4f} score={train_stats['score']:.4f} "
                f"floor={train_stats['floor']:.4f} ssim_h={train_stats['ssim_h']:.4f} "
                f"psnr_h={train_stats['psnr_h']:.4f} | "
                f"flip_rate={train_stats['flip_rate']:.3f} "
                f"pert_score={train_stats['pert_score']:.4f} "
                f"linf_mean={train_stats['linf_mean']:.5f}"
            )
        else:
            print(
                f"train  loss={train_stats['loss']:.4f}  "
                f"cw={train_stats['cw']:.4f}  "
                f"ssim_loss={train_stats['ssim']:.5f}  "
                f"psnr_mse={train_stats['psnr']:.6f}"
            )

        run_val = (epoch % max(1, args.val_every) == 0) or (epoch == args.epochs)
        if not run_val:
            continue

        val_stats = validate(generator, classifier, val_loader, device, epsilon=args.val_epsilon)
        print(
            f"val    score={val_stats['score_mean']:.4f}  "   # mean validator reward
            f"pass_rate={val_stats['pass_rate']:.3f}  "        # full gates passed
            f"flip_rate={val_stats['flip_rate']:.3f}  "
            f"ssim={val_stats['ssim_mean']:.4f}  "
            f"psnr={val_stats['psnr_mean']:.2f} dB"
        )

        # Rank by mean validator score (reward), not just pass-rate.
        if val_stats["score_mean"] > best_score:
            best_score = val_stats["score_mean"]
            torch.save(
                {
                    "epoch": epoch,
                    "max_linf": args.max_linf,
                    "generator_state": generator.state_dict(),
                    "val": val_stats,
                },
                args.save,
            )
            print(f"saved  -> {args.save}  (score={best_score:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
