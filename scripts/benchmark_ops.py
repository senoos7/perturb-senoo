#!/usr/bin/env python3
"""Benchmark the OPS attack (pinned to EfficientNetV2-L) with validator scoring.

For each sampled image we replay the full miner -> validator path:

    npy clean -> PNG(uint8) -> decode -> derive true_label (EfficientNetV2-L)
              -> OPS attack -> adv -> PNG(uint8) -> verify_and_score()

so every number (L-inf, RMSE, SSIM, PSNR, perturbation_score, pass/flip) is
exactly what the Perturb validator would compute.

Examples
--------
Pure white-box overfit (MI-FGSM, fast):
    python scripts/benchmark_ops.py --num 20 --num-sample-neighbor 0 --num-sample-operator 0

Full OPS (transfer-robust, heavy):
    python scripts/benchmark_ops.py --num 10 --num-sample-neighbor 10 --num-sample-operator 20
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from local_index import discover_npy_rows, load_label_index  # noqa: E402
from ops_attack import (  # noqa: E402
    QUANT_SAFE_MAX_LINF,
    BudgetConfig,
    build_ops_attack,
    build_ops_budget_attack,
    forwards_per_image,
)
from paths import (  # noqa: E402
    IMAGENET100_LABELS_CACHE,
    IMAGENET100_SAMPLES_DIR,
    IMAGENET100_VAL_LABELS_CACHE,
    IMAGENET100_VAL_SAMPLES_DIR,
)
from perturb_mirror import constants as C  # noqa: E402
from perturb_mirror.image_io import decode_image_b64, encode_image_b64  # noqa: E402
from perturb_mirror.model import resolve_target_index  # noqa: E402
from perturb_mirror.validator import (  # noqa: E402
    build_challenge_spec,
    sample_epsilon,
    score_miner_response,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark OPS (EfficientNetV2-L) with validator scoring")
    # data source
    p.add_argument("--split", choices=["train", "val"], default="val",
                   help="Which prepared .npy split to sample from (default: val)")
    p.add_argument("--samples-dir", type=Path, default=None,
                   help="Override samples dir (defaults by --split)")
    p.add_argument("--label-cache", type=Path, default=None,
                   help="Override label cache (defaults by --split)")
    p.add_argument("--num", type=int, default=20, help="Number of images to benchmark")
    p.add_argument("--rows", type=int, nargs="*", default=None,
                   help="Explicit row ids (overrides random sampling)")
    p.add_argument("--seed", type=int, default=0, help="Sampling/attack seed")
    # objective
    p.add_argument("--objective", choices=["ce", "budget"], default="budget",
                   help="ce = vanilla OPS (max cross-entropy); "
                        "budget = minimum-norm L-inf/SSIM-aware (default)")
    # challenge / scoring
    p.add_argument("--challenge-epsilon", type=float, default=0.0,
                   help="Challenge epsilon; 0 = sample per image in [0.06, 0.2] like the validator")
    # OPS hyperparameters
    p.add_argument("--attack-epsilon", type=float, default=QUANT_SAFE_MAX_LINF,
                   help="OPS L-inf budget (default 7/255, the quant-safe scoring cap)")
    p.add_argument("--num-iter", type=int, default=10, help="[ce] OPS iterations")
    p.add_argument("--num-sample-neighbor", type=int, default=0,
                   help="OPS neighbor samples (0 = off; off is strongest white-box overfit)")
    p.add_argument("--num-sample-operator", type=int, default=0,
                   help="OPS operator samples (0 = off)")
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--decay", type=float, default=1.0)
    p.add_argument("--random-start", action="store_true")
    # budget objective (only used when --objective budget)
    p.add_argument("--max-level", type=int, default=7, help="Max L-inf level in 1/255 units (<=7)")
    p.add_argument("--inner-iters", type=int, default=60, help="PGD steps per L-inf level")
    p.add_argument("--restarts", type=int, default=2, help="PGD restarts per level (>1 adds random init)")
    p.add_argument("--first-level-restarts", type=int, default=8, help="Extra PGD restarts at level 1 (1/255 floor)")
    p.add_argument("--alpha-frac", type=float, default=0.5, help="PGD step size as fraction of eps")
    p.add_argument("--no-sparse", action="store_true", help="Disable L0 sparse refinement")
    p.add_argument("--sparse-steps", type=int, default=50, help="PGD steps per support size in k binary search")
    p.add_argument("--sparse-restarts", type=int, default=2, help="Restarts of support-restricted PGD (>1 random reseed)")
    p.add_argument("--polish-rounds", type=int, default=3, help="Backward-elimination passes after the k search (0 disables)")
    p.add_argument("--flip-margin", type=float, default=2.0, help="Logit margin required when pruning (conservative)")
    p.add_argument("--accept-margin", type=float, default=1.0, help="Logit margin to accept an L-inf level (lower => more 1/255)")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu (auto by default)")
    return p.parse_args()


def resolve_source(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.split == "train":
        samples_dir = args.samples_dir or IMAGENET100_SAMPLES_DIR
        label_cache = args.label_cache or IMAGENET100_LABELS_CACHE
    else:
        samples_dir = args.samples_dir or IMAGENET100_VAL_SAMPLES_DIR
        label_cache = args.label_cache or IMAGENET100_VAL_LABELS_CACHE
    return samples_dir, label_cache


def pick_rows(args: argparse.Namespace, samples_dir: Path) -> list[int]:
    all_rows = discover_npy_rows(samples_dir)
    if args.rows:
        missing = [r for r in args.rows if r not in set(all_rows)]
        if missing:
            raise SystemExit(f"rows not found in {samples_dir}: {missing[:10]}")
        return list(args.rows)
    rng = random.Random(args.seed)
    n = min(args.num, len(all_rows))
    return sorted(rng.sample(all_rows, n))


def load_clean_b64(samples_dir: Path, row: int) -> str:
    """Build the clean image EXACTLY as the validator does.

    The validator (neurons/validator.py::_imagenet100_image_bytes) sends the
    original image re-encoded as JPEG quality=95, base64-encoded -- NOT a
    lossless PNG. Mirror that here so the benchmark's clean baseline carries the
    same JPEG artifacts the miner actually receives and is scored against.
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    arr = np.load(samples_dir / f"{row:07d}.npy")  # CHW float in [0, 1]
    tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).clamp(0.0, 1.0)
    hwc = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    image = Image.fromarray(hwc, mode="RGB")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def main() -> int:
    args = parse_args()
    samples_dir, label_cache = resolve_source(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = pick_rows(args, samples_dir)
    # Label cache is optional here (we re-derive true_label from the model), but
    # load it when present so we can report cache vs model agreement later.
    try:
        _ = load_label_index(label_cache, rows, allow_missing=True)
    except FileNotFoundError:
        pass

    if args.objective == "budget":
        # ~ levels searched (binary search over 1..max_level) * restarts * inner_iters
        levels = max(1, int(round(math.log2(max(1, args.max_level)))) + 1)
        fpi = levels * max(1, args.restarts) * args.inner_iters
        fpi *= max(1, forwards_per_image(1, args.num_sample_neighbor, args.num_sample_operator))
    else:
        fpi = forwards_per_image(args.num_iter, args.num_sample_neighbor, args.num_sample_operator)
    print(f"device              : {device}")
    print(f"split / samples_dir : {args.split}  {samples_dir}")
    print(f"images              : {len(rows)}")
    print(f"objective           : {args.objective}")
    print(f"challenge_epsilon   : {'sampled[0.06,0.2]' if args.challenge_epsilon <= 0 else args.challenge_epsilon}")
    print(f"OPS sampling        : neighbor={args.num_sample_neighbor} operator={args.num_sample_operator}")
    if args.objective == "budget":
        print(f"max_level           : {args.max_level} ({args.max_level}/255 = {args.max_level/255:.5f})")
        print(f"inner_iters/restarts: {args.inner_iters} / {args.restarts} (level-1 restarts={args.first_level_restarts})")
        print(f"sparse              : {not args.no_sparse} (steps={args.sparse_steps}, restarts={args.sparse_restarts}, prune_margin={args.flip_margin})")
        print(f"polish_rounds       : {args.polish_rounds} (backward elimination)")
        print(f"accept_margin       : {args.accept_margin} (lower => more images at 1/255)")
    else:
        print(f"attack_epsilon      : {args.attack_epsilon}")
        print(f"num_iter            : {args.num_iter}")
    print(f"forwards/image (est): ~{fpi}")
    if args.num_sample_neighbor * args.num_sample_operator == 0:
        print("flip direction      : true gradient (sampling off) -> EfficientNetV2-L overfit")
    else:
        print("flip direction      : full OPS (operator+neighbor sampling)")

    if args.objective == "budget":
        cfg = BudgetConfig(
            max_level=args.max_level,
            inner_iters=args.inner_iters,
            restarts=args.restarts,
            first_level_restarts=args.first_level_restarts,
            alpha_frac=args.alpha_frac,
            sparse=not args.no_sparse,
            sparse_steps=args.sparse_steps,
            sparse_restarts=args.sparse_restarts,
            polish_rounds=args.polish_rounds,
            flip_margin=args.flip_margin,
            accept_margin=args.accept_margin,
        )
        attack = build_ops_budget_attack(
            device,
            num_sample_neighbor=args.num_sample_neighbor,
            num_sample_operator=args.num_sample_operator,
            beta=args.beta,
            decay=args.decay,
            budget=cfg,
        )
    else:
        attack = build_ops_attack(
            device,
            epsilon=args.attack_epsilon,
            num_iter=args.num_iter,
            num_sample_neighbor=args.num_sample_neighbor,
            num_sample_operator=args.num_sample_operator,
            beta=args.beta,
            decay=args.decay,
            random_start=args.random_start,
        )
    model = attack.model.model  # raw EfficientNetV2-L for validator-mirror scoring

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    totals = {"score": 0.0, "norm": 0.0, "rmse": 0.0, "ssim": 0.0, "psnr": 0.0, "ms": 0.0}
    pass_count = 0
    flip_count = 0
    reasons: Counter[str] = Counter()

    pbar = tqdm(rows, desc="ops-bench", dynamic_ncols=True)
    for i, row in enumerate(pbar):
        clean_b64 = load_clean_b64(samples_dir, row)
        epsilon = sample_epsilon(args.seed + i) if args.challenge_epsilon <= 0 else float(args.challenge_epsilon)
        challenge = build_challenge_spec(
            clean_image_b64=clean_b64,
            model=model,
            device=device,
            epsilon=epsilon,
            task_id=f"bench-{row}",
        )

        target_index = resolve_target_index(challenge.true_label)
        if target_index is None:
            reasons["unresolved_true_label"] += 1
            continue

        clean = decode_image_b64(clean_b64).to(device).unsqueeze(0)
        label = torch.tensor([target_index], device=device)

        # Per-image deterministic seed: makes each image's attack independent of
        # how much RNG prior images consumed, so runs are reproducible and
        # comparable (no cross-image drift that pushed one image to 2/255 last run).
        torch.manual_seed(args.seed + row)

        started = time.perf_counter()
        delta = attack(clean, label)
        adv = torch.clamp(clean + delta, 0.0, 1.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - started

        adv_b64 = encode_image_b64(adv.squeeze(0))
        result = score_miner_response(
            challenge=challenge,
            perturbed_image_b64=adv_b64,
            model=model,
            device=device,
            process_time_seconds=elapsed_s,
        )

        reasons[result.reason] += 1
        if result.reason == "success":
            pass_count += 1
        if result.model_prediction and result.model_prediction != challenge.true_label:
            flip_count += 1

        totals["score"] += result.score
        totals["norm"] += result.norm
        totals["rmse"] += result.rmse
        totals["ssim"] += result.ssim
        totals["psnr"] += result.psnr_db
        totals["ms"] += elapsed_s * 1000.0

        n = i + 1
        pbar.set_postfix(
            {
                "score": f"{totals['score'] / n:.4f}",
                "pass": f"{pass_count / n:.3f}",
                "flip": f"{flip_count / n:.3f}",
                "ms": f"{totals['ms'] / n:.0f}",
            },
            refresh=False,
        )
    pbar.close()

    denom = max(1, len(rows))
    print("\n==================== OPS benchmark ====================")
    print(f"images              : {len(rows)}")
    print(f"pass_rate (success) : {pass_count / denom:.4f}")
    print(f"flip_rate           : {flip_count / denom:.4f}")
    print(f"mean score          : {totals['score'] / denom:.6f}")
    print(f"mean L-inf          : {totals['norm'] / denom:.6f}  (gate [{C.MIN_LINF_DELTA}, {C.MAX_LINF_DELTA}])")
    print(f"mean RMSE           : {totals['rmse'] / denom:.6f}")
    print(f"mean SSIM           : {totals['ssim'] / denom:.6f}  (min {C.MIN_SSIM})")
    print(f"mean PSNR dB        : {totals['psnr'] / denom:.4f}  (min {C.MIN_PSNR_DB})")
    print(f"mean time/img (ms)  : {totals['ms'] / denom:.1f}  (timeout {C.TIMEOUT_SECONDS}s)")
    print("reasons             :")
    for reason, count in reasons.most_common():
        print(f"  {reason:32s} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
