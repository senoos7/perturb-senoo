#!/usr/bin/env python3
"""Score a miner submission using the same path as PerturbValidator."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

MY_WORK = Path(__file__).resolve().parent.parent
if str(MY_WORK) not in sys.path:
    sys.path.insert(0, str(MY_WORK))

from challenge_io import file_to_b64, imagenet100_row_to_b64, load_imagenet100_dataset
from perturb_mirror import constants as C
from perturb_mirror.model import load_efficientnet_v2_l
from perturb_mirror.scoring import ChallengeSpec, EvaluationResult, ScoringConfig
from perturb_mirror.validator import (
    baseline_miner_forward,
    build_challenge_spec,
    sample_epsilon,
    score_miner_response,
)


def _print_result(result: EvaluationResult, challenge: ChallengeSpec, image_id: str = "") -> None:
    if image_id:
        print(f"image_id            : {image_id}")
    print(f"task_id             : {challenge.task_id}")
    print(f"model_name          : {challenge.model_name}")
    print(f"true_label          : {challenge.true_label!r}")
    print(f"epsilon             : {challenge.epsilon:.6f}")
    print(f"norm_type           : {challenge.norm_type}")
    print(f"timeout_seconds     : {challenge.timeout_seconds}")
    print(f"score               : {result.score:.6f}")
    print(f"reason              : {result.reason}")
    print(f"model_prediction    : {result.model_prediction!r}")
    print(f"response_time_ms    : {result.response_time_ms}")
    print(f"norm (Linf)         : {result.norm:.6f}")
    print(f"rmse                : {result.rmse:.6f}")
    print(f"ssim                : {result.ssim:.6f}  (min {C.MIN_SSIM})")
    print(f"psnr_db             : {result.psnr_db:.4f}  (min {C.MIN_PSNR_DB})")
    if result.reason == "success":
        effective_max = min(challenge.epsilon, C.MAX_LINF_DELTA)
        denom = max(1e-12, effective_max - C.MIN_LINF_DELTA)
        linf_ratio = min(max((result.norm - C.MIN_LINF_DELTA) / denom, 0.0), 1.0)
        rmse_ratio = min(max(result.rmse / max(1e-12, effective_max), 0.0), 1.0)
        linf_score = (1.0 - linf_ratio) ** 2
        rmse_score = (1.0 - rmse_ratio) ** 2
        total_w = C.LINF_COMPONENT_WEIGHT + C.RMSE_COMPONENT_WEIGHT
        perturbation_score = (
            C.LINF_COMPONENT_WEIGHT * linf_score + C.RMSE_COMPONENT_WEIGHT * rmse_score
        ) / max(1e-12, total_w)
        speed_score = 1.0 - min(result.response_time_ms / (challenge.timeout_seconds * 1000.0), 1.0)
        print(f"linf_score          : {linf_score:.6f}")
        print(f"rmse_score          : {rmse_score:.6f}")
        print(f"perturbation_score  : {perturbation_score:.6f}")
        print(f"speed_score         : {speed_score:.6f}")
        print(
            f"weights             : perturb={C.PERTURBATION_WEIGHT} speed={C.SPEED_WEIGHT} "
            f"linf={C.LINF_COMPONENT_WEIGHT} rmse={C.RMSE_COMPONENT_WEIGHT}"
        )


def _resolve_clean_b64(args: argparse.Namespace) -> tuple[str, str]:
    if args.clean and args.imagenet100_row is not None:
        raise SystemExit("Use either --clean or --imagenet100-row, not both")
    if args.clean:
        return "", file_to_b64(Path(args.clean))
    if args.imagenet100_row is not None:
        dataset = load_imagenet100_dataset()
        row = int(args.imagenet100_row)
        if row < 0 or row >= int(dataset.num_rows):
            raise SystemExit(f"--imagenet100-row must be in [0, {int(dataset.num_rows) - 1}]")
        image_id, clean_b64 = imagenet100_row_to_b64(dataset, row)
        return image_id, clean_b64
    raise SystemExit("Provide --clean <path> or --imagenet100-row <N>")


def _resolve_epsilon(args: argparse.Namespace) -> float:
    if args.epsilon is not None:
        return float(args.epsilon)
    if args.seed is not None:
        return sample_epsilon(int(args.seed))
    raise SystemExit("Provide --epsilon or --seed (validator samples epsilon from seed)")


def run(args: argparse.Namespace) -> tuple[EvaluationResult, ChallengeSpec, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_efficientnet_v2_l(device)
    config = ScoringConfig()

    image_id, clean_b64 = _resolve_clean_b64(args)
    epsilon = _resolve_epsilon(args)
    task_id = args.task_id or (f"local-{image_id}" if image_id else f"local-{Path(args.clean).stem}")

    challenge = build_challenge_spec(
        clean_image_b64=clean_b64,
        model=model,
        device=device,
        epsilon=epsilon,
        task_id=task_id,
        timeout_seconds=args.timeout_seconds,
        norm_type="Linf",
    )

    if args.baseline:
        perturbed_b64, response_ms = baseline_miner_forward(
            clean_image_b64=clean_b64,
            true_label=challenge.true_label,
            epsilon=challenge.epsilon,
            min_delta=config.min_linf_delta,
            model=model,
            device=device,
            norm_type=challenge.norm_type,
        )
        if args.save_perturbed:
            out = Path(args.save_perturbed)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(base64.b64decode(perturbed_b64))
            print(f"saved perturbed     : {out}")
        process_time_s = response_ms / 1000.0
    else:
        if not args.perturbed:
            raise SystemExit("Provide --perturbed <path> or use --baseline")
        perturbed_b64 = file_to_b64(Path(args.perturbed))
        process_time_s = args.response_ms / 1000.0

    result = score_miner_response(
        challenge=challenge,
        perturbed_image_b64=perturbed_b64,
        model=model,
        device=device,
        status_code=200,
        process_time_seconds=process_time_s,
        config=config,
    )
    return result, challenge, image_id


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score a miner submission like PerturbValidator (challenge build + verify_and_score)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--clean", help="Clean challenge image path")
    source.add_argument("--imagenet100-row", type=int, help="ImageNet-100 train row (JPEG encoding like validator)")
    parser.add_argument("--perturbed", help="Miner perturbed image path")
    parser.add_argument("--baseline", action="store_true", help="Run stock baseline miner PGD and score it")
    parser.add_argument("--save-perturbed", help="Save baseline perturbed PNG when using --baseline")
    parser.add_argument(
        "--epsilon",
        type=float,
        help="Challenge epsilon (validator uses sample_epsilon(seed) in [0.06, 0.2])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for validator-style epsilon sampling (same formula as _sample_epsilon)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=C.TIMEOUT_SECONDS,
        help=f"Challenge timeout (default {C.TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--response-ms",
        type=int,
        default=1000,
        help="Simulated miner process_time in ms (validator uses dendrite process_time)",
    )
    parser.add_argument("--task-id", help="Optional task id")
    parser.add_argument("--json", action="store_true", help="Print JSON result")
    args = parser.parse_args()

    result, challenge, image_id = run(args)

    if args.json:
        payload = {
            "image_id": image_id or None,
            "challenge": asdict(challenge),
            "result": asdict(result),
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_result(result, challenge, image_id=image_id)

    return 0 if result.reason == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
