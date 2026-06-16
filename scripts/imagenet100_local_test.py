#!/usr/bin/env python3
"""Local ImageNet-100 attack + validator-style scoring (no Bittensor).

Workflow per row (matches production validator + miner):
  1. Load ImageNet-100 row from local HF cache
  2. Encode clean image as JPEG q=95 base64 (validator path)
  3. true_label = EfficientNetV2-L prediction on clean image
  4. Run configured attack (default sma_mi_fgsm)
  5. verify_and_score() with subnet gates and score formula

CPU-friendly: use scripts/cpu_attack.env.example and keep --max-rows small.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perturbnet import constants as C
from perturbnet.attacks import list_attack_algorithms, run_attack, runtime_profile_summary
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.imagenet100_bootstrap import load_imagenet100
from perturbnet.imagenet100_challenge import imagenet_row_to_clean_b64
from perturbnet.imagenet100_labels import load_label_manifest
from perturbnet.model import normalize_prediction_label, predict_label, resolve_target_index
from perturbnet.scoring import ScoreResult, load_scoring_model, verify_and_score


def _load_env_file(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export ") :].strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value


def _configure_attack_logging() -> None:
    log_mode = os.getenv("PERTURB_ATTACK_LOG", "on").strip().lower()
    if log_mode in {"0", "false", "no", "off"}:
        return
    level = logging.DEBUG if log_mode in {"verbose", "debug", "v"} else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def _resolve_true_label(
    *,
    model: torch.nn.Module,
    clean: torch.Tensor,
    row: int,
    label_cache: dict[int, dict],
) -> tuple[str, int | None]:
    cached = label_cache.get(row)
    if cached and cached.get("true_label"):
        true_label = normalize_prediction_label(str(cached["true_label"]))
    else:
        true_label = normalize_prediction_label(predict_label(model, clean))
    target_index = resolve_target_index(true_label)
    return true_label, target_index


def _run_one_row(
    *,
    row: int,
    example: dict,
    model: torch.nn.Module,
    device: torch.device,
    epsilon: float,
    min_delta: float,
    algorithm: str,
    label_cache: dict[int, dict],
) -> ScoreResult | None:
    clean_b64 = imagenet_row_to_clean_b64(example)
    clean = decode_image_b64(clean_b64).to(device)
    true_label, target_index = _resolve_true_label(
        model=model, clean=clean, row=row, label_cache=label_cache
    )
    if target_index is None:
        print(f"  row={row} SKIP unresolved true_label={true_label!r}")
        return None

    t0 = time.perf_counter()
    attack_result = run_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        epsilon=epsilon,
        min_delta=min_delta,
        device=device,
        algorithm=algorithm,
    )
    attack_ms = int((time.perf_counter() - t0) * 1000)
    adv_b64 = encode_image_b64(attack_result.adv)

    return verify_and_score(
        model=model,
        device=device,
        clean_image_b64=clean_b64,
        perturbed_image_b64=adv_b64,
        true_label=true_label,
        epsilon=epsilon,
        response_time_ms=attack_ms,
    )


def main() -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--env-file",
        default=os.getenv("PERTURB_ENV_FILE", ""),
        help="Load attack env vars before importing algorithms (e.g. scripts/cpu_attack.env.example)",
    )
    pre_args, _ = pre.parse_known_args()
    if pre_args.env_file:
        _load_env_file(pre_args.env_file)

    parser = argparse.ArgumentParser(description="ImageNet-100 local attack + validator scoring", parents=[pre])
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=3, help="How many dataset rows to test (keep small on CPU)")
    parser.add_argument("--row", type=int, default=-1, help="Test a single row index (overrides start/max)")
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--min-delta", type=float, default=C.MIN_LINF_DELTA)
    parser.add_argument("--device", default="", help="cpu | cuda (default: auto)")
    parser.add_argument(
        "--algorithm",
        default=os.getenv("PERTURB_ATTACK_ALGORITHM", "sma_mi_fgsm"),
        choices=list_attack_algorithms(),
    )
    parser.add_argument(
        "--labels-manifest",
        type=Path,
        default=ROOT / "data" / "imagenet100" / "labels.jsonl",
        help="Optional cached true_label manifest (skip re-inference if row present)",
    )
    args = parser.parse_args()
    os.environ["PERTURB_ATTACK_ALGORITHM"] = args.algorithm
    _configure_attack_logging()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.row >= 0:
        row_indices = [args.row]
    else:
        if args.max_rows < 1:
            raise SystemExit("--max-rows must be >= 1")
        row_indices = list(range(args.start_row, args.start_row + args.max_rows))

    print("=== ImageNet-100 local attack + validator scoring ===")
    print(f"device={device} algorithm={args.algorithm}")
    print(f"rows={row_indices[0]}..{row_indices[-1]} count={len(row_indices)}")
    print(f"epsilon={args.epsilon} min_delta={args.min_delta}")
    print(f"profile: {runtime_profile_summary(args.algorithm)}")
    print()

    print("[1/3] Load ImageNet-100 from local cache")
    offline = os.getenv("PERTURB_IMAGENET100_OFFLINE", "auto").strip().lower()
    if offline not in {"0", "false", "no", "off"}:
        print("  mode=offline (local Arrow cache; no Hugging Face network calls)")
    dataset = load_imagenet100()
    print(f"  dataset_rows={int(dataset.num_rows)}")
    for row in row_indices:
        if row < 0 or row >= int(dataset.num_rows):
            raise SystemExit(f"row out of range: {row}")

    label_cache = load_label_manifest(args.labels_manifest) if args.labels_manifest.is_file() else {}
    if label_cache:
        print(f"  label_cache={args.labels_manifest} entries={len(label_cache)}")

    print("[2/3] Load EfficientNetV2-L")
    model = load_scoring_model(device)

    print("[3/3] Attack + score")
    results: list[tuple[int, ScoreResult]] = []
    for row in row_indices:
        print(f"-- row={row} --")
        score = _run_one_row(
            row=row,
            example=dataset[row],
            model=model,
            device=device,
            epsilon=args.epsilon,
            min_delta=args.min_delta,
            algorithm=args.algorithm,
            label_cache=label_cache,
        )
        if score is None:
            continue
        results.append((row, score))
        print(
            f"  score={score.score:.6f} reason={score.reason} "
            f"attack_ms={score.response_time_ms} norm={score.norm:.6f} rmse={score.rmse:.6f} "
            f"ssim={score.ssim:.6f} psnr_db={score.psnr_db:.4f} new_label={score.model_prediction}"
        )

    print()
    if not results:
        print("No rows scored.", file=sys.stderr)
        return 1

    reasons = Counter(score.reason for _, score in results)
    successes = sum(1 for _, score in results if score.reason == "success")
    mean_score = sum(score.score for _, score in results if score.reason == "success") / max(successes, 1)
    print(f"Summary: tested={len(results)} success={successes} mean_score={mean_score:.6f}")
    print(f"reasons={dict(reasons)}")

    return 0 if successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
