#!/usr/bin/env python3
"""
Local end-to-end Perturb workflow (no Bittensor chain, no miner axon).

Simulates: challenge build -> miner attack -> validator scoring.

For ImageNet-100 from local cache, prefer:
  bash ./scripts/run_imagenet100_local_test.sh --max-rows 3
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perturbnet import constants as C
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import normalize_prediction_label, predict_label, resolve_target_index
from perturbnet.scoring import load_scoring_model, verify_and_score


def _load_image_b64(path: str) -> str:
    with open(path, "rb") as handle:
        raw = handle.read()
    if not raw:
        raise ValueError(f"empty image: {path}")
    return base64.b64encode(raw).decode("utf-8")


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

    from perturbnet.attacks import list_attack_algorithms, run_attack, runtime_profile_summary

    parser = argparse.ArgumentParser(description="Local Perturb end-to-end test (single image file)", parents=[pre])
    parser.add_argument(
        "--image",
        default=os.path.join(ROOT, C.DEFAULT_TEST_IMAGE_RELATIVE_PATH),
        help="Path to challenge image (offline smoke image only)",
    )
    parser.add_argument("--epsilon", type=float, default=0.12, help="Challenge epsilon")
    parser.add_argument("--min-delta", type=float, default=C.MIN_LINF_DELTA, help="Minimum Linf")
    parser.add_argument(
        "--algorithm",
        default=os.getenv("PERTURB_ATTACK_ALGORITHM", "sma_mi_fgsm"),
        choices=list_attack_algorithms(),
        help="Attack algorithm (or set PERTURB_ATTACK_ALGORITHM)",
    )
    parser.add_argument("--device", default="", help="cpu | cuda (default: auto)")
    args = parser.parse_args()
    os.environ["PERTURB_ATTACK_ALGORITHM"] = args.algorithm
    _configure_attack_logging()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Perturb local end-to-end test (single image) ===")
    print(f"device={device}")
    print(f"algorithm={args.algorithm}")
    print(f"image={args.image}")
    print(f"epsilon={args.epsilon} min_delta={args.min_delta}")
    print()

    print("[1/4] Load challenge image")
    clean_b64 = _load_image_b64(args.image)
    clean = decode_image_b64(clean_b64).to(device)
    print(f"  shape={tuple(clean.shape)}")

    print("[2/4] Load EfficientNetV2-L + predict clean label (validator true_label)")
    model = load_scoring_model(device)
    true_label = normalize_prediction_label(predict_label(model, clean))
    target_index = resolve_target_index(true_label)
    print(f"  true_label={true_label} target_index={target_index}")
    if target_index is None:
        raise RuntimeError(f"Unknown label for EfficientNet index map: {true_label}")

    print(f"profile: {runtime_profile_summary(args.algorithm)}")
    print(f"[3/4] Miner attack ({args.algorithm})")
    t0 = time.perf_counter()
    attack_result = run_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        epsilon=args.epsilon,
        min_delta=args.min_delta,
        device=device,
        algorithm=args.algorithm,
    )
    attack_ms = int((time.perf_counter() - t0) * 1000)
    adv_b64 = encode_image_b64(attack_result.adv)
    print(
        f"  attack_time_ms={attack_ms} algorithm={attack_result.algorithm_key} "
        f"linf={attack_result.linf:.6f} rmse={attack_result.rmse:.6f} pred_idx={attack_result.pred_index}"
    )

    print("[4/4] Validator score (verify_and_score)")
    score_result = verify_and_score(
        model=model,
        device=device,
        clean_image_b64=clean_b64,
        perturbed_image_b64=adv_b64,
        true_label=true_label,
        epsilon=args.epsilon,
        response_time_ms=attack_ms,
    )
    print(
        "  "
        f"score={score_result.score:.6f} reason={score_result.reason} "
        f"norm={score_result.norm:.6f} rmse={score_result.rmse:.6f} "
        f"ssim={score_result.ssim:.6f} psnr_db={score_result.psnr_db:.4f} "
        f"new_label={score_result.model_prediction}"
    )
    print()
    if score_result.reason == "success":
        print("PASS: local end-to-end workflow succeeded.")
        return 0
    print("FAIL: attack or scoring gate failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
