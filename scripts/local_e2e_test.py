#!/usr/bin/env python3
"""
Local end-to-end Perturb workflow (no Bittensor chain, no miner axon).

Simulates: challenge build -> miner attack -> validator scoring.
"""
from __future__ import annotations

import argparse
import base64
import math
import os
import sys
import time
from dataclasses import dataclass

import requests
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perturbnet import constants as C
from perturbnet.attacks import list_attack_algorithms, run_attack, runtime_profile_summary
from perturbnet.image_io import decode_image_b64, encode_image_b64
from perturbnet.model import (
    load_efficientnet_v2_l,
    normalize_prediction_label,
    predict_label,
    resolve_target_index,
)


@dataclass
class ScoreResult:
    score: float
    reason: str
    model_prediction: str = ""
    response_time_ms: int = 0
    norm: float = 0.0
    rmse: float = 0.0
    epsilon: float = 0.0
    ssim: float = 0.0
    psnr_db: float = 0.0


def _compute_ssim(x_clean: torch.Tensor, x_adv: torch.Tensor, kernel_size: int = 11) -> float:
    padding = kernel_size // 2
    x = x_clean.unsqueeze(0)
    y = x_adv.unsqueeze(0)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return float((numerator / (denominator + 1e-12)).mean().item())


def _compute_psnr_db(x_clean: torch.Tensor, x_adv: torch.Tensor) -> float:
    mse = float(torch.mean((x_adv - x_clean) ** 2).item())
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def verify_and_score(
    *,
    model: torch.nn.Module,
    device: torch.device,
    clean_image_b64: str,
    perturbed_image_b64: str,
    true_label: str,
    epsilon: float,
    response_time_ms: int,
) -> ScoreResult:
    """Mirror of neurons/validator.py verify_and_score (Linf path)."""
    try:
        x_clean = decode_image_b64(clean_image_b64).to(device)
        x_adv = decode_image_b64(perturbed_image_b64).to(device)
    except Exception as exc:
        return ScoreResult(score=0.0, reason=f"decode_failed:{exc}", response_time_ms=response_time_ms)

    if x_adv.shape != x_clean.shape:
        return ScoreResult(score=0.0, reason="shape_mismatch", response_time_ms=response_time_ms)
    if x_adv.min().item() < 0.0 or x_adv.max().item() > 1.0:
        return ScoreResult(score=0.0, reason="value_out_of_range", response_time_ms=response_time_ms)

    try:
        prediction = predict_label(model, x_adv)
    except Exception as exc:
        return ScoreResult(score=0.0, reason=f"model_inference_failed:{exc}", response_time_ms=response_time_ms)

    norm = float((x_adv - x_clean).abs().max().item())
    if norm < C.MIN_LINF_DELTA:
        return ScoreResult(
            score=0.0,
            reason="below_min_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            epsilon=epsilon,
        )

    effective_max_delta = min(float(epsilon), C.MAX_LINF_DELTA)
    if norm > effective_max_delta:
        return ScoreResult(
            score=0.0,
            reason="above_max_delta",
            model_prediction=prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=epsilon,
        )

    normalized_prediction = normalize_prediction_label(prediction)
    if normalized_prediction == normalize_prediction_label(true_label):
        return ScoreResult(
            score=0.0,
            reason="label_match_with_original",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item()),
            epsilon=epsilon,
        )

    rmse = float(torch.sqrt(torch.mean((x_adv - x_clean) ** 2)).item())
    ssim = _compute_ssim(x_clean, x_adv)
    if ssim < C.MIN_SSIM:
        return ScoreResult(
            score=0.0,
            reason="below_min_ssim",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=epsilon,
            ssim=ssim,
        )

    psnr_db = _compute_psnr_db(x_clean, x_adv)
    if C.MIN_PSNR_DB > 0.0 and psnr_db < C.MIN_PSNR_DB:
        return ScoreResult(
            score=0.0,
            reason="below_min_psnr_db",
            model_prediction=normalized_prediction,
            response_time_ms=response_time_ms,
            norm=norm,
            rmse=rmse,
            epsilon=epsilon,
            ssim=ssim,
            psnr_db=psnr_db,
        )

    denom = max(1e-12, effective_max_delta - C.MIN_LINF_DELTA)
    linf_ratio = min(max((norm - C.MIN_LINF_DELTA) / denom, 0.0), 1.0)
    linf_score = (1.0 - linf_ratio) ** 2
    rmse_ratio = min(max(rmse / max(1e-12, effective_max_delta), 0.0), 1.0)
    rmse_score = (1.0 - rmse_ratio) ** 2
    total_weight = max(1e-12, C.LINF_COMPONENT_WEIGHT + C.RMSE_COMPONENT_WEIGHT)
    perturbation_score = (
        (C.LINF_COMPONENT_WEIGHT * linf_score) + (C.RMSE_COMPONENT_WEIGHT * rmse_score)
    ) / total_weight
    time_ratio = response_time_ms / (C.TIMEOUT_SECONDS * 1000.0)
    speed_score = 1.0 - min(time_ratio, 1.0)
    score = C.PERTURBATION_WEIGHT * perturbation_score + C.SPEED_WEIGHT * speed_score

    return ScoreResult(
        score=float(score),
        reason="success",
        model_prediction=normalized_prediction,
        response_time_ms=response_time_ms,
        norm=norm,
        rmse=rmse,
        epsilon=epsilon,
        ssim=ssim,
        psnr_db=psnr_db,
    )


def _load_image_b64(path: str) -> str:
    with open(path, "rb") as handle:
        raw = handle.read()
    if not raw:
        raise ValueError(f"empty image: {path}")
    return base64.b64encode(raw).decode("utf-8")


def _llm_verify(
    llm_endpoint: str,
    prediction: str,
    target: str,
    llm_model: str,
) -> tuple[bool, str]:
    url = llm_endpoint.rstrip("/")
    if not url.endswith("/verify-label"):
        url = f"{url}/verify-label"
    response = requests.post(
        url,
        json={"prediction": prediction, "target_label": target, "llm_model": llm_model},
        timeout=C.LLM_ENDPOINT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or "is_match" not in data:
        raise RuntimeError("llm endpoint returned invalid payload")
    return bool(data["is_match"]), str(data.get("reason", ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Perturb end-to-end test")
    parser.add_argument(
        "--image",
        default=os.path.join(ROOT, C.FALLBACK_IMAGE_RELATIVE_PATH),
        help="Path to challenge image (default: assets/dog_1.jpg)",
    )
    parser.add_argument("--prompt", default=C.FALLBACK_LABEL, help="Broad label for LLM verify")
    parser.add_argument("--epsilon", type=float, default=0.12, help="Challenge epsilon")
    parser.add_argument("--min-delta", type=float, default=C.MIN_LINF_DELTA, help="Minimum Linf")
    parser.add_argument("--llm-endpoint", default="http://127.0.0.1:8081", help="LLM endpoint base URL")
    parser.add_argument("--llm-model", default="qwen2.5:1.5b-instruct")
    parser.add_argument(
        "--algorithm",
        default=os.getenv("PERTURB_ATTACK_ALGORITHM", "sma_mi_fgsm"),
        choices=list_attack_algorithms(),
        help="Attack algorithm (or set PERTURB_ATTACK_ALGORITHM)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM semantic verify (attack + score only)",
    )
    args = parser.parse_args()

    print("=== Perturb local end-to-end test ===")
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"image={args.image}")
    print(f"prompt={args.prompt} epsilon={args.epsilon} min_delta={args.min_delta}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[1/5] Load challenge image")
    clean_b64 = _load_image_b64(args.image)
    clean = decode_image_b64(clean_b64).to(device)
    print(f"  shape={tuple(clean.shape)}")

    print("[2/5] Load EfficientNetV2-L + predict clean label")
    model = load_efficientnet_v2_l(device)
    true_label = normalize_prediction_label(predict_label(model, clean))
    target_index = resolve_target_index(true_label)
    print(f"  true_label={true_label} target_index={target_index}")
    if target_index is None:
        raise RuntimeError(f"Unknown label for EfficientNet index map: {true_label}")

    print("[3/5] LLM semantic verify (validator challenge gate)")
    if args.skip_llm:
        print("  skipped (--skip-llm)")
    else:
        try:
            ok, reason = _llm_verify(args.llm_endpoint, true_label, args.prompt, args.llm_model)
            print(f"  is_match={ok} reason={reason}")
            if not ok:
                raise RuntimeError("LLM rejected challenge (prompt does not match model label)")
        except requests.RequestException as exc:
            print(f"  LLM unavailable ({exc}); re-run with --skip-llm for attack-only test")

    print(f"profile: {runtime_profile_summary()}")
    print(f"[4/5] Miner attack ({args.algorithm})")
    t0 = time.perf_counter()
    result = run_attack(
        clean=clean,
        model=model,
        target_index=target_index,
        epsilon=args.epsilon,
        min_delta=args.min_delta,
        device=device,
        algorithm=args.algorithm,
    )
    attack_ms = int((time.perf_counter() - t0) * 1000)
    adv_b64 = encode_image_b64(result.adv)
    print(
        f"  attack_time_ms={attack_ms} algorithm={result.algorithm_key} "
        f"linf={result.linf:.6f} rmse={result.rmse:.6f} pred_idx={result.pred_index}"
    )

    print("[5/5] Validator score (verify_and_score)")
    result = verify_and_score(
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
        f"score={result.score:.6f} reason={result.reason} "
        f"norm={result.norm:.6f} rmse={result.rmse:.6f} "
        f"ssim={result.ssim:.6f} psnr_db={result.psnr_db:.4f} "
        f"new_label={result.model_prediction}"
    )
    print()
    if result.reason == "success":
        print("PASS: local end-to-end workflow succeeded.")
        return 0
    print("FAIL: attack or scoring gate failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
