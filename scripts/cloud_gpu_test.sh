#!/usr/bin/env bash
# Run on cloud GPU server after upload. ImageNet-100 local cache + validator scoring.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"

DEFAULT_ENV_FILE="$SCRIPT_DIR/miner.test.env"
FALLBACK_ENV_FILE="$SCRIPT_DIR/miner.env"
EXAMPLE_ENV_FILE="$SCRIPT_DIR/miner.test.env.example"
ENV_FILE="${PERTURB_ENV_FILE:-$DEFAULT_ENV_FILE}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      if [[ $# -lt 2 ]]; then
        echo "Usage: bash ./scripts/cloud_gpu_test.sh [--env-file PATH] [imagenet100_local_test.py args...]"
        exit 1
      fi
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -f "$ENV_FILE" && "$ENV_FILE" == "$DEFAULT_ENV_FILE" && -f "$FALLBACK_ENV_FILE" ]]; then
  ENV_FILE="$FALLBACK_ENV_FILE"
fi

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo "=== loaded env: $ENV_FILE ==="
elif [[ -f "$EXAMPLE_ENV_FILE" ]]; then
  echo "No env file found. Create one from template:"
  echo "  cp \"$EXAMPLE_ENV_FILE\" \"$DEFAULT_ENV_FILE\""
  echo "Continuing with built-in defaults..."
else
  echo "No env file found; continuing with built-in defaults..."
fi

export PERTURB_GPU_PROFILE="${PERTURB_GPU_PROFILE:-on}"
export PERTURB_ATTACK_ALGORITHM="${PERTURB_ATTACK_ALGORITHM:-sma_mi_fgsm}"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q -e .

echo "=== GPU check ==="
python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device=", torch.cuda.get_device_name(0))
    print("vram_gb=", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
PY

echo ""
echo "=== attack profile ==="
echo "algorithm=${PERTURB_ATTACK_ALGORITHM}"
echo "gpu_profile=${PERTURB_GPU_PROFILE}"

# Default: 3 ImageNet-100 rows, cuda when available (override with --max-rows etc.)
DEFAULT_ARGS=(--max-rows 3 --device cuda --algorithm "${PERTURB_ATTACK_ALGORITHM}")
if [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  bash "$SCRIPT_DIR/run_imagenet100_local_test.sh" --env-file "$ENV_FILE" "${DEFAULT_ARGS[@]}"
else
  bash "$SCRIPT_DIR/run_imagenet100_local_test.sh" --env-file "$ENV_FILE" "${EXTRA_ARGS[@]}"
fi

echo ""
echo "Copy the score/norm/rmse/attack_ms lines above and share them back."
