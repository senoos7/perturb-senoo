#!/usr/bin/env bash
# Local ImageNet-100 + SMA-MI-FGSM + validator-style scoring (CPU ok).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"

ENV_FILE="${PERTURB_ENV_FILE:-$SCRIPT_DIR/cpu_attack.env.example}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo "=== loaded env: $ENV_FILE ==="
fi

PY="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  if "$ROOT_DIR/.venv/bin/python" -c "import torch" 2>/dev/null; then
    PY="$ROOT_DIR/.venv/bin/python"
  fi
fi
if ! "$PY" -c "import torch" 2>/dev/null; then
  if command -v python3 >/dev/null 2>&1 && python3 -c "import torch" 2>/dev/null; then
    PY="python3"
  else
    echo "ERROR: no Python with torch found. Install: pip install torch torchvision" >&2
    exit 1
  fi
fi

echo "=== device check (python=$PY) ==="
"$PY" - <<'PY'
import torch
print("torch=", torch.__version__)
print("cuda_available=", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device=", torch.cuda.get_device_name(0))
print("device_count=", torch.cuda.device_count())
PY

# Default to cuda when available and caller did not pass --device
if [[ " $* " != *" --device "* ]]; then
  if "$PY" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    set -- "$@" --device cuda
    echo "=== auto-selected --device cuda ==="
  else
    set -- "$@" --device cpu
    echo "=== no CUDA driver/GPU — using --device cpu ==="
  fi
fi

echo ""
exec "$PY" "$SCRIPT_DIR/imagenet100_local_test.py" --env-file "$ENV_FILE" "$@"
