#!/usr/bin/env bash
# Run on cloud GPU server after upload. Prints metrics to share back.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PERTURB_GPU_PROFILE="${PERTURB_GPU_PROFILE:-on}"

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
echo "=== attack algorithm: \${PERTURB_ATTACK_ALGORITHM:-sma_mi_fgsm} ==="
python scripts/local_e2e_test.py --skip-llm "$@"

echo ""
echo "Copy the score/norm/rmse/attack_time_ms lines above and share them back."
