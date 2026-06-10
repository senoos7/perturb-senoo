#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
ENV_FILE="$SCRIPT_DIR/miner.env"
EXAMPLE_ENV_FILE="$SCRIPT_DIR/miner.env.example"
GPU_EXAMPLE_ENV_FILE="$SCRIPT_DIR/miner.env.gpu.example"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "Create it from template:"
  echo "  cp \"$GPU_EXAMPLE_ENV_FILE\" \"$ENV_FILE\""
  echo "  # or: cp \"$EXAMPLE_ENV_FILE\" \"$ENV_FILE\""
  echo "Then edit WALLET_HOTKEY if your hotkey name is not 'default'."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

WALLET_NAME="${WALLET_NAME:-}"
WALLET_HOTKEY="${WALLET_HOTKEY:-}"
WALLET_HOTKEY_SS58="${WALLET_HOTKEY_SS58:-}"
NETUID="${NETUID:-1}"
NETWORK="${NETWORK:-local}"
MINER_EXTRA_ARGS="${MINER_EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
MINER_PORT="${MINER_PORT:-9000}"

if [[ -z "$WALLET_NAME" || -z "$WALLET_HOTKEY" ]]; then
  echo "WALLET_NAME and WALLET_HOTKEY must be set in $ENV_FILE"
  exit 1
fi

# Bittensor expects hotkey name, not SS58 (5Hz...).
if [[ "$WALLET_HOTKEY" =~ ^5[A-HJ-NP-Za-km-z1-9]{40,}$ ]]; then
  echo "ERROR: WALLET_HOTKEY looks like an SS58 address."
  echo "Set WALLET_HOTKEY to the hotkey folder name (e.g. default)."
  echo "Keep the address in WALLET_HOTKEY_SS58 for reference."
  if [[ -n "$WALLET_HOTKEY_SS58" ]]; then
    echo "Expected SS58: $WALLET_HOTKEY_SS58"
  fi
  echo "List hotkeys: btcli wallet overview --wallet.name $WALLET_NAME"
  exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter not found: $PYTHON_BIN"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "Miner wallet: name=$WALLET_NAME hotkey=$WALLET_HOTKEY netuid=$NETUID network=$NETWORK port=$MINER_PORT"
if [[ -n "$WALLET_HOTKEY_SS58" ]]; then
  echo "Registered hotkey SS58 (reference): $WALLET_HOTKEY_SS58"
fi

if [[ "${1:-}" == "--foreground" ]]; then
  echo "Starting miner in foreground..."
  python neurons/miner.py \
    $MINER_EXTRA_ARGS \
    --netuid "$NETUID" \
    --network "$NETWORK" \
    --wallet.name "$WALLET_NAME" \
    --wallet.hotkey "$WALLET_HOTKEY" \
    --axon.port "$MINER_PORT" \
    --log-level "$LOG_LEVEL"
  exit 0
fi

echo "Starting miner with PM2..."
if pm2 describe perturb-miner >/dev/null 2>&1; then
  pm2 delete perturb-miner
fi
pm2 start ".venv/bin/python" --name perturb-miner -- \
  neurons/miner.py \
  $MINER_EXTRA_ARGS \
  --netuid "$NETUID" \
  --network "$NETWORK" \
  --wallet.name "$WALLET_NAME" \
  --wallet.hotkey "$WALLET_HOTKEY" \
  --axon.port "$MINER_PORT" \
  --log-level "$LOG_LEVEL"
pm2 save
pm2 status perturb-miner
