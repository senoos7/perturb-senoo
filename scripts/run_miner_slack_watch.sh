#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_DIR="$ROOT_DIR/scripts"
ENV_FILE="$SCRIPT_DIR/miner_slack.env"
EXAMPLE_ENV_FILE="$SCRIPT_DIR/miner_slack.env.example"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE"
  echo "Create it from template:"
  echo "  cp \"$EXAMPLE_ENV_FILE\" \"$ENV_FILE\""
  echo "Then set SLACK_WEBHOOK_URL to your Slack Incoming Webhook."
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${SLACK_WEBHOOK_URL:-}" ]]; then
  echo "SLACK_WEBHOOK_URL is empty in $ENV_FILE"
  exit 1
fi

export MINER_SLACK_ENV="$ENV_FILE"
export PM2_MINER_PROCESS="${PM2_MINER_PROCESS:-perturb-miner}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/miner_log_reporter.py" \
  --env-file "$ENV_FILE" \
  --pm2-process "$PM2_MINER_PROCESS" \
  --heartbeat-seconds "${SLACK_HEARTBEAT_SECONDS:-1800}"
