#!/usr/bin/env bash
set -euo pipefail

# Run the optimizer in a way that's friendly to non-interactive environments (systemd/nohup).
# You can override everything via environment variables.
#
# Examples:
#   OPT_METHOD=random OPT_CANDIDATES=50000 OPT_STAGE2_TOPK=2000 OPT_NO_WRITE=1 ./scripts/run_optimize.sh
#   BYBIT_OPT_STAGE1_MAX_DD_PCT=50 BYBIT_OPT_STAGE1_MAX_TURNOVER=10 ./scripts/run_optimize.sh

cd "$(dirname "$0")/.."

BIN="${BIN:-.venv/bin/bybit-xsreversal}"
if [[ ! -x "$BIN" ]]; then
  # Fallback to PATH if user is already in an activated venv.
  BIN="bybit-xsreversal"
fi

CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"
OPT_METHOD="${OPT_METHOD:-random}"              # random | grid
OPT_CANDIDATES="${OPT_CANDIDATES:-50000}"      # only used for random; grid ignores unless you set --candidates
OPT_STAGE2_TOPK="${OPT_STAGE2_TOPK:-2000}"
OPT_SEED="${OPT_SEED:-42}"
OPT_NO_WRITE="${OPT_NO_WRITE:-1}"              # 1=true, 0=false

TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/optimize-$TS.log}"

ARGS=(--config "$CONFIG_PATH" optimize --method "$OPT_METHOD" --stage2-topk "$OPT_STAGE2_TOPK" --seed "$OPT_SEED")

if [[ "$OPT_METHOD" == "random" ]]; then
  ARGS+=(--candidates "$OPT_CANDIDATES")
fi

if [[ "$OPT_NO_WRITE" == "1" ]]; then
  ARGS+=(--no-write)
fi

echo "[run_optimize] starting: $BIN ${ARGS[*]}" | tee -a "$LOG_FILE"
echo "[run_optimize] log: $LOG_FILE" | tee -a "$LOG_FILE"
"$BIN" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"


