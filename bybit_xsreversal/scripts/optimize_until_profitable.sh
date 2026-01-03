#!/usr/bin/env bash
set -euo pipefail

# Run optimizer across seeds until a "profitable" Stage2 result is found, then stop.
# "Profitable" here means Stage2-selected Sharpe >= BYBIT_OPT_MIN_SHARPE (same threshold the optimizer already uses).
#
# This is designed for long unattended runs (tmux/systemd/nohup).
#
# Environment knobs (all optional):
#   CONFIG_PATH=config/config.yaml
#   OPT_LEVEL=deep
#   OPT_METHOD=random|grid
#   OPT_CANDIDATES=50000        (random only)
#   OPT_STAGE2_TOPK=2000
#   START_SEED=1
#   MAX_SEEDS=100
#   SEED_STEP=1
#   STOP_ON_ACCEPT=1            (1=true, 0=false)
#
# Profit gate:
#   BYBIT_OPT_MIN_SHARPE=0.0    (set >0 to require profitable Stage2 Sharpe)
#
# Optional OOS gates (if you enabled them in optimizer.py):
#   BYBIT_OPT_REQUIRE_OOS=1
#   BYBIT_OPT_MIN_OOS_SHARPE=0.0
#   BYBIT_OPT_MIN_OOS_CAGR=0.0

cd "$(dirname "$0")/.."

BIN="${BIN:-.venv/bin/bybit-xsreversal}"
if [[ ! -x "$BIN" ]]; then
  BIN="bybit-xsreversal"
fi

CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"
OPT_LEVEL="${OPT_LEVEL:-deep}"
OPT_METHOD="${OPT_METHOD:-random}"
OPT_CANDIDATES="${OPT_CANDIDATES:-50000}"
OPT_STAGE2_TOPK="${OPT_STAGE2_TOPK:-2000}"

START_SEED="${START_SEED:-1}"
MAX_SEEDS="${MAX_SEEDS:-100}"
SEED_STEP="${SEED_STEP:-1}"
STOP_ON_ACCEPT="${STOP_ON_ACCEPT:-1}"

TS="$(date -u +%Y%m%d-%H%M%S)"
BASE_OUT="outputs/optimize/untilprof-$TS"
mkdir -p "$BASE_OUT"

min_sh="${BYBIT_OPT_MIN_SHARPE:-0.0}"

echo "[untilprof] base_out=$BASE_OUT"
echo "[untilprof] threshold: BYBIT_OPT_MIN_SHARPE=$min_sh"
echo "[untilprof] seed sweep: start=$START_SEED step=$SEED_STEP max_seeds=$MAX_SEEDS"

seed="$START_SEED"
count=0
while [[ "$count" -lt "$MAX_SEEDS" ]]; do
  count=$((count + 1))
  outdir="$BASE_OUT/seed-$seed"
  mkdir -p "$outdir"

  echo
  echo "[untilprof] === seed=$seed ($count/$MAX_SEEDS) outdir=$outdir ==="

  args=(--config "$CONFIG_PATH" optimize --output-dir "$outdir" --level "$OPT_LEVEL" --method "$OPT_METHOD" --stage2-topk "$OPT_STAGE2_TOPK" --seed "$seed" --no-write --no-progress)
  if [[ "$OPT_METHOD" == "random" ]]; then
    args+=(--candidates "$OPT_CANDIDATES")
  fi

  # Run optimizer
  "$BIN" "${args[@]}"

  # Determine acceptance based on Stage2-selected Sharpe in best.json
  # best.json is written even when rejected_by_threshold.
  if [[ -f "$outdir/best.json" ]]; then
    sh=$(python - <<PY
import json, math
p = r"$outdir/best.json"
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)
v = obj.get("sharpe")
try:
    x = float(v)
except Exception:
    x = float("nan")
print(x)
PY
)
  else
    sh="nan"
  fi

  python - <<PY
import math
sh = float("$sh") if "$sh" not in ("", "nan") else float("nan")
min_sh = float("$min_sh") if "$min_sh" not in ("", "nan") else 0.0
ok = (math.isfinite(sh) and sh >= min_sh)
print(f"[untilprof] stage2_sharpe={sh:.6g} min_sharpe={min_sh:.6g} accepted={ok}")
PY

  ok=$(python - <<PY
import math
sh = float("$sh") if "$sh" not in ("", "nan") else float("nan")
min_sh = float("$min_sh") if "$min_sh" not in ("", "nan") else 0.0
print("1" if (math.isfinite(sh) and sh >= min_sh) else "0")
PY
)

  if [[ "$ok" == "1" ]]; then
    echo "[untilprof] ACCEPTED seed=$seed (best.json: $outdir/best.json)"
    if [[ "$STOP_ON_ACCEPT" == "1" ]]; then
      echo "[untilprof] stopping on first accepted seed"
      exit 0
    fi
  fi

  seed=$((seed + SEED_STEP))
done

echo "[untilprof] finished: no accepted seed found after $MAX_SEEDS seeds"
exit 0


