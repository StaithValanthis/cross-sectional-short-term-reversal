#!/usr/bin/env bash
set -euo pipefail

# Run optimizer across seeds, collect ALL profitable (OOS) configs, then pick the best one.
# "Profitable" here means OOS Sharpe >= BYBIT_OPT_MIN_OOS_SHARPE and OOS CAGR >= BYBIT_OPT_MIN_OOS_CAGR.
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
#
# OOS Profit gates:
#   BYBIT_OPT_MIN_OOS_SHARPE=0.0    (require OOS Sharpe >= this)
#   BYBIT_OPT_MIN_OOS_CAGR=0.0      (require OOS CAGR >= this)
#   BYBIT_OPT_REQUIRE_OOS=1         (must have OOS evaluation)

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

TS="$(date -u +%Y%m%d-%H%M%S)"
BASE_OUT="outputs/optimize/untilprof-$TS"
mkdir -p "$BASE_OUT"

min_oos_sh="${BYBIT_OPT_MIN_OOS_SHARPE:-0.0}"
min_oos_cagr="${BYBIT_OPT_MIN_OOS_CAGR:-0.0}"
require_oos="${BYBIT_OPT_REQUIRE_OOS:-1}"

echo "[untilprof] base_out=$BASE_OUT"
echo "[untilprof] OOS thresholds: BYBIT_OPT_MIN_OOS_SHARPE=$min_oos_sh BYBIT_OPT_MIN_OOS_CAGR=$min_oos_cagr"
echo "[untilprof] seed sweep: start=$START_SEED step=$SEED_STEP max_seeds=$MAX_SEEDS"

seed="$START_SEED"
count=0
profitable_list=()

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

  # Check OOS profitability from oos_best.json
  if [[ -f "$outdir/oos_best.json" ]]; then
    result=$(python - <<PY
import json, math
p = r"$outdir/oos_best.json"
try:
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    test_metrics = obj.get("test", {})
    oos_sh = test_metrics.get("sharpe")
    oos_cagr = test_metrics.get("cagr")
    try:
        sh_f = float(oos_sh) if oos_sh is not None else float("nan")
        cagr_f = float(oos_cagr) if oos_cagr is not None else float("nan")
    except Exception:
        sh_f = float("nan")
        cagr_f = float("nan")
    print(f"{sh_f:.6g}\t{cagr_f:.6g}")
except Exception as e:
    print("nan\tnan")
PY
)
    oos_sh=$(echo "$result" | cut -f1)
    oos_cagr=$(echo "$result" | cut -f2)
  else
    oos_sh="nan"
    oos_cagr="nan"
  fi

  python - <<PY
import math
oos_sh = float("$oos_sh") if "$oos_sh" not in ("", "nan") else float("nan")
oos_cagr = float("$oos_cagr") if "$oos_cagr" not in ("", "nan") else float("nan")
min_sh = float("$min_oos_sh") if "$min_oos_sh" not in ("", "nan") else 0.0
min_cagr = float("$min_oos_cagr") if "$min_oos_cagr" not in ("", "nan") else 0.0
has_oos = "$require_oos" != "0" and math.isfinite(oos_sh) and math.isfinite(oos_cagr)
ok = has_oos and oos_sh >= min_sh and oos_cagr >= min_cagr
print(f"[untilprof] oos_sharpe={oos_sh:.6g} oos_cagr={oos_cagr:.6g} accepted={ok}")
PY

  ok=$(python - <<PY
import math
oos_sh = float("$oos_sh") if "$oos_sh" not in ("", "nan") else float("nan")
oos_cagr = float("$oos_cagr") if "$oos_cagr" not in ("", "nan") else float("nan")
min_sh = float("$min_oos_sh") if "$min_oos_sh" not in ("", "nan") else 0.0
min_cagr = float("$min_oos_cagr") if "$min_oos_cagr" not in ("", "nan") else 0.0
has_oos = "$require_oos" != "0" and math.isfinite(oos_sh) and math.isfinite(oos_cagr)
print("1" if (has_oos and oos_sh >= min_sh and oos_cagr >= min_cagr) else "0")
PY
)

  if [[ "$ok" == "1" ]]; then
    echo "[untilprof] ACCEPTED seed=$seed (OOS Sharpe=$oos_sh CAGR=$oos_cagr)"
    profitable_list+=("$seed|$oos_sh|$oos_cagr|$outdir")
  else
    echo "[untilprof] rejected seed=$seed (OOS below thresholds or missing)"
  fi

  seed=$((seed + SEED_STEP))
done

echo
echo "[untilprof] === Summary: found ${#profitable_list[@]} profitable configs ==="

if [[ ${#profitable_list[@]} -eq 0 ]]; then
  echo "[untilprof] No profitable configs found after $MAX_SEEDS seeds"
  exit 1
fi

# Rank by OOS Sharpe (descending) and pick the best
best_seed=""
best_sh=""
best_cagr=""
best_outdir=""

for entry in "${profitable_list[@]}"; do
  IFS='|' read -r s sh cagr outd <<< "$entry"
  if [[ -z "$best_seed" ]] || python -c "exit(0 if float('$sh') > float('$best_sh') else 1)"; then
    best_seed="$s"
    best_sh="$sh"
    best_cagr="$cagr"
    best_outdir="$outd"
  fi
done

echo "[untilprof] BEST: seed=$best_seed OOS Sharpe=$best_sh OOS CAGR=$best_cagr"
echo "[untilprof] best config: $best_outdir/best.json"
echo "[untilprof] best OOS: $best_outdir/oos_best.json"

# Write summary JSON
python - <<PY > "$BASE_OUT/summary.json"
import json
profitable = []
for entry in """${profitable_list[@]}""".split():
    if not entry:
        continue
    parts = entry.split("|")
    if len(parts) >= 4:
        profitable.append({
            "seed": int(parts[0]),
            "oos_sharpe": float(parts[1]),
            "oos_cagr": float(parts[2]),
            "outdir": parts[3],
        })
profitable.sort(key=lambda x: x["oos_sharpe"], reverse=True)
summary = {
    "total_seeds": $count,
    "profitable_count": len(profitable),
    "best": profitable[0] if profitable else None,
    "all_profitable": profitable,
}
with open(r"$BASE_OUT/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2))
PY

echo "[untilprof] summary written: $BASE_OUT/summary.json"
echo "[untilprof] To apply the best config, run:"
echo "  bybit-xsreversal --config $CONFIG_PATH optimize --output-dir $best_outdir --level $OPT_LEVEL --method $OPT_METHOD --stage2-topk $OPT_STAGE2_TOPK --seed $best_seed"

exit 0
