## Bybit XS Reversal Bot (USDT Perps)

Production-grade Python 3.11+ trading system for **cross-sectional short-term reversal** (“yesterday’s losers bounce”) on Bybit **USDT linear perpetuals**:

- **Backtest**: daily rebalanced cross-sectional long/short (or long-only) portfolio with fees + slippage + turnover.
- **Live**: daily scheduler that computes targets from the last *complete* UTC daily candle, then rebalances positions with maker-biased execution, risk checks, and full audit snapshots.

### Safety / Disclaimer

This is real trading software. Run **testnet first**, use small size, and understand Bybit perps mechanics (leverage, liquidation, funding, position mode). You are responsible for losses.

---

## Setup

### 1) Install

From `bybit_xsreversal/`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -e .
```

### 2) Configure keys

- Copy `bybit_xsreversal/.env.example` to `.env` and fill keys.
- Edit `bybit_xsreversal/config/config.yaml`.

### 3) Run backtest

```bash
bybit-xsreversal backtest --config config/config.yaml
```

Outputs are written to `bybit_xsreversal/outputs/backtest/<timestamp>/`.

### 4) Run live (testnet first)

Dry-run (prints intended orders only):

```bash
bybit-xsreversal live --config config/config.yaml --dry-run
```

Live trading:

```bash
bybit-xsreversal live --config config/config.yaml
```

Live snapshots are written to `bybit_xsreversal/outputs/live/<timestamp>/rebalance_snapshot.json`.

### Run optimizer manually

The optimizer runs in two stages:
- **Stage 1**: fast screen over many candidates (vectorized)
- **Stage 2**: re-evaluates top candidates with the **full backtest logic** (same core signal code as live)

Run a deep optimization without modifying your config:

```bash
bybit-xsreversal optimize --config config/config.yaml --level deep --method random --seed 42 --no-write
```

Useful environment overrides:
- `BYBIT_OPT_WINDOW_DAYS`: rolling history window for optimization (default 365)
- `BYBIT_OPT_TRAIN_FRAC`: train fraction for OOS split (default 0.7)
- `BYBIT_OPT_MIN_TEST_DAYS`: minimum test days (default 60)
- `BYBIT_OPT_MIN_SHARPE`: minimum Sharpe to accept as “good enough” to write back (default 0.0)
- `BYBIT_OPT_TESTNET`: use testnet market data for optimization (default false; mainnet recommended)

Results are written to `outputs/optimize/<timestamp>/` including `best.json` and (if enabled) OOS metrics.

### Run optimizer without keeping an SSH connection open

If your internet/SSH session drops, a normal foreground process may be terminated. Use one of these:

#### Option A: `tmux` (recommended)

```bash
tmux new -s opt
./scripts/run_optimize.sh
# detach: Ctrl-b then d
```

Reattach later:

```bash
tmux attach -t opt
```

#### Option B: `nohup`

```bash
nohup ./scripts/run_optimize.sh > outputs/logs/optimize-nohup.log 2>&1 & disown
```

#### Option B2: run seeds until you find a profitable Stage2 result

This will iterate seeds and stop on the first run whose **Stage2-selected Sharpe** meets your threshold:

```bash
chmod +x ./scripts/optimize_until_profitable.sh
export BYBIT_OPT_MIN_SHARPE=0.5
export OPT_METHOD=random
export OPT_CANDIDATES=10000
export OPT_STAGE2_TOPK=400
export START_SEED=1
export MAX_SEEDS=50
./scripts/optimize_until_profitable.sh
```

Outputs are written under `outputs/optimize/untilprof-<timestamp>/seed-<n>/`.

#### Option C: systemd one-shot service

This runs the optimizer as a background service (survives SSH disconnects).

1) Copy the template and edit paths:

```bash
sudo cp systemd/bybit_xsreversal-optimize.service /etc/systemd/system/bybit_xsreversal-optimize.service
sudo nano /etc/systemd/system/bybit_xsreversal-optimize.service
```

Set:
- `WorkingDirectory=.../bybit_xsreversal`
- `EnvironmentFile=.../bybit_xsreversal/.env`
- `ExecStart=.../bybit_xsreversal/scripts/run_optimize.sh`

2) Reload and start:

```bash
sudo systemctl daemon-reload
sudo systemctl start bybit_xsreversal-optimize.service
```

3) Watch logs:

```bash
journalctl -u bybit_xsreversal-optimize.service -f
```

The wrapper writes a file log to `outputs/logs/optimize-<timestamp>.log`.

### Minimum order size behavior (exchange constraints)

Some perps have exchange minimums (e.g. `ETHUSDT` min qty) that can exceed your intended notional when equity is small.
By default, the bot **skips** orders that would be truncated to zero.

If you are running a dedicated sub-account and want the bot to **bump tiny orders up to minQty** (only when it stays
within your `max_leverage_per_symbol` cap), set:

- `execution.bump_to_min_qty: true`
- `execution.bump_mode: respect_cap`

If you want it to **always** use the exchange minimums (minQty / min order value) even when that exceeds your
per-symbol cap (use with care, especially on small accounts), set:

- `execution.bump_to_min_qty: true`
- `execution.bump_mode: force_exchange_min`

Operational notes:
- The live process is a **scheduler** that must keep running until the rebalance time. If the process/service is stopped, the rebalance will not happen.
- If the bot starts *after* the scheduled time and it hasn’t rebalanced yet for that day, it will **catch up immediately** on startup (and log that it’s doing so).
- Useful flags:
  - `--run-once`: run a single rebalance immediately and exit
  - `--force`: ignore `interval_days` state and force a rebalance (still respects risk checks)
- Make sure you’re looking at the correct Bybit environment:
  - `exchange.testnet` in `config.yaml` controls whether orders go to testnet vs mainnet
  - `BYBIT_TESTNET=true|false` (if set) overrides the config at runtime and is logged on startup
- If you want the bot to **flatten all existing positions** when the strategy produces an empty target book, set:
  - `rebalance.flatten_on_empty_targets: true`

---

## Notes on Daily Candle Alignment (UTC)

Bybit `interval=D` klines are aligned to **UTC day boundaries**. This bot trades at `rebalance.time_utc` (default `00:05`) and uses the **last complete daily candle** (yesterday’s close) to avoid look-ahead.

---

## Project Layout

The implementation matches the requested layout under `bybit_xsreversal/`:

- `src/data/bybit_client.py`: Bybit v5 REST client (auth, retries, rate-limit handling)
- `src/data/market_data.py`: candles, tickers, orderbook spread/depth filters
- `src/strategy/xs_reversal.py`: ranking + vol scaling + target weights
- `src/backtest/backtester.py`: daily rebalance simulator
- `src/live.py`: scheduler + rebalance runner


