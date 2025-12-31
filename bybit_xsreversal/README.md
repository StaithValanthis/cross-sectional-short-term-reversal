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


