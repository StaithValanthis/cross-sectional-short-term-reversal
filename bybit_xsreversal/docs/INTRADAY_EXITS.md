## Intraday Exits (exit-only loop)

This bot’s baseline behavior is **daily rebalance** using **1D candles**. Intraday exits add an optional, **exit-only** loop that runs between rebalances and can **close** positions using **reduce-only** orders.

### What it does

- **Does not** compute targets or open new positions.
- Every `interval_minutes`, it:
  - Fetches open positions (USDT linear perps only)
  - Fetches 1H candles (trigger) and 4H candles (ATR)
  - Evaluates configured exit rules
  - If an exit is triggered, submits a **reduce-only** order to **fully close** the position

### Triggers (opt-in)

You can enable any combination (the engine emits **at most one** exit per symbol per cycle, with priority):

1. **Time stop**: after `time_stop_hours` (optionally only if unrealized PnL < 0)
2. **Fixed ATR stop**: stop at `entry ± fixed_atr_k * ATR`
3. **Trailing ATR stop**: stop at `extreme ∓ trailing_atr_k * ATR`
4. **Breakeven stop**: once favorable excursion ≥ `breakeven_trigger_atr_t * ATR`, set stop to breakeven (includes costs)

### Candle usage

- **Trigger candles**: `candle_interval_trigger` (default `"60"` = 1H)
- **ATR candles**: `atr_interval` (default `"240"` = 4H)
- ATR is Wilder’s ATR over `atr_period`.

### State file

Configured by `intraday_exits.state_path` (default `outputs/state/intraday_exits_state.json`).

Stored fields (per symbol):
- `trailing_extreme`: max high (long) / min low (short) observed since entry state began
- `last_entry_price`: last known entry price
- `entry_ts`: best-effort entry timestamp (set when first seen; Bybit position payload does not reliably provide open time)

### Safety / constraints

- **Hedge mode not supported**: if any open position has `positionIdx` in (1,2), the intraday cycle is skipped.
- **Kill-switch aware**: if `RiskManager` kill switch triggers, intraday cycle is skipped.
- **Dry run by default**: set `intraday_exits.dry_run=false` to actually place orders.
- **Exit order type**: default `market` for reliability.

### Monitoring

Each cycle logs counters:
- `positions_checked`
- `decisions_triggered`
- `exits_placed`
- `skipped_hedge_mode`
- `skipped_kill_switch`
- `missing_candles`
- `missing_entry_price`

### Rollback

- Set `intraday_exits.enabled: false`
- Restart the process

