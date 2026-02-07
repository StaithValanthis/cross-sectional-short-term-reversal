## Config updates

All config lives in `bybit_xsreversal/config/config.yaml` and is validated by `bybit_xsreversal/src/config.py`.

### New: `intraday_exits` (defaults OFF)

Key fields:
- `intraday_exits.enabled`: enable intraday loop (exit-only)
- `intraday_exits.dry_run`: log-only (recommended first)
- `intraday_exits.interval_minutes`: scheduler interval (>= 5)
- `intraday_exits.candle_interval_trigger`: default `"60"` (1H)
- `intraday_exits.atr_interval`: default `"240"` (4H)
- `intraday_exits.atr_period`: ATR period (>= 2)
- `intraday_exits.fixed_atr_stop_enabled`, `fixed_atr_k`
- `intraday_exits.trailing_atr_stop_enabled`, `trailing_atr_k`
- `intraday_exits.stop_to_breakeven_enabled`, `breakeven_trigger_atr_t`, `breakeven_costs_bps`
- `intraday_exits.time_stop_enabled`, `time_stop_hours`, `time_stop_only_if_unprofitable`
- `intraday_exits.exit_order_type`: `"market"` (default) or `"limit"`
- `intraday_exits.state_path`: persisted state file

### New: turnover cap (optional)

Under `rebalance`:
- `max_turnover_per_rebalance_equity_mult`: `null` disables; otherwise cap = mult * equity
- `turnover_cap_mode`: `"scale"` or `"skip"`

Risk exits and reconciliation closes bypass this cap.

### New: execution churn guardrails / observability

Under `execution`:
- `max_cancel_replace_per_symbol`: after N cancel/replace cycles, forced closes go direct market
- `log_execution_quality`: logs executor counters per rebalance

### New: startup leverage alignment (opt-in)

Under `exchange`:
- `set_leverage_on_startup`: `false` by default
- `leverage`: string passed to Bybit (e.g. `"5"`)
- `leverage_apply_mode`: `"universe"` or `"positions"`
- `leverage_state_path`: idempotency file (prevents spam on restart)

### New: risk hardening

Under `risk`:
- `daily_loss_limit_pct_tier2`: `null` disables
- `max_consecutive_loss_days`: `0` disables

### New: regime observability + optional dynamic gross scaling

Under `filters.regime_filter`:
- `log_actions`
- `dynamic_gross_scale_enabled`
- `dynamic_gross_scale_min`, `dynamic_gross_scale_max`

### New: config profiles (overlay)

Top-level:
- `config_profile: "default"` (default)

If set to e.g. `candidate_1`, the loader merges:
- `config/config.yaml` + `config/profiles/profile_candidate_1.yaml`

