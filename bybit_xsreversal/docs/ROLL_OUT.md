## Rollout order (safe)

### 1) Enable logging only (prod, no behavior change)

- Ensure `execution.log_execution_quality: true`
- Ensure `intraday_exits.enabled: false`

Deploy and confirm normal daily rebalance works.

### 2) Testnet: intraday exits dry-run

- `exchange.testnet: true`
- `intraday_exits.enabled: true`
- `intraday_exits.dry_run: true`
- Enable ONE exit type at a time (start with fixed ATR)

Monitor:
- `decisions_triggered` count
- `would_exit` logs and stop levels

### 3) Testnet: reduce-only exits live (conservative)

- `intraday_exits.dry_run: false`
- Keep `exit_order_type: market`
- Use conservative parameters (larger ATR multipliers)

### 4) Prod deploy with flags off, then gradual enable

- Deploy with:
  - `intraday_exits.enabled: false`
  - `exchange.set_leverage_on_startup: false`
- After stability:
  - enable `intraday_exits.enabled: true` with `dry_run: true` for a day
  - then enable `dry_run: false`

### Rollback

- Set `intraday_exits.enabled: false`
- Restart process

