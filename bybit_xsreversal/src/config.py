from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class ExchangeConfig(BaseModel):
    api_key_env: str = Field(default="BYBIT_API_KEY")
    api_secret_env: str = Field(default="BYBIT_API_SECRET")
    testnet: bool = True
    category: Literal["linear"] = "linear"

    # Startup-only leverage alignment (opt-in)
    set_leverage_on_startup: bool = False
    leverage: str = "5"
    leverage_apply_mode: Literal["universe", "positions"] = "universe"
    leverage_symbols_max: int = 200
    leverage_state_path: str = "outputs/state/leverage_state.json"

    @field_validator("category")
    @classmethod
    def _category_linear_only(cls, v: str) -> str:
        if str(v) != "linear":
            raise ValueError("Only USDT linear perps are supported: exchange.category must be 'linear'")
        return v


class LiquidityBucketConfig(BaseModel):
    enabled: bool = False
    exclude_top_k: int = 0
    min_rank: int = 1
    max_rank: int = 10_000


class UniverseConfig(BaseModel):
    top_n_by_volume: int = 80
    exclude_symbols: list[str] = Field(default_factory=list)
    include_symbols: list[str] = Field(default_factory=list)
    min_24h_quote_volume: float = 0.0
    min_open_interest: float = 0.0
    min_history_days: int = 120
    liquidity_bucket: LiquidityBucketConfig = Field(default_factory=LiquidityBucketConfig)


class SignalConfig(BaseModel):
    lookback_days: Literal[1, 2, 3, 5] = 1
    long_quantile: float = 0.1
    short_quantile: float = 0.1
    long_only: bool = False

    @model_validator(mode="after")
    def _validate_quantiles(self) -> "SignalConfig":
        if not (0.0 < float(self.long_quantile) < 1.0):
            raise ValueError("signal.long_quantile must be in (0,1)")
        if not (0.0 < float(self.short_quantile) < 1.0):
            raise ValueError("signal.short_quantile must be in (0,1)")
        return self


class RebalanceConfig(BaseModel):
    frequency: Literal["daily"] = "daily"
    time_utc: str = "00:05"  # HH:MM
    candle_close_delay_seconds: int = 30
    startup_grace_seconds: int = 300  # if process starts slightly late, still run "today"
    interval_days: int = 1
    rebalance_fraction: float = 1.0  # 1.0 = full rebalance, 0.5 = half-way toward target
    min_weight_change_bps: float = 5.0  # skip trades below this weight delta (bps of equity)
    flatten_on_empty_targets: bool = False  # if true, close all positions when target book is empty

    # Turnover / fee guardrail (optional). Applies to NON-risk orders only.
    max_turnover_per_rebalance_equity_mult: float | None = None
    turnover_cap_mode: Literal["scale", "skip"] = "scale"


class SizingConfig(BaseModel):
    target_gross_leverage: float = 1.0
    vol_lookback_days: int = 14
    max_leverage_per_symbol: float = 0.20
    max_notional_per_symbol: float = 25_000.0
    min_notional_per_symbol: float = 15.0


class RegimeFilterConfig(BaseModel):
    enabled: bool = True
    use_market_regime: bool = True
    market_proxy_symbol: str = "BTCUSDT"
    market_adx_threshold: float = 25.0
    symbol_adx_threshold: float = 30.0
    ema_fast: int = 20
    ema_slow: int = 50
    action: Literal["skip", "scale_down", "switch_to_momentum"] = "scale_down"
    scale_factor: float = 0.35

    # Observability + optional dynamic gross scaling
    log_actions: bool = False
    dynamic_gross_scale_enabled: bool = False
    dynamic_gross_scale_min: float = 1.0
    dynamic_gross_scale_max: float = 5.0


class FiltersConfig(BaseModel):
    max_spread_bps: float = 8.0
    min_orderbook_depth_usd: float = 5_000.0
    depth_band_bps: float = 15.0
    regime_filter: RegimeFilterConfig = Field(default_factory=RegimeFilterConfig)


class FundingFilterConfig(BaseModel):
    enabled: bool = False
    max_abs_daily_funding_rate: float = 0.003  # 30 bps/day approx (sum of 8h rates)
    use_mainnet_data_even_on_testnet: bool = True


class FundingConfig(BaseModel):
    model_in_backtest: bool = True
    filter: FundingFilterConfig = Field(default_factory=FundingFilterConfig)


class ExecutionConfig(BaseModel):
    order_type: Literal["limit", "market"] = "limit"
    maker_bias: bool = True
    post_only: bool = True
    max_order_age_seconds: int = 20
    price_offset_bps: float = 2.0
    ioc_fallback: bool = True
    cancel_open_orders: Literal["none", "bot_only", "symbols", "all"] = "bot_only"
    bump_to_min_qty: bool = False
    bump_mode: Literal["respect_cap", "force_exchange_min"] = "respect_cap"
    min_order_value_usdt: float | None = 5.0

    # Churn reduction + observability
    max_cancel_replace_per_symbol: int = 3
    log_execution_quality: bool = True


class VolSpikeConfig(BaseModel):
    enabled: bool = False
    lookback_days: int = 14
    threshold_multiplier: float = 2.5


class RiskConfig(BaseModel):
    daily_loss_limit_pct: float = 2.5
    daily_loss_limit_pct_tier2: float | None = None
    max_drawdown_pct: float = 20.0
    max_turnover: float = 2.0
    kill_switch_enabled: bool = True
    max_consecutive_loss_days: int = 0

    # Soft per-position exits (applied at rebalance time by overriding target_notional -> 0)
    max_hold_days: int = 0
    max_loss_per_position_pct_equity: float = 0.0
    cooldown_days_after_forced_exit: int = 0
    stop_new_trades_on_vol_spike: VolSpikeConfig = Field(default_factory=VolSpikeConfig)


class IntradayExitsConfig(BaseModel):
    enabled: bool = False
    dry_run: bool = True
    interval_minutes: int = 60
    category: Literal["linear"] = "linear"
    state_path: str = "outputs/state/intraday_exits_state.json"

    candle_interval_trigger: str = "60"  # 1H
    atr_interval: str = "240"  # 4H
    atr_period: int = 14
    min_bars_trigger: int = 3
    min_bars_atr: int = 30

    fixed_atr_stop_enabled: bool = False
    fixed_atr_k: float = 2.0

    trailing_atr_stop_enabled: bool = False
    trailing_atr_k: float = 2.5

    stop_to_breakeven_enabled: bool = False
    breakeven_trigger_atr_t: float = 1.5
    breakeven_costs_bps: float = 6.0

    time_stop_enabled: bool = False
    time_stop_hours: int = 24
    time_stop_only_if_unprofitable: bool = True

    use_intrabar_trigger: bool = True
    use_last_price_trigger: bool = False
    max_positions_per_cycle: int = 200

    exit_order_type: Literal["market", "limit"] = "market"
    exit_price_offset_bps: float = 0.0
    cancel_existing_exit_orders: bool = True
    min_notional_to_exit_usd: float = 5.0

    @model_validator(mode="after")
    def _validate_intraday(self) -> "IntradayExitsConfig":
        if str(self.category) != "linear":
            raise ValueError("intraday_exits.category must be 'linear'")
        if int(self.interval_minutes) < 5:
            raise ValueError("intraday_exits.interval_minutes must be >= 5")
        if int(self.atr_period) < 2:
            raise ValueError("intraday_exits.atr_period must be >= 2")
        if float(self.fixed_atr_k) <= 0:
            raise ValueError("intraday_exits.fixed_atr_k must be > 0")
        if float(self.trailing_atr_k) <= 0:
            raise ValueError("intraday_exits.trailing_atr_k must be > 0")
        if float(self.breakeven_trigger_atr_t) <= 0:
            raise ValueError("intraday_exits.breakeven_trigger_atr_t must be > 0")
        if int(self.time_stop_hours) < 1 or int(self.time_stop_hours) > 168:
            raise ValueError("intraday_exits.time_stop_hours must be between 1 and 168")
        if int(self.max_positions_per_cycle) <= 0:
            raise ValueError("intraday_exits.max_positions_per_cycle must be > 0")
        if float(self.min_notional_to_exit_usd) < 0:
            raise ValueError("intraday_exits.min_notional_to_exit_usd must be >= 0")
        return self


class BacktestConfig(BaseModel):
    start_date: str
    end_date: str
    initial_equity: float = 10_000.0
    taker_fee_bps: float = 6.0
    maker_fee_bps: float = 1.0
    slippage_bps: float = 3.0
    borrow_cost_bps: float = 0.0
    allow_partial_fills: bool = True
    cache_dir: str = "data_cache"


class BotConfig(BaseModel):
    # Optional overlay profile applied by the loader (base config + profiles/<config_profile>.yaml).
    config_profile: str = "default"

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    intraday_exits: IntradayExitsConfig = Field(default_factory=IntradayExitsConfig)

    backtest: BacktestConfig


def load_yaml_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")
    return data


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = dict(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> BotConfig:
    raw = load_yaml_config(path)

    profile = str(raw.get("config_profile") or "default").strip()
    if profile and profile != "default":
        prof_path = Path(path).parent / "profiles" / f"profile_{profile}.yaml"
        if prof_path.exists():
            overlay = load_yaml_config(prof_path)
            raw = _deep_merge_dicts(raw, overlay)
        else:
            raise ValueError(f"config_profile='{profile}' requested but profile file not found: {prof_path.resolve()}")

    try:
        return BotConfig.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid config at {Path(path).resolve()}:\n{e}"
        raise ValueError(msg) from e
