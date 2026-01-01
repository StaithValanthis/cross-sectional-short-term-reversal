from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError


class ExchangeConfig(BaseModel):
    api_key_env: str = Field(default="BYBIT_API_KEY")
    api_secret_env: str = Field(default="BYBIT_API_SECRET")
    testnet: bool = True
    category: Literal["linear"] = "linear"


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


class RebalanceConfig(BaseModel):
    frequency: Literal["daily"] = "daily"
    time_utc: str = "00:05"  # HH:MM
    candle_close_delay_seconds: int = 30
    startup_grace_seconds: int = 300  # if process starts slightly late, still run "today"
    interval_days: int = 1
    rebalance_fraction: float = 1.0  # 1.0 = full rebalance, 0.5 = half-way toward target
    min_weight_change_bps: float = 5.0  # skip trades below this weight delta (bps of equity)
    flatten_on_empty_targets: bool = False  # if true, close all positions when target book is empty


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


class VolSpikeConfig(BaseModel):
    enabled: bool = False
    lookback_days: int = 14
    threshold_multiplier: float = 2.5


class RiskConfig(BaseModel):
    daily_loss_limit_pct: float = 2.5
    max_drawdown_pct: float = 20.0
    max_turnover: float = 2.0
    kill_switch_enabled: bool = True
    stop_new_trades_on_vol_spike: VolSpikeConfig = Field(default_factory=VolSpikeConfig)


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
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    rebalance: RebalanceConfig = Field(default_factory=RebalanceConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    funding: FundingConfig = Field(default_factory=FundingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
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


def load_config(path: str | Path) -> BotConfig:
    raw = load_yaml_config(path)
    try:
        return BotConfig.model_validate(raw)
    except ValidationError as e:
        msg = f"Invalid config at {Path(path).resolve()}:\n{e}"
        raise ValueError(msg) from e


