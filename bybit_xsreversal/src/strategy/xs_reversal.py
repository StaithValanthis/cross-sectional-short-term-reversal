from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BotConfig
from src.strategy.portfolio import PortfolioTargets, build_dollar_neutral_weights, weights_to_notionals
from src.strategy.regime import regime_gate


@dataclass(frozen=True)
class RebalanceSnapshot:
    asof: str
    universe: list[str]
    selected_longs: list[str]
    selected_shorts: list[str]
    returns: dict[str, float]
    vol: dict[str, float]
    raw_weights: dict[str, float]
    final_weights: dict[str, float]
    filters: dict[str, Any]


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def _rolling_vol(close: pd.Series, lookback: int) -> pd.Series:
    r = _daily_returns(close)
    return r.rolling(lookback, min_periods=lookback).std(ddof=0)


def compute_targets_from_daily_candles(
    *,
    candles: dict[str, pd.DataFrame],
    config: BotConfig,
    equity_usd: float,
    asof: datetime | None = None,
    market_proxy_candles: pd.DataFrame | None = None,
    current_weights: dict[str, float] | None = None,
    funding_daily_rate: dict[str, float] | None = None,
) -> tuple[PortfolioTargets, RebalanceSnapshot]:
    """
    Shared core used by backtest and live:
    - compute lookback return ranks (losers long, winners short)
    - inverse-vol scaling
    - dollar-neutral weights + constraints
    - optional regime gating (ADX/EMA)
    """
    if asof is None:
        # Use the last common candle timestamp
        common = sorted(set.intersection(*(set(df.index) for df in candles.values())))
        if not common:
            raise ValueError("No common timestamps across symbols.")
        asof = common[-1]
    asof = asof.astimezone(UTC)

    lookback = int(config.signal.lookback_days)
    vol_lb = int(config.sizing.vol_lookback_days)
    min_hist = int(max(config.universe.min_history_days, 30))

    returns: dict[str, float] = {}
    vol: dict[str, float] = {}
    exclusions: dict[str, str] = {}

    for sym, df in candles.items():
        df = df.sort_index()
        if asof not in df.index:
            exclusions[sym] = "missing_asof_candle"
            continue
        sub = df.loc[:asof].copy()
        if len(sub) < max(lookback + 1, vol_lb + 2, min_hist):
            exclusions[sym] = "insufficient_history"
            continue
        close = sub["close"].astype(float)
        ret = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1.0)
        v = _rolling_vol(close, vol_lb).iloc[-1]
        if not np.isfinite(ret) or not np.isfinite(v) or v <= 0:
            exclusions[sym] = "bad_ret_or_vol"
            continue
        returns[sym] = ret
        vol[sym] = float(v)

    universe = sorted(returns.keys())
    if len(universe) < 5:
        raise ValueError(f"Universe too small after data checks: {len(universe)}")

    # Optional funding filter (daily aggregate funding rate)
    # Only apply if it won't leave us with too few symbols (< 5)
    if config.funding.filter.enabled and funding_daily_rate is not None:
        max_abs = float(config.funding.filter.max_abs_daily_funding_rate)
        symbols_with_funding = [s for s in universe if funding_daily_rate.get(s) is not None]
        symbols_without_funding = [s for s in universe if funding_daily_rate.get(s) is None]
        
        # Apply filter only if we'll have enough symbols left
        filtered_out: list[str] = []
        for sym in symbols_with_funding:
            fr = funding_daily_rate.get(sym)
            if fr is not None and abs(float(fr)) > max_abs:
                filtered_out.append(sym)
        
        remaining_count = len(symbols_without_funding) + len(symbols_with_funding) - len(filtered_out)
        if remaining_count >= 5:
            # Safe to apply filter
            for sym in filtered_out:
                exclusions[sym] = "funding_filter"
                returns.pop(sym, None)
                vol.pop(sym, None)
            universe = sorted(returns.keys())
        else:
            # Filter too strict; skip it to preserve universe size
            logger.debug(
                "Funding filter would leave {} symbols (< 5); skipping filter to preserve universe",
                remaining_count
            )

    # Cross-sectional rank by lookback return (reversal by default)
    ranked = sorted(universe, key=lambda s: returns[s])
    n = len(ranked)

    q_long = float(config.signal.long_quantile)
    q_short = float(config.signal.short_quantile)
    k_long = max(1, int(np.floor(n * q_long)))
    k_short = max(1, int(np.floor(n * q_short)))

    selected_longs = ranked[:k_long]
    selected_shorts = ranked[-k_short:] if not config.signal.long_only else []

    # Regime-aware signal direction: in strong market trend we can switch to momentum.
    signal_mode: str = "reversal"
    rf = config.filters.regime_filter
    market_df = market_proxy_candles if (rf.enabled and rf.use_market_regime) else None
    market_decision = None
    if rf.enabled and market_df is not None:
        mdec = regime_gate(
            symbol_df=market_df,
            market_df=None,
            symbol_adx_threshold=rf.market_adx_threshold,
            market_adx_threshold=rf.market_adx_threshold,
            ema_fast=rf.ema_fast,
            ema_slow=rf.ema_slow,
            action=rf.action,
            scale_factor=rf.scale_factor,
        )
        market_decision = mdec
        if bool(getattr(rf, "log_actions", False)):
            logger.info("Regime (market) decision={} scale={} meta={}", mdec.action, float(mdec.scale), mdec.meta)
        if mdec.action == "switch_to_momentum":
            signal_mode = "momentum"
            # Flip selection: long winners, short losers
            selected_longs = ranked[-k_long:]
            selected_shorts = ranked[:k_short] if not config.signal.long_only else []

    # Inverse vol scores (equal risk)
    long_scores = {s: 1.0 / vol[s] for s in selected_longs}
    short_scores = {s: 1.0 / vol[s] for s in selected_shorts}

    # Optional dynamic gross scaling (opt-in): adjust target gross based on market regime scale.
    base_gross = float(config.sizing.target_gross_leverage)
    eff_gross = base_gross
    if bool(getattr(rf, "dynamic_gross_scale_enabled", False)) and market_decision is not None:
        try:
            eff_gross = float(base_gross) * float(market_decision.scale)
            eff_gross = max(float(getattr(rf, "dynamic_gross_scale_min", 1.0)), min(float(getattr(rf, "dynamic_gross_scale_max", 5.0)), eff_gross))
        except Exception:
            eff_gross = base_gross
        if bool(getattr(rf, "log_actions", False)):
            logger.info("Dynamic gross scaling: base_gross={} effective_gross={}", base_gross, eff_gross)

    raw_weights = build_dollar_neutral_weights(
        long_scores=long_scores,
        short_scores=short_scores,
        target_gross_leverage=float(eff_gross),
        max_abs_weight_per_symbol=float(config.sizing.max_leverage_per_symbol),
        long_only=bool(config.signal.long_only),
    )

    # Optional regime gating: scale all weights down (or skip) in strong trends
    final_weights = dict(raw_weights)
    regime_meta: dict[str, Any] = {}
    if rf.enabled:
        # Market regime decision (single) can scale the whole book.
        if market_df is not None:
            mdec = market_decision or regime_gate(
                symbol_df=market_df,
                market_df=None,
                symbol_adx_threshold=rf.market_adx_threshold,
                market_adx_threshold=rf.market_adx_threshold,
                ema_fast=rf.ema_fast,
                ema_slow=rf.ema_slow,
                action=rf.action,
                scale_factor=rf.scale_factor,
            )
            regime_meta["market"] = {"decision": mdec.action, "scale": mdec.scale, **mdec.meta}
            regime_meta["effective_target_gross_leverage"] = float(eff_gross)
            # If dynamic gross scaling is enabled, we already applied market scale via eff_gross.
            apply_market_scale_to_weights = not bool(getattr(rf, "dynamic_gross_scale_enabled", False))

            if mdec.scale == 0.0:
                final_weights = {k: 0.0 for k in final_weights}
            elif apply_market_scale_to_weights:
                final_weights = {k: v * mdec.scale for k, v in final_weights.items()}

        # Symbol regime (optional): scale per-symbol
        per_symbol: dict[str, Any] = {}
        for sym in list(final_weights.keys()):
            sdec = regime_gate(
                symbol_df=candles[sym].loc[:asof],
                market_df=market_df,
                symbol_adx_threshold=rf.symbol_adx_threshold,
                market_adx_threshold=rf.market_adx_threshold,
                ema_fast=rf.ema_fast,
                ema_slow=rf.ema_slow,
                action=rf.action,
                scale_factor=rf.scale_factor,
            )
            per_symbol[sym] = {"decision": sdec.action, "scale": sdec.scale, **sdec.meta}
            final_weights[sym] = float(final_weights[sym]) * float(sdec.scale)
        regime_meta["per_symbol"] = per_symbol
        if bool(getattr(rf, "log_actions", False)):
            # Avoid logging thousands of lines: only log decisions for symbols we actually trade.
            for sym, info in list(per_symbol.items())[:200]:
                logger.info("Regime (symbol) {} decision={} scale={} meta={}", sym, info.get("decision"), info.get("scale"), {k: info.get(k) for k in ("symbol_adx", "symbol_ema_slope", "market_adx")})

    # Drop tiny weights induced by scaling
    final_weights = {k: float(v) for k, v in final_weights.items() if abs(float(v)) > 1e-8}

    # Turnover controls: partial rebalance + thresholding (shared with live/backtest)
    if current_weights is not None:
        frac = float(config.rebalance.rebalance_fraction)
        frac = max(0.0, min(1.0, frac))
        thresh = float(config.rebalance.min_weight_change_bps) / 10_000.0
        union = set(current_weights) | set(final_weights)
        blended: dict[str, float] = {}
        for s in union:
            cur = float(current_weights.get(s, 0.0))
            tgt = float(final_weights.get(s, 0.0))
            new = cur + frac * (tgt - cur)
            if abs(new - cur) < thresh:
                new = cur
            if abs(new) > 1e-10:
                blended[s] = new
        # Safety: don't exceed target gross; scale down only (never scale up)
        gross = sum(abs(v) for v in blended.values())
        cap = float(config.sizing.target_gross_leverage)
        if gross > cap and gross > 0:
            scale = cap / gross
            blended = {k: v * scale for k, v in blended.items()}
        final_weights = blended

    notionals = weights_to_notionals(
        final_weights,
        equity_usd=float(equity_usd),
        max_notional_per_symbol=float(config.sizing.max_notional_per_symbol),
        min_notional_per_symbol=float(config.sizing.min_notional_per_symbol),
    )

    # Remove symbols dropped by min notional
    final_weights = {k: final_weights[k] for k in notionals.keys()}

    meta: dict[str, Any] = {
        "asof": asof.isoformat(),
        "n_universe_pre": n,
        "n_universe_post": len(universe),
        "k_long": k_long,
        "k_short": 0 if config.signal.long_only else k_short,
        "exclusions": exclusions,
        "signal_mode": signal_mode,
        "regime": regime_meta,
    }

    snap = RebalanceSnapshot(
        asof=asof.isoformat(),
        universe=universe,
        selected_longs=selected_longs,
        selected_shorts=selected_shorts,
        returns={k: float(v) for k, v in returns.items()},
        vol={k: float(v) for k, v in vol.items()},
        raw_weights={k: float(v) for k, v in raw_weights.items()},
        final_weights={k: float(v) for k, v in final_weights.items()},
        filters=meta,
    )

    if not notionals:
        # This can be normal (e.g., min notional filter, partial-rebalance threshold, regime scaling).
        # Keep it at DEBUG to avoid flooding logs during backtests/optimization.
        logger.debug("All targets filtered out; no notionals to trade.")

    return PortfolioTargets(weights=final_weights, notionals_usd=notionals, meta=meta), snap


