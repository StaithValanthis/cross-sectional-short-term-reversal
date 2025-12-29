from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BotConfig
from src.data.market_data import MarketData
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.strategy.xs_reversal import compute_targets_from_daily_candles


def _parse_date(d: str) -> datetime:
    # YYYY-MM-DD in UTC
    dt = datetime.fromisoformat(d)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _buffer_days(cfg: BotConfig) -> int:
    rf = cfg.filters.regime_filter
    return int(max(cfg.sizing.vol_lookback_days + 10, cfg.signal.lookback_days + 5, rf.ema_slow + 20, 40))


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    daily_turnover: pd.Series
    metrics: BacktestMetrics
    meta: dict[str, Any]


def run_backtest(cfg: BotConfig, md: MarketData, outputs_dir: str | Path) -> BacktestResult:
    start = _parse_date(cfg.backtest.start_date)
    end = _parse_date(cfg.backtest.end_date)
    if end <= start:
        raise ValueError("backtest.end_date must be > start_date")

    buffer = _buffer_days(cfg)
    fetch_start = start - timedelta(days=buffer)
    fetch_end = end + timedelta(days=2)

    symbols = md.get_liquidity_ranked_symbols()
    if not symbols:
        raise ValueError("Universe empty (no symbols after filters).")
    md.cache_universe_snapshot(symbols=symbols, meta={"top_n": cfg.universe.top_n_by_volume})

    logger.info("Backtest universe size: {}", len(symbols))
    logger.info("Fetching daily candles (cached) for {} -> {}", fetch_start.date(), fetch_end.date())

    candles: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            candles[s] = md.get_daily_candles(s, fetch_start, fetch_end, use_cache=True, cache_write=True)
        except Exception as e:
            logger.warning("Skipping {}: candle fetch failed: {}", s, e)

    # Funding daily rates (optional)
    funding_daily: dict[str, pd.Series] = {}
    force_mainnet_funding = bool(cfg.funding.filter.use_mainnet_data_even_on_testnet and cfg.exchange.testnet)
    if cfg.funding.model_in_backtest:
        logger.info("Fetching funding history (cached) for backtest window (may take a while on first run)...")
        for s in symbols:
            try:
                funding_daily[s] = md.get_daily_funding_rate(s, fetch_start, fetch_end, force_mainnet=force_mainnet_funding)
            except Exception as e:
                logger.warning("Funding unavailable for {}: {}", s, e)

    # Optional market proxy candles (BTC) for regime gating
    market_df = None
    proxy = cfg.filters.regime_filter.market_proxy_symbol
    if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime and proxy:
        try:
            market_df = md.get_daily_candles(proxy, fetch_start, fetch_end, use_cache=True, cache_write=True)
        except Exception as e:
            logger.warning("Market proxy {} unavailable; disabling market regime gate: {}", proxy, e)

    # Build calendar from market proxy if present, else intersection of all symbols
    if market_df is not None and not market_df.empty:
        calendar = market_df.index
    else:
        idx_sets = [set(df.index) for df in candles.values() if not df.empty]
        calendar = pd.DatetimeIndex(sorted(set.intersection(*idx_sets))) if idx_sets else pd.DatetimeIndex([])

    calendar = calendar[(calendar >= start) & (calendar <= end)].sort_values()
    if len(calendar) < 5:
        raise ValueError("Not enough daily bars in backtest range after alignment.")

    fee_bps = cfg.backtest.maker_fee_bps if (cfg.execution.order_type == "limit" and cfg.execution.post_only) else cfg.backtest.taker_fee_bps
    slip_bps = float(cfg.backtest.slippage_bps)
    tc_bps = float(fee_bps) + slip_bps

    equity = float(cfg.backtest.initial_equity)
    prev_weights: dict[str, float] = {}
    interval_days = max(1, int(cfg.rebalance.interval_days))

    equity_curve: list[tuple[datetime, float]] = []
    rets: list[tuple[datetime, float]] = []
    turnover_list: list[tuple[datetime, float]] = []

    for i in range(0, len(calendar) - 1):
        asof = calendar[i]
        nxt = calendar[i + 1]

        # Rebalance only every interval_days; hold weights otherwise.
        if i % interval_days == 0:
            day_candles = {s: df.loc[:asof] for s, df in candles.items() if not df.empty and asof in df.index}
            if len(day_candles) < 5:
                continue

            # Funding filter input for this day (optional)
            fr_today: dict[str, float] = {}
            if cfg.funding.filter.enabled and funding_daily:
                day_key = asof.floor("D")
                for s in day_candles.keys():
                    ser = funding_daily.get(s)
                    if ser is not None and not ser.empty and day_key in ser.index:
                        fr_today[s] = float(ser.loc[day_key])

            targets, _snap = compute_targets_from_daily_candles(
                candles=day_candles,
                config=cfg,
                equity_usd=equity,
                asof=asof.to_pydatetime(),
                market_proxy_candles=market_df.loc[:asof] if market_df is not None else None,
                current_weights=prev_weights,
                funding_daily_rate=fr_today if fr_today else None,
            )
            w = targets.weights
        else:
            w = dict(prev_weights)

        # Turnover in weight space
        syms = set(prev_weights) | set(w)
        turnover = float(sum(abs(w.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in syms))
        turnover_list.append((asof.to_pydatetime(), turnover))

        traded_notional = turnover * equity
        tc_cost = traded_notional * (tc_bps / 10_000.0)

        # Next day close-to-close returns for each held symbol
        port_ret = 0.0
        for s, ws in w.items():
            df = candles.get(s)
            if df is None or nxt not in df.index or asof not in df.index:
                continue
            px0 = float(df.loc[asof, "close"])
            px1 = float(df.loc[nxt, "close"])
            r = px1 / px0 - 1.0
            port_ret += float(ws) * float(r)

        # Funding PnL approximation (daily aggregate funding rate applied to start-of-period notional)
        funding_pnl = 0.0
        if cfg.funding.model_in_backtest and funding_daily:
            day_key = nxt.floor("D")
            for s, ws in w.items():
                ser = funding_daily.get(s)
                if ser is None or ser.empty or day_key not in ser.index:
                    continue
                fr = float(ser.loc[day_key])
                notional = float(ws) * float(equity)
                funding_pnl += -notional * fr

        equity = equity * (1.0 + port_ret) - tc_cost + funding_pnl
        equity_curve.append((nxt.to_pydatetime(), equity))
        rets.append((nxt.to_pydatetime(), port_ret + (funding_pnl / max(1e-12, equity)) - (tc_cost / max(1e-12, equity))))

        prev_weights = w

    eq = pd.Series({d: v for d, v in equity_curve}).sort_index()
    dr = pd.Series({d: v for d, v in rets}).sort_index()
    to = pd.Series({d: v for d, v in turnover_list}).sort_index()
    m = compute_metrics(eq, dr, to)

    meta: dict[str, Any] = {
        "symbols": symbols,
        "fee_bps": fee_bps,
        "slippage_bps": slip_bps,
        "turnover_cost_bps": tc_bps,
        "buffer_days": buffer,
        "interval_days": interval_days,
        "funding_modeled": bool(cfg.funding.model_in_backtest),
        "funding_force_mainnet": bool(force_mainnet_funding),
    }

    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eq.to_csv(out_dir / "equity.csv", header=["equity"])
    dr.to_csv(out_dir / "daily_returns.csv", header=["ret"])
    to.to_csv(out_dir / "daily_turnover.csv", header=["turnover"])

    # metrics JSON
    import json

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "profit_factor": m.profit_factor,
                "win_rate": m.win_rate,
                "avg_daily_turnover": m.avg_daily_turnover,
                "meta": meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("Backtest outputs written to {}", out_dir.resolve())
    logger.info("Metrics: CAGR={:.2%} Sharpe={:.2f} MaxDD={:.2%} PF={:.2f}", m.cagr, m.sharpe, m.max_drawdown, m.profit_factor)

    return BacktestResult(equity=eq, daily_returns=dr, daily_turnover=to, metrics=m, meta=meta)


