from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.config import BotConfig
from src.data.market_data import MarketData
from src.strategy.portfolio import PortfolioTargets
from src.strategy.xs_reversal import compute_targets_from_daily_candles


def _parse_date(d: str) -> datetime:
    dt = datetime.fromisoformat(d)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _buffer_days(cfg: BotConfig) -> int:
    rf = cfg.filters.regime_filter
    return int(
        max(
            cfg.sizing.vol_lookback_days + 10,
            cfg.signal.lookback_days + 5,
            rf.ema_slow + 20,
            int(cfg.universe.min_history_days) + 5,
            40,
        )
    )


def _calendar_from_any(candles: dict[str, pd.DataFrame], start: datetime, end: datetime) -> pd.DatetimeIndex:
    non_empty = [df for df in candles.values() if df is not None and not df.empty]
    if not non_empty:
        return pd.DatetimeIndex([])
    df_best = max(non_empty, key=lambda d: len(d.index))
    cal = pd.DatetimeIndex(df_best.index)
    return cal[(cal >= start) & (cal <= end)].sort_values()


def _sanitize_daily_candles(symbol: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    issues = {"duplicate_rows_removed": 0, "non_monotonic_fixed": 0}
    if df is None or df.empty:
        return pd.DataFrame(), issues

    out = df.copy()
    if not out.index.is_monotonic_increasing:
        out = out.sort_index()
        issues["non_monotonic_fixed"] = 1
    dup_count = int(out.index.duplicated(keep="last").sum())
    if dup_count > 0:
        out = out[~out.index.duplicated(keep="last")]
        issues["duplicate_rows_removed"] = dup_count
        logger.warning("Backtest data quality: removed {} duplicate daily candles for {}", dup_count, symbol)
    return out, issues


def _recent_daily_gap_count(window: pd.DataFrame) -> int:
    if window.empty:
        return 0
    days = pd.DatetimeIndex(window.index).floor("D")
    expected = pd.date_range(days[0], days[-1], freq="D", tz=UTC)
    return max(0, len(expected) - len(days))


def _liquidity_proxy_at_asof(hist: pd.DataFrame, asof: datetime) -> float:
    try:
        row = hist.loc[asof]
    except KeyError:
        return float("nan")
    turnover = row.get("turnover", np.nan)
    if pd.notna(turnover) and np.isfinite(float(turnover)) and float(turnover) > 0:
        return float(turnover)
    close = row.get("close", np.nan)
    volume = row.get("volume", np.nan)
    if pd.notna(close) and pd.notna(volume):
        liq = float(close) * float(volume)
        if np.isfinite(liq) and liq > 0:
            return liq
    return float("nan")


def _build_historical_day_candles(
    *,
    candles: dict[str, pd.DataFrame],
    asof: datetime,
    cfg: BotConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    min_hist = int(max(cfg.universe.min_history_days, cfg.signal.lookback_days + 2, cfg.sizing.vol_lookback_days + 2, 30))
    excluded_counts: dict[str, int] = defaultdict(int)
    eligible: list[tuple[str, float, pd.DataFrame]] = []

    for sym, df in candles.items():
        if df is None or df.empty:
            excluded_counts["empty_candles"] += 1
            continue
        if asof not in df.index:
            excluded_counts["missing_asof_bar"] += 1
            continue

        hist = df.loc[:asof]
        if len(hist) < min_hist:
            excluded_counts["insufficient_history"] += 1
            continue

        recent = hist.iloc[-min_hist:]
        if _recent_daily_gap_count(recent) > 0:
            excluded_counts["recent_gap"] += 1
            continue

        try:
            close_px = float(hist.loc[asof, "close"])
        except Exception:
            excluded_counts["invalid_close"] += 1
            continue
        if not np.isfinite(close_px) or close_px <= 0:
            excluded_counts["invalid_close"] += 1
            continue

        liq = _liquidity_proxy_at_asof(hist, asof)
        if not np.isfinite(liq) or liq <= 0:
            excluded_counts["missing_liquidity_proxy"] += 1
            continue

        eligible.append((sym, liq, hist))

    eligible.sort(key=lambda x: x[1], reverse=True)
    top_n = int(max(1, cfg.universe.top_n_by_volume))
    selected = eligible[:top_n]
    out = {sym: hist for sym, _liq, hist in selected}
    info = {
        "eligible_before_top_n": len(eligible),
        "selected": len(out),
        "excluded_counts": dict(excluded_counts),
    }
    return out, info


def _resolve_execution_cost_bps(cfg: BotConfig) -> tuple[float, str]:
    scenario = str(getattr(cfg.backtest, "execution_scenario", "mixed"))
    maker = float(cfg.backtest.maker_fee_bps)
    taker = float(cfg.backtest.taker_fee_bps)
    if scenario == "optimistic_maker":
        return maker, scenario
    if scenario == "conservative_taker":
        return taker, scenario
    return (maker + taker) / 2.0, "mixed"


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    daily_returns: pd.Series
    daily_turnover: pd.Series
    metrics: BacktestMetrics
    meta: dict[str, Any]


def run_backtest(cfg: BotConfig, md: MarketData, outputs_dir: str | Path) -> BacktestResult:
    return run_backtest_with_target_transform(cfg=cfg, md=md, outputs_dir=outputs_dir)


def run_backtest_with_target_transform(
    *,
    cfg: BotConfig,
    md: MarketData,
    outputs_dir: str | Path,
    target_transform: Any | None = None,
) -> BacktestResult:
    if bool(getattr(cfg.backtest, "allow_partial_fills", False)):
        raise ValueError(
            "backtest.allow_partial_fills is exposed in config but not modeled in the backtester; set it to false."
        )

    start = _parse_date(cfg.backtest.start_date)
    end = _parse_date(cfg.backtest.end_date)
    if end <= start:
        raise ValueError("backtest.end_date must be > start_date")

    buffer = _buffer_days(cfg)
    fetch_start = start - timedelta(days=buffer)
    fetch_end = end + timedelta(days=2)

    symbols_seed = md.get_liquidity_ranked_symbols()
    if not symbols_seed:
        raise ValueError("Universe empty (no symbols after filters).")
    md.cache_universe_snapshot(
        symbols=symbols_seed,
        meta={
            "top_n": cfg.universe.top_n_by_volume,
            "mode": "current_symbol_seed_only",
            "warning": "Historical universe remains limited to symbols available in the current downloaded seed.",
        },
    )

    logger.info("Backtest symbol seed size: {}", len(symbols_seed))
    logger.info("Fetching daily candles (cached) for {} -> {}", fetch_start.date(), fetch_end.date())

    candles: dict[str, pd.DataFrame] = {}
    data_quality_counts: dict[str, int] = defaultdict(int)
    for s in symbols_seed:
        try:
            raw = md.get_daily_candles(s, fetch_start, fetch_end, use_cache=True, cache_write=True)
            clean, issues = _sanitize_daily_candles(s, raw)
            candles[s] = clean
            for k, v in issues.items():
                data_quality_counts[k] += int(v)
        except Exception as e:
            logger.warning("Skipping {}: candle fetch failed: {}", s, e)
            data_quality_counts["candle_fetch_failures"] += 1

    funding_daily: dict[str, pd.Series] = {}
    force_mainnet_funding = bool(cfg.funding.filter.use_mainnet_data_even_on_testnet and cfg.exchange.testnet)
    if cfg.funding.model_in_backtest:
        logger.info("Fetching funding history (cached) for backtest window (may take a while on first run)...")
        for s in symbols_seed:
            try:
                funding_daily[s] = md.get_daily_funding_rate(s, fetch_start, fetch_end, force_mainnet=force_mainnet_funding)
            except Exception as e:
                logger.warning("Funding unavailable for {}: {}", s, e)
                data_quality_counts["funding_unavailable"] += 1

    market_df = None
    proxy = cfg.filters.regime_filter.market_proxy_symbol
    if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime and proxy:
        try:
            market_raw = md.get_daily_candles(proxy, fetch_start, fetch_end, use_cache=True, cache_write=True)
            market_df, issues = _sanitize_daily_candles(proxy, market_raw)
            for k, v in issues.items():
                data_quality_counts[f"market_{k}"] += int(v)
        except Exception as e:
            logger.warning("Market proxy {} unavailable; disabling market regime gate: {}", proxy, e)

    if market_df is not None and not market_df.empty:
        calendar = pd.DatetimeIndex(market_df.index)
        calendar = calendar[(calendar >= start) & (calendar <= end)].sort_values()
    else:
        calendar = _calendar_from_any(candles, start, end)
    if len(calendar) < 5:
        raise ValueError("Not enough daily bars in backtest range after alignment.")

    fee_bps, execution_scenario = _resolve_execution_cost_bps(cfg)
    slip_bps = float(cfg.backtest.slippage_bps)
    tc_bps = float(fee_bps) + slip_bps
    borrow_bps = float(getattr(cfg.backtest, "borrow_cost_bps", 0.0) or 0.0)

    warnings: list[str] = [
        "Historical universe is date-aware within downloaded candles, but still limited to the current symbol seed.",
        "Execution latency and intraday spread dynamics are not modeled; slippage_bps is the conservative proxy.",
    ]
    if borrow_bps <= 0.0:
        warnings.append("Short borrow cost is disabled (borrow_cost_bps <= 0).")

    equity = float(cfg.backtest.initial_equity)
    initial_equity = equity
    prev_weights: dict[str, float] = {}
    interval_days = max(1, int(cfg.rebalance.interval_days))

    equity_curve: list[tuple[datetime, float]] = []
    rets: list[tuple[datetime, float]] = []
    turnover_list: list[tuple[datetime, float]] = []
    gross_returns_list: list[tuple[datetime, float]] = []
    daily_cost_rows: list[dict[str, Any]] = []
    symbol_pnl: dict[str, float] = defaultdict(float)
    exclusion_counts: dict[str, int] = defaultdict(int)
    long_pnl_total = 0.0
    short_pnl_total = 0.0
    rebalance_count = 0
    trade_days = 0

    for i in range(0, len(calendar) - 1):
        asof = calendar[i]
        nxt = calendar[i + 1]
        w = dict(prev_weights)

        if i % interval_days == 0:
            day_candles, day_info = _build_historical_day_candles(candles=candles, asof=asof.to_pydatetime(), cfg=cfg)
            for k, v in day_info["excluded_counts"].items():
                exclusion_counts[k] += int(v)
            if len(day_candles) < 5:
                exclusion_counts["universe_too_small"] += 1
            else:
                fr_today: dict[str, float] = {}
                if cfg.funding.filter.enabled and funding_daily:
                    day_key = asof.floor("D")
                    for s in day_candles.keys():
                        ser = funding_daily.get(s)
                        if ser is not None and not ser.empty and day_key in ser.index:
                            fr_today[s] = float(ser.loc[day_key])

                try:
                    targets, _snap = compute_targets_from_daily_candles(
                        candles=day_candles,
                        config=cfg,
                        equity_usd=equity,
                        asof=asof.to_pydatetime(),
                        market_proxy_candles=market_df.loc[:asof] if market_df is not None else None,
                        current_weights=prev_weights,
                        funding_daily_rate=fr_today if fr_today else None,
                    )
                except ValueError as e:
                    if "Universe too small" in str(e):
                        exclusion_counts["strategy_universe_too_small"] += 1
                    else:
                        raise
                else:
                    if target_transform is not None:
                        transformed = target_transform(
                            weights=dict(targets.weights),
                            notionals=dict(targets.notionals_usd),
                            equity_usd=float(equity),
                            cfg=cfg,
                            asof=asof.to_pydatetime(),
                            meta=dict(targets.meta),
                        )
                        if not isinstance(transformed, PortfolioTargets):
                            raise TypeError("target_transform must return PortfolioTargets")
                        targets = transformed
                    w = dict(targets.weights)
                    rebalance_count += 1

        syms = set(prev_weights) | set(w)
        turnover = float(sum(abs(w.get(s, 0.0) - prev_weights.get(s, 0.0)) for s in syms))
        turnover_list.append((asof.to_pydatetime(), turnover))
        if turnover > 0:
            trade_days += 1

        equity_start = equity
        traded_notional = turnover * equity_start
        tc_cost = traded_notional * (tc_bps / 10_000.0)

        port_ret = 0.0
        gross_pnl = 0.0
        long_pnl_day = 0.0
        short_pnl_day = 0.0
        missing_return_symbols = 0
        for s, ws in w.items():
            df = candles.get(s)
            if df is None or nxt not in df.index or asof not in df.index:
                missing_return_symbols += 1
                continue
            px0 = float(df.loc[asof, "close"])
            px1 = float(df.loc[nxt, "close"])
            r = px1 / px0 - 1.0
            ret_contrib = float(ws) * float(r)
            pnl_contrib = equity_start * ret_contrib
            port_ret += ret_contrib
            gross_pnl += pnl_contrib
            symbol_pnl[s] += pnl_contrib
            if float(ws) >= 0:
                long_pnl_day += pnl_contrib
            else:
                short_pnl_day += pnl_contrib

        if missing_return_symbols:
            exclusion_counts["missing_next_bar_for_held_symbol"] += missing_return_symbols

        funding_pnl = 0.0
        if cfg.funding.model_in_backtest and funding_daily:
            day_key = nxt.floor("D")
            for s, ws in w.items():
                ser = funding_daily.get(s)
                if ser is None or ser.empty or day_key not in ser.index:
                    continue
                fr = float(ser.loc[day_key])
                notional = float(ws) * float(equity_start)
                funding_pnl += -notional * fr

        short_gross_notional = float(sum(abs(ws) for s, ws in w.items() if ws < 0.0)) * equity_start
        borrow_cost = short_gross_notional * (borrow_bps / 10_000.0)

        equity = equity_start + gross_pnl - tc_cost + funding_pnl - borrow_cost
        net_ret = (equity - equity_start) / max(1e-12, equity_start)
        gross_returns_list.append((nxt.to_pydatetime(), port_ret))
        equity_curve.append((nxt.to_pydatetime(), equity))
        rets.append((nxt.to_pydatetime(), net_ret))
        daily_cost_rows.append(
            {
                "date": nxt.to_pydatetime(),
                "turnover": turnover,
                "traded_notional_usd": traded_notional,
                "gross_pnl_usd": gross_pnl,
                "trading_cost_usd": tc_cost,
                "funding_pnl_usd": funding_pnl,
                "borrow_cost_usd": borrow_cost,
                "long_pnl_usd": long_pnl_day,
                "short_pnl_usd": short_pnl_day,
            }
        )
        long_pnl_total += long_pnl_day
        short_pnl_total += short_pnl_day
        prev_weights = w

    eq = pd.Series({d: v for d, v in equity_curve}).sort_index()
    dr = pd.Series({d: v for d, v in rets}).sort_index()
    to = pd.Series({d: v for d, v in turnover_list}).sort_index()
    gross_dr = pd.Series({d: v for d, v in gross_returns_list}).sort_index()
    m = compute_metrics(eq, dr, to)

    symbol_contrib_series = pd.Series(symbol_pnl).sort_values(ascending=False)
    cost_df = pd.DataFrame(daily_cost_rows)
    if not cost_df.empty:
        cost_df = cost_df.sort_values("date")

    cost_summary = {
        "gross_pnl_total_usd": float(cost_df["gross_pnl_usd"].sum()) if not cost_df.empty else 0.0,
        "trading_cost_total_usd": float(cost_df["trading_cost_usd"].sum()) if not cost_df.empty else 0.0,
        "funding_pnl_total_usd": float(cost_df["funding_pnl_usd"].sum()) if not cost_df.empty else 0.0,
        "borrow_cost_total_usd": float(cost_df["borrow_cost_usd"].sum()) if not cost_df.empty else 0.0,
    }

    meta: dict[str, Any] = {
        "symbols_seed": symbols_seed,
        "execution_scenario": execution_scenario,
        "fee_bps_assumed": fee_bps,
        "slippage_bps": slip_bps,
        "turnover_cost_bps": tc_bps,
        "borrow_cost_bps_daily": borrow_bps,
        "buffer_days": buffer,
        "interval_days": interval_days,
        "funding_modeled": bool(cfg.funding.model_in_backtest),
        "funding_force_mainnet": bool(force_mainnet_funding),
        "historical_universe_mode": "historical_turnover_ranked_with_current_symbol_seed",
        "warnings": warnings,
        "data_quality": {
            "sanitization_counts": dict(data_quality_counts),
            "excluded_counts": dict(exclusion_counts),
        },
        "gross_return_total": float(eq.iloc[-1] - initial_equity + cost_summary["trading_cost_total_usd"] - cost_summary["funding_pnl_total_usd"] + cost_summary["borrow_cost_total_usd"]) / initial_equity if len(eq) else 0.0,
        "net_return_total": float(eq.iloc[-1] / initial_equity - 1.0) if len(eq) else 0.0,
        "rebalance_count": int(rebalance_count),
        "trade_days": int(trade_days),
        "long_pnl_total_usd": float(long_pnl_total),
        "short_pnl_total_usd": float(short_pnl_total),
        "cost_summary": cost_summary,
        "top_symbol_contributions_usd": {str(k): float(v) for k, v in symbol_contrib_series.head(10).items()},
    }

    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eq.to_csv(out_dir / "equity.csv", header=["equity"])
    dr.to_csv(out_dir / "daily_returns.csv", header=["ret"])
    gross_dr.to_csv(out_dir / "gross_daily_returns.csv", header=["gross_ret"])
    to.to_csv(out_dir / "daily_turnover.csv", header=["turnover"])
    if not cost_df.empty:
        cost_df.to_csv(out_dir / "daily_costs.csv", index=False)
    if not symbol_contrib_series.empty:
        symbol_contrib_series.rename("pnl_usd").to_csv(out_dir / "symbol_contributions.csv", header=True)

    import json

    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "cagr": m.cagr,
                "sharpe": m.sharpe,
                "sortino": m.sortino,
                "max_drawdown": m.max_drawdown,
                "calmar": m.calmar,
                "profit_factor": m.profit_factor,
                "win_rate": m.win_rate,
                "avg_daily_turnover": m.avg_daily_turnover,
                "rebalance_count": int(rebalance_count),
                "trade_days": int(trade_days),
                "gross_return_total": meta["gross_return_total"],
                "net_return_total": meta["net_return_total"],
                "long_pnl_total_usd": float(long_pnl_total),
                "short_pnl_total_usd": float(short_pnl_total),
                "cost_summary": cost_summary,
                "top_symbol_contributions_usd": meta["top_symbol_contributions_usd"],
                "meta": meta,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    logger.info("Backtest outputs written to {}", out_dir.resolve())
    logger.info(
        "Metrics: CAGR={:.2%} Sharpe={:.2f} Sortino={:.2f} MaxDD={:.2%} Turnover={:.3f}",
        m.cagr,
        m.sharpe,
        m.sortino,
        m.max_drawdown,
        m.avg_daily_turnover,
    )

    return BacktestResult(equity=eq, daily_returns=dr, daily_turnover=to, metrics=m, meta=meta)
