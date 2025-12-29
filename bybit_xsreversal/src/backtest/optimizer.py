from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.metrics import compute_metrics
from src.config import BotConfig, load_config, load_yaml_config
from src.data.bybit_client import BybitClient
from src.data.market_data import MarketData
from src.strategy.xs_reversal import compute_targets_from_daily_candles


@dataclass(frozen=True)
class Candidate:
    lookback_days: int
    long_quantile: float
    short_quantile: float
    target_gross_leverage: float
    rebalance_time_utc: str


def _parse_date(d: str) -> datetime:
    dt = datetime.fromisoformat(d)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _default_opt_window_days() -> int:
    return 365


def _ensure_backtest_range(cfg: BotConfig) -> tuple[datetime, datetime]:
    # Optimizer should be robust on fresh installs; prefer a recent rolling window
    # instead of relying on whatever backtest start/end happen to be set to.
    # Allow override via env: BYBIT_OPT_WINDOW_DAYS (int).
    win_env = os.getenv("BYBIT_OPT_WINDOW_DAYS", "").strip()
    try:
        win_days = int(win_env) if win_env else _default_opt_window_days()
    except Exception:
        win_days = _default_opt_window_days()

    end = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    start = end - timedelta(days=win_days)
    return start, end


def _buffer_days(cfg: BotConfig) -> int:
    rf = cfg.filters.regime_filter
    return int(max(cfg.sizing.vol_lookback_days + 10, cfg.signal.lookback_days + 5, rf.ema_slow + 20, 40))


def _calendar_from_any(candles: dict[str, pd.DataFrame], start: datetime, end: datetime) -> pd.DatetimeIndex:
    """
    Build a robust daily calendar.
    Avoid using full intersection across all symbols (too brittle when some symbols have sparse history).
    Prefer the longest available history among the downloaded candle sets.
    """
    non_empty = [df for df in candles.values() if df is not None and not df.empty]
    if not non_empty:
        return pd.DatetimeIndex([])
    df_best = max(non_empty, key=lambda d: len(d.index))
    cal = pd.DatetimeIndex(df_best.index)
    cal = cal[(cal >= start) & (cal <= end)].sort_values()
    return cal


def _simulate(
    *,
    cfg: BotConfig,
    candles: dict[str, pd.DataFrame],
    market_df: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fee_bps = cfg.backtest.maker_fee_bps if (cfg.execution.order_type == "limit" and cfg.execution.post_only) else cfg.backtest.taker_fee_bps
    tc_bps = float(fee_bps) + float(cfg.backtest.slippage_bps)

    equity = float(cfg.backtest.initial_equity)
    prev_w: dict[str, float] = {}
    eq_curve: list[tuple[datetime, float]] = []
    dr: list[tuple[datetime, float]] = []
    to: list[tuple[datetime, float]] = []

    for i in range(0, len(calendar) - 1):
        asof = calendar[i]
        nxt = calendar[i + 1]

        day_candles = {s: df.loc[:asof] for s, df in candles.items() if asof in df.index}
        if len(day_candles) < 5:
            continue

        targets, _ = compute_targets_from_daily_candles(
            candles=day_candles,
            config=cfg,
            equity_usd=equity,
            asof=asof.to_pydatetime(),
            market_proxy_candles=market_df.loc[:asof] if market_df is not None else None,
        )
        w = targets.weights
        syms = set(prev_w) | set(w)
        turnover = float(sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in syms))
        traded_notional = turnover * equity
        tc_cost = traded_notional * (tc_bps / 10_000.0)

        port_ret = 0.0
        for s, ws in w.items():
            df = candles.get(s)
            if df is None or nxt not in df.index or asof not in df.index:
                continue
            px0 = float(df.loc[asof, "close"])
            px1 = float(df.loc[nxt, "close"])
            r = px1 / px0 - 1.0
            port_ret += float(ws) * float(r)

        equity = equity * (1.0 + port_ret) - tc_cost
        eq_curve.append((nxt.to_pydatetime(), equity))
        dr.append((nxt.to_pydatetime(), port_ret - (tc_cost / max(1e-12, equity))))
        to.append((asof.to_pydatetime(), turnover))
        prev_w = w

    eq = pd.Series({d: v for d, v in eq_curve}).sort_index()
    daily_ret = pd.Series({d: v for d, v in dr}).sort_index()
    daily_turn = pd.Series({d: v for d, v in to}).sort_index()
    return eq, daily_ret, daily_turn


def _candidates(*, fast: bool, default_time_utc: str) -> Iterable[Candidate]:
    lookbacks = [1, 2, 3, 5] if not fast else [1, 2, 3]
    qs = [0.1, 0.15, 0.2] if not fast else [0.1, 0.2]
    levs = [0.5, 1.0, 1.5] if not fast else [1.0, 1.5]
    times = [default_time_utc]  # keep fixed; scheduling is operational not alpha

    for lb, q, lev, t in itertools.product(lookbacks, qs, levs, times):
        yield Candidate(
            lookback_days=int(lb),
            long_quantile=float(q),
            short_quantile=float(q),
            target_gross_leverage=float(lev),
            rebalance_time_utc=str(t),
        )


def optimize_config(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    fast: bool,
) -> dict[str, Any]:
    """
    Optimize a small parameter grid and write best params back to config.yaml.
    Returns summary dict.
    """
    cfg = load_config(config_path)
    start, end = _ensure_backtest_range(cfg)
    buf = _buffer_days(cfg)
    fetch_start = start - timedelta(days=buf)
    fetch_end = end + timedelta(days=2)

    # Optimization should generally use MAINNET historical data even if you trade on testnet.
    # Some testnet instruments have sparse/absent history, which makes optimization unreliable.
    opt_testnet_env = os.getenv("BYBIT_OPT_TESTNET", "").strip().lower()
    if opt_testnet_env in ("1", "true", "yes", "y"):
        data_testnet = True
    elif opt_testnet_env in ("0", "false", "no", "n"):
        data_testnet = False
    else:
        data_testnet = False
    client = BybitClient(auth=None, testnet=data_testnet)
    if data_testnet != cfg.exchange.testnet:
        logger.warning("Optimizer using testnet={} for market data (trading testnet={})", data_testnet, cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    try:
        symbols = md.get_liquidity_ranked_symbols()
        candles: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                candles[s] = md.get_daily_candles(s, fetch_start, fetch_end, use_cache=True, cache_write=True)
            except Exception as e:
                logger.warning("Skipping {}: {}", s, e)

        market_df = None
        if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime:
            proxy = cfg.filters.regime_filter.market_proxy_symbol
            try:
                market_df = md.get_daily_candles(proxy, fetch_start, fetch_end, use_cache=True, cache_write=True)
            except Exception as e:
                logger.warning("Market proxy unavailable: {}", e)

        if market_df is not None and not market_df.empty:
            cal = market_df.index
            cal = cal[(cal >= start) & (cal <= end)].sort_values()
        else:
            cal = _calendar_from_any(candles, start, end)
        if len(cal) < 30:
            raise ValueError("Not enough aligned daily bars for optimization window.")

        best = None
        best_score = -float("inf")
        rows: list[dict[str, Any]] = []

        for cand in _candidates(fast=fast, default_time_utc=cfg.rebalance.time_utc):
            trial = cfg.model_copy(deep=True)
            trial.signal.lookback_days = int(cand.lookback_days)  # type: ignore[assignment]
            trial.signal.long_quantile = float(cand.long_quantile)
            trial.signal.short_quantile = float(cand.short_quantile)
            trial.sizing.target_gross_leverage = float(cand.target_gross_leverage)
            trial.rebalance.time_utc = str(cand.rebalance_time_utc)

            eq, dr, to = _simulate(cfg=trial, candles=candles, market_df=market_df, calendar=cal)
            m = compute_metrics(eq, dr, to)

            # Hard reject pathological candidates (equity blowups, nonsensical DD)
            if eq.empty or float(eq.min()) <= 0.0:
                continue
            if float(m.max_drawdown) < -0.95:
                continue

            # Objective: Sharpe, with penalty for big drawdown and high turnover
            dd_pen = max(0.0, abs(m.max_drawdown) - (trial.risk.max_drawdown_pct / 100.0))
            score = float(m.sharpe) - 2.0 * float(dd_pen) - 0.1 * float(m.avg_daily_turnover)

            row = {
                "candidate": cand.__dict__,
                "score": score,
                "sharpe": m.sharpe,
                "cagr": m.cagr,
                "max_drawdown": m.max_drawdown,
                "avg_daily_turnover": m.avg_daily_turnover,
            }
            rows.append(row)

            if score > best_score and np.isfinite(score):
                best_score = score
                best = (cand, row)

        if best is None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "optimization_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            logger.error(
                "Optimization found no feasible candidates. Leaving config unchanged. Results: {}",
                (out / "optimization_results.json").resolve(),
            )
            return {
                "status": "no_feasible_candidate",
                "output_dir": str(out.resolve()),
                "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                "universe_size": len(symbols),
                "candidates": len(rows),
            }

        best_cand, best_row = best

        # Patch config.yaml (deep merge)
        raw = load_yaml_config(config_path)
        raw.setdefault("signal", {})
        raw.setdefault("rebalance", {})
        raw.setdefault("sizing", {})

        raw["signal"]["lookback_days"] = int(best_cand.lookback_days)
        raw["signal"]["long_quantile"] = float(best_cand.long_quantile)
        raw["signal"]["short_quantile"] = float(best_cand.short_quantile)
        raw["rebalance"]["time_utc"] = str(best_cand.rebalance_time_utc)
        raw["sizing"]["target_gross_leverage"] = float(best_cand.target_gross_leverage)

        import yaml

        Path(config_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "optimization_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        (out / "best.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")

        logger.info("Optimization complete. Best score={:.3f} sharpe={:.3f} maxDD={:.2%}", best_score, best_row["sharpe"], best_row["max_drawdown"])
        return {
            "best": best_row,
            "output_dir": str(out.resolve()),
            "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
            "universe_size": len(symbols),
            "candidates": len(rows),
        }
    finally:
        client.close()


