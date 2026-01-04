from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal
import time

import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

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
    vol_lookback_days: int
    interval_days: int
    rebalance_fraction: float
    min_weight_change_bps: float
    regime_action: str
    funding_filter_enabled: bool
    funding_max_abs_daily_rate: float


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
    # Must include enough pre-history to satisfy cfg.universe.min_history_days,
    # otherwise the first rebalance day will produce an empty universe and stage2 rejects all candidates.
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


def _level_to_budget(level: str) -> int:
    level = (level or "").strip().lower()
    if level == "quick":
        return 60
    if level == "deep":
        return 800
    return 250  # standard


def _level_to_stage2_topk(level: str) -> int:
    level = (level or "").strip().lower()
    if level == "quick":
        return 25
    if level == "deep":
        return 150
    return 75


def _simulate_candidate_full(
    *,
    cfg: BotConfig,
    candles: dict[str, pd.DataFrame],
    market_df: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
    funding_daily: dict[str, pd.Series] | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Full (slow) simulation consistent with the main backtester logic:
    - uses compute_targets_from_daily_candles (shared with live)
    - applies fees+slippage on turnover
    """
    fee_bps = cfg.backtest.maker_fee_bps if (cfg.execution.order_type == "limit" and cfg.execution.post_only) else cfg.backtest.taker_fee_bps
    tc_bps = float(fee_bps) + float(cfg.backtest.slippage_bps)

    equity = float(cfg.backtest.initial_equity)
    prev_w: dict[str, float] = {}
    eq_curve: list[tuple[datetime, float]] = []
    dr: list[tuple[datetime, float]] = []
    to: list[tuple[datetime, float]] = []
    interval_days = max(1, int(cfg.rebalance.interval_days))

    for i in range(0, len(calendar) - 1):
        asof = calendar[i]
        nxt = calendar[i + 1]

        if i % interval_days == 0:
            day_candles = {s: df.loc[:asof] for s, df in candles.items() if asof in df.index}
            if len(day_candles) < 5:
                continue

            fr_today: dict[str, float] = {}
            if cfg.funding.filter.enabled and funding_daily:
                day_key = asof.floor("D")
                for s in day_candles.keys():
                    ser = funding_daily.get(s)
                    if ser is not None and not ser.empty and day_key in ser.index:
                        fr_today[s] = float(ser.loc[day_key])

            try:
                targets, _ = compute_targets_from_daily_candles(
                    candles=day_candles,
                    config=cfg,
                    equity_usd=equity,
                    asof=asof.to_pydatetime(),
                    market_proxy_candles=market_df.loc[:asof] if market_df is not None else None,
                    current_weights=prev_w,
                    funding_daily_rate=fr_today if fr_today else None,
                )
                w = targets.weights
            except ValueError as e:
                # Robustness: if this particular day has too few eligible symbols (e.g. sparse history),
                # just hold current weights and continue rather than rejecting the entire candidate.
                if "Universe too small" in str(e):
                    w = dict(prev_w)
                else:
                    raise
        else:
            w = dict(prev_w)
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
        eq_curve.append((nxt.to_pydatetime(), equity))
        dr.append((nxt.to_pydatetime(), port_ret + (funding_pnl / max(1e-12, equity)) - (tc_cost / max(1e-12, equity))))
        to.append((asof.to_pydatetime(), turnover))
        prev_w = w

    eq = pd.Series({d: v for d, v in eq_curve}).sort_index()
    daily_ret = pd.Series({d: v for d, v in dr}).sort_index()
    daily_turn = pd.Series({d: v for d, v in to}).sort_index()
    return eq, daily_ret, daily_turn


def _metrics_ok(m: Any) -> bool:
    """
    Guard for stage2: treat NaN/inf metrics as invalid so stage2 can skip the candidate.
    """
    try:
        import numpy as _np

        return bool(
            _np.isfinite(float(m.sharpe))
            and _np.isfinite(float(m.cagr))
            and _np.isfinite(float(m.max_drawdown))
            and _np.isfinite(float(m.avg_daily_turnover))
        )
    except Exception:
        return False


def _random_candidates(
    *,
    n: int,
    default_time_utc: str,
    seed: int,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    lookbacks = np.array([1, 2, 3, 5], dtype=int)
    vol_lbs = np.array([14, 21, 30], dtype=int)
    # Typical quantiles for decile-ish reversal; allow some exploration
    q_low, q_high = 0.05, 0.25
    # Leverage exploration (gross)
    lev_low, lev_high = 0.5, 1.75

    interval_choices = np.array([1, 2, 3], dtype=int)
    frac_low, frac_high = 0.25, 1.0
    thresh_choices = np.array([0.0, 2.0, 5.0, 10.0, 20.0], dtype=float)  # bps
    regime_actions = ["scale_down", "switch_to_momentum"]
    funding_enabled_choices = [False, True]
    funding_max_choices = np.array([0.001, 0.002, 0.003, 0.005], dtype=float)

    out: list[Candidate] = []
    for _ in range(n):
        lb = int(rng.choice(lookbacks))
        vlb = int(rng.choice(vol_lbs))
        q = float(rng.uniform(q_low, q_high))
        # discretize to 0.01 to keep config readable
        q = round(q, 2)
        lev = float(rng.uniform(lev_low, lev_high))
        lev = round(lev, 2)
        interval = int(rng.choice(interval_choices))
        frac = float(rng.uniform(frac_low, frac_high))
        frac = round(frac, 2)
        thresh = float(rng.choice(thresh_choices))
        regime_action = str(rng.choice(regime_actions))
        fund_en = bool(rng.choice(funding_enabled_choices))
        fund_max = float(rng.choice(funding_max_choices))
        out.append(
            Candidate(
                lookback_days=lb,
                long_quantile=q,
                short_quantile=q,
                target_gross_leverage=lev,
                rebalance_time_utc=str(default_time_utc),
                vol_lookback_days=vlb,
                interval_days=interval,
                rebalance_fraction=frac,
                min_weight_change_bps=thresh,
                regime_action=regime_action,
                funding_filter_enabled=fund_en,
                funding_max_abs_daily_rate=fund_max,
            )
        )
    # de-dup
    uniq = {
        (
            c.lookback_days,
            c.long_quantile,
            c.target_gross_leverage,
            c.vol_lookback_days,
            c.interval_days,
            c.rebalance_fraction,
            c.min_weight_change_bps,
            c.regime_action,
            c.funding_filter_enabled,
            c.funding_max_abs_daily_rate,
        ): c
        for c in out
    }
    return list(uniq.values())


def _grid_candidates(*, default_time_utc: str) -> list[Candidate]:
    lookbacks = [1, 2, 3, 5]
    qs = [0.05, 0.10, 0.15, 0.20]
    levs = [0.5, 0.75, 1.0, 1.25, 1.5]
    vol_lbs = [14, 30]
    intervals = [1, 2]
    fracs = [0.5, 1.0]
    thresh_bps = [0.0, 5.0, 10.0]
    regime_actions = ["scale_down", "switch_to_momentum"]
    funding_enabled = [False, True]
    funding_max = [0.002, 0.003]
    out: list[Candidate] = []
    for lb, q, lev, vlb, interval, frac, thr, ra, fe, fm in itertools.product(
        lookbacks, qs, levs, vol_lbs, intervals, fracs, thresh_bps, regime_actions, funding_enabled, funding_max
    ):
        out.append(
            Candidate(
                lookback_days=int(lb),
                long_quantile=float(q),
                short_quantile=float(q),
                target_gross_leverage=float(lev),
                rebalance_time_utc=str(default_time_utc),
                vol_lookback_days=int(vlb),
                interval_days=int(interval),
                rebalance_fraction=float(frac),
                min_weight_change_bps=float(thr),
                regime_action=str(ra),
                funding_filter_enabled=bool(fe),
                funding_max_abs_daily_rate=float(fm),
            )
        )
    return out


def _prepare_close_matrix(
    candles: dict[str, pd.DataFrame],
    *,
    start: datetime,
    end: datetime,
    min_history_days: int,
    max_symbols: int = 60,
) -> pd.DataFrame:
    """
    Build a close price matrix (date x symbol) using symbols with enough history,
    preferring the longest-history symbols to improve optimization stability.
    """
    rows: list[tuple[str, pd.Series]] = []
    for sym, df in candles.items():
        if df is None or df.empty:
            continue
        sub = df.loc[(df.index >= start) & (df.index <= end), "close"].dropna().astype(float)
        if len(sub) < min_history_days:
            continue
        rows.append((sym, sub))
    # Prefer longest history
    rows.sort(key=lambda x: len(x[1]), reverse=True)
    rows = rows[:max_symbols]
    if not rows:
        return pd.DataFrame()
    # Align on union of dates, then later we will drop days with too many NaNs.
    close = pd.concat({sym: s for sym, s in rows}, axis=1).sort_index()
    return close.loc[(close.index >= start) & (close.index <= end)]


def _simulate_candidate_vectorized(
    *,
    close: pd.DataFrame,
    lookback_days: int,
    vol_lookback_days: int,
    q: float,
    gross_leverage: float,
    interval_days: int,
    rebalance_fraction: float,
    min_weight_change_bps: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
    use_maker: bool,
    initial_equity: float,
    max_dd_limit: float,
    max_turnover: float,
    min_symbols: int = 10,
) -> dict[str, Any] | None:
    """
    Fast daily rebalance simulation for candidate selection:
    - cross-sectional rank on lookback return
    - inverse-vol weights
    - dollar-neutral LS (long losers, short winners)
    - turnover/fees modeled in weight space
    Returns metrics dict (with sharpe, cagr, max_drawdown, avg_daily_turnover) on success,
    or dict with reject_reason (and optional diagnostic fields) on failure, or None for legacy compatibility.
    """
    if close.empty:
        return {"reject_reason": "empty_close_data"}
    px = close.copy()
    # Require at least min_symbols available per day
    valid_counts = px.notna().sum(axis=1)
    px = px.loc[valid_counts >= min_symbols]
    if len(px) < (vol_lookback_days + lookback_days + 20):
        return {"reject_reason": "insufficient_history"}

    # daily returns and lookback returns
    # pandas FutureWarning: default fill_method='pad' is deprecated.
    # We do NOT want implicit forward-filling across missing symbol histories.
    r1 = px.pct_change(fill_method=None)
    r_lb = px / px.shift(lookback_days) - 1.0
    vol = r1.rolling(vol_lookback_days, min_periods=vol_lookback_days).std(ddof=0)

    interval_days = max(1, int(interval_days))
    rebalance_fraction = max(0.0, min(1.0, float(rebalance_fraction)))
    min_w_delta = float(min_weight_change_bps) / 10_000.0

    # Weights matrix (T x N)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    q = float(q)
    prev_w = pd.Series(0.0, index=px.columns)
    for t in range(len(px.index)):
        dt = px.index[t]
        if t % interval_days != 0:
            weights.loc[dt] = prev_w.values
            continue
        ret_row = r_lb.loc[dt]
        vol_row = vol.loc[dt]
        # eligible symbols
        mask = ret_row.notna() & vol_row.notna() & (vol_row > 0)
        if mask.sum() < min_symbols:
            continue
        rets = ret_row[mask].astype(float)
        vols = vol_row[mask].astype(float)

        # thresholds
        lo_thr = np.nanquantile(rets.values, q)
        hi_thr = np.nanquantile(rets.values, 1.0 - q)
        longs = rets[rets <= lo_thr].index
        shorts = rets[rets >= hi_thr].index
        if len(longs) == 0 or len(shorts) == 0:
            continue

        invv_l = (1.0 / vols.loc[longs]).replace([np.inf, -np.inf], np.nan).dropna()
        invv_s = (1.0 / vols.loc[shorts]).replace([np.inf, -np.inf], np.nan).dropna()
        if invv_l.empty or invv_s.empty:
            continue

        w_l = invv_l / invv_l.abs().sum()
        w_s = invv_s / invv_s.abs().sum()
        w_l = w_l * (gross_leverage / 2.0)
        w_s = w_s * (gross_leverage / 2.0)
        w = pd.Series(0.0, index=px.columns)
        w.loc[w_l.index] = w_l.values
        w.loc[w_s.index] = -w_s.values
        # Turnover control approximation: partial rebalance + threshold
        w_new = prev_w + rebalance_fraction * (w - prev_w)
        if min_w_delta > 0:
            keep = (w_new - prev_w).abs() < min_w_delta
            w_new[keep] = prev_w[keep]
        # Scale down if gross exceeds cap
        gross = float(w_new.abs().sum())
        if gross > gross_leverage and gross > 0:
            w_new = w_new * (gross_leverage / gross)
        weights.loc[dt] = w_new.values
        prev_w = w_new

    weights = weights.dropna(how="all")
    if weights.empty or len(weights) < 30:
        return {"reject_reason": "insufficient_trading_days"}

    # Turnover / costs
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    avg_turnover = float(dw.mean())
    if avg_turnover > float(max_turnover):
        return {"reject_reason": "max_turnover_exceeded", "avg_turnover": avg_turnover, "max_turnover": float(max_turnover)}

    fee_bps = float(maker_fee_bps) if use_maker else float(taker_fee_bps)
    tc_bps = fee_bps + float(slippage_bps)

    # Portfolio daily return: sum(w_{t} * r1_{t+1})
    aligned = r1.loc[weights.index].shift(-1)
    port_ret = (weights * aligned).sum(axis=1).dropna()
    if port_ret.empty:
        return {"reject_reason": "no_portfolio_returns"}

    # apply transaction costs on traded notional in equity terms
    tc = dw.loc[port_ret.index] * (tc_bps / 10_000.0)
    net_ret = port_ret - tc

    equity = initial_equity * (1.0 + net_ret).cumprod()
    if float(equity.min()) <= 0.0:
        return {"reject_reason": "equity_below_zero"}

    # drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    if abs(max_dd) > float(max_dd_limit):
        return {"reject_reason": "max_drawdown_exceeded", "max_dd": max_dd, "max_dd_limit": float(max_dd_limit)}

    # sharpe
    if net_ret.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(365) * net_ret.mean() / net_ret.std(ddof=0))

    # CAGR on window length
    years = max(1e-9, (len(equity) - 1) / 365.0)
    cagr = float((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0)
    
    # Calmar ratio: CAGR / abs(max_drawdown)
    calmar = 0.0
    if max_dd != 0.0 and np.isfinite(max_dd):
        dd_abs = abs(max_dd)
        if dd_abs > 0:
            calmar = float(cagr / dd_abs) if np.isfinite(cagr) else 0.0
    
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "calmar": calmar, "avg_daily_turnover": avg_turnover}


def optimize_config(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    level: str = "standard",
    candidates: int | None = None,
    method: Literal["random", "grid"] = "random",
    seed: int = 42,
    stage2_topk: int | None = None,
    show_progress: bool = True,
    write_config: bool = True,
) -> dict[str, Any]:
    """
    Optimize a small parameter grid and (optionally) write best params back to config.yaml.
    Returns summary dict.
    
    Supports walk-forward analysis via environment variables:
    - BYBIT_OPT_WALK_FORWARD=1: Enable walk-forward mode
    - BYBIT_OPT_WF_NUM_WINDOWS=5: Number of rolling windows (default: 5)
    - BYBIT_OPT_WF_WINDOW_STEP_DAYS=30: Days to step between windows (default: 30)
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

        # -----------------------
        # Walk-Forward Analysis (if enabled)
        # -----------------------
        wf_enabled_env = os.getenv("BYBIT_OPT_WALK_FORWARD", "").strip().lower()
        wf_enabled = wf_enabled_env in ("1", "true", "yes", "y")
        
        if wf_enabled:
            # Walk-forward: create multiple rolling windows
            wf_num_windows_env = os.getenv("BYBIT_OPT_WF_NUM_WINDOWS", "").strip()
            wf_step_days_env = os.getenv("BYBIT_OPT_WF_WINDOW_STEP_DAYS", "").strip()
            try:
                wf_num_windows = int(wf_num_windows_env) if wf_num_windows_env else 5
            except Exception:
                wf_num_windows = 5
            try:
                wf_step_days = int(wf_step_days_env) if wf_step_days_env else 30
            except Exception:
                wf_step_days = 30
            
            train_frac_env = os.getenv("BYBIT_OPT_TRAIN_FRAC", "").strip()
            try:
                train_frac = float(train_frac_env) if train_frac_env else 0.7
            except Exception:
                train_frac = 0.7
            train_frac = float(min(0.9, max(0.5, train_frac)))
            
            min_test_days_env = os.getenv("BYBIT_OPT_MIN_TEST_DAYS", "").strip()
            try:
                min_test_days = int(min_test_days_env) if min_test_days_env else 60
            except Exception:
                min_test_days = 60
            min_test_days = max(30, min_test_days)
            
            # Create rolling windows
            windows: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
            train_len = int(np.floor(len(cal) * train_frac))
            train_len = max(60, train_len)  # Minimum train window
            
            for w in range(wf_num_windows):
                window_start_idx = w * wf_step_days
                if window_start_idx + train_len >= len(cal):
                    break  # Not enough data for this window
                
                train_end_idx = window_start_idx + train_len
                test_end_idx = min(train_end_idx + min_test_days, len(cal))
                
                if test_end_idx - train_end_idx < min_test_days:
                    break  # Not enough test data
                
                cal_train_w = cal[window_start_idx:train_end_idx]
                cal_test_w = cal[train_end_idx:test_end_idx]
                windows.append((cal_train_w, cal_test_w))
            
            if not windows:
                logger.warning("Walk-forward enabled but no valid windows could be created. Falling back to single window.")
                wf_enabled = False
            else:
                logger.info(
                    "Walk-forward analysis: {} windows, step={} days, train_len={} days, test_len={} days",
                    len(windows),
                    wf_step_days,
                    train_len,
                    len(windows[0][1]) if windows else min_test_days,
                )
        
        # -----------------------
        # Train/Test split (OOS) - Single window mode
        # -----------------------
        if not wf_enabled:
            train_frac_env = os.getenv("BYBIT_OPT_TRAIN_FRAC", "").strip()
            try:
                train_frac = float(train_frac_env) if train_frac_env else 0.7
            except Exception:
                train_frac = 0.7
            train_frac = float(min(0.9, max(0.5, train_frac)))

            min_test_days_env = os.getenv("BYBIT_OPT_MIN_TEST_DAYS", "").strip()
            try:
                min_test_days = int(min_test_days_env) if min_test_days_env else 60
            except Exception:
                min_test_days = 60
            min_test_days = max(30, min_test_days)

            split_idx = int(np.floor(len(cal) * train_frac))
            split_idx = max(2, min(split_idx, len(cal) - 2))
            cal_train = cal[:split_idx]
            cal_test = cal[split_idx:]
            if len(cal_test) < min_test_days:
                # If the window is too small to support a useful OOS slice, fall back to a single window.
                logger.warning(
                    "Optimization window too small for test split (cal={} train={} test={} < min_test_days={}); disabling OOS split for this run.",
                    len(cal),
                    len(cal_train),
                    len(cal_test),
                    min_test_days,
                )
                cal_train = cal
                cal_test = pd.DatetimeIndex([])
            else:
                logger.info(
                    "Optimizer train/test split: train_days={} ({} -> {}), test_days={} ({} -> {})",
                    len(cal_train),
                    cal_train[0].date().isoformat(),
                    cal_train[-1].date().isoformat(),
                    len(cal_test),
                    cal_test[0].date().isoformat(),
                    cal_test[-1].date().isoformat(),
                )
            windows = [(cal_train, cal_test)]  # Single window for compatibility

        # Walk-forward: process each window and aggregate results
        all_window_results: list[dict[str, Any]] = []
        
        for window_idx, (cal_train_w, cal_test_w) in enumerate(windows):
            window_prefix = f"[WF Window {window_idx + 1}/{len(windows)}] " if wf_enabled and len(windows) > 1 else ""
            if wf_enabled and len(windows) > 1:
                logger.info(
                    "{}Processing window: train={} ({} -> {}), test={} ({} -> {})",
                    window_prefix,
                    len(cal_train_w),
                    cal_train_w[0].date().isoformat(),
                    cal_train_w[-1].date().isoformat(),
                    len(cal_test_w),
                    cal_test_w[0].date().isoformat() if len(cal_test_w) > 0 else "N/A",
                    cal_test_w[-1].date().isoformat() if len(cal_test_w) > 0 else "N/A",
                )
            
            # Use this window's train/test split
            cal_train = cal_train_w
            cal_test = cal_test_w
            
            # Prepare data matrix once for fast candidate evaluation.
            # IMPORTANT: This history requirement must be aligned with stage2's shared strategy logic,
            # which enforces cfg.universe.min_history_days. If stage1 uses a looser min history,
            # it may select candidates that cannot trade at all in stage2 (empty universe -> no results).
            min_hist = int(max(80, int(cfg.universe.min_history_days), 2 * max(cfg.sizing.vol_lookback_days, 30)))
            window_start = cal_train[0].to_pydatetime()
            window_end = cal_test[-1].to_pydatetime() if len(cal_test) > 0 else cal_train[-1].to_pydatetime()
            close = _prepare_close_matrix(candles, start=window_start, end=window_end, min_history_days=min_hist, max_symbols=60)
            opt_symbols = list(close.columns) if close is not None and not close.empty else list(candles.keys())
            # Keep stage2 universe consistent with stage1 (reduces sparse-history issues in stage2)
            candles_stage2 = {s: candles[s] for s in opt_symbols if s in candles and candles[s] is not None and not candles[s].empty}

            # For stage2, use calendar based on the optimization symbol subset (more stable).
            cal_stage2 = _calendar_from_any(candles_stage2, window_start, window_end)
            if len(cal_stage2) >= 30:
                cal_window = cal_stage2
            else:
                cal_window = cal_train.append(cal_test) if len(cal_test) > 0 else cal_train

            # Stage1 should optimize on TRAIN only (to avoid leaking test data into selection)
            if close is not None and not close.empty:
                close_train = close.loc[(close.index >= cal_train[0]) & (close.index <= cal_train[-1])]
            else:
                close_train = close

            # Stage1 feasibility gates (used only for the fast screening model).
        # These can be overridden via env vars to avoid rejecting every candidate due to overly strict constraints.
        # - BYBIT_OPT_STAGE1_MAX_DD_PCT: e.g. "50" means allow up to 50% drawdown in stage1 screen
        # - BYBIT_OPT_STAGE1_MAX_TURNOVER: e.g. "10" means allow avg daily turnover up to 10 (weight-space)
        dd_env = os.getenv("BYBIT_OPT_STAGE1_MAX_DD_PCT", "").strip()
        to_env = os.getenv("BYBIT_OPT_STAGE1_MAX_TURNOVER", "").strip()
        try:
            stage1_max_dd = float(dd_env) / 100.0 if dd_env else float(cfg.risk.max_drawdown_pct) / 100.0
        except Exception:
            stage1_max_dd = float(cfg.risk.max_drawdown_pct) / 100.0
        try:
            stage1_max_turnover = float(to_env) if to_env else float(cfg.risk.max_turnover)
        except Exception:
            stage1_max_turnover = float(cfg.risk.max_turnover)
            logger.info(
                "{}Stage1 feasibility gates: max_dd_limit={:.2%} max_turnover={:.3f} (env overrides: BYBIT_OPT_STAGE1_MAX_DD_PCT={}, BYBIT_OPT_STAGE1_MAX_TURNOVER={})",
                window_prefix,
                float(stage1_max_dd),
                float(stage1_max_turnover),
                dd_env or "n/a",
                to_env or "n/a",
            )

            # Funding daily rates cache for stage2 (and for optional funding filter)
            funding_daily: dict[str, pd.Series] = {}
            force_mainnet_funding = bool(cfg.funding.filter.use_mainnet_data_even_on_testnet and cfg.exchange.testnet)
            if cfg.funding.model_in_backtest or cfg.funding.filter.enabled:
                for s in opt_symbols:
                    try:
                        funding_daily[s] = md.get_daily_funding_rate(s, fetch_start, fetch_end, force_mainnet=force_mainnet_funding)
                    except Exception:
                        continue

            best = None
            best_key: tuple[float, float, float, float] | None = None
            rows: list[dict[str, Any]] = []

            budget = int(candidates) if candidates is not None else _level_to_budget(level)
            if method == "grid":
                cand_list = _grid_candidates(default_time_utc=cfg.rebalance.time_utc)
            else:
                # Use window-specific seed to ensure different candidates per window
                window_seed = seed + window_idx * 1000
                cand_list = _random_candidates(n=budget, default_time_utc=cfg.rebalance.time_utc, seed=window_seed)
            # If grid, respect budget if provided
            if candidates is not None:
                cand_list = cand_list[: int(candidates)]
            # If random produced fewer uniques than budget, that's OK.
            candidates_list = cand_list

            # Progress bar + ETA
            # Rich Progress bars may not render in non-interactive terminals or when output is redirected.
            # We'll create the progress context, but if it doesn't render, the heartbeat logs will still show progress.
            progress_ctx = (
                Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]optimize[/bold]"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("best_sharpe={task.fields[best_sharpe]}"),
                    TextColumn("best_dd={task.fields[best_dd]}"),
                    TextColumn("best_cagr={task.fields[best_cagr]}"),
                    transient=False,
                )
                if show_progress
                else None
            )

            def fmt_or_na(x: float | None, fmt: str) -> str:
                if x is None or not np.isfinite(x):
                    return "n/a"
                return format(float(x), fmt)

            if progress_ctx is None:
                task_id = None
            else:
                progress_ctx.start()
                task_id = progress_ctx.add_task(
                    "optimize",
                    total=len(candidates_list),
                    best_sharpe="n/a",
                    best_dd="n/a",
                    best_cagr="n/a",
                )

            # When progress bars are disabled (e.g. systemd / --no-progress), the optimizer can appear "stuck"
            # for hours. Emit lightweight periodic logs so the operator can confirm it's making progress.
            hb_every_env = os.getenv("BYBIT_OPT_STAGE1_LOG_EVERY", "").strip()
            hb_secs_env = os.getenv("BYBIT_OPT_STAGE1_LOG_SECS", "").strip()
            try:
                hb_every = int(hb_every_env) if hb_every_env else 500
            except Exception:
                hb_every = 500
            try:
                hb_secs = float(hb_secs_env) if hb_secs_env else 300.0
            except Exception:
                hb_secs = 300.0
            hb_every = max(1, hb_every)
            hb_secs = max(5.0, hb_secs)
            t0 = time.time()
            last_hb = t0

            try:
                for i, cand in enumerate(candidates_list, start=1):
                    # Evaluate candidate quickly (vectorized)
                    use_maker = bool(cfg.execution.order_type == "limit" and cfg.execution.post_only)
                    m = _simulate_candidate_vectorized(
                        close=close_train,
                        lookback_days=int(cand.lookback_days),
                        vol_lookback_days=int(cand.vol_lookback_days),
                        q=float(cand.long_quantile),
                        gross_leverage=float(cand.target_gross_leverage),
                        interval_days=int(cand.interval_days),
                        rebalance_fraction=float(cand.rebalance_fraction),
                        min_weight_change_bps=float(cand.min_weight_change_bps),
                        maker_fee_bps=float(cfg.backtest.maker_fee_bps),
                        taker_fee_bps=float(cfg.backtest.taker_fee_bps),
                        slippage_bps=float(cfg.backtest.slippage_bps),
                        use_maker=use_maker,
                        initial_equity=float(cfg.backtest.initial_equity),
                        max_dd_limit=float(stage1_max_dd),
                        max_turnover=float(stage1_max_turnover),
                    )

                    # Record candidate metrics even if rejected (for debugging)
                    if m is None or "reject_reason" in m:
                        reject_reason = m.get("reject_reason", "unknown") if m is not None else "unknown"
                        row = {
                            "candidate": cand.__dict__,
                            "sharpe": float("nan"),
                            "cagr": float("nan"),
                            "max_drawdown": float(m.get("max_dd", float("nan"))) if m is not None and "max_dd" in m else float("nan"),
                            "avg_daily_turnover": float(m.get("avg_turnover", float("nan"))) if m is not None and "avg_turnover" in m else float("nan"),
                            "rejected": True,
                            "reject_reason": reject_reason,
                        }
                        # Store limit values for diagnostic logging
                        if m is not None:
                            if "max_dd_limit" in m:
                                row["max_dd_limit"] = float(m["max_dd_limit"])
                            if "max_turnover" in m:
                                row["max_turnover"] = float(m["max_turnover"])
                        rows.append(row)
                        if progress_ctx is not None and task_id is not None:
                            progress_ctx.advance(task_id, 1)
                        continue

                    row = {
                        "candidate": cand.__dict__,
                        "sharpe": float(m["sharpe"]),
                        "cagr": float(m["cagr"]),
                        "max_drawdown": float(m["max_drawdown"]),
                        "calmar": float(m.get("calmar", 0.0)),
                        "avg_daily_turnover": float(m["avg_daily_turnover"]),
                        "rejected": False,
                        "reject_reason": None,
                    }

                    # Composite objective function (configurable via env vars)
                    # Default weights: sharpe=0.4, calmar=0.3, cagr=0.2, sortino=0.0, turnover_penalty=0.1
                    # Set BYBIT_OPT_OBJ_WEIGHT_SHARPE, BYBIT_OPT_OBJ_WEIGHT_CALMAR, etc. to customize
                    w_sharpe_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_SHARPE", "").strip()
                    w_calmar_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CALMAR", "").strip()
                    w_cagr_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CAGR", "").strip()
                    w_turnover_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_TURNOVER", "").strip()
                    use_composite_env = os.getenv("BYBIT_OPT_USE_COMPOSITE", "").strip().lower()
                    
                    try:
                        w_sharpe = float(w_sharpe_env) if w_sharpe_env else 0.4
                        w_calmar = float(w_calmar_env) if w_calmar_env else 0.3
                        w_cagr = float(w_cagr_env) if w_cagr_env else 0.2
                        w_turnover = float(w_turnover_env) if w_turnover_env else 0.1
                    except Exception:
                        w_sharpe, w_calmar, w_cagr, w_turnover = 0.4, 0.3, 0.2, 0.1
                    
                    use_composite = use_composite_env in ("1", "true", "yes", "y")
                    
                    if use_composite:
                        # Composite score: weighted combination of normalized metrics
                        # Normalize: sharpe (typically -2 to +3), calmar (typically -5 to +10), cagr (typically -1 to +1), turnover (typically 0 to 5)
                        sharpe_norm = float(row["sharpe"])  # Already in reasonable scale
                        calmar_norm = float(row["calmar"]) if np.isfinite(row["calmar"]) else 0.0
                        cagr_norm = float(row["cagr"]) * 100.0  # Scale CAGR to percentage points for better balance
                        turnover_norm = -float(row["avg_daily_turnover"])  # Penalty (negative)
                        
                        composite_score = (
                            w_sharpe * sharpe_norm +
                            w_calmar * calmar_norm +
                            w_cagr * cagr_norm +
                            w_turnover * turnover_norm
                        )
                        row["composite_score"] = composite_score
                        # For ranking: higher composite_score is better, so negate for "smaller is better" key
                        key = (-composite_score,)
                    else:
                        # Legacy lexicographic objective:
                        #   1) maximize Sharpe
                        #   2) maximize CAGR
                        #   3) minimize drawdown magnitude
                        #   4) minimize turnover
                        key = (-float(row["sharpe"]), -float(row["cagr"]), abs(float(row["max_drawdown"])), float(row["avg_daily_turnover"]))
                        # Also keep a scalar score for reporting/debugging
                        score = float(row["sharpe"]) + 0.25 * float(row["cagr"]) - 0.05 * float(row["avg_daily_turnover"])
                        row["score"] = score
                    rows.append(row)

                    if best_key is None or key < best_key:
                        best_key = key
                        best = (cand, row)
                        if progress_ctx is not None and task_id is not None:
                            progress_ctx.update(
                                task_id,
                                best_sharpe=fmt_or_na(float(row.get("sharpe", float("nan"))), ".3f"),
                                best_dd=fmt_or_na(float(row.get("max_drawdown", float("nan"))), ".2%"),
                                best_cagr=fmt_or_na(float(row.get("cagr", float("nan"))), ".2%"),
                            )

                    if progress_ctx is not None and task_id is not None:
                        progress_ctx.advance(task_id, 1)
                    else:
                        # Heartbeat logs for non-interactive environments
                        now = time.time()
                        if (i % hb_every == 0) or ((now - last_hb) >= hb_secs):
                            last_hb = now
                            best_sh = fmt_or_na(float(best[1].get("sharpe", float("nan"))), ".3f") if best is not None else "n/a"
                            best_dd = fmt_or_na(float(best[1].get("max_drawdown", float("nan"))), ".2%") if best is not None else "n/a"
                            best_cg = fmt_or_na(float(best[1].get("cagr", float("nan"))), ".2%") if best is not None else "n/a"
                            logger.info(
                                "{}Stage1 progress: {}/{} ({:.1%}) elapsed={}s best_sharpe={} best_dd={} best_cagr={} (set BYBIT_OPT_STAGE1_LOG_EVERY / BYBIT_OPT_STAGE1_LOG_SECS)",
                                window_prefix,
                                i,
                                len(candidates_list),
                                float(i) / float(max(1, len(candidates_list))),
                                int(now - t0),
                                best_sh,
                                best_dd,
                                best_cg,
                            )
            finally:
                if progress_ctx is not None:
                    progress_ctx.stop()

            out = Path(output_dir)
            if wf_enabled and len(windows) > 1:
                out = out / f"window_{window_idx + 1}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "stage1_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

            if best is None:
                # Helpful diagnostics
                if close_train is None or getattr(close_train, "empty", False):
                    logger.error("{}Stage1 close_train matrix is empty; cannot evaluate candidates (check data window/min_history).", window_prefix)
                try:
                    rej = [r.get("reject_reason") for r in rows if r.get("rejected")]
                    counts: dict[str, int] = {}
                    for x in rej:
                        k = str(x or "unknown")
                        counts[k] = counts.get(k, 0) + 1
                    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    logger.error("{}Stage1 rejection reasons (top 5): {}", window_prefix, top)
                except Exception as e:
                    logger.debug("Failed to generate rejection diagnostics: {}", e)
                logger.warning(
                    "{}Optimization found no feasible candidates. Skipping this window.",
                    window_prefix,
                )
                # Skip this window in walk-forward mode, otherwise return error
                if wf_enabled:
                    # Skip to next window in walk-forward analysis
                    # (We'll continue the loop naturally by not executing the rest of this iteration)
                    pass
                else:
                    # For single window, return error
                    return {
                        "status": "no_feasible_candidate",
                        "output_dir": str(out.resolve()),
                        "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                        "universe_size": len(symbols),
                        "candidates": len(rows),
                    }
                # If wf_enabled, skip the rest of this iteration
                if wf_enabled:
                    # Collect empty result for this window and continue
                    window_result = {
                        "window_idx": window_idx,
                        "train_start": cal_train[0].date().isoformat(),
                        "train_end": cal_train[-1].date().isoformat(),
                        "test_start": cal_test[0].date().isoformat() if len(cal_test) > 0 else None,
                        "test_end": cal_test[-1].date().isoformat() if len(cal_test) > 0 else None,
                        "best_candidate": None,
                        "train_metrics": None,
                        "oos_metrics": None,
                        "skipped": True,
                        "reason": "no_feasible_candidates",
                    }
                    all_window_results.append(window_result)
                    # Use continue here - it should be recognized as being in the loop
                    continue

            best_cand, best_row = best

            # Stage 2: re-evaluate top-K candidates with the full backtester logic.
            feasible = [r for r in rows if (not r.get("rejected")) and np.isfinite(float(r.get("sharpe", float("nan"))))]
            # Sort by composite_score if available, otherwise by sharpe
            use_composite_env = os.getenv("BYBIT_OPT_USE_COMPOSITE", "").strip().lower()
            use_composite = use_composite_env in ("1", "true", "yes", "y")
            if use_composite:
                feasible.sort(key=lambda r: float(r.get("composite_score", -1e9)), reverse=True)
            else:
                feasible.sort(key=lambda r: float(r.get("sharpe", -1e9)), reverse=True)

            if not feasible:
                logger.warning("{}Stage1 found {} feasible candidates, but all were rejected. Skipping Stage2.", window_prefix, len(rows))
                if wf_enabled:
                    continue
                return {
                    "status": "no_feasible_candidate",
                    "output_dir": str(out.resolve()),
                    "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                    "universe_size": len(symbols),
                    "candidates": len(rows),
                    "feasible": 0,
                }

            k2 = int(stage2_topk) if stage2_topk is not None else _level_to_stage2_topk(level)
            k2 = min(k2, len(feasible))  # Don't exceed available feasible candidates
            if k2 == 0:
                logger.warning("{}No feasible candidates available for Stage2 (feasible={} total={}).", window_prefix, len(feasible), len(rows))
                if wf_enabled:
                    continue
                return {
                    "status": "no_feasible_candidate",
                    "output_dir": str(out.resolve()),
                    "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                    "universe_size": len(symbols),
                    "candidates": len(rows),
                    "feasible": len(feasible),
                }
            stage2_candidates = [Candidate(**feasible[i]["candidate"]) for i in range(k2)]

            stage2_rows: list[dict[str, Any]] = []
            stage2_best: tuple[Candidate, dict[str, Any]] | None = None
            stage2_best_key: tuple[float, float, float, float] | None = None

            if show_progress:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold]stage2[/bold] full backtest"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("best_sharpe={task.fields[best_sharpe]}"),
                    TextColumn("best_dd={task.fields[best_dd]}"),
                    TextColumn("best_cagr={task.fields[best_cagr]}"),
                ) as p2:
                    t2 = p2.add_task("stage2", total=len(stage2_candidates), best_sharpe="n/a", best_dd="n/a", best_cagr="n/a")
                    for cand2 in stage2_candidates:
                        row2: dict[str, Any] | None = None
                        try:
                            trial = cfg.model_copy(deep=True)
                            trial.signal.lookback_days = int(cand2.lookback_days)  # type: ignore[assignment]
                            trial.signal.long_quantile = float(cand2.long_quantile)
                            trial.signal.short_quantile = float(cand2.short_quantile)
                            trial.sizing.target_gross_leverage = float(cand2.target_gross_leverage)
                            trial.sizing.vol_lookback_days = int(cand2.vol_lookback_days)
                            trial.rebalance.time_utc = str(cand2.rebalance_time_utc)
                            trial.rebalance.interval_days = int(cand2.interval_days)
                            trial.rebalance.rebalance_fraction = float(cand2.rebalance_fraction)
                            trial.rebalance.min_weight_change_bps = float(cand2.min_weight_change_bps)
                            trial.filters.regime_filter.action = str(cand2.regime_action)  # type: ignore[assignment]
                            trial.funding.filter.enabled = bool(cand2.funding_filter_enabled)
                            trial.funding.filter.max_abs_daily_funding_rate = float(cand2.funding_max_abs_daily_rate)

                            eq2, dr2, to2 = _simulate_candidate_full(cfg=trial, candles=candles_stage2, market_df=market_df, calendar=cal_train, funding_daily=funding_daily)
                            m2 = compute_metrics(eq2, dr2, to2)
                            # Reject degenerate curves: if we barely traded / produced too few daily points,
                            # annualized metrics (especially CAGR) can become meaningless.
                            min_pts_env = os.getenv("BYBIT_OPT_MIN_POINTS", "").strip()
                            try:
                                min_pts = int(min_pts_env) if min_pts_env else 90
                            except Exception:
                                min_pts = 90
                            min_pts = max(30, min_pts)
                            if eq2.empty or dr2.empty or len(eq2) < min_pts or len(dr2) < min_pts or not _metrics_ok(m2):
                                raise ValueError("stage2_invalid_metrics")
                            
                            # Use same composite objective as Stage1 if enabled
                            use_composite_env = os.getenv("BYBIT_OPT_USE_COMPOSITE", "").strip().lower()
                            use_composite = use_composite_env in ("1", "true", "yes", "y")
                            
                            if use_composite:
                                w_sharpe_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_SHARPE", "").strip()
                                w_calmar_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CALMAR", "").strip()
                                w_cagr_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CAGR", "").strip()
                                w_turnover_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_TURNOVER", "").strip()
                                try:
                                    w_sharpe = float(w_sharpe_env) if w_sharpe_env else 0.4
                                    w_calmar = float(w_calmar_env) if w_calmar_env else 0.3
                                    w_cagr = float(w_cagr_env) if w_cagr_env else 0.2
                                    w_turnover = float(w_turnover_env) if w_turnover_env else 0.1
                                except Exception:
                                    w_sharpe, w_calmar, w_cagr, w_turnover = 0.4, 0.3, 0.2, 0.1
                                
                                sharpe_norm = float(m2.sharpe)
                                calmar_norm = float(m2.calmar) if np.isfinite(m2.calmar) else 0.0
                                cagr_norm = float(m2.cagr) * 100.0
                                turnover_norm = -float(m2.avg_daily_turnover)
                                
                                composite_score = (
                                    w_sharpe * sharpe_norm +
                                    w_calmar * calmar_norm +
                                    w_cagr * cagr_norm +
                                    w_turnover * turnover_norm
                                )
                                key2 = (-composite_score,)
                            else:
                                key2 = (-float(m2.sharpe), -float(m2.cagr), abs(float(m2.max_drawdown)), float(m2.avg_daily_turnover))
                            
                            row2 = {
                                "candidate": cand2.__dict__,
                                "sharpe": float(m2.sharpe),
                                "cagr": float(m2.cagr),
                                "max_drawdown": float(m2.max_drawdown),
                                "calmar": float(m2.calmar),
                                "sortino": float(m2.sortino),
                                "profit_factor": float(m2.profit_factor),
                                "win_rate": float(m2.win_rate),
                                "avg_daily_turnover": float(m2.avg_daily_turnover),
                            }
                            if use_composite:
                                row2["composite_score"] = composite_score
                            if stage2_best_key is None or key2 < stage2_best_key:
                                stage2_best_key = key2
                                stage2_best = (cand2, row2)
                                p2.update(
                                    t2,
                                    best_sharpe=fmt_or_na(float(row2.get("sharpe", float("nan"))), ".3f"),
                                    best_dd=fmt_or_na(float(row2.get("max_drawdown", float("nan"))), ".2%"),
                                    best_cagr=fmt_or_na(float(row2.get("cagr", float("nan"))), ".2%"),
                                )
                        except (ValueError, KeyError, IndexError) as e:
                            err_msg = str(e)
                            if "Universe too small" in err_msg:
                                logger.debug("Stage2 candidate {} rejected: empty universe (likely funding/min_history filter too strict)", cand2.__dict__)
                            else:
                                logger.debug("Stage2 candidate {} rejected: {}", cand2.__dict__, err_msg)
                            row2 = {
                                "candidate": cand2.__dict__,
                                "rejected": True,
                                "error": err_msg,
                                "sharpe": float("nan"),
                                "cagr": float("nan"),
                                "max_drawdown": float("nan"),
                                "avg_daily_turnover": float("nan"),
                            }
                        except Exception as e:
                            logger.warning("Stage2 candidate {} failed with unexpected error: {}", cand2.__dict__, e)
                            row2 = {
                                "candidate": cand2.__dict__,
                                "rejected": True,
                                "error": str(e),
                                "sharpe": float("nan"),
                                "cagr": float("nan"),
                                "max_drawdown": float("nan"),
                                "avg_daily_turnover": float("nan"),
                            }
                        finally:
                            if row2 is None:
                                row2 = {
                                    "candidate": cand2.__dict__,
                                    "rejected": True,
                                    "error": "stage2_unknown_error",
                                    "sharpe": float("nan"),
                                    "cagr": float("nan"),
                                    "max_drawdown": float("nan"),
                                    "avg_daily_turnover": float("nan"),
                                }
                        stage2_rows.append(row2)
                        p2.advance(t2, 1)
            else:
                for cand2 in stage2_candidates:
                    row2: dict[str, Any] | None = None
                    try:
                        trial = cfg.model_copy(deep=True)
                        trial.signal.lookback_days = int(cand2.lookback_days)  # type: ignore[assignment]
                        trial.signal.long_quantile = float(cand2.long_quantile)
                        trial.signal.short_quantile = float(cand2.short_quantile)
                        trial.sizing.target_gross_leverage = float(cand2.target_gross_leverage)
                        trial.sizing.vol_lookback_days = int(cand2.vol_lookback_days)
                        trial.rebalance.time_utc = str(cand2.rebalance_time_utc)
                        trial.rebalance.interval_days = int(cand2.interval_days)
                        trial.rebalance.rebalance_fraction = float(cand2.rebalance_fraction)
                        trial.rebalance.min_weight_change_bps = float(cand2.min_weight_change_bps)
                        trial.filters.regime_filter.action = str(cand2.regime_action)  # type: ignore[assignment]
                        trial.funding.filter.enabled = bool(cand2.funding_filter_enabled)
                        trial.funding.filter.max_abs_daily_funding_rate = float(cand2.funding_max_abs_daily_rate)

                        eq2, dr2, to2 = _simulate_candidate_full(cfg=trial, candles=candles_stage2, market_df=market_df, calendar=cal_train, funding_daily=funding_daily)
                        m2 = compute_metrics(eq2, dr2, to2)
                        min_pts_env = os.getenv("BYBIT_OPT_MIN_POINTS", "").strip()
                        try:
                            min_pts = int(min_pts_env) if min_pts_env else 90
                        except Exception:
                            min_pts = 90
                        min_pts = max(30, min_pts)
                        if eq2.empty or dr2.empty or len(eq2) < min_pts or len(dr2) < min_pts or not _metrics_ok(m2):
                            raise ValueError("stage2_invalid_metrics")
                        
                        # Use same composite objective as Stage1 if enabled
                        use_composite_env = os.getenv("BYBIT_OPT_USE_COMPOSITE", "").strip().lower()
                        use_composite = use_composite_env in ("1", "true", "yes", "y")
                        
                        if use_composite:
                            w_sharpe_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_SHARPE", "").strip()
                            w_calmar_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CALMAR", "").strip()
                            w_cagr_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CAGR", "").strip()
                            w_turnover_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_TURNOVER", "").strip()
                            try:
                                w_sharpe = float(w_sharpe_env) if w_sharpe_env else 0.4
                                w_calmar = float(w_calmar_env) if w_calmar_env else 0.3
                                w_cagr = float(w_cagr_env) if w_cagr_env else 0.2
                                w_turnover = float(w_turnover_env) if w_turnover_env else 0.1
                            except Exception:
                                w_sharpe, w_calmar, w_cagr, w_turnover = 0.4, 0.3, 0.2, 0.1
                            
                            sharpe_norm = float(m2.sharpe)
                            calmar_norm = float(m2.calmar) if np.isfinite(m2.calmar) else 0.0
                            cagr_norm = float(m2.cagr) * 100.0
                            turnover_norm = -float(m2.avg_daily_turnover)
                            
                            composite_score = (
                                w_sharpe * sharpe_norm +
                                w_calmar * calmar_norm +
                                w_cagr * cagr_norm +
                                w_turnover * turnover_norm
                            )
                            key2 = (-composite_score,)
                        else:
                            key2 = (-float(m2.sharpe), -float(m2.cagr), abs(float(m2.max_drawdown)), float(m2.avg_daily_turnover))
                        
                        row2 = {
                            "candidate": cand2.__dict__,
                            "sharpe": float(m2.sharpe),
                            "cagr": float(m2.cagr),
                            "max_drawdown": float(m2.max_drawdown),
                            "calmar": float(m2.calmar),
                            "sortino": float(m2.sortino),
                            "profit_factor": float(m2.profit_factor),
                            "win_rate": float(m2.win_rate),
                            "avg_daily_turnover": float(m2.avg_daily_turnover),
                        }
                        if use_composite:
                            row2["composite_score"] = composite_score
                        if stage2_best_key is None or key2 < stage2_best_key:
                            stage2_best_key = key2
                            stage2_best = (cand2, row2)
                    except (ValueError, KeyError, IndexError) as e:
                        err_msg = str(e)
                        if "Universe too small" in err_msg:
                            logger.debug("Stage2 candidate {} rejected: empty universe (likely funding/min_history filter too strict)", cand2.__dict__)
                        else:
                            logger.debug("Stage2 candidate {} rejected: {}", cand2.__dict__, err_msg)
                        row2 = {
                            "candidate": cand2.__dict__,
                            "rejected": True,
                            "error": err_msg,
                            "sharpe": float("nan"),
                            "cagr": float("nan"),
                            "max_drawdown": float("nan"),
                            "avg_daily_turnover": float("nan"),
                        }
                    except Exception as e:
                        logger.warning("Stage2 candidate {} failed with unexpected error: {}", cand2.__dict__, e)
                        row2 = {
                            "candidate": cand2.__dict__,
                            "rejected": True,
                            "error": str(e),
                            "sharpe": float("nan"),
                            "cagr": float("nan"),
                            "max_drawdown": float("nan"),
                            "avg_daily_turnover": float("nan"),
                        }
                    finally:
                        if row2 is None:
                            row2 = {
                                "candidate": cand2.__dict__,
                                "rejected": True,
                                "error": "stage2_unknown_error",
                                "sharpe": float("nan"),
                                "cagr": float("nan"),
                                "max_drawdown": float("nan"),
                                "avg_daily_turnover": float("nan"),
                            }
                        stage2_rows.append(row2)

            (out / "stage2_results.json").write_text(json.dumps(stage2_rows, indent=2, sort_keys=True), encoding="utf-8")

            # Use Stage 2 best if available, otherwise fall back to Stage 1
            if stage2_best is not None:
                best_cand, best_row = stage2_best
                logger.info("{}Stage2 selected best candidate based on full backtest metrics.", window_prefix)
            else:
                logger.warning("{}Stage2 had no results; falling back to stage1 selection.", window_prefix)
                # best_cand and best_row are already set from Stage 1 above

            # Evaluate the selected params Out-Of-Sample (test window) for sanity.
            oos_metrics: dict[str, Any] | None = None
            m_oos: Any | None = None
            if len(cal_test) >= 2:
                try:
                    trial = cfg.model_copy(deep=True)
                    trial.signal.lookback_days = int(best_cand.lookback_days)  # type: ignore[assignment]
                    trial.signal.long_quantile = float(best_cand.long_quantile)
                    trial.signal.short_quantile = float(best_cand.short_quantile)
                    trial.sizing.target_gross_leverage = float(best_cand.target_gross_leverage)
                    trial.sizing.vol_lookback_days = int(best_cand.vol_lookback_days)
                    trial.rebalance.time_utc = str(best_cand.rebalance_time_utc)
                    trial.rebalance.interval_days = int(best_cand.interval_days)
                    trial.rebalance.rebalance_fraction = float(best_cand.rebalance_fraction)
                    trial.rebalance.min_weight_change_bps = float(best_cand.min_weight_change_bps)
                    trial.filters.regime_filter.action = str(best_cand.regime_action)  # type: ignore[assignment]
                    trial.funding.filter.enabled = bool(best_cand.funding_filter_enabled)
                    trial.funding.filter.max_abs_daily_funding_rate = float(best_cand.funding_max_abs_daily_rate)

                    # IMPORTANT: evaluate OOS in a *stateful* way by running a single contiguous simulation
                    # across train+test, then slicing metrics to the test window. This avoids resetting
                    # weights/equity at the test boundary (which can distort OOS significantly).
                    cal_full = cal_train.append(cal_test)
                    eq_full, dr_full, to_full = _simulate_candidate_full(
                        cfg=trial,
                        candles=candles_stage2,
                        market_df=market_df,
                        calendar=cal_full,
                        funding_daily=funding_daily,
                    )

                    test_start_dt = cal_test[0].to_pydatetime()
                    eq_oos = eq_full.loc[eq_full.index >= test_start_dt]
                    dr_oos = dr_full.loc[dr_full.index >= test_start_dt]
                    to_oos = to_full.loc[to_full.index >= test_start_dt]

                    m_oos = compute_metrics(eq_oos, dr_oos, to_oos)
                    oos_metrics = {
                        "sharpe": float(m_oos.sharpe),
                        "cagr": float(m_oos.cagr),
                        "max_drawdown": float(m_oos.max_drawdown),
                        "calmar": float(m_oos.calmar),
                        "sortino": float(m_oos.sortino),
                        "profit_factor": float(m_oos.profit_factor),
                        "win_rate": float(m_oos.win_rate),
                        "avg_daily_turnover": float(m_oos.avg_daily_turnover),
                        "test_days": int(len(eq_oos)),
                        "test_start": cal_test[0].date().isoformat(),
                        "test_end": cal_test[-1].date().isoformat(),
                    }
                    
                    # Overfitting detection: compare train vs test metrics
                    train_sharpe = float(best_row.get("sharpe", 0.0))
                    train_cagr = float(best_row.get("cagr", 0.0))
                    train_calmar = float(best_row.get("calmar", 0.0))
                    test_sharpe = float(m_oos.sharpe)
                    test_cagr = float(m_oos.cagr)
                    test_calmar = float(m_oos.calmar)
                    
                    overfitting_flags = []
                    if np.isfinite(train_sharpe) and np.isfinite(test_sharpe) and train_sharpe > 0:
                        sharpe_degradation = (train_sharpe - test_sharpe) / train_sharpe
                        if sharpe_degradation > 0.5:  # Test Sharpe is < 50% of train Sharpe
                            overfitting_flags.append(f"Sharpe degradation: {sharpe_degradation:.1%} (train={train_sharpe:.3f} test={test_sharpe:.3f})")
                    
                    if np.isfinite(train_cagr) and np.isfinite(test_cagr) and train_cagr > 0:
                        cagr_degradation = (train_cagr - test_cagr) / train_cagr
                        if cagr_degradation > 0.5:  # Test CAGR is < 50% of train CAGR
                            overfitting_flags.append(f"CAGR degradation: {cagr_degradation:.1%} (train={train_cagr:.2%} test={test_cagr:.2%})")
                    
                    if np.isfinite(train_calmar) and np.isfinite(test_calmar) and train_calmar > 0:
                        calmar_degradation = (train_calmar - test_calmar) / train_calmar
                        if calmar_degradation > 0.5:  # Test Calmar is < 50% of train Calmar
                            overfitting_flags.append(f"Calmar degradation: {calmar_degradation:.1%} (train={train_calmar:.3f} test={test_calmar:.3f})")
                    
                    overfitting_warning = None
                    if overfitting_flags:
                        overfitting_warning = " | ".join(overfitting_flags)
                        logger.warning("{}Potential overfitting detected: {}", window_prefix, overfitting_warning)
                    
                    oos_metrics["overfitting_warning"] = overfitting_warning
                    oos_metrics["train_vs_test"] = {
                        "sharpe": {"train": train_sharpe, "test": test_sharpe, "degradation_pct": float((train_sharpe - test_sharpe) / train_sharpe * 100) if train_sharpe > 0 and np.isfinite(train_sharpe) and np.isfinite(test_sharpe) else None},
                        "cagr": {"train": train_cagr, "test": test_cagr, "degradation_pct": float((train_cagr - test_cagr) / train_cagr * 100) if train_cagr > 0 and np.isfinite(train_cagr) and np.isfinite(test_cagr) else None},
                        "calmar": {"train": train_calmar, "test": test_calmar, "degradation_pct": float((train_calmar - test_calmar) / train_calmar * 100) if train_calmar > 0 and np.isfinite(train_calmar) and np.isfinite(test_calmar) else None},
                    }
                    
                    (out / "oos_best.json").write_text(
                        json.dumps(
                            {
                                "split": {
                                    "train_days": int(len(cal_train)),
                                    "train_start": cal_train[0].date().isoformat(),
                                    "train_end": cal_train[-1].date().isoformat(),
                                    "test_days": int(len(cal_test)),
                                    "test_start": cal_test[0].date().isoformat(),
                                    "test_end": cal_test[-1].date().isoformat(),
                                },
                                "best_candidate": best_cand.__dict__,
                                "train": best_row,
                                "test": oos_metrics,
                            },
                            indent=2,
                            sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                    logger.info(
                        "{}OOS (test) metrics for selected params: Sharpe={:.3f} CAGR={:.2%} MaxDD={:.2%} Calmar={:.3f} Turnover={:.3f} Days={}",
                        window_prefix,
                        float(m_oos.sharpe),
                        float(m_oos.cagr),
                        float(m_oos.max_drawdown),
                        float(m_oos.calmar),
                        float(m_oos.avg_daily_turnover),
                        int(len(eq_oos)),
                    )
                except Exception as e:
                    logger.warning("{}OOS evaluation failed; continuing without OOS metrics: {}", window_prefix, e)
                    m_oos = None

            # Debug: summarize stage2 outcomes (why did we get no results?)
            try:
                total2 = len(stage2_rows)
                ok2 = [r for r in stage2_rows if not r.get("rejected") and np.isfinite(float(r.get("sharpe", float("nan"))))]
                rej2 = [r for r in stage2_rows if r.get("rejected")]
                if not ok2:
                    # Count by error string (top few)
                    err_counts: dict[str, int] = {}
                    for r in rej2:
                        err = str(r.get("error") or "unknown")
                        err_counts[err] = err_counts.get(err, 0) + 1
                    top_err = sorted(err_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
                    logger.warning(
                        "{}Stage2 produced 0 successful candidates (total={} rejected={}). Top reject reasons: {}",
                        window_prefix,
                        total2,
                        len(rej2),
                        top_err,
                    )
                else:
                    logger.info("{}Stage2 produced {} successful candidates out of {}", window_prefix, len(ok2), total2)
            except Exception:
                pass

            # Helpful transparency: show top-by-sharpe in logs (stage2 if available).
            try:
                df = pd.DataFrame(stage2_rows if stage2_rows else [r for r in rows if not r.get("rejected")])
                if not df.empty:
                    if "rejected" in df.columns:
                        df = df[df["rejected"] != True]  # noqa: E712
                    # Drop rows with NaN/inf sharpe
                    df = df[pd.to_numeric(df["sharpe"], errors="coerce").notna()]
                    top_sh = df.sort_values("sharpe", ascending=False).head(5)[
                        ["sharpe", "cagr", "max_drawdown", "avg_daily_turnover", "candidate"]
                    ]
                    logger.info("{}Top 5 by Sharpe:\n{}", window_prefix, top_sh.to_string(index=False))
            except Exception:
                pass

            # Guardrail: don't write params that are statistically/financially "worse than nothing".
            # Default: reject negative Sharpe (BYBIT_OPT_MIN_SHARPE=0.0).
            min_sharpe_env = os.getenv("BYBIT_OPT_MIN_SHARPE", "").strip()
            try:
                min_sharpe = float(min_sharpe_env) if min_sharpe_env else 0.0
            except Exception:
                min_sharpe = 0.0
            best_sharpe = float(best_row.get("sharpe", 0.0))
            if not np.isfinite(best_sharpe) or best_sharpe < min_sharpe:
                logger.warning(
                    "{}Optimizer best Sharpe {:.3f} < min {:.3f}; skipping this window.",
                    window_prefix,
                    best_sharpe,
                    min_sharpe,
                )
                if wf_enabled:
                    continue
                return {
                    "status": "rejected_by_threshold",
                    "min_sharpe": min_sharpe,
                    "best": best_row,
                    "output_dir": str(out.resolve()),
                    "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                    "universe_size": len(symbols),
                    "candidates": len(rows),
                }

            # Optional: OOS profitability gate (recommended when you want the optimizer to only "accept" configs
            # that are profitable out-of-sample).
            #
            # Defaults:
            #   BYBIT_OPT_REQUIRE_OOS=0  (do not reject if OOS isn't available)
            #   BYBIT_OPT_MIN_OOS_SHARPE=0.0
            #   BYBIT_OPT_MIN_OOS_CAGR=0.0
            require_oos_env = os.getenv("BYBIT_OPT_REQUIRE_OOS", "").strip().lower()
            require_oos = require_oos_env in ("1", "true", "yes", "y")
            min_oos_sh_env = os.getenv("BYBIT_OPT_MIN_OOS_SHARPE", "").strip()
            min_oos_cagr_env = os.getenv("BYBIT_OPT_MIN_OOS_CAGR", "").strip()
            try:
                min_oos_sharpe = float(min_oos_sh_env) if min_oos_sh_env else 0.0
            except Exception:
                min_oos_sharpe = 0.0
            try:
                min_oos_cagr = float(min_oos_cagr_env) if min_oos_cagr_env else 0.0
            except Exception:
                min_oos_cagr = 0.0

            if require_oos and m_oos is None:
                logger.warning("{}Optimizer rejected: OOS metrics required (BYBIT_OPT_REQUIRE_OOS=1) but were unavailable.", window_prefix)
                if wf_enabled:
                    continue
                return {
                    "status": "rejected_by_oos_threshold",
                    "reason": "oos_metrics_unavailable",
                    "min_oos_sharpe": min_oos_sharpe,
                    "min_oos_cagr": min_oos_cagr,
                    "output_dir": str(out.resolve()),
                    "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                    "universe_size": len(symbols),
                    "candidates": len(rows),
                }

            if require_oos and m_oos is not None:
                oos_sh = float(m_oos.sharpe)
                oos_cg = float(m_oos.cagr)
                if (not np.isfinite(oos_sh)) or (not np.isfinite(oos_cg)) or (oos_sh < min_oos_sharpe) or (oos_cg < min_oos_cagr):
                    logger.warning(
                        "{}Optimizer rejected by OOS thresholds: OOS Sharpe {:.3f} (min {:.3f}), OOS CAGR {:.2%} (min {:.2%}).",
                        window_prefix,
                        oos_sh,
                        min_oos_sharpe,
                        oos_cg,
                        min_oos_cagr,
                    )
                    if wf_enabled:
                        continue
                    return {
                        "status": "rejected_by_oos_threshold",
                        "min_oos_sharpe": min_oos_sharpe,
                        "min_oos_cagr": min_oos_cagr,
                        "oos_sharpe": oos_sh,
                        "oos_cagr": oos_cg,
                        "output_dir": str(out.resolve()),
                        "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                        "universe_size": len(symbols),
                        "candidates": len(rows),
                    }

            # Collect window result for walk-forward aggregation
            window_result = {
                "window_idx": window_idx,
                "train_start": cal_train[0].date().isoformat(),
                "train_end": cal_train[-1].date().isoformat(),
                "test_start": cal_test[0].date().isoformat() if len(cal_test) > 0 else None,
                "test_end": cal_test[-1].date().isoformat() if len(cal_test) > 0 else None,
                "best_candidate": best_cand.__dict__,
                "train_metrics": best_row,
                "oos_metrics": oos_metrics,
            }
            all_window_results.append(window_result)
            
            # For single window mode, write config and return immediately
            if not wf_enabled:
                if write_config:
                    # Patch config.yaml (deep merge)
                    raw = load_yaml_config(config_path)
                    raw.setdefault("signal", {})
                    raw.setdefault("rebalance", {})
                    raw.setdefault("sizing", {})

                    raw["signal"]["lookback_days"] = int(best_cand.lookback_days)
                    raw["signal"]["long_quantile"] = float(best_cand.long_quantile)
                    raw["signal"]["short_quantile"] = float(best_cand.short_quantile)
                    raw["rebalance"]["time_utc"] = str(best_cand.rebalance_time_utc)
                    raw["rebalance"]["interval_days"] = int(best_cand.interval_days)
                    raw["rebalance"]["rebalance_fraction"] = float(best_cand.rebalance_fraction)
                    raw["rebalance"]["min_weight_change_bps"] = float(best_cand.min_weight_change_bps)
                    raw["sizing"]["target_gross_leverage"] = float(best_cand.target_gross_leverage)
                    raw["sizing"]["vol_lookback_days"] = int(best_cand.vol_lookback_days)
                    raw.setdefault("filters", {}).setdefault("regime_filter", {})
                    raw["filters"]["regime_filter"]["action"] = str(best_cand.regime_action)
                    raw.setdefault("funding", {}).setdefault("filter", {})
                    raw["funding"]["filter"]["enabled"] = bool(best_cand.funding_filter_enabled)
                    raw["funding"]["filter"]["max_abs_daily_funding_rate"] = float(best_cand.funding_max_abs_daily_rate)

                    import yaml

                    Path(config_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
                else:
                    logger.warning("write_config=false: NOT writing params back to {} (see best.json in output dir).", Path(config_path).resolve())

                (out / "best.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")

                logger.info(
                    "Optimization complete. Best sharpe={:.3f} cagr={:.2%} maxDD={:.2%} turnover={:.3f} params={}",
                    float(best_row.get("sharpe", 0.0)),
                    float(best_row.get("cagr", 0.0)),
                    float(best_row.get("max_drawdown", 0.0)),
                    float(best_row.get("avg_daily_turnover", 0.0)),
                    best_cand.__dict__,
                )
                return {
                    "best": best_row,
                    "oos": oos_metrics,
                    "output_dir": str(out.resolve()),
                    "window": {"start": window_start.date().isoformat(), "end": window_end.date().isoformat()},
                    "universe_size": len(symbols),
                    "candidates": len(rows),
                    "evaluated": len(candidates_list),
                    "stage2_topk": int(stage2_topk) if stage2_topk is not None else _level_to_stage2_topk(level),
                    "write_config": bool(write_config),
                }
        
        # Walk-forward aggregation: select best candidate across all windows
        if wf_enabled and len(all_window_results) > 0:
            logger.info("Aggregating results across {} walk-forward windows...", len(all_window_results))
            
            # Count how many times each candidate appears (robustness metric)
            candidate_counts: dict[tuple, list[dict[str, Any]]] = {}
            for wr in all_window_results:
                cand_dict = wr["best_candidate"]
                # Create a hashable key from candidate params
                cand_key = (
                    cand_dict.get("lookback_days"),
                    cand_dict.get("long_quantile"),
                    cand_dict.get("short_quantile"),
                    cand_dict.get("target_gross_leverage"),
                    cand_dict.get("vol_lookback_days"),
                    cand_dict.get("interval_days"),
                    cand_dict.get("rebalance_fraction"),
                    cand_dict.get("min_weight_change_bps"),
                    cand_dict.get("regime_action"),
                    cand_dict.get("funding_filter_enabled"),
                    cand_dict.get("funding_max_abs_daily_rate"),
                )
                if cand_key not in candidate_counts:
                    candidate_counts[cand_key] = []
                candidate_counts[cand_key].append(wr)
            
            # Rank candidates by average OOS Sharpe (or composite score if enabled)
            use_composite_env = os.getenv("BYBIT_OPT_USE_COMPOSITE", "").strip().lower()
            use_composite = use_composite_env in ("1", "true", "yes", "y")
            
            candidate_scores: list[tuple[tuple, float, int, dict[str, Any]]] = []
            for cand_key, window_results in candidate_counts.items():
                oos_sharpes = []
                oos_calmars = []
                oos_cagrs = []
                for wr in window_results:
                    oos = wr.get("oos_metrics")
                    if oos and oos.get("sharpe") is not None:
                        sh = float(oos.get("sharpe", 0.0))
                        if np.isfinite(sh):
                            oos_sharpes.append(sh)
                    if oos and oos.get("calmar") is not None:
                        cm = float(oos.get("calmar", 0.0))
                        if np.isfinite(cm):
                            oos_calmars.append(cm)
                    if oos and oos.get("cagr") is not None:
                        cg = float(oos.get("cagr", 0.0))
                        if np.isfinite(cg):
                            oos_cagrs.append(cg)
                
                if oos_sharpes:
                    avg_sharpe = float(np.mean(oos_sharpes))
                    avg_calmar = float(np.mean(oos_calmars)) if oos_calmars else 0.0
                    avg_cagr = float(np.mean(oos_cagrs)) if oos_cagrs else 0.0
                    appearance_count = len(window_results)
                    
                    if use_composite:
                        w_sharpe_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_SHARPE", "").strip()
                        w_calmar_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CALMAR", "").strip()
                        w_cagr_env = os.getenv("BYBIT_OPT_OBJ_WEIGHT_CAGR", "").strip()
                        try:
                            w_sharpe = float(w_sharpe_env) if w_sharpe_env else 0.4
                            w_calmar = float(w_calmar_env) if w_calmar_env else 0.3
                            w_cagr = float(w_cagr_env) if w_cagr_env else 0.2
                        except Exception:
                            w_sharpe, w_calmar, w_cagr = 0.4, 0.3, 0.2
                        
                        score = w_sharpe * avg_sharpe + w_calmar * avg_calmar + w_cagr * avg_cagr * 100.0
                    else:
                        score = avg_sharpe
                    
                    # Boost score by appearance count (robustness bonus)
                    score = score * (1.0 + 0.1 * (appearance_count - 1))
                    candidate_scores.append((cand_key, score, appearance_count, window_results[0]))
            
            if candidate_scores:
                # Sort by score (descending), then by appearance count
                candidate_scores.sort(key=lambda x: (-x[1], -x[2]))
                best_key, best_score, best_count, best_window_result = candidate_scores[0]
                
                logger.info(
                    "Walk-forward best candidate: avg_OOS_Sharpe={:.3f} (appeared in {}/{} windows), params={}",
                    best_score / (1.0 + 0.1 * (best_count - 1)) if best_count > 1 else best_score,
                    best_count,
                    len(all_window_results),
                    best_window_result["best_candidate"],
                )
                
                # Update best_cand and best_row for final write
                best_cand = Candidate(**best_window_result["best_candidate"])
                best_row = best_window_result["train_metrics"]
                oos_metrics = best_window_result["oos_metrics"]
                
                # Save walk-forward summary
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                wf_summary = {
                    "num_windows": len(all_window_results),
                    "best_candidate": best_cand.__dict__,
                    "best_score": best_score,
                    "appearance_count": best_count,
                    "all_windows": all_window_results,
                    "candidate_rankings": [
                        {
                            "candidate": wr["best_candidate"],
                            "score": score,
                            "appearance_count": count,
                            "avg_oos_sharpe": float(np.mean([w.get("oos_metrics", {}).get("sharpe", 0.0) for w in candidate_counts[key] if w.get("oos_metrics", {}).get("sharpe") is not None])),
                        }
                        for key, score, count, wr in candidate_scores[:10]  # Top 10
                    ],
                }
                (out / "walkforward_summary.json").write_text(json.dumps(wf_summary, indent=2, sort_keys=True), encoding="utf-8")
            else:
                logger.warning("Walk-forward: No valid candidates with OOS metrics across windows. Using last window result.")
                best_window_result = all_window_results[-1]
                best_cand = Candidate(**best_window_result["best_candidate"])
                best_row = best_window_result["train_metrics"]
                oos_metrics = best_window_result["oos_metrics"]
            
            # Write final config
            if write_config:
                raw = load_yaml_config(config_path)
                raw.setdefault("signal", {})
                raw.setdefault("rebalance", {})
                raw.setdefault("sizing", {})

                raw["signal"]["lookback_days"] = int(best_cand.lookback_days)
                raw["signal"]["long_quantile"] = float(best_cand.long_quantile)
                raw["signal"]["short_quantile"] = float(best_cand.short_quantile)
                raw["rebalance"]["time_utc"] = str(best_cand.rebalance_time_utc)
                raw["rebalance"]["interval_days"] = int(best_cand.interval_days)
                raw["rebalance"]["rebalance_fraction"] = float(best_cand.rebalance_fraction)
                raw["rebalance"]["min_weight_change_bps"] = float(best_cand.min_weight_change_bps)
                raw["sizing"]["target_gross_leverage"] = float(best_cand.target_gross_leverage)
                raw["sizing"]["vol_lookback_days"] = int(best_cand.vol_lookback_days)
                raw.setdefault("filters", {}).setdefault("regime_filter", {})
                raw["filters"]["regime_filter"]["action"] = str(best_cand.regime_action)
                raw.setdefault("funding", {}).setdefault("filter", {})
                raw["funding"]["filter"]["enabled"] = bool(best_cand.funding_filter_enabled)
                raw["funding"]["filter"]["max_abs_daily_funding_rate"] = float(best_cand.funding_max_abs_daily_rate)

                import yaml

                Path(config_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "best.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")
            
            logger.info(
                "Walk-forward optimization complete. Best sharpe={:.3f} cagr={:.2%} maxDD={:.2%} turnover={:.3f} params={}",
                float(best_row.get("sharpe", 0.0)),
                float(best_row.get("cagr", 0.0)),
                float(best_row.get("max_drawdown", 0.0)),
                float(best_row.get("avg_daily_turnover", 0.0)),
                best_cand.__dict__,
            )
            return {
                "best": best_row,
                "oos": oos_metrics,
                "output_dir": str(out.resolve()),
                "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                "universe_size": len(symbols),
                "candidates": len(rows) if 'rows' in locals() else 0,
                "evaluated": len(candidates_list) if 'candidates_list' in locals() else 0,
                "stage2_topk": int(stage2_topk) if stage2_topk is not None else _level_to_stage2_topk(level),
                "write_config": bool(write_config),
                "walkforward": {"enabled": True, "num_windows": len(all_window_results)},
            }
        
        # Fallback: if walk-forward was enabled but no results, return error
        if wf_enabled and len(all_window_results) == 0:
            return {
                "status": "no_feasible_candidate",
                "output_dir": str(Path(output_dir).resolve()),
                "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                "universe_size": len(symbols),
                "candidates": 0,
                "walkforward": {"enabled": True, "num_windows": 0},
            }
    finally:
        client.close()


