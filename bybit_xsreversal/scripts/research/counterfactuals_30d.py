#!/usr/bin/env python3
"""
Phase D: Counterfactual exit / trade-management simulations.
Read-only. Uses trades_enriched + candles from trade_forensics_30d.py.
Run from repo root:
  python3 bybit_xsreversal/scripts/research/counterfactuals_30d.py --days 30 --warmup_days 120

Outputs:
  - outputs/research_30d/counterfactuals_summary.csv
  - outputs/research_30d/equity_curves.csv
  - outputs/research_30d/stopout_regret_detail.csv

Uses episodes (from trade_forensics) when available; falls back to trades with 24h assumed entry.
ATR(14) on 4H; stop trigger on 1H intrabar (long: candle.low <= stop, short: candle.high >= stop).
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SCRIPT_DIR.parent.parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from loguru import logger

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


def _ensure_deps() -> None:
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy are required. pip install pandas numpy")


def _ensure_linear_category(cfg, args_category: str | None) -> str:
    """Research scripts: USDT linear perps only. Validate/warn and return 'linear'."""
    config_cat = getattr(getattr(cfg, "exchange", None), "category", None) if cfg else None
    config_cat = config_cat or "linear"
    cat = (args_category or config_cat or "linear").strip().lower()
    if cat != "linear":
        logger.warning("Research pipeline is for USDT linear perps only; forcing category='linear' (was {})", cat)
        return "linear"
    return "linear"


def _load_trades(out_dir: Path) -> pd.DataFrame:
    p_parquet = out_dir / "trades_enriched.parquet"
    p_csv = out_dir / "trades_enriched.csv"
    if p_parquet.exists():
        df = pd.read_parquet(p_parquet)
    elif p_csv.exists():
        df = pd.read_csv(p_csv)
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    else:
        raise FileNotFoundError(f"Run trade_forensics_30d.py first. Missing {p_parquet} or {p_csv}")
    return df


def _load_episodes(out_dir: Path) -> pd.DataFrame | None:
    """Load episodes.parquet if present (episode-based analysis)."""
    for name in ("episodes.parquet", "episodes.csv"):
        p = out_dir / name
        if p.exists():
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
                for col in ("entry_ts", "exit_ts"):
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], utc=True)
            return df
    return None


def _load_research_summary(out_dir: Path) -> dict | None:
    """Load research_30d_summary.json if present (n_fills, total_realized_pnl_usd, etc.)."""
    p = out_dir / "research_30d_summary.json"
    if not p.exists():
        return None
    try:
        import json
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def _load_candles(out_dir: Path, interval: str) -> dict[str, pd.DataFrame]:
    base = out_dir / "candles" / interval
    result = {}
    if not base.exists():
        return result
    for p in base.iterdir():
        if p.suffix in (".parquet", ".csv"):
            sym = p.stem
            if p.suffix == ".parquet":
                result[sym] = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
                if "ts_open_utc" in df.columns:
                    df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                result[sym] = df
    return result


def atr_series(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()


def _atr_at_entry(candles_4h: pd.DataFrame, entry_ts: pd.Timestamp, n: int = 14) -> float:
    """ATR(14) from 4H candles: use last bar at or before entry_ts."""
    if candles_4h is None or candles_4h.empty or "high" not in candles_4h.columns:
        return float("nan")
    c = candles_4h.copy()
    if "ts_open_utc" in c.columns and not c["ts_open_utc"].empty:
        c = c.set_index(pd.to_datetime(c["ts_open_utc"], utc=True))
    before = c[c.index <= entry_ts].tail(max(n, 20))
    if len(before) < n:
        return float("nan")
    atr = atr_series(before["high"], before["low"], before["close"], n)
    return float(atr.iloc[-1]) if not atr.empty else float("nan")


def _episode_candles_1h(candles_1h: dict[str, pd.DataFrame], symbol: str, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp) -> pd.DataFrame:
    """1H candles for symbol in [entry_ts, exit_ts], sorted by time."""
    sym = str(symbol).upper()
    df = candles_1h.get(sym)
    if df is None or df.empty or "high" not in df.columns or "low" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    if "ts_open_utc" in df.columns:
        df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
        mask = (df["ts_open_utc"] >= entry_ts) & (df["ts_open_utc"] <= exit_ts)
        return df.loc[mask].sort_values("ts_open_utc").reset_index(drop=True)
    return pd.DataFrame()


def _exit_cost_usd(notional_usd: float, taker_fee_bps: float, slippage_bps: float) -> float:
    return notional_usd * (taker_fee_bps + slippage_bps) / 10000.0


def _get_research_costs(cfg) -> tuple[float, float, float, float, float, float]:
    """Return (taker_fee_bps, maker_fee_bps, base_slippage_bps, spread_proxy_alpha, spread_proxy_min_bps, spread_proxy_max_bps). Use research_costs if set else backtest; conservative taker default."""
    rc = getattr(cfg, "research_costs", None)
    bt = getattr(cfg, "backtest", None)
    taker = getattr(rc, "taker_fee_bps", None) if rc else None
    if taker is None and bt:
        taker = getattr(bt, "taker_fee_bps", 6.0)
    taker = float(taker or 6.0)
    maker = float(getattr(rc, "maker_fee_bps", 1.0) if rc else (getattr(bt, "maker_fee_bps", 1.0) if bt else 1.0))
    base_slip = float(getattr(rc, "base_slippage_bps", 2.0) if rc else (getattr(bt, "slippage_bps", 2.0) if bt else 2.0))
    alpha = float(getattr(rc, "spread_proxy_alpha", 0.25) if rc else 0.25)
    min_bps = float(getattr(rc, "spread_proxy_min_bps", 1.0) if rc else 1.0)
    max_bps = float(getattr(rc, "spread_proxy_max_bps", 50.0) if rc else 50.0)
    return taker, maker, base_slip, alpha, min_bps, max_bps


def _spread_proxy_bps(high: float, low: float, close: float, alpha: float, min_bps: float, max_bps: float) -> float:
    """spread_proxy_bps = clamp((high-low)/close * 10000 * alpha, min_bps, max_bps)."""
    if close is None or close <= 0:
        return min_bps
    raw = (float(high) - float(low)) / float(close) * 10000.0 * alpha
    return float(np.clip(raw, min_bps, max_bps))


def _simulate_episode_fixed_stop_intrabar(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_vwap: float,
    exit_vwap: float,
    side: str,
    qty: float,
    atr: float,
    k_atr: float,
    bars_1h: pd.DataFrame,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[float, str, pd.Timestamp | None, dict | None]:
    """LONG: stop triggers if candle.low <= stop. SHORT: if candle.high >= stop. Returns (pnl_usd, exit_reason, trigger_ts, first_trigger_candle)."""
    is_long = str(side).strip().lower() == "long"
    entry_vwap = float(entry_vwap)
    exit_vwap = float(exit_vwap)
    qty = float(qty)
    if atr <= 0 or np.isnan(atr) or bars_1h.empty:
        notional = exit_vwap * qty
        pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
        return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None
    if is_long:
        stop_price = entry_vwap - k_atr * atr
        for _, row in bars_1h.iterrows():
            if float(row["low"]) <= stop_price:
                notional = stop_price * qty
                pnl = (stop_price - entry_vwap) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
    else:
        stop_price = entry_vwap + k_atr * atr
        for _, row in bars_1h.iterrows():
            if float(row["high"]) >= stop_price:
                notional = stop_price * qty
                pnl = (entry_vwap - stop_price) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
    notional = exit_vwap * qty
    pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
    return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None


def _simulate_episode_trailing_intrabar(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_vwap: float,
    exit_vwap: float,
    side: str,
    qty: float,
    atr: float,
    k_atr: float,
    bars_1h: pd.DataFrame,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[float, str, pd.Timestamp | None, dict | None]:
    """Trailing: long trail = best_high - k*ATR, trigger when bar.low <= trail; short symmetric."""
    is_long = str(side).strip().lower() == "long"
    entry_vwap = float(entry_vwap)
    exit_vwap = float(exit_vwap)
    qty = float(qty)
    if atr <= 0 or np.isnan(atr) or bars_1h.empty:
        notional = exit_vwap * qty
        pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
        return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None
    best = entry_vwap if is_long else entry_vwap
    for _, row in bars_1h.iterrows():
        h, l_ = float(row["high"]), float(row["low"])
        if is_long:
            best = max(best, h)
            trail = best - k_atr * atr
            if l_ <= trail:
                notional = trail * qty
                pnl = (trail - entry_vwap) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "trail_stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
        else:
            best = min(best, l_)
            trail = best + k_atr * atr
            if h >= trail:
                notional = trail * qty
                pnl = (entry_vwap - trail) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "trail_stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
    notional = exit_vwap * qty
    pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
    return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None


def _simulate_episode_stop_to_be_intrabar(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_vwap: float,
    exit_vwap: float,
    side: str,
    qty: float,
    atr: float,
    t_atr: float,
    bars_1h: pd.DataFrame,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[float, str, pd.Timestamp | None, dict | None]:
    """Move stop to breakeven (+costs) when MFE >= t*ATR; then check intrabar."""
    is_long = str(side).strip().lower() == "long"
    entry_vwap = float(entry_vwap)
    exit_vwap = float(exit_vwap)
    qty = float(qty)
    cost_bps = taker_fee_bps + slippage_bps
    be_price = entry_vwap * (1 + cost_bps / 10000.0) if is_long else entry_vwap * (1 - cost_bps / 10000.0)
    if atr <= 0 or np.isnan(atr) or bars_1h.empty:
        notional = exit_vwap * qty
        pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
        return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None
    stop_price = None
    best = entry_vwap if is_long else entry_vwap
    for _, row in bars_1h.iterrows():
        h, l_ = float(row["high"]), float(row["low"])
        if is_long:
            best = max(best, h)
            if best - entry_vwap >= t_atr * atr:
                stop_price = be_price
            if stop_price is not None and l_ <= stop_price:
                notional = stop_price * qty
                pnl = (stop_price - entry_vwap) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "be_stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
        else:
            best = min(best, l_)
            if entry_vwap - best >= t_atr * atr:
                stop_price = be_price
            if stop_price is not None and h >= stop_price:
                notional = stop_price * qty
                pnl = (entry_vwap - stop_price) * qty - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                ts = row.get("ts_open_utc")
                return pnl, "be_stop", pd.Timestamp(ts) if ts is not None else None, row.to_dict()
    notional = exit_vwap * qty
    pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
    return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None


def _simulate_episode_time_stop_intrabar(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_vwap: float,
    exit_vwap: float,
    side: str,
    qty: float,
    max_hours: float,
    bars_1h: pd.DataFrame,
    taker_fee_bps: float,
    slippage_bps: float,
) -> tuple[float, str, pd.Timestamp | None, dict | None]:
    """Exit after max_hours if not profitable; use next bar open (or actual exit)."""
    is_long = str(side).strip().lower() == "long"
    entry_vwap = float(entry_vwap)
    exit_vwap = float(exit_vwap)
    qty = float(qty)
    if bars_1h.empty:
        notional = exit_vwap * qty
        pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
        return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None
    for _, row in bars_1h.iterrows():
        ts = row.get("ts_open_utc")
        if ts is None:
            continue
        elapsed = (pd.Timestamp(ts) - entry_ts).total_seconds() / 3600.0
        if elapsed >= max_hours:
            exit_p = float(row.get("open", row.get("close", exit_vwap)))
            notional = exit_p * qty
            pnl = (exit_p - entry_vwap) * qty if is_long else (entry_vwap - exit_p) * qty
            if pnl <= 0:
                pnl = pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps)
                return pnl, "time_stop", pd.Timestamp(ts), row.to_dict()
    notional = exit_vwap * qty
    pnl = (exit_vwap - entry_vwap) * qty if is_long else (entry_vwap - exit_vwap) * qty
    return pnl - _exit_cost_usd(notional, taker_fee_bps, slippage_bps), "actual", None, None


def simulate_fixed_stop(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    side: str,
    qty: float,
    atr: float,
    k_atr: float,
) -> tuple[float, str]:
    """Returns (pnl_usd, exit_reason)."""
    if atr <= 0 or np.isnan(atr):
        return (exit_price - entry_price) * qty if side and str(side).lower() != "sell" else (entry_price - exit_price) * qty, "actual"
    if side and str(side).lower() == "sell":
        stop_price = entry_price + k_atr * atr
        if exit_price >= stop_price:
            return (entry_price - stop_price) * qty, "stop"
    else:
        stop_price = entry_price - k_atr * atr
        if exit_price <= stop_price:
            return (stop_price - entry_price) * qty, "stop"
    return (exit_price - entry_price) * qty if side and str(side).lower() != "sell" else (entry_price - exit_price) * qty, "actual"


def simulate_trailing_stop(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    side: str,
    qty: float,
    high_series: pd.Series,
    low_series: pd.Series,
    atr: float,
    k_atr: float,
) -> tuple[float, str]:
    """Trail from peak favorable by k*ATR. Simplified: use max(high) and min(low) in window."""
    if atr <= 0 or np.isnan(atr) or high_series.empty:
        pnl = (exit_price - entry_price) * qty if side and str(side).lower() != "sell" else (entry_price - exit_price) * qty
        return pnl, "actual"
    if side and str(side).lower() == "sell":
        best = low_series.min()
        trail = best + k_atr * atr
        if exit_price >= trail:
            return (entry_price - trail) * qty, "trail_stop"
        return (entry_price - exit_price) * qty, "actual"
    else:
        best = high_series.max()
        trail = best - k_atr * atr
        if exit_price <= trail:
            return (trail - entry_price) * qty, "trail_stop"
        return (exit_price - entry_price) * qty, "actual"


def run_counterfactuals_episodes(
    episodes_df: pd.DataFrame,
    candles_1h: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame],
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    base_slippage_bps: float | None = None,
    spread_proxy_alpha: float = 0.25,
    spread_proxy_min_bps: float = 1.0,
    spread_proxy_max_bps: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simulate stops on EPISODES with 1H intrabar trigger (long: low<=stop, short: high>=stop).
    ATR(14) from 4H. Slippage = base_slippage_bps + spread_proxy from last bar (if base_slippage_bps set).
    Returns (counterfactuals_summary, equity_curves, stopout_regret_detail).
    """
    if episodes_df.empty:
        summary = pd.DataFrame(columns=["model", "param", "net_pnl_usd", "max_dd_pct", "win_rate", "profit_factor", "stop_out_regret_pct", "n_trades", "avg_fee_bps", "avg_slippage_bps"])
        equity = pd.DataFrame(columns=["model", "date", "equity"])
        regret = pd.DataFrame(columns=["episode_id", "model", "param", "actual_pnl", "sim_pnl", "exit_reason", "winner_stopped_early"])
        return summary, equity, regret

    base_slip = base_slippage_bps if base_slippage_bps is not None else slippage_bps
    use_spread_proxy = base_slippage_bps is not None

    baseline_pnls = (episodes_df["realized_pnl_usd"].fillna(0) - episodes_df["fees_usd"].fillna(0)).values
    baseline_net = float(np.sum(baseline_pnls))
    n_ep = len(episodes_df)
    wins_baseline = int(np.sum(baseline_pnls > 0))
    start_equity = 10000.0
    results: list[dict] = []
    equity_curves: list[dict] = []
    regret_rows: list[dict] = []

    def add_equity(name: str, pnls: np.ndarray) -> float:
        cum = np.cumsum(pnls)
        eq = start_equity + cum
        peak = np.maximum.accumulate(eq)
        dd = (eq / peak - 1.0) * 100.0
        for i in range(len(eq)):
            ts = episodes_df["exit_ts"].iloc[i] if i < len(episodes_df) else None
            equity_curves.append({"model": name, "date": pd.Timestamp(ts).date() if ts is not None else None, "equity": float(eq[i]), "cum_pnl": float(cum[i])})
        return float(np.min(dd)) if len(dd) else 0.0

    results.append({"model": "baseline", "param": "", "net_pnl_usd": baseline_net, "max_dd_pct": add_equity("baseline", baseline_pnls), "win_rate": (baseline_pnls > 0).mean(), "profit_factor": _pf(pd.Series(baseline_pnls)), "stop_out_regret_pct": 0.0, "n_trades": n_ep, "avg_fee_bps": taker_fee_bps, "avg_slippage_bps": base_slip})

    fixed_k_grid = [1.0, 1.5, 2.0, 2.5]
    trailing_k_grid = [1.5, 2.0, 2.5]
    be_t_grid = [1.0, 1.5, 2.0]
    time_h_grid = [6, 12, 24, 36]

    def _slippage_for_episode(bars_1h: pd.DataFrame) -> float:
        if not use_spread_proxy or bars_1h.empty:
            return base_slip
        last = bars_1h.iloc[-1]
        sp = _spread_proxy_bps(float(last["high"]), float(last["low"]), float(last["close"]), spread_proxy_alpha, spread_proxy_min_bps, spread_proxy_max_bps)
        return base_slip + sp

    for k_atr in fixed_k_grid:
        pnls = []
        slippage_list: list[float] = []
        for _, ep in episodes_df.iterrows():
            sym = str(ep["symbol"]).upper()
            entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
            exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
            bars_1h = _episode_candles_1h(candles_1h, sym, entry_ts, exit_ts)
            slip_bps = _slippage_for_episode(bars_1h)
            slippage_list.append(slip_bps)
            atr = _atr_at_entry(candles_4h.get(sym), entry_ts, 14) if candles_4h else float("nan")
            pnl, reason, _, _ = _simulate_episode_fixed_stop_intrabar(
                entry_ts, exit_ts, ep["entry_vwap"], ep["exit_vwap"], ep["side"], ep["max_abs_position"],
                atr, k_atr, bars_1h, taker_fee_bps, slip_bps,
            )
            pnls.append(pnl)
            actual_pnl = float(ep["realized_pnl_usd"] or 0) - float(ep["fees_usd"] or 0)
            regret_rows.append({"episode_id": ep["episode_id"], "model": "fixed_stop", "param": f"k={k_atr}", "actual_pnl": actual_pnl, "sim_pnl": pnl, "exit_reason": reason, "winner_stopped_early": (actual_pnl > 0 and pnl < actual_pnl and reason != "actual")})
        pnls = np.array(pnls)
        s = pd.Series(pnls)
        regret_pct = (wins_baseline - (pnls > 0).sum()) / max(1, wins_baseline) * 100.0
        avg_slip = float(np.mean(slippage_list)) if slippage_list else base_slip
        results.append({"model": "fixed_stop", "param": f"k={k_atr}", "net_pnl_usd": float(pnls.sum()), "max_dd_pct": add_equity(f"fixed_stop_{k_atr}ATR", pnls), "win_rate": (pnls > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret_pct, "n_trades": n_ep, "avg_fee_bps": taker_fee_bps, "avg_slippage_bps": avg_slip})

    for k_atr in trailing_k_grid:
        pnls = []
        slippage_list = []
        for _, ep in episodes_df.iterrows():
            sym = str(ep["symbol"]).upper()
            entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
            exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
            bars_1h = _episode_candles_1h(candles_1h, sym, entry_ts, exit_ts)
            slip_bps = _slippage_for_episode(bars_1h)
            slippage_list.append(slip_bps)
            atr = _atr_at_entry(candles_4h.get(sym), entry_ts, 14) if candles_4h else float("nan")
            pnl, reason, _, _ = _simulate_episode_trailing_intrabar(
                entry_ts, exit_ts, ep["entry_vwap"], ep["exit_vwap"], ep["side"], ep["max_abs_position"],
                atr, k_atr, bars_1h, taker_fee_bps, slip_bps,
            )
            pnls.append(pnl)
            actual_pnl = float(ep["realized_pnl_usd"] or 0) - float(ep["fees_usd"] or 0)
            regret_rows.append({"episode_id": ep["episode_id"], "model": "trailing_stop", "param": f"k={k_atr}", "actual_pnl": actual_pnl, "sim_pnl": pnl, "exit_reason": reason, "winner_stopped_early": (actual_pnl > 0 and pnl < actual_pnl and reason != "actual")})
        pnls = np.array(pnls)
        s = pd.Series(pnls)
        regret_pct = (wins_baseline - (pnls > 0).sum()) / max(1, wins_baseline) * 100.0
        avg_slip = float(np.mean(slippage_list)) if slippage_list else base_slip
        results.append({"model": "trailing_stop", "param": f"k={k_atr}", "net_pnl_usd": float(pnls.sum()), "max_dd_pct": add_equity(f"trailing_{k_atr}ATR", pnls), "win_rate": (pnls > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret_pct, "n_trades": n_ep, "avg_fee_bps": taker_fee_bps, "avg_slippage_bps": avg_slip})

    for t_atr in be_t_grid:
        pnls = []
        slippage_list = []
        for _, ep in episodes_df.iterrows():
            sym = str(ep["symbol"]).upper()
            entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
            exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
            bars_1h = _episode_candles_1h(candles_1h, sym, entry_ts, exit_ts)
            slip_bps = _slippage_for_episode(bars_1h)
            slippage_list.append(slip_bps)
            atr = _atr_at_entry(candles_4h.get(sym), entry_ts, 14) if candles_4h else float("nan")
            pnl, reason, _, _ = _simulate_episode_stop_to_be_intrabar(
                entry_ts, exit_ts, ep["entry_vwap"], ep["exit_vwap"], ep["side"], ep["max_abs_position"],
                atr, t_atr, bars_1h, taker_fee_bps, slip_bps,
            )
            pnls.append(pnl)
            actual_pnl = float(ep["realized_pnl_usd"] or 0) - float(ep["fees_usd"] or 0)
            regret_rows.append({"episode_id": ep["episode_id"], "model": "stop_to_be", "param": f"t={t_atr}", "actual_pnl": actual_pnl, "sim_pnl": pnl, "exit_reason": reason, "winner_stopped_early": (actual_pnl > 0 and pnl < actual_pnl and reason != "actual")})
        pnls = np.array(pnls)
        s = pd.Series(pnls)
        regret_pct = (wins_baseline - (pnls > 0).sum()) / max(1, wins_baseline) * 100.0
        avg_slip = float(np.mean(slippage_list)) if slippage_list else base_slip
        results.append({"model": "stop_to_be", "param": f"t={t_atr}", "net_pnl_usd": float(pnls.sum()), "max_dd_pct": add_equity(f"stop_to_be_t{t_atr}", pnls), "win_rate": (pnls > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret_pct, "n_trades": n_ep, "avg_fee_bps": taker_fee_bps, "avg_slippage_bps": avg_slip})

    for max_h in time_h_grid:
        pnls = []
        slippage_list = []
        for _, ep in episodes_df.iterrows():
            sym = str(ep["symbol"]).upper()
            entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
            exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
            bars_1h = _episode_candles_1h(candles_1h, sym, entry_ts, exit_ts)
            slip_bps = _slippage_for_episode(bars_1h)
            slippage_list.append(slip_bps)
            pnl, reason, _, _ = _simulate_episode_time_stop_intrabar(
                entry_ts, exit_ts, ep["entry_vwap"], ep["exit_vwap"], ep["side"], ep["max_abs_position"],
                max_h, bars_1h, taker_fee_bps, slip_bps,
            )
            pnls.append(pnl)
            actual_pnl = float(ep["realized_pnl_usd"] or 0) - float(ep["fees_usd"] or 0)
            regret_rows.append({"episode_id": ep["episode_id"], "model": "time_stop", "param": f"N={max_h}h", "actual_pnl": actual_pnl, "sim_pnl": pnl, "exit_reason": reason, "winner_stopped_early": (actual_pnl > 0 and pnl < actual_pnl and reason != "actual")})
        pnls = np.array(pnls)
        s = pd.Series(pnls)
        regret_pct = (wins_baseline - (pnls > 0).sum()) / max(1, wins_baseline) * 100.0
        avg_slip = float(np.mean(slippage_list)) if slippage_list else base_slip
        results.append({"model": "time_stop", "param": f"N={max_h}h", "net_pnl_usd": float(pnls.sum()), "max_dd_pct": add_equity(f"time_stop_{max_h}h", pnls), "win_rate": (pnls > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret_pct, "n_trades": n_ep, "avg_fee_bps": taker_fee_bps, "avg_slippage_bps": avg_slip})

    summary_df = pd.DataFrame(results)
    equity_df = pd.DataFrame(equity_curves)
    regret_df = pd.DataFrame(regret_rows)
    return summary_df, equity_df, regret_df


def run_counterfactuals(
    trades: pd.DataFrame,
    candles_4h: dict[str, pd.DataFrame],
    assumed_entry_hours: int = 24,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each trade (close), assume entry = assumed_entry_hours before.
    Simulate: fixed stop (1.5 ATR, 2 ATR), trailing (1.5, 2), time exit 6h/12h, stop-to-BE, partial TP.
    Return (counterfactuals_summary, equity_curves).
    """
    if trades.empty:
        summary = pd.DataFrame(columns=["model", "param", "net_pnl_usd", "max_dd_pct", "win_rate", "profit_factor", "stop_out_regret_pct", "n_trades"])
        equity = pd.DataFrame(columns=["model", "date", "equity"])
        return summary, equity

    baseline_pnls = trades["realized_pnl_usd"].fillna(0) - trades["fee_usd"].fillna(0)
    baseline_net = float(baseline_pnls.sum())
    n_trades = len(trades)

    results: list[dict[str, Any]] = []
    equity_curves: list[dict[str, Any]] = []

    def add_equity(name: str, pnls: pd.Series, start_equity: float = 10000.0) -> float:
        cum = pnls.cumsum()
        eq = start_equity + cum
        peak = eq.cummax()
        dd = (eq / peak - 1.0) * 100.0
        for i in range(len(eq)):
            ts = trades["timestamp_utc"].iloc[i] if i < len(trades) else None
            v = eq.iloc[i]
            c = cum.iloc[i] if i < len(cum) else 0
            equity_curves.append({"model": name, "date": pd.Timestamp(ts).date() if ts is not None and hasattr(pd.Timestamp(ts), "date") else str(ts), "equity": float(v), "cum_pnl": float(c)})
        return float(dd.min()) if not dd.empty else 0.0

    start_equity = 10000.0
    results.append({"model": "baseline", "param": "", "net_pnl_usd": baseline_net, "max_dd_pct": add_equity("baseline", baseline_pnls, start_equity), "win_rate": (baseline_pnls > 0).mean() if n_trades else 0, "profit_factor": _pf(baseline_pnls), "stop_out_regret_pct": 0.0, "n_trades": n_trades})
    wins_baseline = (baseline_pnls > 0).sum()

    sim_pnls_list: list[tuple[str, str, pd.Series]] = []
    for k_atr in [1.5, 2.0]:
        pnls = []
        for i, row in trades.iterrows():
            sym = str(row.get("symbol", "")).upper()
            exit_ts = pd.Timestamp(row["timestamp_utc"]).tz_localize(UTC) if row["timestamp_utc"].tzinfo is None else pd.Timestamp(row["timestamp_utc"])
            entry_ts = exit_ts - pd.Timedelta(hours=assumed_entry_hours)
            entry_price = row.get("price")
            exit_price = row.get("price")
            side = row.get("side")
            qty = float(row.get("qty") or 0)
            cand = candles_4h.get(sym)
            atr_val = np.nan
            if cand is not None and len(cand) >= 14 and "high" in cand.columns:
                atr_val = atr_series(cand["high"], cand["low"], cand["close"], 14).iloc[-1]
            pnl, _ = simulate_fixed_stop(entry_ts, exit_ts, float(entry_price or 0), float(exit_price or 0), side, qty, atr_val, k_atr)
            fee = float(row.get("fee_usd") or 0)
            pnls.append(pnl - fee)
        s = pd.Series(pnls)
        sim_pnls_list.append((f"fixed_stop_{k_atr}ATR", f"k={k_atr}", s))
        regret = (wins_baseline - (s > 0).sum()) / max(1, wins_baseline) * 100.0
        results.append({"model": "fixed_stop", "param": f"k={k_atr}", "net_pnl_usd": float(s.sum()), "max_dd_pct": add_equity(f"fixed_stop_{k_atr}ATR", s, start_equity), "win_rate": (s > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret, "n_trades": n_trades})

    for k_atr in [1.5, 2.0]:
        pnls = []
        for i, row in trades.iterrows():
            sym = str(row.get("symbol", "")).upper()
            exit_ts = pd.Timestamp(row["timestamp_utc"]).tz_localize(UTC) if row["timestamp_utc"].tzinfo is None else pd.Timestamp(row["timestamp_utc"])
            entry_ts = exit_ts - pd.Timedelta(hours=assumed_entry_hours)
            entry_price = row.get("price")
            exit_price = row.get("price")
            side = row.get("side")
            qty = float(row.get("qty") or 0)
            cand = candles_4h.get(sym)
            atr_val = np.nan
            high_s = pd.Series(dtype=float)
            low_s = pd.Series(dtype=float)
            if cand is not None and "ts_open_utc" in cand.columns and "high" in cand.columns:
                cand = cand.set_index("ts_open_utc") if "ts_open_utc" in cand.columns else cand
                mask = (cand.index >= entry_ts) & (cand.index <= exit_ts)
                sub = cand.loc[mask]
                if not sub.empty:
                    high_s = sub["high"]
                    low_s = sub["low"]
                    if len(sub) >= 14:
                        atr_val = atr_series(sub["high"], sub["low"], sub["close"], 14).iloc[-1]
            pnl, _ = simulate_trailing_stop(entry_ts, exit_ts, float(entry_price or 0), float(exit_price or 0), side, qty, high_s, low_s, atr_val, k_atr)
            fee = float(row.get("fee_usd") or 0)
            pnls.append(pnl - fee)
        s = pd.Series(pnls)
        regret = (wins_baseline - (s > 0).sum()) / max(1, wins_baseline) * 100.0
        results.append({"model": "trailing_stop", "param": f"k={k_atr}", "net_pnl_usd": float(s.sum()), "max_dd_pct": add_equity(f"trailing_{k_atr}ATR", s, start_equity), "win_rate": (s > 0).mean(), "profit_factor": _pf(s), "stop_out_regret_pct": regret, "n_trades": n_trades})

    summary_df = pd.DataFrame(results)
    equity_df = pd.DataFrame(equity_curves)
    return summary_df, equity_df


def _pf(series: pd.Series) -> float:
    gross_profit = series[series > 0].sum()
    gross_loss = series[series < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / abs(gross_loss))


def _load_config(config_path: Path):
    from src.config import load_config
    return load_config(config_path)


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description="Phase D: Counterfactual exit simulations (read-only).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--warmup_days", type=int, default=120)
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d")
    parser.add_argument("--category", type=str, default=None, help="Bybit category (default: linear; enforced linear for research)")
    parser.add_argument("--include-open-episodes", action="store_true", help="Run counterfactuals on open episodes too (default: closed only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    if not out_dir.exists():
        raise FileNotFoundError(f"Run trade_forensics_30d.py first. Missing {out_dir}")

    config_path = _PACKAGE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        config_path = _REPO_ROOT / "bybit_xsreversal" / "config" / "config.yaml"
    cfg = _load_config(config_path) if config_path.exists() else None
    category = _ensure_linear_category(cfg, args.category)

    taker_fee_bps = 6.0
    slippage_bps = 2.0
    base_slippage_bps: float | None = None
    spread_proxy_alpha = 0.25
    spread_proxy_min_bps = 1.0
    spread_proxy_max_bps = 50.0
    if cfg:
        taker_fee_bps, _, base_slip, spread_proxy_alpha, spread_proxy_min_bps, spread_proxy_max_bps = _get_research_costs(cfg)
        base_slippage_bps = base_slip
        slippage_bps = base_slip

    summary_meta = _load_research_summary(out_dir)
    if summary_meta:
        logger.info(
            "Total bot activity (all fills): {} fills, {:.2f} USD realized. Use this for performance; counterfactuals below are episode-based.",
            summary_meta.get("n_fills", 0), summary_meta.get("total_realized_pnl_usd", 0),
        )

    episodes_df = _load_episodes(out_dir)
    if episodes_df is not None and not episodes_df.empty and "closed" in episodes_df.columns:
        closed_only = episodes_df[episodes_df["closed"] == True]
        n_open = (episodes_df["closed"] == False).sum()
        if args.include_open_episodes:
            logger.info("Using all episodes: {} closed, {} open", len(closed_only), n_open)
        else:
            episodes_df = closed_only.reset_index(drop=True)
            if n_open > 0:
                logger.info("Using closed episodes only ({}). {} open episodes excluded; use --include-open-episodes to include.", len(episodes_df), n_open)

    candles_1h = _load_candles(out_dir, "1H")
    candles_4h = _load_candles(out_dir, "4H")
    if not candles_4h:
        logger.warning("No 4H candles; counterfactuals will use fallback (no ATR).")

    if episodes_df is not None and not episodes_df.empty and (candles_1h or candles_4h):
        summary_df, equity_df, regret_df = run_counterfactuals_episodes(
            episodes_df, candles_1h or candles_4h, candles_4h,
            taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps,
            base_slippage_bps=base_slippage_bps, spread_proxy_alpha=spread_proxy_alpha,
            spread_proxy_min_bps=spread_proxy_min_bps, spread_proxy_max_bps=spread_proxy_max_bps,
        )
        summary_df.to_csv(out_dir / "counterfactuals_summary.csv", index=False)
        equity_df.to_csv(out_dir / "equity_curves.csv", index=False)
        regret_df.to_csv(out_dir / "stopout_regret_detail.csv", index=False)
        logger.info("Wrote counterfactuals_summary.csv, equity_curves.csv, stopout_regret_detail.csv (episode-based, intrabar 1H)")
        for _, ep in episodes_df.sample(n=min(2, len(episodes_df)), replace=False).iterrows():
            sym = str(ep["symbol"]).upper()
            entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
            exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
            bars_1h = _episode_candles_1h(candles_1h or candles_4h, sym, entry_ts, exit_ts)
            atr = _atr_at_entry(candles_4h.get(sym) if candles_4h else None, entry_ts, 14)
            pnl, reason, trigger_ts, trigger_row = _simulate_episode_fixed_stop_intrabar(
                entry_ts, exit_ts, ep["entry_vwap"], ep["exit_vwap"], ep["side"], ep["max_abs_position"],
                atr, 1.5, bars_1h, taker_fee_bps, slippage_bps,
            )
            logger.info("Validation episode {}: exit_reason={} trigger_ts={} first_trigger_candle={}", ep["episode_id"], reason, trigger_ts, trigger_row)
    else:
        trades = _load_trades(out_dir)
        summary_df, equity_df = run_counterfactuals(trades, candles_4h, assumed_entry_hours=24)
        summary_df.to_csv(out_dir / "counterfactuals_summary.csv", index=False)
        equity_df.to_csv(out_dir / "equity_curves.csv", index=False)
        logger.info("Wrote counterfactuals_summary.csv and equity_curves.csv (trade-based, 24h assumed entry)")


if __name__ == "__main__":
    main()