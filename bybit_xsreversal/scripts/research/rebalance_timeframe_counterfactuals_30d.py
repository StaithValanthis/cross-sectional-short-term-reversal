#!/usr/bin/env python3
"""
Rebalance frequency + signal timeframe counterfactuals (24h vs 12h vs 6h; 1D vs 4H).
Read-only. Uses candles from outputs/research_30d/candles (run trade_forensics first).
Run from repo root:
  python3 bybit_xsreversal/scripts/research/rebalance_timeframe_counterfactuals_30d.py --days 30 --warmup_days 120

Outputs:
  - outputs/research_30d/rebalance_timeframe_summary.csv
  - outputs/research_30d/rebalance_timeframe_equity_curves.csv
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
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None


def _ensure_deps() -> None:
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy required. pip install pandas numpy")


def _ensure_linear_category(cfg, args_category: str | None) -> str:
    config_cat = getattr(getattr(cfg, "exchange", None), "category", None) if cfg else None
    config_cat = config_cat or "linear"
    cat = (args_category or config_cat or "linear").strip().lower()
    if cat != "linear":
        logger.warning("Research pipeline is for USDT linear perps only; forcing category='linear' (was {})", cat)
        return "linear"
    return "linear"


def _load_config(config_path: Path):
    from src.config import load_config
    return load_config(config_path)


def _load_candles(out_dir: Path, interval: str) -> dict[str, pd.DataFrame]:
    base = out_dir / "candles" / interval
    result = {}
    if not base.exists():
        return result
    for p in base.iterdir():
        if p.suffix in (".parquet", ".csv"):
            sym = p.stem
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            if "ts_open_utc" in df.columns:
                df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                df = df.sort_values("ts_open_utc").reset_index(drop=True)
            result[sym] = df
    return result


def _bars_per_day(interval: str) -> int:
    if interval == "1D":
        return 1
    if interval == "4H":
        return 6
    if interval == "1H":
        return 24
    return 1


def _rebalance_timestamps(
    start_dt: datetime,
    end_dt: datetime,
    freq_hours: int,
) -> list[datetime]:
    """Generate rebalance timestamps at freq_hours (24, 12, or 6)."""
    out = []
    t = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if freq_hours == 24:
        while t <= end_dt:
            out.append(t)
            t += timedelta(days=1)
    else:
        while t <= end_dt:
            out.append(t)
            t += timedelta(hours=freq_hours)
    return [ts for ts in out if start_dt <= ts <= end_dt]


def _research_target_engine(
    candles: dict[str, pd.DataFrame],
    asof: datetime,
    lookback_bars: int,
    vol_lookback_bars: int,
    long_quantile: float,
    short_quantile: float,
    target_gross_leverage: float,
    max_leverage_per_symbol: float,
    equity_usd: float,
    current_weights: dict[str, float],
    rebalance_fraction: float,
    min_weight_change_bps: float,
    max_notional_per_symbol: float,
    min_notional_per_symbol: float,
) -> dict[str, float]:
    """
    Simplified xs_reversal target engine: return notionals_usd.
    Candles: symbol -> df with ts_open_utc, close. asof = last bar time to use.
    """
    returns: dict[str, float] = {}
    vol: dict[str, float] = {}
    for sym, df in candles.items():
        if df.empty or "close" not in df.columns or "ts_open_utc" not in df.columns:
            continue
        df = df[df["ts_open_utc"] <= asof].sort_values("ts_open_utc").tail(max(lookback_bars, vol_lookback_bars) + 10)
        if len(df) < max(lookback_bars + 1, vol_lookback_bars + 2):
            continue
        close = df["close"].astype(float)
        ret = float(close.iloc[-1] / close.iloc[-1 - lookback_bars] - 1.0)
        r = close.pct_change().dropna()
        v = r.rolling(vol_lookback_bars, min_periods=vol_lookback_bars).std(ddof=0).iloc[-1]
        if not np.isfinite(ret) or not np.isfinite(v) or v <= 0:
            continue
        returns[sym] = ret
        vol[sym] = float(v)
    universe = sorted(returns.keys())
    if len(universe) < 5:
        return {}
    ranked = sorted(universe, key=lambda s: returns[s])
    n = len(ranked)
    k_long = max(1, int(np.floor(n * long_quantile)))
    k_short = max(1, int(np.floor(n * short_quantile)))
    selected_longs = ranked[:k_long]
    selected_shorts = ranked[-k_short:]
    long_scores = {s: 1.0 / vol[s] for s in selected_longs}
    short_scores = {s: 1.0 / vol[s] for s in selected_shorts}

    def _norm(w: dict, target: float) -> dict:
        s = sum(abs(v) for v in w.values())
        if s <= 0:
            return w
        return {k: v * target / s for k, v in w.items()}

    def _cap(w: dict, cap_abs: float, target: float) -> dict:
        w = {k: np.sign(v) * min(abs(v), cap_abs) for k, v in w.items()}
        return _norm(w, target)

    w_long = _norm(long_scores, target_gross_leverage / 2.0)
    w_short = _norm(short_scores, target_gross_leverage / 2.0)
    w_long = _cap(w_long, max_leverage_per_symbol, target_gross_leverage / 2.0)
    w_short = _cap(w_short, max_leverage_per_symbol, target_gross_leverage / 2.0)
    raw_weights = dict(w_long)
    raw_weights.update({k: -abs(v) for k, v in w_short.items()})

    thresh = min_weight_change_bps / 10_000.0
    union = set(current_weights) | set(raw_weights)
    blended = {}
    for s in union:
        cur = current_weights.get(s, 0.0)
        tgt = raw_weights.get(s, 0.0)
        new = cur + rebalance_fraction * (tgt - cur)
        if abs(new - cur) < thresh:
            new = cur
        if abs(new) > 1e-10:
            blended[s] = new
    gross = sum(abs(v) for v in blended.values())
    if gross > target_gross_leverage and gross > 0:
        scale = target_gross_leverage / gross
        blended = {k: v * scale for k, v in blended.items()}

    notionals = {}
    for sym, w in blended.items():
        notional = w * equity_usd
        if abs(notional) < min_notional_per_symbol:
            continue
        if abs(notional) > max_notional_per_symbol:
            notional = np.sign(notional) * max_notional_per_symbol
        notionals[sym] = float(notional)
    return notionals


def _price_at(candles: dict[str, pd.DataFrame], sym: str, ts: datetime) -> float:
    """Last close at or before ts for symbol."""
    df = candles.get(sym)
    if df is None or df.empty or "close" not in df.columns or "ts_open_utc" not in df.columns:
        return 0.0
    df = df[df["ts_open_utc"] <= ts].sort_values("ts_open_utc")
    if df.empty:
        return 0.0
    return float(df.iloc[-1]["close"])


def _next_candle_open(candles: dict[str, pd.DataFrame], asof: datetime, interval: str) -> dict[str, float]:
    """Price at next candle open after asof (fill model)."""
    out = {}
    for sym, df in candles.items():
        if df.empty or "open" not in df.columns or "ts_open_utc" not in df.columns:
            continue
        df = df[df["ts_open_utc"] > asof].head(1)
        if df.empty:
            continue
        out[sym] = float(df.iloc[0]["open"])
    return out


def _next_candle_vwap(candles: dict[str, pd.DataFrame], asof: datetime) -> dict[str, float]:
    """VWAP of next candle after asof: (open+high+low+close)/4."""
    out = {}
    for sym, df in candles.items():
        if df.empty or "open" not in df.columns or "ts_open_utc" not in df.columns:
            continue
        df = df[df["ts_open_utc"] > asof].head(1)
        if df.empty:
            continue
        r = df.iloc[0]
        vwap = (float(r["open"]) + float(r["high"]) + float(r["low"]) + float(r["close"])) / 4.0
        out[sym] = vwap
    return out


def _spread_proxy_bps(high: float, low: float, close: float, alpha: float, min_bps: float, max_bps: float) -> float:
    if close is None or close <= 0:
        return min_bps
    raw = (high - low) / close * 10000.0 * alpha
    return float(np.clip(raw, min_bps, max_bps))


def _next_candle_fill_and_slippage(
    candles: dict[str, pd.DataFrame],
    asof: datetime,
    fill_model: str,
    base_slippage_bps: float,
    spread_alpha: float,
    spread_min_bps: float,
    spread_max_bps: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (fill_prices, slippage_bps_per_sym). fill_model in ('open', 'vwap')."""
    prices = {}
    slippage_bps = {}
    for sym, df in candles.items():
        if df.empty or "open" not in df.columns or "ts_open_utc" not in df.columns:
            continue
        df = df[df["ts_open_utc"] > asof].head(1)
        if df.empty:
            continue
        r = df.iloc[0]
        o, h, l_, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        if fill_model == "vwap":
            prices[sym] = (o + h + l_ + c) / 4.0
        else:
            prices[sym] = o
        sp = _spread_proxy_bps(h, l_, c, spread_alpha, spread_min_bps, spread_max_bps)
        slippage_bps[sym] = base_slippage_bps + sp
    return prices, slippage_bps


def run_rebalance_timeframe_sim(
    candles_1d: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame],
    config,
    eval_days: int,
    warmup_days: int,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    base_slippage_bps: float | None = None,
    spread_proxy_alpha: float = 0.25,
    spread_proxy_min_bps: float = 1.0,
    spread_proxy_max_bps: float = 50.0,
    fill_model: str = "open",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run variants: rebalance 24h/12h/6h x signal 1D/4H.
    Fill at next candle open (fill_model=open) or VWAP (fill_model=vwap).
    Slippage = base_slippage_bps + spread_proxy from fill bar. Apply fees + slippage.
    Returns (summary_df, equity_curves_df).
    """
    base_slip = base_slippage_bps if base_slippage_bps is not None else slippage_bps
    fill_model = (fill_model or "open").strip().lower()
    if fill_model not in ("open", "vwap"):
        fill_model = "open"
    end_dt = datetime.now(tz=UTC)
    start_dt = end_dt - timedelta(days=eval_days)
    warmup_start = start_dt - timedelta(days=warmup_days)
    start_equity = 10000.0

    lookback_days = int(getattr(config.signal, "lookback_days", 1) or 1)
    vol_lookback_days = int(getattr(config.sizing, "vol_lookback_days", 14) or 14)
    long_q = float(getattr(config.signal, "long_quantile", 0.2) or 0.2)
    short_q = float(getattr(config.signal, "short_quantile", 0.2) or 0.2)
    gross = float(getattr(config.sizing, "target_gross_leverage", 5.0) or 5.0)
    max_lev = float(getattr(config.sizing, "max_leverage_per_symbol", 0.2) or 0.2)
    frac = float(getattr(config.rebalance, "rebalance_fraction", 0.36) or 0.36)
    min_bps = float(getattr(config.rebalance, "min_weight_change_bps", 400) or 400)
    max_notional = float(getattr(config.sizing, "max_notional_per_symbol", 25000) or 25000)
    min_notional = float(getattr(config.sizing, "min_notional_per_symbol", 5) or 5)

    summary_rows = []
    equity_rows = []

    for rebalance_hours in [24, 12, 6]:
        for signal_tf in ["1D", "4H"]:
            candles = candles_1d if signal_tf == "1D" else candles_4h
            if not candles:
                continue
            bars_per_day = _bars_per_day(signal_tf)
            lookback_bars = max(1, lookback_days * bars_per_day)
            vol_lookback_bars = max(2, vol_lookback_days * bars_per_day)
            timestamps = _rebalance_timestamps(start_dt, end_dt, rebalance_hours)
            if not timestamps:
                continue

            cash = start_equity
            positions: dict[str, float] = {}
            current_weights: dict[str, float] = {}
            cum_fees = 0.0
            total_notional = 0.0
            n_trades = 0
            variant = f"rebal_{rebalance_hours}h_signal_{signal_tf}"
            equity_curve = [{"variant": variant, "date": start_dt.date(), "equity": start_equity, "cum_pnl": 0.0}]

            for ts in timestamps:
                equity_before = cash + sum(positions.get(s, 0) * _price_at(candles, s, ts) for s in positions)
                notionals = _research_target_engine(
                    candles,
                    ts,
                    lookback_bars,
                    vol_lookback_bars,
                    long_q,
                    short_q,
                    gross,
                    max_lev,
                    equity_before if equity_before > 0 else start_equity,
                    current_weights,
                    frac,
                    min_bps,
                    max_notional,
                    min_notional,
                )
                prices, slippage_bps_per_sym = _next_candle_fill_and_slippage(
                    candles, ts, fill_model, base_slip, spread_proxy_alpha, spread_proxy_min_bps, spread_proxy_max_bps,
                )
                if not prices:
                    equity_curve.append({"variant": variant, "date": ts.date() if hasattr(ts, "date") else ts, "equity": equity_before, "cum_pnl": equity_before - start_equity})
                    continue
                target_positions: dict[str, float] = {}
                for sym, notional in notionals.items():
                    px = prices.get(sym)
                    if px is not None and px > 0:
                        target_positions[sym] = notional / px
                for sym in set(positions) | set(target_positions):
                    cur = positions.get(sym, 0.0)
                    tgt = target_positions.get(sym, 0.0)
                    delta = tgt - cur
                    if abs(delta) < 1e-12:
                        continue
                    px = prices.get(sym)
                    if px is None or px <= 0:
                        continue
                    notional = abs(delta * px)
                    total_notional += notional
                    slip_bps = slippage_bps_per_sym.get(sym, base_slip)
                    cost = notional * (taker_fee_bps + slip_bps) / 10000.0
                    cum_fees += cost
                    n_trades += 1
                    cash -= delta * px + cost
                    positions[sym] = tgt
                for sym in list(positions):
                    if abs(positions[sym]) < 1e-12:
                        del positions[sym]
                equity_after = cash + sum(positions.get(s, 0) * prices.get(s, 0) for s in positions if s in prices)
                current_weights = {s: (positions[s] * prices.get(s, 0)) / equity_after if equity_after > 0 else 0 for s in positions if s in prices}
                equity_curve.append({"variant": variant, "date": ts.date() if hasattr(ts, "date") else ts, "equity": equity_after, "cum_pnl": equity_after - start_equity})
            if equity_curve:
                net_pnl = equity_curve[-1]["equity"] - start_equity
                eq_series = [e["equity"] for e in equity_curve]
                peak = eq_series[0]
                max_dd = 0.0
                for eq in eq_series:
                    peak = max(peak, eq)
                    if peak > 0:
                        max_dd = min(max_dd, (eq / peak - 1.0) * 100.0)
                avg_fee_bps = taker_fee_bps
                avg_slippage_bps = (cum_fees / total_notional * 10000.0 - taker_fee_bps) if total_notional > 0 else base_slip
                summary_rows.append({
                    "variant": variant,
                    "rebalance_hours": rebalance_hours,
                    "signal_tf": signal_tf,
                    "net_pnl_usd": net_pnl,
                    "max_dd_pct": max_dd,
                    "fees_usd": cum_fees,
                    "n_trades": n_trades,
                    "final_equity": equity_curve[-1]["equity"],
                    "avg_fee_bps": avg_fee_bps,
                    "avg_slippage_bps": round(avg_slippage_bps, 2),
                })
                equity_rows.extend(equity_curve)

    summary_df = pd.DataFrame(summary_rows)
    equity_df = pd.DataFrame(equity_rows)
    return summary_df, equity_df


def score_config(
    candles_1d: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame],
    param_overrides: dict[str, Any],
    eval_days: int = 30,
    warmup_days: int = 120,
    taker_fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> tuple[float, float, float, int]:
    """
    Run 24h rebalance / 1D signal sim with given param overrides. Used as scoring function for param search.
    param_overrides: lookback_days, long_quantile, short_quantile, vol_lookback_days,
      rebalance_fraction, min_weight_change_bps, target_gross_leverage (optional).
    Returns (net_pnl_usd, max_dd_pct, fees_usd, n_trades).
    """
    class Cfg:
        pass
    cfg = Cfg()
    cfg.signal = type("S", (), {"lookback_days": param_overrides.get("lookback_days", 1), "long_quantile": param_overrides.get("long_quantile", 0.2), "short_quantile": param_overrides.get("short_quantile", 0.2)})()
    cfg.sizing = type("Sz", (), {"vol_lookback_days": param_overrides.get("vol_lookback_days", 14), "target_gross_leverage": param_overrides.get("target_gross_leverage", 5.0), "max_leverage_per_symbol": 0.2, "max_notional_per_symbol": 25000, "min_notional_per_symbol": 5})()
    cfg.rebalance = type("R", (), {"rebalance_fraction": param_overrides.get("rebalance_fraction", 0.36), "min_weight_change_bps": param_overrides.get("min_weight_change_bps", 400)})()
    summary, _ = run_rebalance_timeframe_sim(candles_1d, candles_4h, cfg, eval_days=eval_days, warmup_days=warmup_days, taker_fee_bps=taker_fee_bps, slippage_bps=slippage_bps)
    row = summary[summary["variant"] == "rebal_24h_signal_1D"]
    if row.empty:
        return -1e9, 0.0, 0.0, 0
    r = row.iloc[0]
    return float(r["net_pnl_usd"]), float(r["max_dd_pct"]), float(r["fees_usd"]), int(r["n_trades"])


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description="Rebalance + timeframe counterfactuals (read-only).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--warmup_days", type=int, default=120)
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--fill_model", type=str, default="open", choices=["open", "vwap"], help="Fill at next candle open or VWAP")
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
    base_slippage_bps = 2.0
    spread_proxy_alpha = 0.25
    spread_proxy_min_bps = 1.0
    spread_proxy_max_bps = 50.0
    if cfg:
        rc = getattr(cfg, "research_costs", None)
        bt = getattr(cfg, "backtest", None)
        taker_fee_bps = float(getattr(rc, "taker_fee_bps", None) or getattr(bt, "taker_fee_bps", 6.0) or 6.0)
        base_slippage_bps = float(getattr(rc, "base_slippage_bps", None) or getattr(bt, "slippage_bps", 2.0) or 2.0)
        spread_proxy_alpha = float(getattr(rc, "spread_proxy_alpha", 0.25) or 0.25)
        spread_proxy_min_bps = float(getattr(rc, "spread_proxy_min_bps", 1.0) or 1.0)
        spread_proxy_max_bps = float(getattr(rc, "spread_proxy_max_bps", 50.0) or 50.0)

    candles_1d = _load_candles(out_dir, "1D")
    candles_4h = _load_candles(out_dir, "4H")
    if not candles_1d and not candles_4h:
        logger.warning("No 1D or 4H candles; run trade_forensics_30d.py first with candle download.")
        summary_df = pd.DataFrame(columns=["variant", "rebalance_hours", "signal_tf", "net_pnl_usd", "max_dd_pct", "fees_usd", "n_trades", "final_equity", "avg_fee_bps", "avg_slippage_bps"])
        equity_df = pd.DataFrame(columns=["variant", "date", "equity", "cum_pnl"])
    else:
        summary_df, equity_df = run_rebalance_timeframe_sim(
            candles_1d, candles_4h, cfg or type("Cfg", (), {"signal": type("S", (), {"lookback_days": 1, "long_quantile": 0.2, "short_quantile": 0.2})(), "sizing": type("Sz", (), {"vol_lookback_days": 14, "target_gross_leverage": 5.0, "max_leverage_per_symbol": 0.2, "max_notional_per_symbol": 25000, "min_notional_per_symbol": 5})(), "rebalance": type("R", (), {"rebalance_fraction": 0.36, "min_weight_change_bps": 400})()})(),
            eval_days=args.days,
            warmup_days=args.warmup_days,
            taker_fee_bps=taker_fee_bps,
            base_slippage_bps=base_slippage_bps,
            spread_proxy_alpha=spread_proxy_alpha,
            spread_proxy_min_bps=spread_proxy_min_bps,
            spread_proxy_max_bps=spread_proxy_max_bps,
            fill_model=args.fill_model,
        )
    summary_df.to_csv(out_dir / "rebalance_timeframe_summary.csv", index=False)
    equity_df.to_csv(out_dir / "rebalance_timeframe_equity_curves.csv", index=False)
    logger.info("Wrote rebalance_timeframe_summary.csv and rebalance_timeframe_equity_curves.csv to {}", out_dir)


if __name__ == "__main__":
    main()