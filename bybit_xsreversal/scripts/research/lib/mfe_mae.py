ubuntu@instance-20250817-2147:~/cross-sectional-short-term-reversal/bybit_xsreversal/scripts/research/lib$ cat mfe_mae.py
"""
MFE/MAE over real episode windows using 1H (or 4H fallback) candle HIGH/LOW.

LONG:
  MAE = min(low) - entry_vwap  (adverse = price went down)
  MFE = max(high) - entry_vwap (favorable = price went up)
SHORT:
  MAE = entry_vwap - max(high) (adverse = price went up)
  MFE = entry_vwap - min(low) (favorable = price went down)

Also: time_to_mae, time_to_mfe, mfe_giveback = MFE - (exit_vwap - entry_vwap) for longs, etc.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "ts_open_utc" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["ts_open_utc"]):
        df = df.copy()
        df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
    return df


def mfe_mae_for_episode(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_vwap: float,
    exit_vwap: float,
    side: str,
    max_abs_position: float,
    candles: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute MFE, MAE (price then USD), time_to_mae, time_to_mfe, mfe_giveback.
    Candles: ts_open_utc, high, low, close.
    """
    out = {
        "mfe_price": None,
        "mae_price": None,
        "mfe_usd": None,
        "mae_usd": None,
        "time_to_mae_h": None,
        "time_to_mfe_h": None,
        "mfe_giveback_price": None,
        "mfe_giveback_usd": None,
    }
    candles = _ensure_ts(candles)
    if candles.empty or "high" not in candles.columns or "low" not in candles.columns:
        return out
    entry_ts = pd.Timestamp(entry_ts).tz_localize("UTC") if getattr(entry_ts, "tzinfo", None) is None else pd.Timestamp(entry_ts)
    exit_ts = pd.Timestamp(exit_ts).tz_localize("UTC") if getattr(exit_ts, "tzinfo", None) is None else pd.Timestamp(exit_ts)
    mask = (candles["ts_open_utc"] >= entry_ts) & (candles["ts_open_utc"] <= exit_ts)
    sub = candles.loc[mask]
    if sub.empty:
        return out
    entry_vwap = float(entry_vwap)
    exit_vwap = float(exit_vwap)
    qty = float(max_abs_position)
    is_long = str(side).strip().lower() == "long"
    if is_long:
        mfe_price = float(sub["high"].max() - entry_vwap)
        mae_price = float(sub["low"].min() - entry_vwap)
        mae_ts = sub.loc[sub["low"].idxmin(), "ts_open_utc"]
        mfe_ts = sub.loc[sub["high"].idxmax(), "ts_open_utc"]
    else:
        mfe_price = float(entry_vwap - sub["low"].min())
        mae_price = float(entry_vwap - sub["high"].max())
        mae_ts = sub.loc[sub["high"].idxmax(), "ts_open_utc"]
        mfe_ts = sub.loc[sub["low"].idxmin(), "ts_open_utc"]
    out["mfe_price"] = mfe_price
    out["mae_price"] = mae_price
    out["mfe_usd"] = mfe_price * qty
    out["mae_usd"] = mae_price * qty
    try:
        out["time_to_mae_h"] = (pd.Timestamp(mae_ts) - entry_ts).total_seconds() / 3600.0
        out["time_to_mfe_h"] = (pd.Timestamp(mfe_ts) - entry_ts).total_seconds() / 3600.0
    except Exception:
        pass
    realized_price = exit_vwap - entry_vwap if is_long else entry_vwap - exit_vwap
    out["mfe_giveback_price"] = mfe_price - realized_price if mfe_price > realized_price else 0.0
    out["mfe_giveback_usd"] = out["mfe_giveback_price"] * qty
    return out


def compute_mfe_mae_by_episode(
    episodes_df: pd.DataFrame,
    candles_1h: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """For each episode, compute MFE/MAE from 1H candles; fallback 4H if 1H missing."""
    if episodes_df.empty:
        return pd.DataFrame(columns=[
            "episode_id", "symbol", "side", "entry_ts", "exit_ts", "entry_vwap", "exit_vwap",
            "mfe_usd", "mae_usd", "time_to_mfe_h", "time_to_mae_h", "mfe_giveback_usd",
        ])
    rows = []
    for _, ep in episodes_df.iterrows():
        sym = str(ep.get("symbol", "")).upper()
        cand = candles_1h.get(sym)
        if (cand is None or cand.empty) and candles_4h:
            cand = candles_4h.get(sym)
        res = mfe_mae_for_episode(
            ep["entry_ts"],
            ep["exit_ts"],
            ep["entry_vwap"],
            ep["exit_vwap"],
            ep["side"],
            ep.get("max_abs_position", 0) or 0,
            cand if cand is not None else pd.DataFrame(),
        )
        rows.append({
            "episode_id": ep["episode_id"],
            "symbol": ep["symbol"],
            "side": ep["side"],
            "entry_ts": ep["entry_ts"],
            "exit_ts": ep["exit_ts"],
            "entry_vwap": ep["entry_vwap"],
            "exit_vwap": ep["exit_vwap"],
            "mfe_usd": res["mfe_usd"],
            "mae_usd": res["mae_usd"],
            "time_to_mfe_h": res["time_to_mfe_h"],
            "time_to_mae_h": res["time_to_mae_h"],
            "mfe_giveback_usd": res["mfe_giveback_usd"],
        })
    return pd.DataFrame(rows)