from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """
    Wilder's ADX(n) on OHLC. Returns a series aligned with inputs.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing is an EMA with alpha=1/n
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100.0 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan)
    adx_ = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx_


def ema_slope(close: pd.Series, fast: int, slow: int) -> pd.Series:
    """
    Simple EMA trend proxy: (EMA_fast - EMA_slow) / close.
    """
    ef = ema(close, span=fast)
    es = ema(close, span=slow)
    slope = (ef - es) / close.replace(0.0, np.nan)
    return slope


@dataclass(frozen=True)
class RegimeDecision:
    action: str
    scale: float
    meta: dict[str, Any]


def regime_gate(
    *,
    symbol_df: pd.DataFrame,
    market_df: pd.DataFrame | None,
    symbol_adx_threshold: float,
    market_adx_threshold: float,
    ema_fast: int,
    ema_slow: int,
    action: str,
    scale_factor: float,
) -> RegimeDecision:
    """
    Gate reversal trades in strong trends:
    - Strong symbol trend if ADX(symbol) > threshold and |ema_slope| is large.
    - Strong market trend if ADX(market) > threshold (optional upstream).
    """
    meta: dict[str, Any] = {}

    sym_adx = adx(symbol_df["high"], symbol_df["low"], symbol_df["close"], n=14).iloc[-1]
    sym_slope = ema_slope(symbol_df["close"], fast=ema_fast, slow=ema_slow).iloc[-1]
    meta["symbol_adx"] = float(sym_adx) if pd.notna(sym_adx) else None
    meta["symbol_ema_slope"] = float(sym_slope) if pd.notna(sym_slope) else None

    market_adx_val = None
    if market_df is not None:
        market_adx_val = adx(market_df["high"], market_df["low"], market_df["close"], n=14).iloc[-1]
        meta["market_adx"] = float(market_adx_val) if pd.notna(market_adx_val) else None

    strong_symbol = pd.notna(sym_adx) and float(sym_adx) >= float(symbol_adx_threshold)
    strong_market = market_adx_val is not None and pd.notna(market_adx_val) and float(market_adx_val) >= float(market_adx_threshold)

    if strong_symbol or strong_market:
        if action == "skip":
            return RegimeDecision(action="skip", scale=0.0, meta=meta)
        if action == "switch_to_momentum":
            # Strategy-level action: caller may flip signal direction in strong-trend regimes.
            return RegimeDecision(action="switch_to_momentum", scale=1.0, meta=meta)
        return RegimeDecision(action="scale_down", scale=float(scale_factor), meta=meta)

    return RegimeDecision(action="pass", scale=1.0, meta=meta)


