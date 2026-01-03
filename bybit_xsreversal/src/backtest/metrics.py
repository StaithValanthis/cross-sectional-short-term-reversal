from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestMetrics:
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float  # CAGR / abs(max_drawdown), or 0 if max_drawdown == 0
    profit_factor: float
    win_rate: float
    avg_daily_turnover: float


def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def _cagr(equity: pd.Series, periods_per_year: int = 365) -> float:
    if len(equity) < 2:
        return 0.0
    e0 = float(equity.iloc[0])
    e1 = float(equity.iloc[-1])
    # Robustness: if equity goes non-positive, CAGR is not well-defined.
    if e0 <= 0.0 or e1 <= 0.0:
        return -1.0
    total = e1 / e0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    if total <= 0.0:
        return -1.0
    return float(total ** (1.0 / years) - 1.0)


def _sharpe(r: pd.Series, periods_per_year: int = 365) -> float:
    r = r.dropna()
    if r.std(ddof=0) == 0 or r.empty:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std(ddof=0))


def _sortino(r: pd.Series, periods_per_year: int = 365) -> float:
    r = r.dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    denom = downside.std(ddof=0)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / denom)


def _profit_factor(r: pd.Series) -> float:
    r = r.dropna()
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg == 0:
        return float("inf") if pos > 0 else 0.0
    return float(pos / neg)


def _calmar(cagr: float, max_drawdown: float) -> float:
    """Calmar ratio: CAGR / abs(max_drawdown). Higher is better."""
    if max_drawdown == 0.0 or not np.isfinite(max_drawdown):
        return 0.0 if cagr == 0.0 else float("inf") if cagr > 0 else float("-inf")
    dd_abs = abs(max_drawdown)
    if dd_abs == 0.0:
        return 0.0 if cagr == 0.0 else float("inf") if cagr > 0 else float("-inf")
    return float(cagr / dd_abs)


def compute_metrics(equity: pd.Series, daily_returns: pd.Series, daily_turnover: pd.Series) -> BacktestMetrics:
    daily_returns = daily_returns.dropna()
    win_rate = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
    cagr = _cagr(equity)
    max_dd = _max_drawdown(equity)
    return BacktestMetrics(
        cagr=cagr,
        sharpe=_sharpe(daily_returns),
        sortino=_sortino(daily_returns),
        max_drawdown=max_dd,
        calmar=_calmar(cagr, max_dd),
        profit_factor=_profit_factor(daily_returns),
        win_rate=win_rate,
        avg_daily_turnover=float(daily_turnover.dropna().mean()) if len(daily_turnover) else 0.0,
    )


