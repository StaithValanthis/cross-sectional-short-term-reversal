from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PortfolioTargets:
    weights: dict[str, float]          # signed weights (notional / equity)
    notionals_usd: dict[str, float]    # signed notionals
    meta: dict[str, Any]


def _normalize(weights: dict[str, float], target_sum_abs: float) -> dict[str, float]:
    s = float(sum(abs(w) for w in weights.values()))
    if s <= 0:
        return {k: 0.0 for k in weights}
    scale = target_sum_abs / s
    return {k: float(v) * scale for k, v in weights.items()}


def _apply_cap_and_renormalize(
    weights: dict[str, float],
    *,
    cap_abs: float,
    target_sum_abs: float,
) -> dict[str, float]:
    """
    Clip abs(weight) to cap_abs, then renormalize remaining degrees of freedom to target_sum_abs.
    Iterative "waterfilling" style to keep feasibility.
    """
    if target_sum_abs <= 0:
        return {k: 0.0 for k in weights}
    if cap_abs <= 0:
        return {k: 0.0 for k in weights}

    w = dict(weights)
    for _ in range(10):
        # Clip
        for k, v in list(w.items()):
            if abs(v) > cap_abs:
                w[k] = float(np.sign(v) * cap_abs)
        s = sum(abs(v) for v in w.values())
        if s == 0:
            return {k: 0.0 for k in w}
        if abs(s - target_sum_abs) < 1e-10:
            return w
        # If clipped sum is already below target, scale up unclipped names only is complex; scale all is OK since clip will re-bind.
        scale = target_sum_abs / s
        w = {k: v * scale for k, v in w.items()}
        # Converges quickly in practice for modest caps.
    return w


def build_dollar_neutral_weights(
    *,
    long_scores: dict[str, float],
    short_scores: dict[str, float],
    target_gross_leverage: float,
    max_abs_weight_per_symbol: float,
    long_only: bool,
) -> dict[str, float]:
    """
    Convert positive "scores" (e.g., inverse-vol) into signed weights.
    - Long-only: sum(abs(weights)) == target_gross_leverage
    - L/S: sum(long weights) == target_gross_leverage/2 and sum(abs(short weights)) == target_gross_leverage/2
    """
    target_gross_leverage = float(target_gross_leverage)
    max_abs_weight_per_symbol = float(max_abs_weight_per_symbol)
    if target_gross_leverage <= 0:
        return {}

    longs = {k: max(0.0, float(v)) for k, v in long_scores.items()}
    shorts = {k: max(0.0, float(v)) for k, v in short_scores.items()}

    if long_only:
        w = _normalize(longs, target_sum_abs=target_gross_leverage)
        w = _apply_cap_and_renormalize(w, cap_abs=max_abs_weight_per_symbol, target_sum_abs=target_gross_leverage)
        return w

    # Long/short: allocate half gross to each side
    w_long = _normalize(longs, target_sum_abs=target_gross_leverage / 2.0)
    w_short = _normalize(shorts, target_sum_abs=target_gross_leverage / 2.0)
    w_long = _apply_cap_and_renormalize(w_long, cap_abs=max_abs_weight_per_symbol, target_sum_abs=target_gross_leverage / 2.0)
    w_short = _apply_cap_and_renormalize(w_short, cap_abs=max_abs_weight_per_symbol, target_sum_abs=target_gross_leverage / 2.0)

    out: dict[str, float] = {}
    out.update(w_long)
    out.update({k: -abs(v) for k, v in w_short.items()})
    return out


def weights_to_notionals(
    weights: dict[str, float],
    *,
    equity_usd: float,
    max_notional_per_symbol: float,
    min_notional_per_symbol: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, w in weights.items():
        notional = float(w) * float(equity_usd)
        if abs(notional) < float(min_notional_per_symbol):
            continue
        if abs(notional) > float(max_notional_per_symbol):
            notional = float(np.sign(notional) * float(max_notional_per_symbol))
        out[sym] = notional
    return out


