from __future__ import annotations

import math


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b == 0.0:
        return default
    return a / b


def is_finite(x: float) -> bool:
    return math.isfinite(x)


