from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import orjson
import pandas as pd
from loguru import logger

from src.config import BotConfig, IntradayExitsConfig
from src.data.market_data import normalize_symbol


SideSign = Literal[1, -1]  # 1=long, -1=short


@dataclass(frozen=True)
class ExitDecision:
    symbol: str
    side: Literal["Buy", "Sell"]  # side of the reduce-only EXIT order
    qty_to_close: float  # full close
    reduce_only: bool
    reason: str  # "risk_atr_fixed" | "risk_atr_trailing" | "risk_breakeven" | "risk_time_stop"
    stop_type: str
    stop_price: float | None
    trigger_price: float | None
    atr: float | None
    entry_price: float | None


@dataclass
class IntradayStateSymbol:
    trailing_extreme: float | None = None
    last_entry_price: float | None = None
    last_qty: float | None = None
    entry_ts: str | None = None  # ISO
    last_update_ts: str | None = None  # ISO


@dataclass
class IntradayExitState:
    version: int
    last_run_ts: str | None
    per_symbol: dict[str, IntradayStateSymbol]


def load_intraday_state(path: str | Path) -> IntradayExitState:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        return IntradayExitState(version=1, last_run_ts=None, per_symbol={})
    try:
        raw = orjson.loads(p.read_bytes())
    except Exception as e:
        logger.warning("Failed to parse intraday state {}; reinitializing: {}", p, e)
        return IntradayExitState(version=1, last_run_ts=None, per_symbol={})

    per: dict[str, IntradayStateSymbol] = {}
    for sym, meta in (raw.get("per_symbol") or {}).items():
        s = normalize_symbol(str(sym))
        if not s or not isinstance(meta, dict):
            continue
        per[s] = IntradayStateSymbol(
            trailing_extreme=_safe_float(meta.get("trailing_extreme")),
            last_entry_price=_safe_float(meta.get("last_entry_price")),
            last_qty=_safe_float(meta.get("last_qty")),
            entry_ts=str(meta.get("entry_ts")) if meta.get("entry_ts") else None,
            last_update_ts=str(meta.get("last_update_ts")) if meta.get("last_update_ts") else None,
        )

    return IntradayExitState(
        version=int(raw.get("version") or 1),
        last_run_ts=str(raw.get("last_run_ts")) if raw.get("last_run_ts") else None,
        per_symbol=per,
    )


def save_intraday_state(path: str | Path, st: IntradayExitState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": int(st.version),
        "last_run_ts": st.last_run_ts,
        "per_symbol": {
            s: {
                "trailing_extreme": m.trailing_extreme,
                "last_entry_price": m.last_entry_price,
                "last_qty": m.last_qty,
                "entry_ts": m.entry_ts,
                "last_update_ts": m.last_update_ts,
            }
            for s, m in sorted(st.per_symbol.items())
        },
    }
    p.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


# ---------------- Pure, unit-testable helpers ----------------


def compute_atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float | None:
    """
    Wilder ATR over a series. Returns the LAST ATR value.
    """
    period = int(period)
    if period < 2:
        raise ValueError("period must be >= 2")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high/low/close must be same length")
    n = len(high)
    if n < period + 1:
        return None

    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    tr = tr[1:]  # first TR undefined-ish; drop to match typical definition
    if len(tr) < period:
        return None

    # Initial ATR: SMA of first `period` TRs
    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / float(period)
    for v in tr[period:]:
        atr = atr + alpha * (float(v) - atr)
    return float(atr)


def fixed_stop_price(entry: float, atr: float, k: float, side_sign: SideSign) -> float:
    if side_sign == 1:
        return float(entry - k * atr)
    return float(entry + k * atr)


def update_trailing_extreme(prev_extreme: float | None, bar_high: float, bar_low: float, side_sign: SideSign) -> float:
    if prev_extreme is None:
        return float(bar_high) if side_sign == 1 else float(bar_low)
    if side_sign == 1:
        return float(max(float(prev_extreme), float(bar_high)))
    return float(min(float(prev_extreme), float(bar_low)))


def trailing_stop_price(extreme: float, atr: float, k: float, side_sign: SideSign) -> float:
    if side_sign == 1:
        return float(extreme - k * atr)
    return float(extreme + k * atr)


def breakeven_stop(entry: float, costs_bps: float, side_sign: SideSign) -> float:
    # Approx: include fees+slippage as bps of notional.
    c = float(costs_bps) / 10_000.0
    if side_sign == 1:
        return float(entry * (1.0 + c))
    return float(entry * (1.0 - c))


def trigger_intrabar(side_sign: SideSign, bar_high: float, bar_low: float, stop: float) -> bool:
    if side_sign == 1:
        return float(bar_low) <= float(stop)
    return float(bar_high) >= float(stop)


def trigger_last(side_sign: SideSign, last_price: float, stop: float) -> bool:
    if side_sign == 1:
        return float(last_price) <= float(stop)
    return float(last_price) >= float(stop)


def time_stop_trigger(
    entry_ts: str | None,
    now_ts: datetime,
    hours: int,
    only_if_unprofitable: bool,
    unrealized_pnl: float | None,
) -> bool:
    if not entry_ts:
        return False
    try:
        dt = datetime.fromisoformat(entry_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        dt = dt.astimezone(UTC)
    except Exception:
        return False

    held = now_ts.astimezone(UTC) - dt
    if held < timedelta(hours=int(hours)):
        return False
    if only_if_unprofitable and unrealized_pnl is not None:
        return float(unrealized_pnl) < 0.0
    return True


def _side_sign_from_position_qty(qty_signed: float) -> SideSign:
    return 1 if float(qty_signed) > 0 else -1


def _exit_side_for_position(qty_signed: float) -> Literal["Buy", "Sell"]:
    # To close: long -> Sell, short -> Buy
    return "Sell" if float(qty_signed) > 0 else "Buy"


def evaluate_intraday_exit_decisions(
    *,
    cfg: IntradayExitsConfig,
    positions: list[dict[str, Any]],
    candles_1h_by_symbol: dict[str, pd.DataFrame],
    candles_4h_by_symbol: dict[str, pd.DataFrame],
    last_prices_by_symbol: dict[str, float] | None,
    state: IntradayExitState,
    now_ts: datetime | None = None,
) -> list[ExitDecision]:
    """
    Exit-only evaluator. Emits at most one ExitDecision per symbol per cycle.
    """
    now = (now_ts or datetime.now(tz=UTC)).astimezone(UTC)
    out: list[ExitDecision] = []

    # Cap how many positions we process per cycle (operational safety).
    pos_list = list(positions or [])[: max(0, int(cfg.max_positions_per_cycle))]

    for p in pos_list:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue

        # Skip hedge-mode symbols: intraday engine is one-way only (like rebalance).
        try:
            pos_idx = int(p.get("positionIdx") or 0)
        except Exception:
            pos_idx = 0
        if pos_idx in (1, 2):
            continue

        side = str(p.get("side", "")).strip()
        size = float(p.get("size") or 0.0)
        if abs(size) <= 1e-12:
            continue
        qty_signed = float(size) if side == "Buy" else -float(size)

        mark = _safe_float(p.get("markPrice") or p.get("avgPrice") or 0.0) or 0.0
        notional = abs(float(qty_signed) * float(mark)) if float(mark) > 0 else None
        if notional is not None and notional + 1e-9 < float(cfg.min_notional_to_exit_usd):
            continue

        entry = _safe_float(p.get("avgPrice") or p.get("avgEntryPrice") or p.get("entryPrice"))
        st_sym = state.per_symbol.get(sym) or IntradayStateSymbol()
        if entry is None:
            entry = st_sym.last_entry_price
        if entry is None:
            logger.warning("Intraday exits: missing entry price for {}; skipping", sym)
            continue

        side_sign = _side_sign_from_position_qty(qty_signed)

        # Pull candles
        c1 = candles_1h_by_symbol.get(sym)
        c4 = candles_4h_by_symbol.get(sym)
        if c1 is None or c4 is None:
            logger.warning("Intraday exits: missing candles for {}; skipping", sym)
            continue
        if len(c1) < int(cfg.min_bars_trigger) or len(c4) < int(cfg.min_bars_atr):
            logger.warning("Intraday exits: insufficient bars for {} (1h={},4h={}); skipping", sym, len(c1), len(c4))
            continue

        # ATR from 4H
        atr = compute_atr_wilder(
            high=c4["high"].astype(float).to_numpy(),
            low=c4["low"].astype(float).to_numpy(),
            close=c4["close"].astype(float).to_numpy(),
            period=int(cfg.atr_period),
        )
        if atr is None or not np.isfinite(float(atr)) or float(atr) <= 0:
            logger.warning("Intraday exits: bad ATR for {}; skipping", sym)
            continue

        # Update trailing extreme using the last 1H bar
        last_bar = c1.sort_index().iloc[-1]
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        extreme = update_trailing_extreme(st_sym.trailing_extreme, bar_high, bar_low, side_sign)
        st_sym.trailing_extreme = float(extreme)
        st_sym.last_entry_price = float(entry)
        st_sym.last_qty = float(qty_signed)
        st_sym.last_update_ts = now.isoformat()
        state.per_symbol[sym] = st_sym

        # Stop priority: time stop -> fixed ATR -> trailing ATR -> breakeven
        stop_candidates: list[tuple[str, float, str]] = []

        # 1) Time stop
        unreal = _safe_float(p.get("unrealisedPnl") or p.get("unrealizedPnl"))
        if bool(cfg.time_stop_enabled) and time_stop_trigger(
            st_sym.entry_ts,
            now,
            int(cfg.time_stop_hours),
            bool(cfg.time_stop_only_if_unprofitable),
            unreal,
        ):
            stop_candidates.append(("risk_time_stop", float("nan"), "time_stop"))

        # 2) Fixed ATR stop
        if bool(cfg.fixed_atr_stop_enabled):
            sp = fixed_stop_price(float(entry), float(atr), float(cfg.fixed_atr_k), side_sign)
            stop_candidates.append(("risk_atr_fixed", float(sp), "atr_fixed"))

        # 3) Trailing ATR stop
        if bool(cfg.trailing_atr_stop_enabled) and st_sym.trailing_extreme is not None:
            sp = trailing_stop_price(float(st_sym.trailing_extreme), float(atr), float(cfg.trailing_atr_k), side_sign)
            stop_candidates.append(("risk_atr_trailing", float(sp), "atr_trailing"))

        # 4) Breakeven stop (only after favorable move >= t*ATR)
        if bool(cfg.stop_to_breakeven_enabled) and st_sym.trailing_extreme is not None:
            mfe = (float(st_sym.trailing_extreme) - float(entry)) if side_sign == 1 else (float(entry) - float(st_sym.trailing_extreme))
            if float(mfe) >= float(cfg.breakeven_trigger_atr_t) * float(atr):
                sp = breakeven_stop(float(entry), float(cfg.breakeven_costs_bps), side_sign)
                stop_candidates.append(("risk_breakeven", float(sp), "breakeven"))

        if not stop_candidates:
            continue

        reason, stop_price, stop_type = stop_candidates[0]

        # Trigger check
        triggered = False
        trigger_px: float | None = None
        recent = c1.sort_index().iloc[-int(cfg.min_bars_trigger) :]
        if reason == "risk_time_stop":
            triggered = True
        elif bool(cfg.use_intrabar_trigger) and np.isfinite(stop_price):
            for _, r in recent.iterrows():
                if trigger_intrabar(side_sign, float(r["high"]), float(r["low"]), float(stop_price)):
                    triggered = True
                    trigger_px = float(r["low"]) if side_sign == 1 else float(r["high"])
                    break
        elif bool(cfg.use_last_price_trigger) and last_prices_by_symbol is not None and np.isfinite(stop_price):
            lp = last_prices_by_symbol.get(sym)
            if lp is not None and trigger_last(side_sign, float(lp), float(stop_price)):
                triggered = True
                trigger_px = float(lp)

        if not triggered:
            continue

        qty_to_close = abs(float(qty_signed))
        out.append(
            ExitDecision(
                symbol=sym,
                side=_exit_side_for_position(qty_signed),
                qty_to_close=qty_to_close,
                reduce_only=True,
                reason=reason,
                stop_type=stop_type,
                stop_price=(None if (reason == "risk_time_stop") else float(stop_price)),
                trigger_price=trigger_px,
                atr=float(atr),
                entry_price=float(entry),
            )
        )

    state.last_run_ts = now.isoformat()
    return out


def ensure_entry_ts_from_position(*, st: IntradayExitState, positions: list[dict[str, Any]], now_ts: datetime | None = None) -> None:
    """
    Best-effort: if a symbol is newly seen in positions and has no entry_ts in state, set it to now.
    (Bybit position payload doesn't reliably provide opened timestamp.)
    """
    now = (now_ts or datetime.now(tz=UTC)).astimezone(UTC)
    for p in positions or []:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue
        size = float(p.get("size") or 0.0)
        if abs(size) <= 1e-12:
            continue
        m = st.per_symbol.get(sym) or IntradayStateSymbol()
        if not m.entry_ts:
            m.entry_ts = now.isoformat()
        st.per_symbol[sym] = m


def build_intraday_candle_window(cfg: IntradayExitsConfig, now_ts: datetime | None = None) -> tuple[datetime, datetime]:
    """
    Compute a safe candle window for the intraday cycle based on atr_period/min bars.
    """
    now = (now_ts or datetime.now(tz=UTC)).astimezone(UTC)
    # Rough bar widths: 1H and 4H
    need_1h = max(int(cfg.min_bars_trigger) + 2, 8)
    need_4h = max(int(cfg.min_bars_atr) + 2, int(cfg.atr_period) + 5)
    lookback_hours = max(need_1h * 1, need_4h * 4)
    start = now - timedelta(hours=int(lookback_hours) + 4)
    end = now + timedelta(hours=1)
    return start, end

