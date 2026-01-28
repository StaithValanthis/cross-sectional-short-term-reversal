from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import orjson
from loguru import logger

from src.config import RiskConfig
from src.data.market_data import normalize_symbol


def _utc_today() -> date:
    return datetime.now(tz=UTC).date()


def _parse_day(s: str) -> date:
    # s is expected YYYY-MM-DD (UTC day)
    return date.fromisoformat(s)


def _day_str(d: date) -> str:
    return d.isoformat()


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class OpenPositionMeta:
    opened_at_day: str  # YYYY-MM-DD (UTC)
    side_sign: int  # +1 long, -1 short
    cum_realised_at_open: float | None = None


@dataclass
class CooldownMeta:
    last_forced_exit_day: str  # YYYY-MM-DD (UTC)


@dataclass
class PositionsRiskState:
    version: int
    open_positions: dict[str, OpenPositionMeta]
    cooldowns: dict[str, CooldownMeta]


def load_positions_risk_state(path: Path) -> PositionsRiskState:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return PositionsRiskState(version=1, open_positions={}, cooldowns={})
    try:
        raw = orjson.loads(path.read_bytes())
    except Exception as e:
        logger.warning("Failed to parse positions risk state {}; reinitializing: {}", path, e)
        return PositionsRiskState(version=1, open_positions={}, cooldowns={})

    op: dict[str, OpenPositionMeta] = {}
    cd: dict[str, CooldownMeta] = {}

    for sym, meta in (raw.get("open_positions") or {}).items():
        s = normalize_symbol(str(sym))
        if not s or not isinstance(meta, dict):
            continue
        opened = str(meta.get("opened_at_day") or "").strip()
        side = meta.get("side_sign")
        if not opened or side not in (-1, 1):
            continue
        op[s] = OpenPositionMeta(
            opened_at_day=opened,
            side_sign=int(side),
            cum_realised_at_open=_safe_float(meta.get("cum_realised_at_open")),
        )

    for sym, meta in (raw.get("cooldowns") or {}).items():
        s = normalize_symbol(str(sym))
        if not s or not isinstance(meta, dict):
            continue
        last = str(meta.get("last_forced_exit_day") or "").strip()
        if not last:
            continue
        cd[s] = CooldownMeta(last_forced_exit_day=last)

    ver = int(raw.get("version") or 1)
    return PositionsRiskState(version=ver, open_positions=op, cooldowns=cd)


def save_positions_risk_state(path: Path, st: PositionsRiskState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "version": int(st.version),
        "open_positions": {
            s: {
                "opened_at_day": m.opened_at_day,
                "side_sign": int(m.side_sign),
                "cum_realised_at_open": m.cum_realised_at_open,
            }
            for s, m in sorted(st.open_positions.items())
        },
        "cooldowns": {s: {"last_forced_exit_day": m.last_forced_exit_day} for s, m in sorted(st.cooldowns.items())},
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


@dataclass(frozen=True)
class RawPnlMetrics:
    unrealised_pnl: float | None
    cum_realised_pnl: float | None


def extract_pnl_metrics_by_symbol(raw_positions: list[dict[str, Any]]) -> dict[str, RawPnlMetrics]:
    """
    Extract unrealised/cumulative realised PnL by symbol from Bybit v5 position list payload.
    We keep this defensive because field names vary a bit and some accounts omit values.
    """
    out: dict[str, dict[str, float]] = {}
    has: dict[str, dict[str, bool]] = {}

    for p in raw_positions or []:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue
        # Ignore hedge-mode idx=1/2 - this bot is one-way, and hedge mode exits are not supported.
        try:
            pos_idx = int(p.get("positionIdx") or 0)
        except Exception:
            pos_idx = 0
        if pos_idx in (1, 2):
            continue

        # Bybit uses "unrealisedPnl" / "cumRealisedPnl" in many payloads
        u = _safe_float(p.get("unrealisedPnl"))
        if u is None:
            u = _safe_float(p.get("unrealizedPnl"))
        c = _safe_float(p.get("cumRealisedPnl"))
        if c is None:
            c = _safe_float(p.get("cumRealizedPnl"))

        out.setdefault(sym, {"unreal": 0.0, "cum": 0.0})
        has.setdefault(sym, {"unreal": False, "cum": False})
        if u is not None:
            out[sym]["unreal"] += float(u)
            has[sym]["unreal"] = True
        if c is not None:
            out[sym]["cum"] += float(c)
            has[sym]["cum"] = True

    final: dict[str, RawPnlMetrics] = {}
    for sym, agg in out.items():
        final[sym] = RawPnlMetrics(
            unrealised_pnl=float(agg["unreal"]) if has.get(sym, {}).get("unreal") else None,
            cum_realised_pnl=float(agg["cum"]) if has.get(sym, {}).get("cum") else None,
        )
    return final


def update_state_from_positions(
    *,
    st: PositionsRiskState,
    positions: dict[str, Any],  # expects values with .size (signed qty)
    pnl_metrics: dict[str, RawPnlMetrics],
    today: date,
) -> None:
    """
    Update open-position tracking strictly from local state rules:
    - newly seen symbol -> opened_at=today
    - sign flip -> reset opened_at=today
    - flat -> delete open_positions entry
    """
    today_s = _day_str(today)

    # Remove entries for positions that are now flat
    for sym in list(st.open_positions.keys()):
        pos = positions.get(sym)
        cur = float(getattr(pos, "size", 0.0)) if pos is not None else 0.0
        if abs(cur) <= 1e-12:
            st.open_positions.pop(sym, None)

    # Add/update entries for currently open positions
    for sym, pos in positions.items():
        sym = normalize_symbol(sym)
        if not sym:
            continue
        cur = float(getattr(pos, "size", 0.0))
        if abs(cur) <= 1e-12:
            continue
        sign = 1 if cur > 0 else -1

        prev = st.open_positions.get(sym)
        if prev is None or int(prev.side_sign) != int(sign):
            m = pnl_metrics.get(sym)
            st.open_positions[sym] = OpenPositionMeta(
                opened_at_day=today_s,
                side_sign=int(sign),
                cum_realised_at_open=(m.cum_realised_pnl if m is not None else None),
            )


@dataclass(frozen=True)
class RiskExitEvent:
    symbol: str
    kind: str  # "risk_time_stop" | "risk_loss_cap" | "risk_cooldown"
    held_days: int | None
    pnl_since_open: float | None
    loss_pct_equity: float | None
    threshold: float | int | None


def _held_days(opened_at_day: str, today: date) -> int | None:
    try:
        return (today - _parse_day(opened_at_day)).days
    except Exception:
        return None


def evaluate_risk_exits(
    *,
    cfg: RiskConfig,
    st: PositionsRiskState,
    positions: dict[str, Any],  # expects .size
    pnl_metrics: dict[str, RawPnlMetrics],
    equity_usdt: float | None,
    today: date,
) -> tuple[dict[str, str], list[RiskExitEvent]]:
    """
    Returns:
    - force_close_reasons: symbol -> reason string (e.g., "risk_time_stop" or "risk_loss_cap")
    - events: structured details for logging/snapshots
    """
    reasons: dict[str, str] = {}
    events: list[RiskExitEvent] = []

    max_hold_days = int(getattr(cfg, "max_hold_days", 0) or 0)
    loss_cap = float(getattr(cfg, "max_loss_per_position_pct_equity", 0.0) or 0.0)

    eq = float(equity_usdt) if equity_usdt is not None else None
    if eq is not None and eq <= 0:
        eq = None

    for sym, pos in positions.items():
        sym = normalize_symbol(sym)
        if not sym:
            continue
        cur = float(getattr(pos, "size", 0.0))
        if abs(cur) <= 1e-12:
            continue

        meta = st.open_positions.get(sym)
        held = _held_days(meta.opened_at_day, today) if meta is not None else None

        # 1) Time stop
        if max_hold_days > 0 and held is not None and held >= max_hold_days:
            reasons[sym] = "risk_time_stop"
            events.append(
                RiskExitEvent(
                    symbol=sym,
                    kind="risk_time_stop",
                    held_days=int(held),
                    pnl_since_open=None,
                    loss_pct_equity=None,
                    threshold=int(max_hold_days),
                )
            )
            continue  # hard exit wins

        # 2) Loss cap (equity-based)
        if loss_cap > 0 and eq is not None:
            m = pnl_metrics.get(sym)
            unreal = m.unrealised_pnl if m is not None else None
            cum = m.cum_realised_pnl if m is not None else None

            pnl_since_open: float | None = None
            if unreal is not None:
                pnl_since_open = float(unreal)
                if meta is not None and meta.cum_realised_at_open is not None and cum is not None:
                    pnl_since_open = float(unreal) + (float(cum) - float(meta.cum_realised_at_open))

            if pnl_since_open is not None:
                loss = max(0.0, -float(pnl_since_open))
                loss_pct = (loss / float(eq)) * 100.0 if eq > 0 else None
                if loss_pct is not None and loss_pct >= float(loss_cap):
                    reasons[sym] = "risk_loss_cap"
                    events.append(
                        RiskExitEvent(
                            symbol=sym,
                            kind="risk_loss_cap",
                            held_days=int(held) if held is not None else None,
                            pnl_since_open=float(pnl_since_open),
                            loss_pct_equity=float(loss_pct),
                            threshold=float(loss_cap),
                        )
                    )

    return reasons, events


def apply_cooldown_exclusions(
    *,
    cfg: RiskConfig,
    st: PositionsRiskState,
    target_notionals: dict[str, float],
    today: date,
) -> tuple[set[str], list[RiskExitEvent]]:
    """
    Exclude symbols from targets if they are within cooldown window after a forced exit.
    Returns (cooldown_symbols, events).
    """
    cooldown_days = int(getattr(cfg, "cooldown_days_after_forced_exit", 0) or 0)
    if cooldown_days <= 0:
        return set(), []

    out_syms: set[str] = set()
    events: list[RiskExitEvent] = []
    for sym, cd in st.cooldowns.items():
        try:
            days_since = (today - _parse_day(cd.last_forced_exit_day)).days
        except Exception:
            continue
        if days_since < cooldown_days:
            out_syms.add(sym)
            events.append(
                RiskExitEvent(
                    symbol=sym,
                    kind="risk_cooldown",
                    held_days=None,
                    pnl_since_open=None,
                    loss_pct_equity=None,
                    threshold=int(cooldown_days),
                )
            )

    # Only apply to current targets (don’t invent new keys)
    for sym in out_syms:
        if sym in target_notionals:
            target_notionals[sym] = 0.0
    return out_syms, events


def mark_forced_exits_in_state(*, st: PositionsRiskState, symbols: set[str], today: date) -> None:
    if not symbols:
        return
    today_s = _day_str(today)
    for sym in symbols:
        st.cooldowns[normalize_symbol(sym)] = CooldownMeta(last_forced_exit_day=today_s)

