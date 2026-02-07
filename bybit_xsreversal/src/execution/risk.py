from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
from loguru import logger

from src.config import RiskConfig


@dataclass
class RiskState:
    day: str  # YYYY-MM-DD UTC
    start_equity: float
    high_water: float
    kill_switch: bool
    # observability / multi-day controls
    consecutive_loss_days: int = 0
    last_seen_equity: float | None = None


class RiskManager:
    def __init__(self, *, cfg: RiskConfig, state_dir: str | Path) -> None:
        self.cfg = cfg
        self.state_path = Path(state_dir) / "risk_state.json"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state: RiskState | None = None

    def _today(self) -> str:
        now = datetime.now(tz=UTC)
        return f"{now.year:04d}-{now.month:02d}-{now.day:02d}"

    def load_or_init(self, current_equity: float) -> RiskState:
        day = self._today()
        if self.state is not None and self.state.day == day:
            return self.state

        if self.state_path.exists():
            try:
                raw = orjson.loads(self.state_path.read_bytes())
                if raw.get("day") == day:
                    self.state = RiskState(
                        day=day,
                        start_equity=float(raw["start_equity"]),
                        high_water=float(raw.get("high_water", raw["start_equity"])),
                        kill_switch=bool(raw.get("kill_switch", False)),
                        consecutive_loss_days=int(raw.get("consecutive_loss_days", 0) or 0),
                        last_seen_equity=float(raw.get("last_seen_equity")) if raw.get("last_seen_equity") is not None else None,
                    )
                    return self.state
                # Day rollover: carry consecutive-loss counter forward based on prior day's outcome.
                prev_start = float(raw.get("start_equity") or current_equity)
                prev_last = raw.get("last_seen_equity")
                prev_last_f = float(prev_last) if prev_last is not None else prev_start
                prev_pnl = prev_last_f - prev_start
                prev_cons = int(raw.get("consecutive_loss_days", 0) or 0)
                cons = (prev_cons + 1) if float(prev_pnl) < 0 else 0
                self.state = RiskState(
                    day=day,
                    start_equity=float(current_equity),
                    high_water=float(current_equity),
                    kill_switch=False,
                    consecutive_loss_days=int(cons),
                    last_seen_equity=float(current_equity),
                )
                self._persist()
                return self.state
            except Exception as e:
                logger.warning("Failed to parse risk state; reinitializing: {}", e)

        self.state = RiskState(
            day=day,
            start_equity=float(current_equity),
            high_water=float(current_equity),
            kill_switch=False,
            consecutive_loss_days=0,
            last_seen_equity=float(current_equity),
        )
        self._persist()
        return self.state

    def _persist(self) -> None:
        if self.state is None:
            return
        payload: dict[str, Any] = {
            "day": self.state.day,
            "start_equity": self.state.start_equity,
            "high_water": self.state.high_water,
            "kill_switch": self.state.kill_switch,
            "consecutive_loss_days": int(self.state.consecutive_loss_days),
            "last_seen_equity": self.state.last_seen_equity,
        }
        self.state_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    def check(self, current_equity: float) -> tuple[bool, dict[str, Any]]:
        """
        Returns (ok_to_trade, details). If kill switch triggered, ok_to_trade=False.
        """
        if not self.cfg.kill_switch_enabled:
            return True, {"kill_switch_enabled": False}

        st = self.load_or_init(current_equity)
        st.high_water = max(st.high_water, float(current_equity))
        st.last_seen_equity = float(current_equity)

        loss_limit = 1.0 - float(self.cfg.daily_loss_limit_pct) / 100.0
        tier2_pct = getattr(self.cfg, "daily_loss_limit_pct_tier2", None)
        tier2_limit = None
        if tier2_pct is not None:
            try:
                tier2_limit = 1.0 - float(tier2_pct) / 100.0
            except Exception:
                tier2_limit = None
        dd_limit = 1.0 - float(self.cfg.max_drawdown_pct) / 100.0

        daily_ok = float(current_equity) >= float(st.start_equity) * loss_limit
        daily_ok_tier2 = True
        if tier2_limit is not None:
            daily_ok_tier2 = float(current_equity) >= float(st.start_equity) * float(tier2_limit)
        dd_ok = float(current_equity) >= float(st.high_water) * dd_limit
        max_cons = int(getattr(self.cfg, "max_consecutive_loss_days", 0) or 0)
        cons_ok = True if max_cons <= 0 else int(st.consecutive_loss_days) < int(max_cons)

        if not cons_ok:
            st.kill_switch = True
            self._persist()
            return False, {
                "kill_switch": True,
                "reason": "kill_switch_consecutive_loss",
                "consecutive_loss_days": int(st.consecutive_loss_days),
                "max_consecutive_loss_days": int(max_cons),
                "start_equity": st.start_equity,
                "high_water": st.high_water,
                "current_equity": float(current_equity),
            }

        if not daily_ok_tier2:
            st.kill_switch = True
            self._persist()
            return False, {
                "kill_switch": True,
                "reason": "kill_switch_tier2",
                "daily_ok_tier2": daily_ok_tier2,
                "tier2_pct": tier2_pct,
                "start_equity": st.start_equity,
                "high_water": st.high_water,
                "current_equity": float(current_equity),
            }

        if not daily_ok or not dd_ok:
            st.kill_switch = True
            self._persist()
            return False, {
                "kill_switch": True,
                "reason": "kill_switch_tier1",
                "daily_ok": daily_ok,
                "dd_ok": dd_ok,
                "start_equity": st.start_equity,
                "high_water": st.high_water,
                "current_equity": float(current_equity),
            }

        if st.kill_switch:
            # once killed, stays killed for the day
            self._persist()
            return False, {"kill_switch": True, "reason": "already_triggered_today"}

        self._persist()
        return True, {
            "kill_switch": False,
            "start_equity": st.start_equity,
            "high_water": st.high_water,
            "current_equity": float(current_equity),
            "consecutive_loss_days": int(st.consecutive_loss_days),
        }


