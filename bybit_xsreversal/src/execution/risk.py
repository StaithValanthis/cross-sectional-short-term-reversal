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
                    )
                    return self.state
            except Exception as e:
                logger.warning("Failed to parse risk state; reinitializing: {}", e)

        self.state = RiskState(day=day, start_equity=float(current_equity), high_water=float(current_equity), kill_switch=False)
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

        loss_limit = 1.0 - float(self.cfg.daily_loss_limit_pct) / 100.0
        dd_limit = 1.0 - float(self.cfg.max_drawdown_pct) / 100.0

        daily_ok = float(current_equity) >= float(st.start_equity) * loss_limit
        dd_ok = float(current_equity) >= float(st.high_water) * dd_limit

        if not daily_ok or not dd_ok:
            st.kill_switch = True
            self._persist()
            return False, {
                "kill_switch": True,
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
        }


