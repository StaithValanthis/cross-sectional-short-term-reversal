from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import RiskConfig
from src.execution.risk import RiskManager


class RiskManagerTests(unittest.TestCase):
    def test_same_day_loss_limit_triggers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rm = RiskManager(cfg=RiskConfig(daily_loss_limit_pct=2.5, max_drawdown_pct=20.0), state_dir=Path(td))
            ok0, _ = rm.check(1000.0)
            ok1, info = rm.check(970.0)
            self.assertTrue(ok0)
            self.assertFalse(ok1)
            self.assertEqual(info["reason"], "kill_switch_tier1")
            self.assertFalse(info["daily_ok"])

    def test_persistent_high_water_survives_day_rollover(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rm = RiskManager(cfg=RiskConfig(daily_loss_limit_pct=50.0, max_drawdown_pct=20.0), state_dir=Path(td))
            rm.check(1000.0)
            rm.check(1200.0)
            raw = {
                "day": "2000-01-01",
                "start_equity": 1000.0,
                "high_water": 1200.0,
                "kill_switch": False,
                "consecutive_loss_days": 0,
                "last_seen_equity": 1200.0,
            }
            rm.state = None
            rm.state_path.write_text(__import__("json").dumps(raw), encoding="utf-8")

            ok, info = rm.check(900.0)
            self.assertFalse(ok)
            self.assertEqual(info["reason"], "kill_switch_tier1")
            self.assertFalse(info["dd_ok"])

    def test_state_rollover_preserves_portfolio_peak(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            rm = RiskManager(cfg=RiskConfig(daily_loss_limit_pct=50.0, max_drawdown_pct=50.0), state_dir=Path(td))
            raw = {
                "day": "2000-01-01",
                "start_equity": 1000.0,
                "high_water": 1500.0,
                "kill_switch": False,
                "consecutive_loss_days": 0,
                "last_seen_equity": 1400.0,
            }
            rm.state_path.write_text(__import__("json").dumps(raw), encoding="utf-8")
            st = rm.load_or_init(1300.0)
            self.assertEqual(st.high_water, 1500.0)
            self.assertEqual(st.start_equity, 1300.0)


if __name__ == "__main__":
    unittest.main()
