from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.config import BotConfig
from src.execution.rebalance import Position, apply_turnover_cap


class _FakeMD:
    def __init__(self, mid_by_symbol: dict[str, float]) -> None:
        self._mid = dict(mid_by_symbol)

    def get_orderbook_stats(self, symbol: str):
        return SimpleNamespace(mid=float(self._mid[symbol]))


class TestTurnoverCap(unittest.TestCase):
    def _cfg(self, mult: float | None, mode: str = "scale") -> BotConfig:
        raw = {
            "exchange": {"testnet": True, "category": "linear"},
            "rebalance": {"max_turnover_per_rebalance_equity_mult": mult, "turnover_cap_mode": mode},
            "backtest": {"start_date": "2023-01-01", "end_date": "2023-02-01"},
        }
        return BotConfig.model_validate(raw)

    def test_scales_non_risk_deltas(self) -> None:
        cfg = self._cfg(mult=1.0, mode="scale")
        md = _FakeMD({"AAAUSDT": 100.0})
        positions = {"AAAUSDT": Position(symbol="AAAUSDT", size=1.0, mark_price=100.0)}
        targets = {"AAAUSDT": 300.0}
        out, info = apply_turnover_cap(
            cfg=cfg,
            md=md,
            positions_actual=positions,
            target_notionals=targets,
            force_close_reasons=None,
            equity_usdt=100.0,
        )
        self.assertIsNotNone(info)
        # projected turnover = |300-100| = 200, cap = 100 => factor=0.5 => new target 200
        self.assertAlmostEqual(out["AAAUSDT"], 200.0, places=6)

    def test_risk_exits_bypass_cap(self) -> None:
        cfg = self._cfg(mult=1.0, mode="scale")
        md = _FakeMD({"AAAUSDT": 100.0})
        positions = {"AAAUSDT": Position(symbol="AAAUSDT", size=1.0, mark_price=100.0)}
        targets = {"AAAUSDT": 300.0}
        out, info = apply_turnover_cap(
            cfg=cfg,
            md=md,
            positions_actual=positions,
            target_notionals=targets,
            force_close_reasons={"AAAUSDT": "risk_loss_cap"},
            equity_usdt=100.0,
        )
        self.assertIsNone(info)
        self.assertAlmostEqual(out["AAAUSDT"], 300.0, places=6)

    def test_reconciliation_close_bypasses_cap(self) -> None:
        cfg = self._cfg(mult=0.1, mode="skip")
        md = _FakeMD({"AAAUSDT": 100.0})
        positions = {"AAAUSDT": Position(symbol="AAAUSDT", size=1.0, mark_price=100.0)}
        targets = {"AAAUSDT": 0.0}  # close outside universe
        out, info = apply_turnover_cap(
            cfg=cfg,
            md=md,
            positions_actual=positions,
            target_notionals=targets,
            force_close_reasons=None,
            equity_usdt=100.0,
        )
        self.assertIsNone(info)
        self.assertEqual(out["AAAUSDT"], 0.0)


if __name__ == "__main__":
    unittest.main()

