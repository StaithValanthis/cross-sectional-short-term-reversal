import unittest
from types import SimpleNamespace
from datetime import date, timedelta

from src.config import RiskConfig, BotConfig
from src.execution.rebalance import Position, plan_rebalance_orders
from src.execution.risk_exits import (
    PositionsRiskState,
    OpenPositionMeta,
    CooldownMeta,
    RawPnlMetrics,
    apply_cooldown_exclusions,
    evaluate_risk_exits,
    mark_forced_exits_in_state,
    update_state_from_positions,
)
from src.data.market_data import InstrumentMeta


class _StubMD:
    def get_orderbook_stats(self, symbol: str):
        return SimpleNamespace(mid=100.0)

    def get_instrument_meta(self, symbol: str):
        return InstrumentMeta(symbol=symbol, qty_step=0.001, min_qty=0.0, max_qty=None, tick_size=0.01, min_notional=None)


class RiskExitsTests(unittest.TestCase):
    def test_state_update_new_flip_flat(self):
        today = date(2026, 1, 10)
        st = PositionsRiskState(version=1, open_positions={}, cooldowns={})
        pnl = {"ABCUSDT": RawPnlMetrics(unrealised_pnl=0.0, cum_realised_pnl=12.0)}

        # new position
        positions = {"ABCUSDT": Position(symbol="ABCUSDT", size=1.0, mark_price=100.0)}
        update_state_from_positions(st=st, positions=positions, pnl_metrics=pnl, today=today)
        self.assertIn("ABCUSDT", st.open_positions)
        self.assertEqual(st.open_positions["ABCUSDT"].opened_at_day, "2026-01-10")
        self.assertEqual(st.open_positions["ABCUSDT"].side_sign, 1)
        self.assertEqual(st.open_positions["ABCUSDT"].cum_realised_at_open, 12.0)

        # sign flip resets open day
        tomorrow = today + timedelta(days=1)
        positions2 = {"ABCUSDT": Position(symbol="ABCUSDT", size=-2.0, mark_price=100.0)}
        update_state_from_positions(st=st, positions=positions2, pnl_metrics=pnl, today=tomorrow)
        self.assertEqual(st.open_positions["ABCUSDT"].opened_at_day, "2026-01-11")
        self.assertEqual(st.open_positions["ABCUSDT"].side_sign, -1)

        # flat deletes open meta
        positions3 = {"ABCUSDT": Position(symbol="ABCUSDT", size=0.0, mark_price=100.0)}
        update_state_from_positions(st=st, positions=positions3, pnl_metrics=pnl, today=tomorrow)
        self.assertNotIn("ABCUSDT", st.open_positions)

    def test_time_stop_forces_close(self):
        cfg = RiskConfig(max_hold_days=3)
        today = date(2026, 1, 10)
        st = PositionsRiskState(
            version=1,
            open_positions={"ABCUSDT": OpenPositionMeta(opened_at_day="2026-01-07", side_sign=1)},
            cooldowns={},
        )
        positions = {"ABCUSDT": Position(symbol="ABCUSDT", size=1.0, mark_price=100.0)}
        reasons, events = evaluate_risk_exits(cfg=cfg, st=st, positions=positions, pnl_metrics={}, equity_usdt=1000.0, today=today)
        self.assertEqual(reasons.get("ABCUSDT"), "risk_time_stop")
        self.assertTrue(any(e.kind == "risk_time_stop" for e in events))

    def test_loss_cap_forces_close(self):
        cfg = RiskConfig(max_loss_per_position_pct_equity=0.5)
        today = date(2026, 1, 10)
        st = PositionsRiskState(
            version=1,
            open_positions={"ABCUSDT": OpenPositionMeta(opened_at_day="2026-01-09", side_sign=1, cum_realised_at_open=0.0)},
            cooldowns={},
        )
        positions = {"ABCUSDT": Position(symbol="ABCUSDT", size=1.0, mark_price=100.0)}
        pnl = {"ABCUSDT": RawPnlMetrics(unrealised_pnl=-10.0, cum_realised_pnl=0.0)}  # 1% loss on 1000 equity
        reasons, events = evaluate_risk_exits(cfg=cfg, st=st, positions=positions, pnl_metrics=pnl, equity_usdt=1000.0, today=today)
        self.assertEqual(reasons.get("ABCUSDT"), "risk_loss_cap")
        ev = [e for e in events if e.kind == "risk_loss_cap"][0]
        self.assertGreaterEqual(ev.loss_pct_equity or 0.0, 0.5)

    def test_cooldown_excludes_targets(self):
        cfg = RiskConfig(cooldown_days_after_forced_exit=3)
        today = date(2026, 1, 10)
        st = PositionsRiskState(
            version=1,
            open_positions={},
            cooldowns={"ABCUSDT": CooldownMeta(last_forced_exit_day="2026-01-09")},
        )
        targets = {"ABCUSDT": 100.0, "XYZUSDT": -50.0}
        syms, _events = apply_cooldown_exclusions(cfg=cfg, st=st, target_notionals=targets, today=today)
        self.assertIn("ABCUSDT", syms)
        self.assertEqual(targets["ABCUSDT"], 0.0)
        self.assertEqual(targets["XYZUSDT"], -50.0)

    def test_plan_orders_uses_forced_reason_and_closes_full(self):
        # Integration-ish: ensure forced close becomes a full close order with reason, even if target was non-zero
        md = _StubMD()
        cfg = BotConfig.model_validate(
            {
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )
        cur = {"ABCUSDT": Position(symbol="ABCUSDT", size=1.0, mark_price=100.0)}
        # Effective targets overridden to 0
        targets = {"ABCUSDT": 0.0}
        orders = plan_rebalance_orders(
            cfg=cfg,
            md=md,
            current_positions=cur,
            target_notionals=targets,
            force_close_reasons={"ABCUSDT": "risk_time_stop"},
        )
        self.assertEqual(len(orders), 1)
        o = orders[0]
        self.assertTrue(o.reduce_only)
        self.assertEqual(o.reason, "risk_time_stop")
        self.assertEqual(o.side, "Sell")
        self.assertAlmostEqual(o.qty, 1.0, places=9)

    def test_force_close_ignores_min_notional_and_small_delta_gates(self):
        # Closing positions must not be skipped even if notional is tiny vs min_notional_per_symbol.
        md = _StubMD()
        cfg = BotConfig.model_validate(
            {
                "sizing": {"min_notional_per_symbol": 10_000.0},  # absurdly high to trip "already at target" gate
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )
        cur = {"DUSTUSDT": Position(symbol="DUSTUSDT", size=0.001, mark_price=100.0)}  # $0.10 notional
        targets = {"DUSTUSDT": 0.0}
        orders = plan_rebalance_orders(
            cfg=cfg,
            md=md,
            current_positions=cur,
            target_notionals=targets,
            force_close_reasons={"DUSTUSDT": "risk_loss_cap"},
        )
        self.assertEqual(len(orders), 1)
        self.assertTrue(orders[0].reduce_only)
        self.assertEqual(orders[0].reason, "risk_loss_cap")


if __name__ == "__main__":
    unittest.main()

