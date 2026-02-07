from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.execution.intraday_exits import (
    IntradayExitState,
    IntradayStateSymbol,
    breakeven_stop,
    compute_atr_wilder,
    fixed_stop_price,
    time_stop_trigger,
    trailing_stop_price,
    trigger_intrabar,
    update_trailing_extreme,
    evaluate_intraday_exit_decisions,
)
from src.config import IntradayExitsConfig


class TestIntradayExitsPure(unittest.TestCase):
    def test_atr_wilder_simple(self) -> None:
        high = np.array([10, 11, 12, 13], dtype=float)
        low = np.array([9, 10, 11, 12], dtype=float)
        close = np.array([9.5, 10.5, 11.5, 12.5], dtype=float)
        atr = compute_atr_wilder(high, low, close, period=2)
        self.assertIsNotNone(atr)
        self.assertAlmostEqual(float(atr or 0.0), 1.5, places=6)

    def test_fixed_stop_long_short(self) -> None:
        self.assertAlmostEqual(fixed_stop_price(100.0, 2.0, 2.0, 1), 96.0)
        self.assertAlmostEqual(fixed_stop_price(100.0, 2.0, 2.0, -1), 104.0)

    def test_trailing_extreme_update(self) -> None:
        self.assertEqual(update_trailing_extreme(None, 10.0, 9.0, 1), 10.0)
        self.assertEqual(update_trailing_extreme(10.0, 11.0, 10.0, 1), 11.0)
        self.assertEqual(update_trailing_extreme(None, 10.0, 9.0, -1), 9.0)
        self.assertEqual(update_trailing_extreme(9.0, 10.0, 8.5, -1), 8.5)

    def test_trailing_stop_price_long_short(self) -> None:
        self.assertAlmostEqual(trailing_stop_price(110.0, 2.0, 2.5, 1), 105.0)
        self.assertAlmostEqual(trailing_stop_price(90.0, 2.0, 2.5, -1), 95.0)

    def test_breakeven_stop(self) -> None:
        self.assertAlmostEqual(breakeven_stop(100.0, 10.0, 1), 100.1)
        self.assertAlmostEqual(breakeven_stop(100.0, 10.0, -1), 99.9)

    def test_intrabar_trigger(self) -> None:
        self.assertTrue(trigger_intrabar(1, bar_high=101.0, bar_low=99.0, stop=100.0))
        self.assertFalse(trigger_intrabar(1, bar_high=101.0, bar_low=100.5, stop=100.0))
        self.assertTrue(trigger_intrabar(-1, bar_high=101.0, bar_low=99.0, stop=100.0))
        self.assertFalse(trigger_intrabar(-1, bar_high=99.5, bar_low=98.0, stop=100.0))

    def test_time_stop_trigger(self) -> None:
        now = datetime.now(tz=UTC)
        entry_ts = (now - timedelta(hours=25)).isoformat()
        self.assertTrue(time_stop_trigger(entry_ts, now, hours=24, only_if_unprofitable=False, unrealized_pnl=None))
        self.assertTrue(time_stop_trigger(entry_ts, now, hours=24, only_if_unprofitable=True, unrealized_pnl=-1.0))
        self.assertFalse(time_stop_trigger(entry_ts, now, hours=24, only_if_unprofitable=True, unrealized_pnl=1.0))


class TestIntradayEvaluator(unittest.TestCase):
    def test_evaluator_emits_reduce_only_full_close(self) -> None:
        cfg = IntradayExitsConfig(
            enabled=True,
            dry_run=True,
            fixed_atr_stop_enabled=True,
            fixed_atr_k=1.0,
            trailing_atr_stop_enabled=False,
            stop_to_breakeven_enabled=False,
            time_stop_enabled=False,
            use_intrabar_trigger=True,
            min_bars_trigger=1,
            min_bars_atr=3,
            atr_period=2,
            min_notional_to_exit_usd=0.0,
        )
        now = datetime.now(tz=UTC)
        # Position: long 1.0, entry 100
        positions = [{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "avgPrice": "100", "markPrice": "100"}]

        # 4H candles (need >= atr_period+1 = 3 bars)
        c4 = pd.DataFrame(
            {
                "open": [100, 100, 100],
                "high": [101, 101, 101],
                "low": [99, 99, 99],
                "close": [100, 100, 100],
                "volume": [1, 1, 1],
                "turnover": [1, 1, 1],
            },
            index=pd.to_datetime([now - timedelta(hours=8), now - timedelta(hours=4), now], utc=True),
        )
        # 1H last bar crosses stop (entry - 1*atr). atr should be ~2 here, stop ~98, so set low 97.
        c1 = pd.DataFrame(
            {"open": [100], "high": [101], "low": [97], "close": [99], "volume": [1], "turnover": [1]},
            index=pd.to_datetime([now], utc=True),
        )

        st = IntradayExitState(version=1, last_run_ts=None, per_symbol={"AAAUSDT": IntradayStateSymbol(entry_ts=now.isoformat())})
        decisions = evaluate_intraday_exit_decisions(
            cfg=cfg,
            positions=positions,
            candles_1h_by_symbol={"AAAUSDT": c1},
            candles_4h_by_symbol={"AAAUSDT": c4},
            last_prices_by_symbol=None,
            state=st,
            now_ts=now,
        )
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.symbol, "AAAUSDT")
        self.assertTrue(d.reduce_only)
        self.assertEqual(d.qty_to_close, 1.0)
        self.assertEqual(d.side, "Sell")
        self.assertTrue(d.reason.startswith("risk_"))


if __name__ == "__main__":
    unittest.main()

