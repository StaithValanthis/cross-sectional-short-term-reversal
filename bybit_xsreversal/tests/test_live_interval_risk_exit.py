from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orjson

from src.config import BotConfig
from src.live import _run_risk_exit_only_reconcile, run_live


class _FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def get_positions(self, *, category: str = "linear", settle_coin: str = "USDT"):
        return [{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "100", "positionIdx": 0}]

    def get_open_orders(self, *, category: str, symbol: str | None = None, settle_coin: str | None = None):
        return []

    def get_wallet_balance(self, *, account_type: str = "UNIFIED"):
        return {"list": [{"coin": [{"coin": "USDT", "equity": "1000"}], "totalEquity": "1000"}]}


class _FakeRiskManager:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def check(self, current_equity: float):
        return True, {"kill_switch": False, "current_equity": float(current_equity)}


class _FakeLock:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def acquire(self) -> bool:
        return True

    def release(self) -> None:
        return None


class LiveIntervalRiskExitTests(unittest.TestCase):
    def test_run_once_executes_risk_exit_reconcile_when_rebalance_is_skipped(self) -> None:
        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "rebalance": {"interval_days": 3, "time_utc": "00:05"},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )

        captured: dict[str, object] = {}

        with tempfile.TemporaryDirectory() as td:
            cwd0 = os.getcwd()
            try:
                os.chdir(td)
                state_dir = Path("outputs") / "state"
                state_dir.mkdir(parents=True, exist_ok=True)
                state_dir.joinpath("rebalance_state.json").write_bytes(
                    orjson.dumps({"last_rebalance_day": "2026-01-02"}, option=orjson.OPT_INDENT_2)
                )
                os.environ["BYBIT_API_KEY"] = "test"
                os.environ["BYBIT_API_SECRET"] = "test"

                def _fake_run_rebalance(*, cfg, client, md, target_notionals, dry_run):
                    captured["target_notionals"] = dict(target_notionals)
                    captured["dry_run"] = bool(dry_run)
                    return {"orders": [], "targets_effective": dict(target_notionals), "risk_exits": {}, "summary": {"planned_orders": 0}}

                with patch("src.live.BybitClient", _FakeClient), \
                     patch("src.live.RiskManager", _FakeRiskManager), \
                     patch("src.live.FileLock", _FakeLock), \
                     patch("src.live.fetch_equity_usdt", return_value=1000.0), \
                     patch("src.live.run_rebalance", side_effect=_fake_run_rebalance), \
                     patch("src.live.now_utc") as mock_now, \
                     patch("src.live.time.sleep", return_value=None):
                    from datetime import UTC, datetime

                    mock_now.return_value = datetime(2026, 1, 2, 0, 6, tzinfo=UTC)
                    run_live(cfg, dry_run=True, run_once=True, force=False)

                self.assertEqual(captured["target_notionals"], {"AAAUSDT": 100.0})
                self.assertTrue(captured["dry_run"])
            finally:
                os.chdir(cwd0)
                os.environ.pop("BYBIT_API_KEY", None)
                os.environ.pop("BYBIT_API_SECRET", None)

    def test_risk_exit_only_reconcile_closes_position_when_hold_days_exceeded(self) -> None:
        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "risk": {"max_hold_days": 1, "max_loss_per_position_pct_equity": 0.0, "cooldown_days_after_forced_exit": 0},
                "execution": {"ioc_fallback": False, "cancel_open_orders": "none"},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )

        class _FakeMD:
            def get_orderbook_stats(self, symbol: str):
                from types import SimpleNamespace

                return SimpleNamespace(mid=100.0, best_bid=99.5, best_ask=100.5)

            def get_instrument_meta(self, symbol: str):
                from src.data.market_data import InstrumentMeta

                return InstrumentMeta(symbol=symbol, qty_step=0.001, min_qty=0.0, max_qty=None, tick_size=0.01, min_notional=None)

        with tempfile.TemporaryDirectory() as td:
            cwd0 = os.getcwd()
            try:
                os.chdir(td)
                state_dir = Path("outputs") / "state"
                state_dir.mkdir(parents=True, exist_ok=True)
                state_dir.joinpath("positions_state.json").write_bytes(
                    orjson.dumps(
                        {
                            "version": 1,
                            "open_positions": {
                                "AAAUSDT": {
                                    "opened_at_day": "2026-01-01",
                                    "side_sign": 1,
                                    "cum_realised_at_open": 0.0,
                                }
                            },
                            "cooldowns": {},
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )

                with patch("src.live.fetch_equity_usdt", return_value=1000.0), \
                     patch("src.execution.rebalance.datetime") as mock_dt:
                    from datetime import UTC, datetime
                    from src.execution.risk import RiskManager

                    mock_dt.now.return_value = datetime(2026, 1, 3, 0, 6, tzinfo=UTC)
                    mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

                    res = _run_risk_exit_only_reconcile(
                        cfg=cfg,
                        client=_FakeClient(),
                        md=_FakeMD(),
                        risk=RiskManager(cfg=cfg.risk, state_dir=state_dir),
                        dry_run=True,
                    )

                assert res is not None
                self.assertEqual(len(res["orders"]), 1)
                self.assertEqual(res["orders"][0]["reason"], "risk_time_stop")
            finally:
                os.chdir(cwd0)

    def test_risk_exit_only_reconcile_does_not_emit_regular_rebalance_orders(self) -> None:
        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "execution": {"ioc_fallback": False, "cancel_open_orders": "none"},
                "risk": {"max_hold_days": 0, "max_loss_per_position_pct_equity": 0.0, "cooldown_days_after_forced_exit": 0},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )

        class _FakeMD:
            def get_orderbook_stats(self, symbol: str):
                from types import SimpleNamespace

                return SimpleNamespace(mid=100.0, best_bid=99.5, best_ask=100.5)

            def get_instrument_meta(self, symbol: str):
                from src.data.market_data import InstrumentMeta

                return InstrumentMeta(symbol=symbol, qty_step=0.001, min_qty=0.0, max_qty=None, tick_size=0.01, min_notional=None)

        class _MarkSkewClient(_FakeClient):
            def get_positions(self, *, category: str = "linear", settle_coin: str = "USDT"):
                return [{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "90", "positionIdx": 0}]

        with tempfile.TemporaryDirectory() as td:
            cwd0 = os.getcwd()
            try:
                os.chdir(td)
                state_dir = Path("outputs") / "state"
                state_dir.mkdir(parents=True, exist_ok=True)

                with patch("src.live.fetch_equity_usdt", return_value=1000.0):
                    from src.execution.risk import RiskManager

                    res = _run_risk_exit_only_reconcile(
                        cfg=cfg,
                        client=_MarkSkewClient(),
                        md=_FakeMD(),
                        risk=RiskManager(cfg=cfg.risk, state_dir=state_dir),
                        dry_run=True,
                    )

                assert res is not None
                self.assertEqual(res["orders"], [])
            finally:
                os.chdir(cwd0)


if __name__ == "__main__":
    unittest.main()
