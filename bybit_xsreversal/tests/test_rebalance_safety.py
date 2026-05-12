from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.config import BotConfig
from src.data.market_data import InstrumentMeta
from src.execution.rebalance import Position, _signed_open_order_pending_qty, run_rebalance


class _StubMD:
    def get_orderbook_stats(self, symbol: str):
        return SimpleNamespace(mid=100.0, best_bid=99.5, best_ask=100.5)

    def get_instrument_meta(self, symbol: str):
        return InstrumentMeta(symbol=symbol, qty_step=0.001, min_qty=0.0, max_qty=None, tick_size=0.01, min_notional=None)


class _StubClient:
    def __init__(self, positions: list[dict], open_orders_by_symbol: dict[str, list[dict]] | None = None) -> None:
        self._positions = list(positions)
        self._open_orders_by_symbol = dict(open_orders_by_symbol or {})

    def get_positions(self, *, category: str = "linear", settle_coin: str = "USDT"):
        return list(self._positions)

    def get_open_orders(self, *, category: str, symbol: str | None = None, settle_coin: str | None = None):
        if symbol is None:
            out: list[dict] = []
            for orders in self._open_orders_by_symbol.values():
                out.extend(list(orders))
            return out
        return list(self._open_orders_by_symbol.get(symbol, []))

    def get_wallet_balance(self, *, account_type: str = "UNIFIED"):
        return {"list": [{"coin": [{"coin": "USDT", "equity": "1000"}], "totalEquity": "1000"}]}

    def cancel_order(self, *, category: str, symbol: str, order_id: str) -> None:
        return None

    def cancel_all_orders(self, *, category: str, symbol: str | None = None, settle_coin: str | None = "USDT", base_coin: str | None = None, order_filter: str | None = None):
        return {}

    def create_order(self, *, category: str, order: dict):
        return {"orderId": "test-order"}


class PendingOrderExposureTests(unittest.TestCase):
    def test_counts_only_open_quantity(self) -> None:
        qty = _signed_open_order_pending_qty(
            [
                {"side": "Buy", "qty": "2.0", "cumExecQty": "0.0", "orderStatus": "New"},
                {"side": "Sell", "qty": "5.0", "cumExecQty": "2.0", "orderStatus": "PartiallyFilled"},
            ]
        )
        self.assertAlmostEqual(qty, -1.0, places=9)

    def test_filled_cancelled_and_rejected_do_not_count(self) -> None:
        qty = _signed_open_order_pending_qty(
            [
                {"side": "Buy", "qty": "1.0", "cumExecQty": "1.0", "orderStatus": "Filled"},
                {"side": "Buy", "qty": "3.0", "cumExecQty": "0.0", "orderStatus": "Cancelled"},
                {"side": "Sell", "qty": "4.0", "cumExecQty": "0.0", "orderStatus": "Rejected"},
            ]
        )
        self.assertAlmostEqual(qty, 0.0, places=9)

    def test_missing_status_treated_as_open_but_remaining_only(self) -> None:
        qty = _signed_open_order_pending_qty(
            [
                {"side": "Buy", "qty": "2.5", "cumExecQty": "1.0"},
            ]
        )
        self.assertAlmostEqual(qty, 1.5, places=9)

    def test_mixed_position_and_partial_open_order_use_remaining_qty_only(self) -> None:
        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "execution": {"ioc_fallback": False, "cancel_open_orders": "none"},
                "risk": {"max_hold_days": 0, "max_loss_per_position_pct_equity": 0.0, "cooldown_days_after_forced_exit": 0},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )
        client = _StubClient(
            positions=[{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "100", "positionIdx": 0}],
            open_orders_by_symbol={
                "AAAUSDT": [
                    {"symbol": "AAAUSDT", "side": "Buy", "qty": "2", "cumExecQty": "1", "orderStatus": "PartiallyFilled"}
                ]
            },
        )
        res = run_rebalance(cfg=cfg, client=client, md=_StubMD(), target_notionals={"AAAUSDT": 200.0}, dry_run=True)
        self.assertEqual(res["orders"], [])

    def test_pending_sell_order_reduces_effective_long_exposure(self) -> None:
        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "execution": {"ioc_fallback": False, "cancel_open_orders": "none"},
                "risk": {"max_hold_days": 0, "max_loss_per_position_pct_equity": 0.0, "cooldown_days_after_forced_exit": 0},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )
        client = _StubClient(
            positions=[{"symbol": "AAAUSDT", "side": "Buy", "size": "3", "markPrice": "100", "positionIdx": 0}],
            open_orders_by_symbol={
                "AAAUSDT": [
                    {"symbol": "AAAUSDT", "side": "Sell", "qty": "2", "cumExecQty": "1", "orderStatus": "PartiallyFilled"}
                ]
            },
        )
        res = run_rebalance(cfg=cfg, client=client, md=_StubMD(), target_notionals={"AAAUSDT": 200.0}, dry_run=True)
        self.assertEqual(res["orders"], [])


class EmptyTargetFlattenTests(unittest.TestCase):
    def _cfg(self, *, flatten_on_empty_targets: bool) -> BotConfig:
        return BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "rebalance": {"flatten_on_empty_targets": flatten_on_empty_targets, "cancel_open_orders": "none"},
                "execution": {"ioc_fallback": False, "cancel_open_orders": "none"},
                "risk": {"max_hold_days": 0, "max_loss_per_position_pct_equity": 0.0, "cooldown_days_after_forced_exit": 0},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-02"},
            }
        )

    def test_empty_targets_preserve_positions_when_flatten_disabled(self) -> None:
        cfg = self._cfg(flatten_on_empty_targets=False)
        client = _StubClient(
            positions=[{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "100", "positionIdx": 0}],
        )
        res = run_rebalance(cfg=cfg, client=client, md=_StubMD(), target_notionals={}, dry_run=True)
        self.assertEqual(res["orders"], [])
        self.assertIn("AAAUSDT", res["targets_effective"])

    def test_empty_targets_flatten_positions_when_enabled(self) -> None:
        cfg = self._cfg(flatten_on_empty_targets=True)
        client = _StubClient(
            positions=[{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "100", "positionIdx": 0}],
        )
        res = run_rebalance(cfg=cfg, client=client, md=_StubMD(), target_notionals={}, dry_run=True)
        self.assertEqual(len(res["orders"]), 1)
        self.assertEqual(res["orders"][0]["side"], "Sell")

    def test_forced_exit_still_closes_when_flatten_disabled(self) -> None:
        cfg = self._cfg(flatten_on_empty_targets=False)
        client = _StubClient(
            positions=[{"symbol": "AAAUSDT", "side": "Buy", "size": "1", "markPrice": "100", "positionIdx": 0}],
        )
        res = run_rebalance(cfg=cfg, client=client, md=_StubMD(), target_notionals={"AAAUSDT": 0.0}, dry_run=True)
        self.assertEqual(len(res["orders"]), 1)
        self.assertEqual(res["orders"][0]["side"], "Sell")


if __name__ == "__main__":
    unittest.main()
