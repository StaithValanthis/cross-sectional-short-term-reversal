from __future__ import annotations

import tempfile
import unittest

from src.config import BotConfig
from src.live import _apply_leverage_on_startup


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def get_positions(self, *, category: str, settle_coin: str = "USDT"):
        return [
            {"symbol": "AAAUSDT", "side": "Buy", "size": "1"},
            {"symbol": "BBBUSDT", "side": "Sell", "size": "2"},
        ]

    def set_leverage(self, *, category: str, symbol: str, leverage: str) -> None:
        self.calls.append((category, symbol, leverage))


class _FakeMD:
    def get_liquidity_ranked_symbols(self):
        return ["AAAUSDT", "BBBUSDT"]


class TestLeverageStartup(unittest.TestCase):
    def test_startup_only_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            state_path = f"{td}/leverage_state.json"
            cfg = BotConfig.model_validate(
                {
                    "exchange": {
                        "testnet": True,
                        "category": "linear",
                        "set_leverage_on_startup": True,
                        "leverage": "5",
                        "leverage_apply_mode": "positions",
                        "leverage_symbols_max": 200,
                        "leverage_state_path": state_path,
                    },
                    "backtest": {"start_date": "2023-01-01", "end_date": "2023-02-01"},
                }
            )
            client = _FakeClient()
            md = _FakeMD()

            _apply_leverage_on_startup(cfg=cfg, client=client, md=md)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0][0], "linear")

            # Second call should read state file and do nothing
            _apply_leverage_on_startup(cfg=cfg, client=client, md=md)
            self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()

