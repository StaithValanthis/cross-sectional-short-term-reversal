from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.config import BotConfig
from src.strategy.portfolio import PortfolioTargets
from src.backtest.backtester import run_backtest


class _FakeMD:
    def __init__(self, candles_by_symbol: dict[str, pd.DataFrame]) -> None:
        self._candles = dict(candles_by_symbol)
        self.universe_snapshots: list[tuple[list[str], dict]] = []

    def get_liquidity_ranked_symbols(self):
        return list(self._candles.keys())

    def cache_universe_snapshot(self, *, symbols: list[str], meta: dict):
        self.universe_snapshots.append((list(symbols), dict(meta)))

    def get_daily_candles(self, symbol: str, start, end, *, use_cache: bool = True, cache_write: bool = True):
        return self._candles[symbol]


class BacktesterRegressionTests(unittest.TestCase):
    def test_signal_change_updates_holdings_and_turnover(self) -> None:
        dates = pd.date_range("2023-01-01", periods=6, freq="D", tz="UTC")
        candles_by_symbol: dict[str, pd.DataFrame] = {}
        for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"]):
            closes = [100 + idx + i for i in range(len(dates))]
            candles_by_symbol[sym] = pd.DataFrame(
                {
                    "open": closes,
                    "high": [c + 1 for c in closes],
                    "low": [c - 1 for c in closes],
                    "close": closes,
                    "volume": [1000.0] * len(dates),
                    "turnover": [100000.0] * len(dates),
                },
                index=dates,
            )

        cfg = BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "signal": {"lookback_days": 1, "long_quantile": 0.2, "short_quantile": 0.2},
                "filters": {"regime_filter": {"enabled": False, "use_market_regime": False}},
                "funding": {"model_in_backtest": False, "filter": {"enabled": False}},
                "rebalance": {"interval_days": 1},
                "backtest": {"start_date": "2023-01-01", "end_date": "2023-01-06", "initial_equity": 10000.0},
            }
        )
        md = _FakeMD(candles_by_symbol)

        target_a = PortfolioTargets(weights={"AAAUSDT": 0.5, "BBBUSDT": -0.5}, notionals_usd={"AAAUSDT": 5000.0, "BBBUSDT": -5000.0}, meta={})
        target_b = PortfolioTargets(weights={"CCCUSDT": 0.5, "DDDUSDT": -0.5}, notionals_usd={"CCCUSDT": 5000.0, "DDDUSDT": -5000.0}, meta={})
        side_effects = [
            (target_a, SimpleNamespace()),
            (target_b, SimpleNamespace()),
            (target_b, SimpleNamespace()),
            (target_b, SimpleNamespace()),
            (target_b, SimpleNamespace()),
        ]

        with tempfile.TemporaryDirectory() as td, patch("src.backtest.backtester.compute_targets_from_daily_candles", side_effect=side_effects):
            res = run_backtest(cfg, md, td)

        self.assertTrue((res.daily_turnover > 0).any())
        self.assertGreater(float(res.daily_turnover.iloc[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
