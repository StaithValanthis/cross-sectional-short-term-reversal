from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.backtest.backtester import run_backtest
from src.config import BotConfig
from src.strategy.portfolio import PortfolioTargets


def _make_daily_frame(dates: pd.DatetimeIndex, *, base: float, turnover: float = 100000.0) -> pd.DataFrame:
    closes = [base for _ in range(len(dates))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(dates),
            "turnover": [turnover] * len(dates),
        },
        index=dates,
    )


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

    def get_daily_funding_rate(self, symbol: str, start, end, *, force_mainnet=None):
        df = self._candles[symbol]
        return pd.Series(0.0, index=pd.DatetimeIndex(df.index).floor("D"))


class BacktesterPhase2Tests(unittest.TestCase):
    def _base_cfg(self, **backtest_overrides) -> BotConfig:
        backtest = {
            "start_date": "2023-01-07",
            "end_date": "2023-01-10",
            "initial_equity": 10000.0,
            "slippage_bps": 0.0,
            "allow_partial_fills": False,
        }
        backtest.update(backtest_overrides)
        return BotConfig.model_validate(
            {
                "exchange": {"testnet": True, "category": "linear"},
                "universe": {"top_n_by_volume": 6, "min_history_days": 5},
                "signal": {"lookback_days": 1, "long_quantile": 0.2, "short_quantile": 0.2},
                "filters": {"regime_filter": {"enabled": False, "use_market_regime": False}},
                "funding": {"model_in_backtest": False, "filter": {"enabled": False}},
                "rebalance": {"interval_days": 1},
                "backtest": backtest,
            }
        )

    def test_symbols_with_insufficient_history_are_excluded(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        candles["SHORTUSDT"] = _make_daily_frame(dates[-4:], base=111.0)
        md = _FakeMD(candles)
        cfg = self._base_cfg()
        seen_universes: list[set[str]] = []

        def _fake_targets(*, candles, config, equity_usd, asof, market_proxy_candles, current_weights, funding_daily_rate):
            seen_universes.append(set(candles.keys()))
            universe = list(candles.keys())[:5]
            weights = {universe[0]: 0.5, universe[1]: -0.5}
            notionals = {universe[0]: 5000.0, universe[1]: -5000.0}
            return PortfolioTargets(weights=weights, notionals_usd=notionals, meta={}), SimpleNamespace()

        with tempfile.TemporaryDirectory() as td, patch("src.backtest.backtester.compute_targets_from_daily_candles", side_effect=_fake_targets):
            res = run_backtest(cfg, md, td)

        self.assertTrue(seen_universes)
        self.assertTrue(all("SHORTUSDT" not in u for u in seen_universes))
        self.assertGreaterEqual(res.meta["data_quality"]["excluded_counts"].get("insufficient_history", 0), 1)

    def test_recent_gaps_are_excluded_from_historical_universe(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        gap_dates = dates.delete(4)
        candles["GAPUSDT"] = _make_daily_frame(gap_dates, base=120.0)
        md = _FakeMD(candles)
        cfg = self._base_cfg()
        seen_universes: list[set[str]] = []

        def _fake_targets(*, candles, config, equity_usd, asof, market_proxy_candles, current_weights, funding_daily_rate):
            seen_universes.append(set(candles.keys()))
            universe = list(candles.keys())[:5]
            weights = {universe[0]: 0.5, universe[1]: -0.5}
            notionals = {universe[0]: 5000.0, universe[1]: -5000.0}
            return PortfolioTargets(weights=weights, notionals_usd=notionals, meta={}), SimpleNamespace()

        with tempfile.TemporaryDirectory() as td, patch("src.backtest.backtester.compute_targets_from_daily_candles", side_effect=_fake_targets):
            res = run_backtest(cfg, md, td)

        self.assertTrue(seen_universes)
        self.assertTrue(all("GAPUSDT" not in u for u in seen_universes))
        self.assertGreaterEqual(res.meta["data_quality"]["excluded_counts"].get("recent_gap", 0), 1)

    def test_allow_partial_fills_true_is_rejected_as_unsupported(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        md = _FakeMD(candles)
        cfg = self._base_cfg(allow_partial_fills=True)

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "allow_partial_fills"):
                run_backtest(cfg, md, td)

    def test_borrow_cost_reduces_short_backtest_equity(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        md = _FakeMD(candles)
        cfg0 = self._base_cfg(borrow_cost_bps=0.0)
        cfg1 = self._base_cfg(borrow_cost_bps=10.0)
        targets = PortfolioTargets(weights={"AAAUSDT": 0.5, "BBBUSDT": -0.5}, notionals_usd={"AAAUSDT": 5000.0, "BBBUSDT": -5000.0}, meta={})

        with tempfile.TemporaryDirectory() as td0, tempfile.TemporaryDirectory() as td1:
            with patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
                res0 = run_backtest(cfg0, md, td0)
            with patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
                res1 = run_backtest(cfg1, md, td1)

        self.assertLess(float(res1.equity.iloc[-1]), float(res0.equity.iloc[-1]))

    def test_execution_scenarios_change_costs_conservatively(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        md = _FakeMD(candles)
        targets = PortfolioTargets(weights={"AAAUSDT": 0.5, "BBBUSDT": -0.5}, notionals_usd={"AAAUSDT": 5000.0, "BBBUSDT": -5000.0}, meta={})

        cfg_opt = self._base_cfg(execution_scenario="optimistic_maker", maker_fee_bps=1.0, taker_fee_bps=6.0)
        cfg_mix = self._base_cfg(execution_scenario="mixed", maker_fee_bps=1.0, taker_fee_bps=6.0)
        cfg_tak = self._base_cfg(execution_scenario="conservative_taker", maker_fee_bps=1.0, taker_fee_bps=6.0)

        with tempfile.TemporaryDirectory() as td_opt, tempfile.TemporaryDirectory() as td_mix, tempfile.TemporaryDirectory() as td_tak:
            with patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
                res_opt = run_backtest(cfg_opt, md, td_opt)
            with patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
                res_mix = run_backtest(cfg_mix, md, td_mix)
            with patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
                res_tak = run_backtest(cfg_tak, md, td_tak)

        self.assertGreater(float(res_opt.equity.iloc[-1]), float(res_mix.equity.iloc[-1]))
        self.assertGreater(float(res_mix.equity.iloc[-1]), float(res_tak.equity.iloc[-1]))

    def test_backtest_meta_warns_about_survivorship_and_latency_limits(self) -> None:
        dates = pd.date_range("2023-01-01", periods=12, freq="D", tz="UTC")
        candles = {sym: _make_daily_frame(dates, base=100.0 + idx) for idx, sym in enumerate(["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"])}
        md = _FakeMD(candles)
        cfg = self._base_cfg()
        targets = PortfolioTargets(weights={"AAAUSDT": 0.5, "BBBUSDT": -0.5}, notionals_usd={"AAAUSDT": 5000.0, "BBBUSDT": -5000.0}, meta={})

        with tempfile.TemporaryDirectory() as td, patch("src.backtest.backtester.compute_targets_from_daily_candles", return_value=(targets, SimpleNamespace())):
            res = run_backtest(cfg, md, td)

        warnings = "\n".join(res.meta["warnings"])
        self.assertIn("current symbol seed", warnings)
        self.assertIn("Execution latency", warnings)


if __name__ == "__main__":
    unittest.main()
