from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.backtest.metrics import BacktestMetrics
from src.config import BotConfig
from src.research.phase3 import build_phase3_variants, run_phase3_review


def _base_cfg() -> BotConfig:
    return BotConfig.model_validate(
        {
            "exchange": {"testnet": True, "category": "linear"},
            "filters": {"regime_filter": {"enabled": True, "use_market_regime": True, "action": "switch_to_momentum"}},
            "funding": {"model_in_backtest": False, "filter": {"enabled": True, "max_abs_daily_funding_rate": 0.001}},
            "backtest": {"start_date": "2023-01-01", "end_date": "2023-03-01", "initial_equity": 10000.0},
        }
    )


class _FakeMD:
    def get_daily_candles(self, symbol, start, end, *, use_cache=True, cache_write=True):
        dates = pd.date_range("2023-01-01", periods=20, freq="D", tz="UTC")
        return pd.DataFrame({"close": range(100, 120)}, index=dates)

    def get_daily_funding_rate(self, symbol, start, end, *, force_mainnet=None):
        dates = pd.date_range("2023-01-02", periods=19, freq="D", tz="UTC")
        return pd.Series(0.0, index=dates)


def _fake_result(variant_name: str):
    dates = pd.date_range("2023-01-02", periods=19, freq="D", tz="UTC")
    daily_ret = pd.Series([0.001] * len(dates), index=dates)
    equity = pd.Series([10000.0 * (1.001 ** (i + 1)) for i in range(len(dates))], index=dates)
    turnover = pd.Series([0.1] * len(dates), index=dates)
    metrics = BacktestMetrics(
        cagr=0.10,
        sharpe=1.2,
        sortino=1.4,
        max_drawdown=-0.05,
        calmar=2.0,
        profit_factor=1.3,
        win_rate=0.55,
        avg_daily_turnover=0.1,
    )
    meta = {
        "execution_scenario": "mixed",
        "net_return_total": 0.12,
        "gross_return_total": 0.15,
        "rebalance_count": 5,
        "trade_days": 5,
        "long_pnl_total_usd": 120.0,
        "short_pnl_total_usd": -20.0 if variant_name == "short_only" else 80.0,
        "cost_summary": {
            "trading_cost_total_usd": 10.0,
            "funding_pnl_total_usd": -3.0,
            "borrow_cost_total_usd": 0.0,
        },
        "top_symbol_contributions_usd": {"AAAUSDT": 60.0, "BBBUSDT": 30.0},
        "warnings": ["w1", "w2"],
    }
    return SimpleNamespace(equity=equity, daily_returns=daily_ret, daily_turnover=turnover, metrics=metrics, meta=meta)


class Phase3ResearchTests(unittest.TestCase):
    def test_variant_generation_does_not_mutate_production_defaults(self) -> None:
        cfg = _base_cfg()
        original = cfg.model_dump(mode="python")
        variants = build_phase3_variants(cfg)
        self.assertGreaterEqual(len(variants), 10)
        self.assertEqual(cfg.model_dump(mode="python"), original)

    def test_disabled_filters_are_disabled_in_variants(self) -> None:
        cfg = _base_cfg()
        variants = {v.name: v for v in build_phase3_variants(cfg)}
        self.assertFalse(variants["no_regime_scaling"].cfg.filters.regime_filter.enabled)
        self.assertFalse(variants["no_funding_filter"].cfg.funding.filter.enabled)
        self.assertIsNone(variants["no_1d_confirmation"].cfg.signal.ret_1d_long_max)

    def test_short_only_variant_has_transform(self) -> None:
        cfg = _base_cfg()
        variants = {v.name: v for v in build_phase3_variants(cfg)}
        self.assertIsNotNone(variants["short_only"].target_transform)
        self.assertTrue(variants["long_only"].cfg.signal.long_only)

    def test_invalid_variant_fails_clearly(self) -> None:
        cfg = _base_cfg()
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "Unknown research variant"):
                run_phase3_review(cfg=cfg, md=_FakeMD(), output_dir=td, variant_names=["does_not_exist"])

    def test_output_files_are_written(self) -> None:
        cfg = _base_cfg()
        md = _FakeMD()

        def _run_variant_side_effect(variant, *, md, output_dir):
            return _fake_result(variant.name)

        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as td, patch("src.research.phase3._run_variant", side_effect=_run_variant_side_effect):
            outputs = run_phase3_review(cfg=cfg, md=md, output_dir=td, variant_names=["baseline", "long_only", "short_only"])

        for path in outputs.values():
            self.assertTrue(Path(path).exists())


if __name__ == "__main__":
    unittest.main()
