#!/usr/bin/env python3
"""
Phase E: Timeframe & parameter improvement tests (constrained, last-30d focus).
Read-only. Uses trades_enriched + candles; optionally runs approximate delta backtest.
Run from repo root:
  python3 bybit_xsreversal/scripts/research/optimize_params_30d.py --days 30 --warmup_days 120

Outputs:
  - outputs/research_30d/param_search_results.csv
  - outputs/research_30d/best_configs.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SCRIPT_DIR.parent.parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from loguru import logger

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

try:
    import yaml
except ImportError:
    yaml = None


def _ensure_deps() -> None:
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy required. pip install pandas numpy")


def _load_config(config_path: Path):
    from src.config import load_config
    return load_config(config_path)


def _ensure_linear_category(cfg, args_category: str | None) -> str:
    """Research scripts: USDT linear perps only. Validate/warn and return 'linear'."""
    config_cat = getattr(getattr(cfg, "exchange", None), "category", None) if cfg else None
    config_cat = config_cat or "linear"
    cat = (args_category or config_cat or "linear").strip().lower()
    if cat != "linear":
        logger.warning("Research pipeline is for USDT linear perps only; forcing category='linear' (was {})", cat)
        return "linear"
    return "linear"


def _load_candles(out_dir: Path, interval: str) -> dict[str, pd.DataFrame]:
    base = out_dir / "candles" / interval
    result = {}
    if not base.exists():
        return result
    for p in base.iterdir():
        if p.suffix in (".parquet", ".csv"):
            sym = p.stem
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            if "ts_open_utc" in df.columns:
                df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                df = df.sort_values("ts_open_utc").reset_index(drop=True)
            result[sym] = df
    return result


def _objective_score(
    net_pnl: float,
    max_dd_pct: float,
    fees: float,
    n_trades: int,
    lambda_dd: float = 0.5,
    mu_fees: float = 0.001,
    min_trades: int = 10,
) -> float:
    """Objective: net_pnl - lambda*max_dd - mu*fees, with min_trades constraint to avoid degenerate configs."""
    if n_trades < min_trades:
        return -1e9
    return net_pnl - lambda_dd * abs(max_dd_pct) - mu_fees * fees


def run_param_search(
    candles_1d: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame],
    out_dir: Path,
    eval_days: int = 30,
    warmup_days: int = 120,
    lambda_dd: float = 0.5,
    mu_fees: float = 0.001,
    min_trades: int = 10,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Bounded grid over strategy params; score each config via Task 5 simulator (24h/1D variant).
    Search space: lookback_days [3,5,7,10,14,21], long_quantile [0.05..0.25], short_quantile [0.75..0.95],
    vol_lookback_days [7,14,21,30], rebalance_fraction [0.15,0.25,0.36,0.50], min_weight_change_bps [200,400,600,800],
    gross_exposure [1,2,3,4,5]. Objective: net_pnl - lambda*max_dd - mu*fees; min_trades constraint.
    """
    from scripts.research.rebalance_timeframe_counterfactuals_30d import score_config

    lookback_days = [3, 5, 7, 10, 14, 21]
    long_quantile = [0.05, 0.10, 0.15, 0.20, 0.25]
    short_quantile = [0.75, 0.80, 0.85, 0.90, 0.95]
    vol_lookback_days = [7, 14, 21, 30]
    rebalance_fraction = [0.15, 0.25, 0.36, 0.50]
    min_weight_change_bps = [200, 400, 600, 800]
    gross_exposure = [1, 2, 3, 4, 5]

    grid = []
    for lb in lookback_days:
        for lq in long_quantile:
            for sq in short_quantile:
                if lq >= sq:
                    continue
                for vl in vol_lookback_days:
                    for rf in rebalance_fraction:
                        for mw in min_weight_change_bps:
                            for ge in gross_exposure:
                                grid.append({
                                    "lookback_days": lb,
                                    "long_quantile": lq,
                                    "short_quantile": sq,
                                    "vol_lookback_days": vl,
                                    "rebalance_fraction": rf,
                                    "min_weight_change_bps": mw,
                                    "target_gross_leverage": ge,
                                })
    import random
    max_configs = 80
    if len(grid) > max_configs:
        random.seed(42)
        grid = random.sample(grid, max_configs)
    if not candles_1d and not candles_4h:
        return pd.DataFrame(columns=["lookback_days", "long_quantile", "short_quantile", "vol_lookback_days", "rebalance_fraction", "min_weight_change_bps", "target_gross_leverage", "net_pnl", "max_dd_pct", "fees", "n_trades", "objective"]), []

    results = []
    for i, g in enumerate(grid):
        try:
            net_pnl, max_dd, fees, n_trades = score_config(
                candles_1d, candles_4h, g, eval_days=eval_days, warmup_days=warmup_days,
            )
        except Exception as e:
            logger.warning("Param run failed for {}: {}", g, e)
            net_pnl, max_dd, fees, n_trades = -1e9, 0.0, 0.0, 0
        obj = _objective_score(net_pnl, max_dd, fees, n_trades, lambda_dd=lambda_dd, mu_fees=mu_fees, min_trades=min_trades)
        results.append({
            **g,
            "net_pnl": net_pnl,
            "max_dd_pct": max_dd,
            "fees": fees,
            "n_trades": n_trades,
            "objective": obj,
        })
        if (i + 1) % 50 == 0:
            logger.info("Param search progress: {}/{}", i + 1, len(grid))
    df = pd.DataFrame(results)
    df = df.sort_values("objective", ascending=False).reset_index(drop=True)
    best = df.head(5).to_dict("records")
    return df, best


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description="Phase E: Parameter search (read-only, constrained).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--warmup_days", type=int, default=120)
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d")
    parser.add_argument("--category", type=str, default=None, help="Bybit category (default: linear; enforced linear for research)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    if not out_dir.exists():
        raise FileNotFoundError(f"Run trade_forensics_30d.py first. Missing {out_dir}")

    config_path = _PACKAGE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        config_path = _REPO_ROOT / "bybit_xsreversal" / "config" / "config.yaml"
    cfg = _load_config(config_path) if config_path.exists() else None
    category = _ensure_linear_category(cfg, args.category)

    candles_1d = _load_candles(out_dir, "1D")
    candles_4h = _load_candles(out_dir, "4H")
    results_df, best_configs = run_param_search(
        candles_1d, candles_4h, out_dir, eval_days=args.days, warmup_days=args.warmup_days,
    )
    results_df.to_csv(out_dir / "param_search_results.csv", index=False)

    if yaml and best_configs:
        with open(out_dir / "best_configs.yaml", "w") as f:
            yaml.dump({"top_configs": best_configs[:5], "note": "Scored via rebalance_timeframe simulator (24h/1D)."}, f, default_flow_style=False)
    logger.info("Wrote param_search_results.csv and best_configs.yaml to {}", out_dir)


if __name__ == "__main__":
    main()