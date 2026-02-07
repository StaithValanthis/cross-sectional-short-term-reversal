#!/usr/bin/env python3
"""
Sensitivity analysis: robustness of best counterfactual and rebalance/timeframe models
to slippage_bps, taker_fee_bps, and fill_model (rebalance only). Optional: exclude top 1 symbol.
Read-only. Outputs: outputs/research_30d/sensitivity_summary.csv

Run from repo root:
  python3 bybit_xsreversal/scripts/research/sensitivity_30d.py --days 30 --warmup_days 120 --top_k 3
"""
from __future__ import annotations

import argparse
import os
import sys
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


def _ensure_deps() -> None:
    if pd is None or np is None:
        raise RuntimeError("pandas and numpy required. pip install pandas numpy")


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
            result[sym] = df
    return result


def _load_episodes(out_dir: Path) -> pd.DataFrame | None:
    for name in ("episodes.parquet", "episodes.csv"):
        p = out_dir / name
        if p.exists():
            if p.suffix == ".parquet":
                return pd.read_parquet(p)
            df = pd.read_csv(p)
            for col in ("entry_ts", "exit_ts"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True)
            return df
    return None


def _load_config(config_path: Path):
    from src.config import load_config
    return load_config(config_path)


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description="Sensitivity analysis (read-only).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--warmup_days", type=int, default=120)
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d")
    parser.add_argument("--top_k", type=int, default=3, help="Top K counterfactual models and rebalance variants")
    parser.add_argument("--exclude_top_symbol", action="store_true", help="Exclude top 1 symbol by PnL in rebalance (fragility check)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    if not out_dir.exists():
        raise FileNotFoundError(f"Missing {out_dir}. Run trade_forensics_30d.py and counterfactuals first.")

    config_path = _PACKAGE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        config_path = _REPO_ROOT / "bybit_xsreversal" / "config" / "config.yaml"
    cfg = _load_config(config_path) if config_path.exists() else None
    actual_taker = 6.0
    if cfg:
        rc = getattr(cfg, "research_costs", None)
        bt = getattr(cfg, "backtest", None)
        actual_taker = float(getattr(rc, "taker_fee_bps", None) or getattr(bt, "taker_fee_bps", 6.0) or 6.0)

    sensitivity_rows: list[dict[str, Any]] = []

    # ---- Counterfactual sensitivity: best 3 models x (slippage_bps, taker_fee_bps) ----
    cf_path = out_dir / "counterfactuals_summary.csv"
    if cf_path.exists():
        cf_df = pd.read_csv(cf_path)
        if not cf_df.empty and "model" in cf_df.columns and "param" in cf_df.columns:
            cf_df = cf_df[cf_df["model"] != "baseline"].copy()
            cf_df["_key"] = cf_df["model"] + "_" + cf_df["param"].astype(str)
            top_cf = cf_df.nlargest(args.top_k, "net_pnl_usd")["_key"].tolist()
            episodes_df = _load_episodes(out_dir)
            candles_1h = _load_candles(out_dir, "1H")
            candles_4h = _load_candles(out_dir, "4H")
            if episodes_df is not None and not episodes_df.empty and (candles_1h or candles_4h):
                from scripts.research.counterfactuals_30d import run_counterfactuals_episodes
                for slippage_bps in [0, 5]:
                    for taker_override in [actual_taker, actual_taker + 2]:
                        summary, _, _ = run_counterfactuals_episodes(
                            episodes_df, candles_1h or candles_4h, candles_4h,
                            taker_fee_bps=taker_override, slippage_bps=float(slippage_bps),
                            base_slippage_bps=float(slippage_bps),
                        )
                        for _, row in summary.iterrows():
                            if row["model"] == "baseline":
                                continue
                            key = row["model"] + "_" + str(row["param"])
                            if key in top_cf:
                                sensitivity_rows.append({
                                    "source": "counterfactual",
                                    "model_or_variant": key,
                                    "slippage_bps": slippage_bps,
                                    "taker_fee_bps": taker_override,
                                    "fill_model": "",
                                    "net_pnl_usd": row["net_pnl_usd"],
                                    "max_dd_pct": row["max_dd_pct"],
                                    "n_trades": row.get("n_trades"),
                                })
            else:
                logger.warning("Episodes or candles missing; skipping counterfactual sensitivity.")
        else:
            logger.warning("counterfactuals_summary.csv missing or empty; skipping counterfactual sensitivity.")
    else:
        logger.warning("counterfactuals_summary.csv not found; skipping counterfactual sensitivity.")

    # ---- Rebalance sensitivity: best 3 variants x (slippage_bps, taker_fee_bps, fill_model) ----
    rb_path = out_dir / "rebalance_timeframe_summary.csv"
    if rb_path.exists():
        rb_df = pd.read_csv(rb_path)
        if not rb_df.empty and "variant" in rb_df.columns:
            top_rb = rb_df.nlargest(args.top_k, "net_pnl_usd")["variant"].tolist()
            candles_1d = _load_candles(out_dir, "1D")
            candles_4h = _load_candles(out_dir, "4H")
            if candles_1d or candles_4h:
                from scripts.research.rebalance_timeframe_counterfactuals_30d import run_rebalance_timeframe_sim
                dummy_cfg = type("Cfg", (), {
                    "signal": type("S", (), {"lookback_days": 1, "long_quantile": 0.2, "short_quantile": 0.2})(),
                    "sizing": type("Sz", (), {"vol_lookback_days": 14, "target_gross_leverage": 5.0, "max_leverage_per_symbol": 0.2, "max_notional_per_symbol": 25000, "min_notional_per_symbol": 5})(),
                    "rebalance": type("R", (), {"rebalance_fraction": 0.36, "min_weight_change_bps": 400})(),
                })()
                for slippage_bps in [0, 5]:
                    for taker_override in [actual_taker, actual_taker + 2]:
                        for fill_model in ["open", "vwap"]:
                            summary, _ = run_rebalance_timeframe_sim(
                                candles_1d, candles_4h, dummy_cfg,
                                eval_days=args.days, warmup_days=args.warmup_days,
                                taker_fee_bps=taker_override,
                                base_slippage_bps=float(slippage_bps),
                                fill_model=fill_model,
                            )
                            for _, row in summary.iterrows():
                                if row["variant"] in top_rb:
                                    sensitivity_rows.append({
                                        "source": "rebalance_timeframe",
                                        "model_or_variant": row["variant"],
                                        "slippage_bps": slippage_bps,
                                        "taker_fee_bps": taker_override,
                                        "fill_model": fill_model,
                                        "net_pnl_usd": row["net_pnl_usd"],
                                        "max_dd_pct": row["max_dd_pct"],
                                        "n_trades": row.get("n_trades"),
                                    })
            else:
                logger.warning("No 1D/4H candles; skipping rebalance sensitivity.")
        else:
            logger.warning("rebalance_timeframe_summary.csv missing or empty; skipping rebalance sensitivity.")
    else:
        logger.warning("rebalance_timeframe_summary.csv not found; skipping rebalance sensitivity.")

    out_df = pd.DataFrame(sensitivity_rows)
    if not out_df.empty:
        out_df.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        logger.info("Wrote sensitivity_summary.csv ({} rows) to {}", len(out_df), out_dir)
    else:
        out_df.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        logger.info("No sensitivity rows; wrote empty sensitivity_summary.csv to {}", out_dir)


if __name__ == "__main__":
    main()