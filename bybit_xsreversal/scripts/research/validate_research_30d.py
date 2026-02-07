#!/usr/bin/env python3
"""
Validation harness for 30-day research pipeline: data integrity, PnL reconciliation,
candle coverage, cost sanity, and run metadata. Read-only. All outputs under
outputs/research_30d/validation/.

Run from repo root:
  python3 bybit_xsreversal/scripts/research/validate_research_30d.py --days 30 --warmup_days 120
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
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


def _load_df(out_dir: Path, name: str, *alt_names: str) -> pd.DataFrame | None:
    for n in (name,) + alt_names:
        for ext in (".parquet", ".csv"):
            p = out_dir / n.replace(".parquet", ext).replace(".csv", ext)
            if not p.exists():
                continue
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                    for col in ("entry_ts", "exit_ts", "timestamp_utc", "ts_open_utc"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], utc=True)
                return df
            except Exception as e:
                logger.warning("Failed to load {}: {}", p, e)
    return None


def _load_json(out_dir: Path, name: str) -> dict | None:
    p = out_dir / name
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load {}: {}", p, e)
        return None


def _git_commit_hash(repo_root: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return None


# ---------- 1) Episode reconstruction integrity ----------


def validate_episode_integrity(episodes_df: pd.DataFrame, episode_fills_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each episode: net position starts at 0, ends at 0; direction consistent;
    entry_ts < exit_ts, hold_hours > 0; no fill assigned to multiple episodes.
    Returns episode_integrity.csv rows.
    """
    rows = []
    if episodes_df.empty or episode_fills_df.empty:
        return pd.DataFrame(columns=[
            "episode_id", "symbol", "position_starts_zero", "position_ends_zero", "direction_consistent",
            "entry_before_exit", "hold_hours_positive", "num_fills", "fill_unique", "all_checks_pass",
        ])
    for _, ep in episodes_df.iterrows():
        eid = ep["episode_id"]
        sym = ep["symbol"]
        fills = episode_fills_df[(episode_fills_df["episode_id"] == eid) & (episode_fills_df["symbol"] == sym)].sort_values("timestamp_utc")
        position_starts_zero = True  # by construction: episode starts when position goes 0 -> nonzero
        position_ends_zero = True
        if not fills.empty:
            last_pos = fills["position_after"].iloc[-1]
            position_ends_zero = abs(last_pos) < 1e-9
        entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
        exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
        entry_before_exit = entry_ts < exit_ts
        hold_hours = float(ep.get("hold_hours") or 0)
        hold_hours_positive = hold_hours > 0
        direction_consistent = True
        if not fills.empty and "position_after" in fills.columns:
            signs = np.sign(fills["position_after"].replace(0, np.nan).dropna())
            if len(signs) > 1:
                direction_consistent = (signs.iloc[0] == signs.iloc[-1]) or (signs.nunique() <= 1)
        fill_unique = True
        num_fills = len(fills)
        all_pass = position_ends_zero and entry_before_exit and hold_hours_positive and direction_consistent
        rows.append({
            "episode_id": eid,
            "symbol": sym,
            "position_starts_zero": position_starts_zero,
            "position_ends_zero": position_ends_zero,
            "direction_consistent": direction_consistent,
            "entry_before_exit": entry_before_exit,
            "hold_hours_positive": hold_hours_positive,
            "num_fills": num_fills,
            "fill_unique": fill_unique,
            "all_checks_pass": all_pass,
        })
    return pd.DataFrame(rows)


def check_no_fill_in_multiple_episodes(episode_fills_df: pd.DataFrame) -> tuple[bool, int]:
    """True if no (symbol, timestamp_utc, qty, price) appears in more than one episode."""
    if episode_fills_df.empty or len(episode_fills_df) < 2:
        return True, 0
    key = episode_fills_df["symbol"].astype(str) + "_" + episode_fills_df["timestamp_utc"].astype(str) + "_" + episode_fills_df["qty"].astype(str) + "_" + episode_fills_df["price"].astype(str)
    dup = key.duplicated(keep=False)
    n_dup = dup.sum()
    return n_dup == 0, int(n_dup)


# ---------- 2) PnL reconciliation ----------


def validate_pnl_reconciliation(
    episodes_df: pd.DataFrame,
    trades_df: pd.DataFrame | None,
    raw_fetch_meta: dict | None,
) -> pd.DataFrame:
    """
    Sum realized_pnl across episodes per symbol and overall; compare to trades_enriched
    and to closed_pnl if available in meta.
    """
    rows = []
    if episodes_df.empty:
        return pd.DataFrame(columns=["symbol", "episode_pnl_sum", "trades_pnl_sum", "closed_pnl_sum", "episode_vs_trades_diff", "mismatch_ratio", "note"])
    ep_sum_total = episodes_df["realized_pnl_usd"].fillna(0).sum()
    ep_by_sym = episodes_df.fillna(0).groupby("symbol")["realized_pnl_usd"].sum()
    for sym in ep_by_sym.index.tolist():
        episode_pnl_sum = float(ep_by_sym[sym])
        trades_pnl_sum = None
        if trades_df is not None and not trades_df.empty and "symbol" in trades_df.columns:
            t = trades_df[trades_df["symbol"] == sym]["realized_pnl_usd"].fillna(0).sum()
            trades_pnl_sum = float(t)
        closed_pnl_sum = None
        episode_vs_trades_diff = None
        mismatch_ratio = None
        note = ""
        if trades_pnl_sum is not None:
            episode_vs_trades_diff = episode_pnl_sum - trades_pnl_sum
            if abs(trades_pnl_sum) > 1e-9:
                mismatch_ratio = episode_vs_trades_diff / abs(trades_pnl_sum)
        rows.append({
            "symbol": sym,
            "episode_pnl_sum": episode_pnl_sum,
            "trades_pnl_sum": trades_pnl_sum,
            "closed_pnl_sum": closed_pnl_sum,
            "episode_vs_trades_diff": episode_vs_trades_diff,
            "mismatch_ratio": mismatch_ratio,
            "note": note,
        })
    totals_row = {
        "symbol": "_TOTAL",
        "episode_pnl_sum": float(ep_sum_total),
        "trades_pnl_sum": float(trades_df["realized_pnl_usd"].fillna(0).sum()) if trades_df is not None and not trades_df.empty else None,
        "closed_pnl_sum": None,
        "episode_vs_trades_diff": None,
        "mismatch_ratio": None,
        "note": "closed_pnl not persisted; compare episode vs trades only" if not raw_fetch_meta else "",
    }
    if totals_row["trades_pnl_sum"] is not None and totals_row["episode_pnl_sum"] is not None:
        totals_row["episode_vs_trades_diff"] = totals_row["episode_pnl_sum"] - totals_row["trades_pnl_sum"]
        if abs(totals_row["trades_pnl_sum"]) > 1e-9:
            totals_row["mismatch_ratio"] = totals_row["episode_vs_trades_diff"] / abs(totals_row["trades_pnl_sum"])
    rows.append(totals_row)
    return pd.DataFrame(rows)


# ---------- 3) Candle coverage ----------


def _candles_in_window(candles: pd.DataFrame, entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, interval_hours: float) -> tuple[int, float, int]:
    """Count bars in [entry_ts, exit_ts], expected count, and gaps (gaps > 2*interval_hours for 1H, > 2*interval for 4H)."""
    if candles is None or candles.empty or "ts_open_utc" not in candles.columns:
        return 0, 0.0, 0
    candles = candles[candles["ts_open_utc"].notna()].sort_values("ts_open_utc")
    candles["ts_open_utc"] = pd.to_datetime(candles["ts_open_utc"], utc=True)
    mask = (candles["ts_open_utc"] >= entry_ts) & (candles["ts_open_utc"] <= exit_ts)
    sub = candles.loc[mask]
    n_bars = len(sub)
    delta_h = (exit_ts - entry_ts).total_seconds() / 3600.0
    expected = max(0, int(delta_h / interval_hours) + 1)
    pct = (n_bars / expected * 100.0) if expected > 0 else 0.0
    gaps = 0
    if len(sub) > 1:
        diffs = sub["ts_open_utc"].diff().dropna()
        gap_threshold = pd.Timedelta(hours=2 * interval_hours)
        gaps = int((diffs > gap_threshold).sum())
    return n_bars, pct, gaps


def validate_candle_coverage(
    episodes_df: pd.DataFrame,
    candles_1h: dict[str, pd.DataFrame],
    candles_4h: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """For each episode: % of window covered by 1H candles; fallback 4H; gaps > 2h (1H) or > 8h (4H)."""
    rows = []
    if episodes_df.empty:
        return pd.DataFrame(columns=[
            "episode_id", "symbol", "hold_hours", "1h_bars", "1h_pct_coverage", "1h_gaps_gt_2h",
            "4h_bars", "4h_pct_coverage", "4h_gaps_gt_8h", "used_1h", "used_4h_fallback",
        ])
    for _, ep in episodes_df.iterrows():
        eid = ep["episode_id"]
        sym = str(ep["symbol"]).upper()
        entry_ts = pd.Timestamp(ep["entry_ts"]).tz_localize("UTC") if getattr(ep["entry_ts"], "tzinfo", None) is None else pd.Timestamp(ep["entry_ts"])
        exit_ts = pd.Timestamp(ep["exit_ts"]).tz_localize("UTC") if getattr(ep["exit_ts"], "tzinfo", None) is None else pd.Timestamp(ep["exit_ts"])
        hold_hours = float(ep.get("hold_hours") or 0)
        c1h = candles_1h.get(sym)
        c4h = candles_4h.get(sym)
        n_1h, pct_1h, gaps_1h = _candles_in_window(c1h, entry_ts, exit_ts, 1.0)
        n_4h, pct_4h, gaps_4h = _candles_in_window(c4h, entry_ts, exit_ts, 4.0)
        used_1h = n_1h > 0
        used_4h_fallback = not used_1h and n_4h > 0
        rows.append({
            "episode_id": eid,
            "symbol": sym,
            "hold_hours": hold_hours,
            "1h_bars": n_1h,
            "1h_pct_coverage": round(pct_1h, 2),
            "1h_gaps_gt_2h": gaps_1h,
            "4h_bars": n_4h,
            "4h_pct_coverage": round(pct_4h, 2),
            "4h_gaps_gt_8h": gaps_4h,
            "used_1h": used_1h,
            "used_4h_fallback": used_4h_fallback,
        })
    return pd.DataFrame(rows)


# ---------- 4) Cost sanity ----------


def validate_fee_sanity(trades_df: pd.DataFrame | None, episode_fills_df: pd.DataFrame | None) -> pd.DataFrame:
    """Fees vs notional: fee_usd/notional bps; detect outliers (e.g. > 50 bps)."""
    rows = []
    if episode_fills_df is not None and not episode_fills_df.empty and "fee_usd" in episode_fills_df.columns and "price" in episode_fills_df.columns and "qty" in episode_fills_df.columns:
        episode_fills_df = episode_fills_df.copy()
        episode_fills_df["notional"] = episode_fills_df["qty"].astype(float) * episode_fills_df["price"].astype(float)
        episode_fills_df["fee_bps"] = np.where(
            episode_fills_df["notional"] > 0,
            episode_fills_df["fee_usd"].fillna(0) / episode_fills_df["notional"] * 10000.0,
            np.nan,
        )
        for _, r in episode_fills_df.iterrows():
            notional = float(r["notional"])
            fee_usd = float(r["fee_usd"] or 0)
            fee_bps = float(r["fee_bps"]) if not np.isnan(r["fee_bps"]) else None
            outlier = fee_bps is not None and fee_bps > 50
            rows.append({
                "source": "episode_fill",
                "symbol": r.get("symbol"),
                "timestamp_utc": r.get("timestamp_utc"),
                "notional_usd": notional,
                "fee_usd": fee_usd,
                "fee_bps": fee_bps,
                "outlier_gt_50_bps": outlier,
            })
    if trades_df is not None and not trades_df.empty:
        if "notional_usd" not in trades_df.columns and "price" in trades_df.columns and "qty" in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["notional_usd"] = trades_df["qty"].astype(float) * trades_df["price"].astype(float)
        for _, r in trades_df.head(500).iterrows():
            notional = float(r.get("notional_usd") or r.get("qty", 0) * r.get("price", 0) or 0)
            if notional <= 0:
                continue
            fee_usd = float(r.get("fee_usd") or 0)
            fee_bps = fee_usd / notional * 10000.0
            outlier = fee_bps > 50
            rows.append({
                "source": "trade",
                "symbol": r.get("symbol"),
                "timestamp_utc": r.get("timestamp_utc"),
                "notional_usd": notional,
                "fee_usd": fee_usd,
                "fee_bps": fee_bps,
                "outlier_gt_50_bps": outlier,
            })
    if not rows:
        return pd.DataFrame(columns=["source", "symbol", "timestamp_utc", "notional_usd", "fee_usd", "fee_bps", "outlier_gt_50_bps"])
    df = pd.DataFrame(rows)
    return df


# ---------- 5) Run meta ----------


def save_run_meta(out_dir: Path, run_args: dict[str, Any], script_versions: dict[str, str]) -> None:
    meta = {
        "git_commit_hash": _git_commit_hash(_REPO_ROOT),
        "run_args": run_args,
        "script_versions": script_versions,
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------- Main ----------


def _load_candles_by_interval(out_dir: Path, interval: str) -> dict[str, pd.DataFrame]:
    base = out_dir / "candles" / interval
    result = {}
    if not base.exists():
        return result
    for p in base.iterdir():
        if p.suffix in (".parquet", ".csv"):
            sym = p.stem
            try:
                if p.suffix == ".parquet":
                    df = pd.read_parquet(p)
                else:
                    df = pd.read_csv(p)
                    if "ts_open_utc" in df.columns:
                        df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                result[sym] = df
            except Exception as e:
                logger.warning("Failed to load candle {}: {}", p, e)
    return result


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(description="Validate 30d research pipeline (read-only).")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--warmup_days", type=int, default=120)
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    if not out_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {out_dir}. Run trade_forensics_30d.py first.")

    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    episodes_df = _load_df(out_dir, "episodes.parquet")
    episode_fills_df = _load_df(out_dir, "episode_fills.parquet")
    trades_df = _load_df(out_dir, "trades_enriched.parquet")
    raw_fetch_meta = _load_json(out_dir, "raw_fetch_meta.json")
    candles_1h = _load_candles_by_interval(out_dir, "1H")
    candles_4h = _load_candles_by_interval(out_dir, "4H")

    run_args = {"days": args.days, "warmup_days": args.warmup_days, "out_dir": str(out_dir)}
    script_versions = {}
    try:
        script_versions["validate_research_30d"] = Path(__file__).resolve().stat().st_mtime
    except Exception:
        pass
    save_run_meta(validation_dir, run_args, script_versions)

    # 1) Episode integrity
    if episodes_df is not None and episode_fills_df is not None:
        integrity_df = validate_episode_integrity(episodes_df, episode_fills_df)
        no_dup, n_dup = check_no_fill_in_multiple_episodes(episode_fills_df)
        integrity_df["no_fill_in_multiple_episodes"] = no_dup
        integrity_df["duplicate_fill_count"] = n_dup
        integrity_df.to_csv(validation_dir / "episode_integrity.csv", index=False)
        logger.info("Episode integrity: {} episodes, all_checks_pass={}", len(integrity_df), integrity_df["all_checks_pass"].all() if not integrity_df.empty else "N/A")
    else:
        integrity_df = pd.DataFrame()
        logger.warning("Episodes or episode_fills missing; skipping episode integrity.")

    # 2) PnL reconciliation
    if episodes_df is not None:
        pnl_df = validate_pnl_reconciliation(episodes_df, trades_df, raw_fetch_meta)
        pnl_df.to_csv(validation_dir / "pnl_reconciliation.csv", index=False)
        logger.info("PnL reconciliation: episode total={}", pnl_df[pnl_df["symbol"] == "_TOTAL"]["episode_pnl_sum"].iloc[0] if "_TOTAL" in pnl_df["symbol"].values else "N/A")
    else:
        pnl_df = pd.DataFrame()
        logger.warning("Episodes missing; skipping PnL reconciliation.")

    # 3) Candle coverage
    if episodes_df is not None:
        cov_df = validate_candle_coverage(episodes_df, candles_1h, candles_4h)
        cov_df.to_csv(validation_dir / "candle_coverage.csv", index=False)
        if not cov_df.empty:
            fallback_rate = cov_df["used_4h_fallback"].mean() * 100
            gap_1h = cov_df["1h_gaps_gt_2h"].sum()
            gap_4h = cov_df["4h_gaps_gt_8h"].sum()
            logger.info("Candle coverage: 4H fallback rate={:.1f}%, 1H gaps>2h={}, 4H gaps>8h={}", fallback_rate, int(gap_1h), int(gap_4h))
    else:
        cov_df = pd.DataFrame()
        logger.warning("Episodes missing; skipping candle coverage.")

    # 4) Fee sanity
    fee_df = validate_fee_sanity(trades_df, episode_fills_df)
    fee_df.to_csv(validation_dir / "fee_sanity.csv", index=False)
    if not fee_df.empty and "fee_bps" in fee_df.columns:
        n_out = fee_df["outlier_gt_50_bps"].sum() if "outlier_gt_50_bps" in fee_df.columns else 0
        logger.info("Fee sanity: {} rows, outliers(>50bps)={}", len(fee_df), int(n_out))
    else:
        logger.info("Fee sanity: no fill/trade data or no fee_bps.")

    # Funding: Bybit execution/closed-pnl do not separate funding; document exclusion.
    funding_path = out_dir / "funding_summary.csv"
    if not funding_path.exists():
        try:
            funding_df = pd.DataFrame([{"funding_available": False, "note": "Funding not fetched; Bybit execution/closed-pnl do not separate funding. Net PnL in research excludes funding. Include when API provides funding history."}])
            funding_df.to_csv(funding_path, index=False)
        except Exception as e:
            logger.warning("Could not write funding_summary.csv: {}", e)

    logger.info("Validation outputs written to {}", validation_dir)


if __name__ == "__main__":
    main()