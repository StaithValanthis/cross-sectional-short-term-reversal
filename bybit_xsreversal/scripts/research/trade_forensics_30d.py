#!/usr/bin/env python3
"""
Phase A + B: Build ground-truth trade dataset (last N days) and pull multi-timeframe candles.
Read-only: no orders placed. Run from repo root:
  python3 bybit_xsreversal/scripts/research/trade_forensics_30d.py --days 30 --warmup_days 120

Outputs:
  - outputs/research_30d/trades_enriched.parquet (or .csv)
  - outputs/research_30d/raw_fetch_meta.json
  - outputs/research_30d/candles/{interval}/{symbol}.parquet (1D, 4H, 1H)
  - outputs/research_30d/per_trade_metrics.csv (Phase C stub)
  - outputs/research_30d/mfe_mae_summary.csv (Phase C stub)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SCRIPT_DIR.parent.parent  # scripts/research -> scripts -> bybit_xsreversal
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from loguru import logger

from scripts.research.lib import episode_recon
from scripts.research.lib import mfe_mae as mfe_mae_lib

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False


def _ensure_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. pip install pandas")


def _need_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _as_bool(x: Any) -> bool | None:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return None


def _load_config(config_path: Path):
    from src.config import load_config
    return load_config(config_path)


def _ensure_linear_category(cfg, args_category: str | None) -> str:
    """Research scripts: USDT linear perps only. Validate/warn and return 'linear'."""
    config_cat = getattr(getattr(cfg, "exchange", None), "category", None) or "linear"
    cat = (args_category or config_cat or "linear").strip().lower()
    if cat != "linear":
        logger.warning("Research pipeline is for USDT linear perps only; forcing category='linear' (was {})", cat)
        return "linear"
    return "linear"


def _create_client(cfg):
    from src.data.bybit_client import BybitAuth, BybitClient
    key = _need_env(cfg.exchange.api_key_env)
    sec = _need_env(cfg.exchange.api_secret_env)
    auth = BybitAuth(api_key=key, api_secret=sec)
    testnet = bool(cfg.exchange.testnet)
    env_testnet = os.getenv("BYBIT_TESTNET", "").strip().lower()
    if env_testnet in ("1", "true", "yes", "y"):
        testnet = True
    elif env_testnet in ("0", "false", "no", "n"):
        testnet = False
    return BybitClient(auth=auth, testnet=testnet)


def _fetch_equity_readonly(client) -> float | None:
    try:
        from src.execution.rebalance import fetch_equity_usdt
        return float(fetch_equity_usdt(client=client))
    except Exception as e:
        logger.warning("Could not fetch current equity: {}", e)
        return None


# ---------- Phase A: Trade ingestion ----------

# Bybit execution/list and order/history limit time range to 7 days per request.
BYBIT_MAX_DAYS_PER_REQUEST = 7


def fetch_trade_sources(client, category: str, start_ms: int, end_ms: int) -> tuple[list[dict], list[dict], list[dict], dict[str, Any]]:
    """Fetch executions, order history, closed PnL. Bybit limits time range to 7 days per request; chunk accordingly."""
    meta: dict[str, Any] = {"endpoints_used": [], "pagination_counts": {}, "params_used": {"category": category, "start_ms": start_ms, "end_ms": end_ms}, "chunks": 0}
    chunk_ms = BYBIT_MAX_DAYS_PER_REQUEST * 24 * 60 * 60 * 1000
    all_executions: list[dict] = []
    all_orders: list[dict] = []
    all_closed: list[dict] = []
    cur_start = start_ms
    while cur_start < end_ms:
        cur_end = min(cur_start + chunk_ms, end_ms)
        exec_chunk = client.get_executions(category=category, start_ms=cur_start, end_ms=cur_end)
        all_executions.extend(exec_chunk)
        time.sleep(0.2)
        order_chunk = client.get_order_history(category=category, start_ms=cur_start, end_ms=cur_end)
        all_orders.extend(order_chunk)
        time.sleep(0.2)
        closed_chunk = client.get_closed_pnl(category=category, start_ms=cur_start, end_ms=cur_end)
        all_closed.extend(closed_chunk)
        meta["chunks"] += 1
        cur_start = cur_end
        time.sleep(0.2)
    meta["endpoints_used"] = ["/v5/execution/list", "/v5/order/history", "/v5/position/closed-pnl"]
    meta["pagination_counts"] = {"executions": len(all_executions), "orders": len(all_orders), "closed_pnl": len(all_closed)}
    return all_executions, all_orders, all_closed, meta


def _timestamp_bucket(ts_ms: int, bucket_sec: int = 60) -> int:
    return (ts_ms // (bucket_sec * 1000)) * (bucket_sec * 1000)


def build_trades_unified(
    executions: list[dict],
    orders: list[dict],
    closed: list[dict],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build unified trade rows with robust linking.
    Primary: (order_id, trade_id). Fallback: (symbol, side, timestamp_bucket, qty, price) approx.
    """
    meta: dict[str, Any] = {"match_quality_counts": {"exact": 0, "approx": 0, "unmatched": 0}, "unmatched_reasons": []}

    order_by_id: dict[str, dict] = {}
    for o in orders or []:
        oid = str(o.get("orderId") or "").strip()
        if oid:
            order_by_id[oid] = {
                "reduce_only": _as_bool(o.get("reduceOnly")),
                "order_type": str(o.get("orderType") or "").strip() or None,
                "order_status": str(o.get("orderStatus") or "").strip() or None,
            }

    closed_by_order: dict[str, float] = defaultdict(float)
    closed_rows_no_oid: list[dict] = []
    for c in closed or []:
        oid = str(c.get("orderId") or "").strip()
        pnl = _safe_float(c.get("closedPnl"))
        sym = str(c.get("symbol") or "").strip().upper()
        ts_ms = int(_safe_float(c.get("updatedTime") or c.get("createdTime") or 0) or 0)
        if oid and pnl is not None:
            closed_by_order[oid] += pnl
        elif pnl is not None and sym:
            closed_rows_no_oid.append({"symbol": sym, "closedPnl": pnl, "ts_ms": ts_ms})

    rows: list[dict[str, Any]] = []
    seen_exact: set[tuple[str, str]] = set()
    for ex in executions or []:
        ts_ms = int(_safe_float(ex.get("tradeTime") or ex.get("execTime") or 0) or 0)
        if ts_ms <= 0:
            continue
        order_id = str(ex.get("orderId") or "").strip() or None
        trade_id = str(ex.get("execId") or ex.get("id") or "").strip() or None
        symbol = str(ex.get("symbol") or "").strip().upper()
        side = str(ex.get("side") or "").strip().title() or None
        qty = _safe_float(ex.get("execQty") or ex.get("qty"))
        price = _safe_float(ex.get("execPrice") or ex.get("price"))
        fee = _safe_float(ex.get("execFee") or ex.get("commission"))
        realized = _safe_float(ex.get("closedPnl"))
        pnl_source = "execution"
        match_quality = "exact"

        if order_id and (order_id, trade_id or "") in seen_exact:
            continue
        if order_id:
            seen_exact.add((order_id, trade_id or ""))

        if realized is None and order_id:
            realized = closed_by_order.get(order_id)
            if realized is not None:
                pnl_source = "merged"
        if realized is None and not order_id and symbol and qty is not None and price is not None:
            bucket = _timestamp_bucket(ts_ms)
            for cr in closed_rows_no_oid:
                if cr["symbol"] == symbol and abs(cr.get("ts_ms", 0) - ts_ms) < 120_000:
                    realized = cr.get("closedPnl")
                    pnl_source = "closed_pnl"
                    match_quality = "approx"
                    break

        if realized is None:
            match_quality = "unmatched"
            meta["unmatched_reasons"].append(f"symbol={symbol} ts={ts_ms} order_id={order_id}")

        order_meta = order_by_id.get(order_id, {}) if order_id else {}
        notional = (qty or 0) * (price or 0)
        rows.append({
            "timestamp_utc": datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "notional_usd": notional,
            "fee_usd": fee,
            "realized_pnl_usd": realized,
            "order_id": order_id,
            "trade_id": trade_id,
            "reduce_only": order_meta.get("reduce_only"),
            "order_type": order_meta.get("order_type"),
            "exec_type": None,
            "pnl_source": pnl_source,
            "match_quality": match_quality,
        })
        meta["match_quality_counts"][match_quality] = meta["match_quality_counts"].get(match_quality, 0) + 1

    df = pd.DataFrame(rows)
    if not df.empty:
        subset_cols = [c for c in ["order_id", "trade_id"] if c in df.columns]
        if subset_cols:
            df = df.sort_values("timestamp_utc").drop_duplicates(subset=subset_cols, keep="first")
        else:
            df = df.sort_values("timestamp_utc")
        df = df.reset_index(drop=True)
    return df, meta


# ---------- Phase B: Candle fetcher ----------

BYBIT_INTERVALS = {"1D": "D", "4H": "240", "1H": "60"}


def fetch_klines(
    client,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """Fetch klines with pagination. Bybit returns newest-first."""
    all_rows: list[list] = []
    cursor_end = end_ms
    for _ in range(50):
        chunk = client.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start_ms=start_ms,
            end_ms=cursor_end,
            limit=1000,
        )
        if not chunk:
            break
        all_rows.extend(chunk)
        oldest_ms = int(chunk[-1][0])
        if oldest_ms <= start_ms:
            break
        cursor_end = oldest_ms - 1
        time.sleep(0.15)
    if not all_rows:
        return pd.DataFrame(columns=["ts_open_utc", "open", "high", "low", "close", "volume"])
    rows = []
    for r in all_rows:
        try:
            ts_ms = int(r[0])
            rows.append({
                "ts_open_utc": datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]) if len(r) > 5 else 0.0,
            })
        except (IndexError, ValueError, TypeError):
            continue
    df = pd.DataFrame(rows).drop_duplicates(subset=["ts_open_utc"]).sort_values("ts_open_utc").reset_index(drop=True)
    return df


def save_candles_cache(out_dir: Path, interval: str, symbol: str, df: pd.DataFrame) -> Path:
    base = out_dir / "candles" / interval
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{symbol}.parquet" if _HAS_PARQUET else base / f"{symbol}.csv"
    if _HAS_PARQUET:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_candles_cached(out_dir: Path, interval: str, symbol: str) -> pd.DataFrame | None:
    p_parquet = out_dir / "candles" / interval / f"{symbol}.parquet"
    p_csv = out_dir / "candles" / interval / f"{symbol}.csv"
    if p_parquet.exists():
        return pd.read_parquet(p_parquet)
    if p_csv.exists():
        df = pd.read_csv(p_csv)
        if "ts_open_utc" in df.columns:
            df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
        return df
    return None


def download_candles_for_symbols(
    client,
    category: str,
    symbols: list[str],
    start_dt: datetime,
    end_dt: datetime,
    out_dir: Path,
    intervals: list[str] = ("1D", "4H", "1H"),
) -> dict[str, dict[str, Any]]:
    """Download and cache candles for each symbol and interval. Return per-interval stats."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    stats: dict[str, dict[str, Any]] = {}
    for interval in intervals:
        bybit_int = BYBIT_INTERVALS.get(interval, interval)
        stats[interval] = {"symbols_ok": 0, "symbols_fail": 0, "gaps": []}
        for sym in symbols:
            try:
                df = fetch_klines(client, category, sym, bybit_int, start_ms, end_ms)
                if df.empty:
                    stats[interval]["symbols_fail"] += 1
                    continue
                save_candles_cache(out_dir, interval, sym, df)
                stats[interval]["symbols_ok"] += 1
                if len(df) > 1:
                    diffs = df["ts_open_utc"].diff().dropna()
                    if interval == "1H":
                        expected = pd.Timedelta(hours=1)
                    elif interval == "4H":
                        expected = pd.Timedelta(hours=4)
                    else:
                        expected = pd.Timedelta(days=1)
                    gaps = diffs[diffs > expected * 1.5]
                    if not gaps.empty:
                        stats[interval]["gaps"].append({"symbol": sym, "count": len(gaps)})
                time.sleep(0.1)
            except Exception as e:
                logger.warning("Candle fetch {} {}: {}", interval, sym, e)
                stats[interval]["symbols_fail"] += 1
    return stats


# ---------- Phase C: per-trade metrics and MFE/MAE ----------

def _mfe_mae_for_episode(
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_price: float,
    side: str,
    qty: float,
    candles: pd.DataFrame,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Compute MFE/MAE in USD, time_to_mfe_h, time_to_exit_h.
    Candles must have ts_open_utc, high, low, close. Window [entry_ts, exit_ts].
    """
    if candles.empty or "ts_open_utc" not in candles.columns:
        return None, None, None, None
    candles = candles.copy()
    if not pd.api.types.is_datetime64_any_dtype(candles["ts_open_utc"]):
        candles["ts_open_utc"] = pd.to_datetime(candles["ts_open_utc"], utc=True)
    mask = (candles["ts_open_utc"] >= entry_ts) & (candles["ts_open_utc"] <= exit_ts)
    sub = candles.loc[mask]
    if sub.empty:
        return None, None, None, None
    entry_price = float(entry_price)
    qty = float(qty)
    if side and str(side).lower() == "sell":
        mfe_price = entry_price - sub["low"].min()
        mae_price = sub["high"].max() - entry_price
    else:
        mfe_price = sub["high"].max() - entry_price
        mae_price = entry_price - sub["low"].min()
    mfe_usd = mfe_price * qty
    mae_usd = mae_price * qty
    high_ts = sub.loc[sub["high"].idxmax(), "ts_open_utc"]
    low_ts = sub.loc[sub["low"].idxmin(), "ts_open_utc"]
    if side and str(side).lower() == "sell":
        mfe_ts = low_ts
    else:
        mfe_ts = high_ts
    try:
        time_to_mfe_h = (pd.Timestamp(mfe_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0
    except Exception:
        time_to_mfe_h = None
    try:
        time_to_exit_h = (pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)).total_seconds() / 3600.0
    except Exception:
        time_to_exit_h = None
    return mfe_usd, mae_usd, time_to_mfe_h, time_to_exit_h


def compute_mfe_mae(
    trades: pd.DataFrame,
    candles_1h: dict[str, pd.DataFrame],
    assumed_entry_hours: int = 24,
) -> pd.DataFrame:
    """
    For each trade (treated as a close), assume entry = assumed_entry_hours before exit;
    entry_price = close of 1H candle at entry time (or nearest). Compute MFE/MAE from 1H candles.
    """
    if trades.empty:
        return pd.DataFrame(columns=["trade_index", "symbol", "timestamp_utc", "side", "qty", "price", "realized_pnl_usd", "mfe_usd", "mae_usd", "time_to_mfe_h", "time_to_exit_h"])
    summary_rows = []
    for i, row in trades.iterrows():
        exit_ts = row.get("timestamp_utc")
        if pd.isna(exit_ts):
            summary_rows.append({**{k: row.get(k) for k in ["symbol", "side", "qty", "price", "realized_pnl_usd"]}, "trade_index": i, "timestamp_utc": exit_ts, "mfe_usd": None, "mae_usd": None, "time_to_mfe_h": None, "time_to_exit_h": None})
            continue
        exit_ts = pd.Timestamp(exit_ts).tz_localize(UTC) if exit_ts.tzinfo is None else pd.Timestamp(exit_ts)
        entry_ts = exit_ts - pd.Timedelta(hours=assumed_entry_hours)
        symbol = row.get("symbol")
        candles = candles_1h.get(str(symbol).upper() if symbol else "") if candles_1h else None
        entry_price = row.get("price")
        if candles is not None and not candles.empty and "close" in candles.columns:
            cand_ts = candles["ts_open_utc"]
            if not pd.api.types.is_datetime64_any_dtype(cand_ts):
                cand_ts = pd.to_datetime(cand_ts, utc=True)
            before = candles[cand_ts <= entry_ts]
            if not before.empty:
                entry_price = float(before.iloc[-1]["close"])
            else:
                entry_price = float(entry_price) if entry_price is not None else None
        else:
            entry_price = float(entry_price) if entry_price is not None else None
        if entry_price is None:
            mfe_usd, mae_usd, time_to_mfe_h, time_to_exit_h = None, None, None, None
        else:
            mfe_usd, mae_usd, time_to_mfe_h, time_to_exit_h = _mfe_mae_for_episode(
                entry_ts, exit_ts, entry_price, row.get("side"), row.get("qty") or 0, candles if candles is not None else pd.DataFrame()
            )
        summary_rows.append({
            "trade_index": i,
            "symbol": symbol,
            "timestamp_utc": exit_ts,
            "side": row.get("side"),
            "qty": row.get("qty"),
            "price": row.get("price"),
            "realized_pnl_usd": row.get("realized_pnl_usd"),
            "mfe_usd": mfe_usd,
            "mae_usd": mae_usd,
            "time_to_mfe_h": time_to_mfe_h,
            "time_to_exit_h": time_to_exit_h,
        })
    return pd.DataFrame(summary_rows)


def compute_mfe_mae_stub(trades: pd.DataFrame, candles_1h: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wrapper: compute MFE/MAE with 24h assumed entry."""
    return compute_mfe_mae(trades, candles_1h, assumed_entry_hours=24)


def main() -> None:
    _ensure_pandas()
    parser = argparse.ArgumentParser(description="Phase A+B: Ground-truth trades + multi-TF candles (read-only).")
    parser.add_argument("--days", type=int, default=30, help="Trade window days (default 30)")
    parser.add_argument("--warmup_days", type=int, default=120, help="Warmup days for candles (default 120)")
    parser.add_argument("--category", type=str, default=None, help="Bybit category (default from config: linear)")
    parser.add_argument("--out_dir", type=str, default="outputs/research_30d", help="Output directory")
    parser.add_argument("--start_equity", type=float, default=None, help="Override start equity if wallet unavailable")
    parser.add_argument("--skip_candles", action="store_true", help="Skip Phase B candle download")
    parser.add_argument("--local", action="store_true", help="Use existing trades_enriched.* and candles only; no Bybit fetch (for repro without API)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = _PACKAGE_DIR / "config" / "config.yaml"
    if not config_path.exists():
        config_path = _REPO_ROOT / "bybit_xsreversal" / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO_ROOT / ".env")
        load_dotenv(_PACKAGE_DIR / ".env")
    except ImportError:
        pass

    cfg = _load_config(config_path)
    category = _ensure_linear_category(cfg, args.category)
    end_ts = datetime.now(tz=UTC)
    trade_start_ts = end_ts - timedelta(days=args.days)
    start_ms = int(trade_start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    if args.local:
        p_parquet = out_dir / "trades_enriched.parquet"
        p_csv = out_dir / "trades_enriched.csv"
        if p_parquet.exists():
            trades_df = pd.read_parquet(p_parquet)
        elif p_csv.exists():
            trades_df = pd.read_csv(p_csv)
            if "timestamp_utc" in trades_df.columns:
                trades_df["timestamp_utc"] = pd.to_datetime(trades_df["timestamp_utc"], utc=True)
        else:
            raise FileNotFoundError("--local requires existing trades_enriched.parquet or trades_enriched.csv in out_dir")
        fetch_meta = {"source": "local", "window_start": trade_start_ts.isoformat(), "window_end": end_ts.isoformat(), "equity_note": "N/A (local run)"}
        with open(out_dir / "raw_fetch_meta.json", "w") as f:
            json.dump(fetch_meta, f, indent=2)
        client = None
    else:
        client = _create_client(cfg)
        try:
            executions, orders, closed, fetch_meta = fetch_trade_sources(client, category, start_ms, end_ms)
            trades_df, link_meta = build_trades_unified(executions, orders, closed)
            fetch_meta["linking"] = link_meta

            closed_pnl_by_order: dict[str, float] = defaultdict(float)
            for c in closed or []:
                oid = str(c.get("orderId") or "").strip()
                pnl = _safe_float(c.get("closedPnl"))
                if oid and pnl is not None:
                    closed_pnl_by_order[oid] += pnl
            fills_df = episode_recon.build_fills_df(executions)
            episodes_df, episode_fills_df, linkage_audit = episode_recon.reconstruct_episodes(fills_df, closed_pnl_by_order)
            fetch_meta["episode_linkage_audit"] = linkage_audit

            start_equity = args.start_equity
            if start_equity is None:
                start_equity = _fetch_equity_readonly(client)
            if start_equity is None:
                fetch_meta["equity_note"] = "start_equity not available; use --start_equity or ROI/DD will be N/A"
            else:
                fetch_meta["start_equity"] = float(start_equity)

            fetch_meta["window_start"] = trade_start_ts.isoformat()
            fetch_meta["window_end"] = end_ts.isoformat()
            fetch_meta["params_used"] = {"category": category, "start_ms": start_ms, "end_ms": end_ms}
            with open(out_dir / "raw_fetch_meta.json", "w") as f:
                json.dump(fetch_meta, f, indent=2)
        except Exception:
            if client:
                client.close()
            raise

    try:

        if trades_df.empty:
            logger.warning("No trades in window; writing empty enriched file.")
            trades_df = pd.DataFrame(columns=[
                "timestamp_utc", "symbol", "side", "qty", "price", "notional_usd", "fee_usd", "realized_pnl_usd",
                "order_id", "trade_id", "reduce_only", "order_type", "exec_type", "pnl_source", "match_quality"
            ])
        if _HAS_PARQUET:
            trades_df.to_parquet(out_dir / "trades_enriched.parquet", index=False)
        else:
            trades_df.to_csv(out_dir / "trades_enriched.csv", index=False)
        logger.info("Wrote trades_enriched ({} rows) to {}", len(trades_df), out_dir)

        if args.local:
            # Rebuild episodes from trades_enriched so we get closed + open episodes (no stale disk copy).
            fills_df = episode_recon.trades_to_fills_df(trades_df)
            if not fills_df.empty:
                episodes_df, episode_fills_df, _ = episode_recon.reconstruct_episodes(fills_df, {}, include_open_episodes=True)
            else:
                episodes_df = pd.DataFrame()
                episode_fills_df = pd.DataFrame()

        # All-fills performance summary (aligns with live bot: total fills, total realized PnL).
        n_fills = len(trades_df)
        total_realized_pnl_usd = float(trades_df["realized_pnl_usd"].fillna(0).sum()) if not trades_df.empty and "realized_pnl_usd" in trades_df.columns else 0.0
        n_episodes_closed = int((episodes_df["closed"] == True).sum()) if not episodes_df.empty and "closed" in episodes_df.columns else (len(episodes_df) if not episodes_df.empty else 0)
        n_episodes_open = int((episodes_df["closed"] == False).sum()) if not episodes_df.empty and "closed" in episodes_df.columns else 0
        summary = {
            "n_fills": n_fills,
            "total_realized_pnl_usd": total_realized_pnl_usd,
            "n_episodes_closed": n_episodes_closed,
            "n_episodes_open": n_episodes_open,
            "n_episodes_total": n_episodes_closed + n_episodes_open,
        }
        with open(out_dir / "research_30d_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            "Performance (all fills): {} fills, {:.2f} USD realized. Episodes: {} closed, {} open (use all-fills for bot performance).",
            n_fills, total_realized_pnl_usd, n_episodes_closed, n_episodes_open,
        )

        if not episodes_df.empty:
            if _HAS_PARQUET:
                episodes_df.to_parquet(out_dir / "episodes.parquet", index=False)
                episode_fills_df.to_parquet(out_dir / "episode_fills.parquet", index=False)
            else:
                episodes_df.to_csv(out_dir / "episodes.csv", index=False)
                episode_fills_df.to_csv(out_dir / "episode_fills.csv", index=False)
            logger.info("Wrote episodes ({} rows) and episode_fills ({} rows)", len(episodes_df), len(episode_fills_df))
            audit = episode_recon.audit_episodes(episodes_df, episode_fills_df, n=5)
            for a in audit:
                logger.info("Episode audit: {} symbol={} position_returns_to_zero={}", a["episode_id"], a["symbol"], a["position_returns_to_zero"])

        candle_stats = None
        symbols_for_candles = list(trades_df["symbol"].dropna().unique()) if not trades_df.empty else []
        if not episodes_df.empty:
            symbols_for_candles = list(set(symbols_for_candles) | set(episodes_df["symbol"].dropna().astype(str).unique()))
        if not args.skip_candles and symbols_for_candles and client is not None:
            candle_start = end_ts - timedelta(days=args.days + args.warmup_days + 5)
            candle_end = end_ts + timedelta(days=1)
            candle_stats = download_candles_for_symbols(client, category, symbols_for_candles, candle_start, candle_end, out_dir)
            with open(out_dir / "raw_fetch_meta.json", "r") as f:
                meta2 = json.load(f)
            meta2["candle_fetch"] = candle_stats
            with open(out_dir / "raw_fetch_meta.json", "w") as f:
                json.dump(meta2, f, indent=2)
            logger.info("Candles cached: {}", candle_stats)

        candles_1h = {}
        if (out_dir / "candles" / "1H").exists():
            for p in (out_dir / "candles" / "1H").iterdir():
                if p.suffix in (".parquet", ".csv"):
                    sym = p.stem
                    if _HAS_PARQUET and p.suffix == ".parquet":
                        candles_1h[sym] = pd.read_parquet(p)
                    else:
                        df = pd.read_csv(p)
                        if "ts_open_utc" in df.columns:
                            df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                        candles_1h[sym] = df
        candles_4h = {}
        if (out_dir / "candles" / "4H").exists():
            for p in (out_dir / "candles" / "4H").iterdir():
                if p.suffix in (".parquet", ".csv"):
                    sym = p.stem
                    if _HAS_PARQUET and p.suffix == ".parquet":
                        candles_4h[sym] = pd.read_parquet(p)
                    else:
                        df = pd.read_csv(p)
                        if "ts_open_utc" in df.columns:
                            df["ts_open_utc"] = pd.to_datetime(df["ts_open_utc"], utc=True)
                        candles_4h[sym] = df

        if not episodes_df.empty:
            per_trade = episodes_df.copy()
            per_trade["signal_score"] = None
            per_trade["realized_vol"] = None
            per_trade["regime_state"] = None
            per_trade.to_csv(out_dir / "per_trade_metrics.csv", index=False)
            mfe_mae_by_ep = mfe_mae_lib.compute_mfe_mae_by_episode(episodes_df, candles_1h, candles_4h)
            mfe_mae_by_ep.to_csv(out_dir / "mfe_mae_by_episode.csv", index=False)
            logger.info("Wrote per_trade_metrics (episode-based, {} rows) and mfe_mae_by_episode.csv", len(per_trade))
        else:
            per_trade = trades_df.copy()
            per_trade["signal_score"] = None
            per_trade["realized_vol"] = None
            per_trade["regime_state"] = None
            per_trade.to_csv(out_dir / "per_trade_metrics.csv", index=False)
            mfe_mae_df = compute_mfe_mae_stub(trades_df, candles_1h)
            mfe_mae_df.to_csv(out_dir / "mfe_mae_summary.csv", index=False)
            logger.info("Wrote per_trade_metrics (trade-based) and mfe_mae_summary.csv (24h assumed entry)")
    finally:
        if client is not None:
            client.close()

    logger.info("Phase A+B complete. Outputs under {}", out_dir)


if __name__ == "__main__":
    main()