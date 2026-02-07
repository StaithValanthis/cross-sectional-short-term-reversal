#!/usr/bin/env python3
"""
Analyze last N days of trading performance (read-only).
Uses local outputs if present, else fetches from Bybit execution/order/closed-pnl APIs.
Run from repo root: python bybit_xsreversal/scripts/analyze_last_30d.py --days 30

Outputs:
  - <out_dir>/trades.csv   (timestamp, symbol, side, qty, price, notional_usd, fee_usd, realized_pnl_usd, order_id, trade_id, reduce_only, order_type, status, source)
  - <out_dir>/daily.csv    (date, realized_pnl_usd, fees_usd, net_pnl_usd, trades_count, turnover_usd)
  - <out_dir>/raw_fetch_meta.json
  - <out_dir>/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Run from repo root: bybit_xsreversal/scripts/analyze_last_30d.py -> parents[1] = bybit_xsreversal
_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _PACKAGE_DIR.parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))

from loguru import logger  # noqa: E402

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


def _ensure_deps() -> None:
    if pd is None:
        raise RuntimeError("pandas is required. Install with: pip install pandas")


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
    from src.config import load_config  # noqa: E402
    return load_config(config_path)


def _create_client(cfg):
    from src.data.bybit_client import BybitAuth, BybitClient  # noqa: E402
    key = _need_env(cfg.exchange.api_key_env)
    sec = _need_env(cfg.exchange.api_secret_env)
    auth = BybitAuth(api_key=key, api_secret=sec)
    testnet_env = os.getenv("BYBIT_TESTNET", "").strip().lower()
    testnet = bool(cfg.exchange.testnet)
    if testnet_env in ("1", "true", "yes", "y"):
        testnet = True
    elif testnet_env in ("0", "false", "no", "n"):
        testnet = False
    return BybitClient(auth=auth, testnet=testnet)


def _fetch_equity_readonly(client) -> float | None:
    """Read-only wallet equity (same as live path)."""
    try:
        from src.execution.rebalance import fetch_equity_usdt  # noqa: E402
        return float(fetch_equity_usdt(client=client))
    except Exception as e:
        logger.warning("Could not fetch current equity: {}", e)
        return None


def _find_local_sources(out_dir: Path) -> tuple[Path | None, Path | None]:
    """Return (trades_csv_path, daily_csv_path) if present."""
    t = out_dir / "trades.csv"
    d = out_dir / "daily.csv"
    return (t if t.exists() and t.stat().st_size > 0 else None, d if d.exists() and d.stat().st_size > 0 else None)


def fetch_bybit(
    client,
    category: str,
    start_ms: int,
    end_ms: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Fetch executions, order history, closed PnL (read-only). Uses existing client retries."""
    executions = client.get_executions(category=category, start_ms=start_ms, end_ms=end_ms)
    orders = client.get_order_history(category=category, start_ms=start_ms, end_ms=end_ms)
    closed = client.get_closed_pnl(category=category, start_ms=start_ms, end_ms=end_ms)
    return executions, orders, closed


def build_trades_df(
    executions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    closed: list[dict[str, Any]],
    tz: timezone,
) -> pd.DataFrame:
    """Build unified trades dataframe. Merge by order_id where possible."""
    order_meta: dict[str, dict[str, Any]] = {}
    for o in orders or []:
        oid = str(o.get("orderId") or "").strip()
        if not oid:
            continue
        order_meta[oid] = {
            "reduce_only": _as_bool(o.get("reduceOnly")),
            "order_type": str(o.get("orderType") or o.get("orderType") or "").strip() or None,
            "order_status": str(o.get("orderStatus") or "").strip() or None,
            "leverage": _safe_float(o.get("leverage")),
        }

    # Closed PnL can be per-symbol/position; Bybit may not always have orderId. Aggregate by orderId when present.
    closed_by_order: dict[str, float] = defaultdict(float)
    for c in closed or []:
        oid = str(c.get("orderId") or "").strip()
        pnl = _safe_float(c.get("closedPnl"))
        if oid and pnl is not None:
            closed_by_order[oid] += pnl

    rows: list[dict[str, Any]] = []
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
        if realized is None and order_id:
            realized = closed_by_order.get(order_id)
        meta = order_meta.get(order_id, {}) if order_id else {}
        reduce_only = meta.get("reduce_only")
        order_type = meta.get("order_type")
        status = meta.get("order_status")
        leverage = meta.get("leverage")

        notional = (qty or 0) * (price or 0)
        source = "merged"
        if realized is not None and order_id and order_id in closed_by_order:
            source = "closed_pnl|executions"
        else:
            source = "executions"

        rows.append({
            "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=tz),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "notional_usd": notional,
            "fee_usd": fee,
            "realized_pnl_usd": realized,
            "order_id": order_id,
            "trade_id": trade_id,
            "reduce_only": reduce_only,
            "order_type": order_type,
            "status": status,
            "leverage": leverage,
            "source": source,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_daily_df(trades: pd.DataFrame, tz: timezone) -> pd.DataFrame:
    """Aggregate trades to daily: date, realized_pnl_usd, fees_usd, net_pnl_usd, trades_count, turnover_usd."""
    if trades.empty:
        return pd.DataFrame(columns=["date", "realized_pnl_usd", "fees_usd", "net_pnl_usd", "trades_count", "turnover_usd"])
    df = trades.copy()
    df["date"] = df["timestamp"].dt.tz_convert(tz).dt.date
    daily = df.groupby("date").agg(
        realized_pnl_usd=("realized_pnl_usd", lambda s: s.fillna(0).sum()),
        fees_usd=("fee_usd", lambda s: s.fillna(0).sum()),
        trades_count=("timestamp", "count"),
        turnover_usd=("notional_usd", lambda s: s.fillna(0).abs().sum()),
    ).reset_index()
    daily["net_pnl_usd"] = daily["realized_pnl_usd"] - daily["fees_usd"]
    return daily


def compute_summary_metrics(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    start_equity: float | None,
    end_equity: float | None,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, Any]:
    """Compute Phase 2 metrics."""
    out: dict[str, Any] = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "start_equity": start_equity,
        "end_equity": end_equity,
    }
    if trades.empty:
        out["status"] = "empty"
        out["message"] = "No trades in window"
        return out

    total_realized = float(trades["realized_pnl_usd"].fillna(0).sum())
    total_fees = float(trades["fee_usd"].fillna(0).sum())
    net_pnl = total_realized - total_fees
    out["total_realized_pnl_usd"] = total_realized
    out["total_fees_usd"] = total_fees
    out["net_pnl_usd"] = net_pnl

    if start_equity is not None and float(start_equity) > 0:
        out["roi_pct"] = 100.0 * net_pnl / float(start_equity)
    elif end_equity is not None and float(end_equity) > 0:
        start_implied = float(end_equity) - net_pnl
        out["roi_pct"] = 100.0 * net_pnl / start_implied if start_implied > 0 else None
        out["start_equity_assumption"] = "reconstructed_from_end_and_net_pnl"
    else:
        out["roi_pct"] = None
        out["start_equity_assumption"] = "unknown"

    days = max(1, (window_end - window_start).days)
    out["trades_per_day"] = len(trades) / days
    out["turnover_usd_total"] = float(trades["notional_usd"].fillna(0).abs().sum())
    out["turnover_per_day_usd"] = out["turnover_usd_total"] / days
    out["avg_trade_notional_usd"] = float(trades["notional_usd"].fillna(0).abs().mean()) if len(trades) else None

    # Trade stats (from closed_pnl perspective: each row can be a fill; wins = realized_pnl > 0)
    wins = trades["realized_pnl_usd"].dropna()
    wins = wins[wins > 0]
    losses = trades["realized_pnl_usd"].dropna()
    losses = losses[losses < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    n_closed = n_wins + n_losses
    out["win_rate"] = (n_wins / n_closed) if n_closed else None
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0
    out["profit_factor"] = (gross_profit / abs(gross_loss)) if gross_loss != 0 else (float("inf") if gross_profit > 0 else None)
    out["avg_win_usd"] = float(wins.mean()) if len(wins) else None
    out["avg_loss_usd"] = float(losses.mean()) if len(losses) else None
    out["expectancy_usd"] = (float(wins.mean()) * (n_wins / n_closed) + float(losses.mean()) * (n_losses / n_closed)) if n_closed and (wins.size or losses.size) else None

    # Daily stats
    if not daily.empty and "net_pnl_usd" in daily.columns:
        daily_rets = daily["net_pnl_usd"]
        out["daily_net_pnl_mean_usd"] = float(daily_rets.mean())
        out["daily_net_pnl_std_usd"] = float(daily_rets.std()) if len(daily_rets) > 1 else None
        if start_equity and float(start_equity) > 0:
            daily_ret_pct = daily_rets / float(start_equity)
            out["daily_return_mean_pct"] = float(daily_ret_pct.mean()) * 100.0
            out["daily_return_std_pct"] = float(daily_ret_pct.std()) * 100.0 if len(daily_ret_pct) > 1 else None
            # Simple Sharpe (annualized): mean(daily_ret) / std(daily_ret) * sqrt(252) if daily
            if out.get("daily_return_std_pct") and out["daily_return_std_pct"] > 0:
                out["sharpe_simple_annual"] = (out["daily_return_mean_pct"] / 100.0) / (out["daily_return_std_pct"] / 100.0) * (252 ** 0.5)
            else:
                out["sharpe_simple_annual"] = None
        else:
            out["daily_return_mean_pct"] = None
            out["daily_return_std_pct"] = None
            out["sharpe_simple_annual"] = None

    # Equity curve for drawdown (reconstructed if no historical equity)
    if start_equity is not None and not daily.empty:
        eq = float(start_equity)
        curve = [eq]
        for _, row in daily.sort_values("date").iterrows():
            eq += float(row.get("net_pnl_usd", 0))
            curve.append(eq)
        peak = curve[0]
        max_dd = 0.0
        for v in curve:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        out["max_drawdown_pct"] = 100.0 * max_dd
    else:
        out["max_drawdown_pct"] = None

    # Attribution: per-symbol
    if not trades.empty and "symbol" in trades.columns:
        by_sym = trades.groupby("symbol").agg(
            net_pnl_usd=("realized_pnl_usd", lambda s: s.fillna(0).sum()),
            fees_usd=("fee_usd", lambda s: s.fillna(0).sum()),
            trades_count=("timestamp", "count"),
        ).reset_index()
        by_sym["net_pnl_usd"] = by_sym["net_pnl_usd"] - by_sym["fees_usd"]
        out["per_symbol"] = by_sym.to_dict(orient="records")

    # Fee drag, concentration, tail
    if total_realized != 0:
        out["fee_drag"] = total_fees / abs(total_realized)
    else:
        out["fee_drag"] = None
    if not daily.empty:
        out["worst_day_net_pnl_usd"] = float(daily["net_pnl_usd"].min())
    if not trades.empty:
        out["worst_trade_realized_pnl_usd"] = float(trades["realized_pnl_usd"].min()) if trades["realized_pnl_usd"].notna().any() else None
        pnl_vals = trades["realized_pnl_usd"].dropna()
        if len(pnl_vals):
            out["p95_loss_usd"] = float(pnl_vals.quantile(0.05))  # 5th pct = tail loss

    # Time slicing: day-of-week, hour (UTC)
    if not trades.empty and "timestamp" in trades.columns:
        ts = pd.to_datetime(trades["timestamp"], utc=True)
        trades_copy = trades.assign(dow=ts.dt.dayofweek, hour=ts.dt.hour)
        net_col = "realized_pnl_usd" if "realized_pnl_usd" in trades_copy.columns else "realized_pnl"
        fee_col = "fee_usd" if "fee_usd" in trades_copy.columns else "fee"
        if net_col in trades_copy.columns:
            trades_copy["net"] = trades_copy[net_col].fillna(0) - trades_copy.get(fee_col, pd.Series(0)).fillna(0)
            out["by_day_of_week"] = trades_copy.groupby("dow")["net"].sum().to_dict()
            out["by_hour_utc"] = trades_copy.groupby("hour")["net"].sum().to_dict()

    return out


def main() -> None:
    _ensure_deps()
    parser = argparse.ArgumentParser(
        description="Analyze last N days of trading performance (read-only). Produces trades.csv, daily.csv, raw_fetch_meta.json, summary.json.",
        epilog="Run from repo root: python bybit_xsreversal/scripts/analyze_last_30d.py --days 30",
    )
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze (default 30)")
    parser.add_argument("--out_dir", type=str, default="outputs/perf_last30d", help="Output directory (default outputs/perf_last30d)")
    parser.add_argument("--timezone", type=str, default="UTC", help="Timezone for dates (default UTC)")
    parser.add_argument("--source", type=str, choices=("auto", "local", "bybit"), default="auto", help="Data source: auto=local then bybit, local=only existing files, bybit=fetch (default auto)")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path (default: bybit_xsreversal/config/config.yaml)")
    args = parser.parse_args()

    # Resolve paths
    out_dir = Path(args.out_dir) if os.path.isabs(args.out_dir) else _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config) if args.config else _PACKAGE_DIR / "config" / "config.yaml"
    if not config_path.is_absolute():
        config_path = _REPO_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    tz = UTC if args.timezone.upper() == "UTC" else timezone.utc  # simplified: use UTC
    window_end = datetime.now(tz=UTC)
    window_start = window_end - timedelta(days=max(1, args.days))
    start_ms = int(window_start.timestamp() * 1000)
    end_ms = int(window_end.timestamp() * 1000)

    # Local sources
    trades_path = out_dir / "trades.csv"
    daily_path = out_dir / "daily.csv"
    local_trades, local_daily = _find_local_sources(out_dir)
    use_local = args.source == "local" or (args.source == "auto" and local_trades is not None)
    if args.source == "local" and local_trades is None:
        logger.warning("Source is 'local' but no local trades.csv found; writing empty outputs.")
        trades_path.write_text("timestamp,symbol,side,qty,price,notional_usd,fee_usd,realized_pnl_usd,order_id,trade_id,reduce_only,order_type,status,source\n")
        daily_path.write_text("date,realized_pnl_usd,fees_usd,net_pnl_usd,trades_count,turnover_usd\n")
        fetch_meta = {"window_start": window_start.isoformat(), "window_end": window_end.isoformat(), "source_used": "local", "warnings": ["No local data"]}
        (out_dir / "raw_fetch_meta.json").write_text(json.dumps(fetch_meta, indent=2))
        (out_dir / "summary.json").write_text(json.dumps(compute_summary_metrics(pd.DataFrame(), pd.DataFrame(), None, None, window_start, window_end), indent=2))
        return

    trades_df: pd.DataFrame = pd.DataFrame()
    fetch_meta: dict[str, Any] = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "timezone": args.timezone,
        "source_used": "local",
        "endpoints_used": [],
        "params_used": {"category": "linear", "start_ms": start_ms, "end_ms": end_ms},
        "pagination_counts": {},
        "missing_fields": [],
        "merge_match_rate": None,
        "warnings": [],
    }

    if use_local and local_trades:
        logger.info("Using local trades from {}", local_trades)
        trades_df = pd.read_csv(local_trades)
        if "timestamp" in trades_df.columns:
            trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True)
        fetch_meta["source_used"] = "local"
    else:
        # Bybit fetch
        try:
            from dotenv import load_dotenv
            load_dotenv(_REPO_ROOT / ".env")
            load_dotenv(_PACKAGE_DIR / ".env")
        except ImportError:
            pass
        cfg = _load_config(config_path)
        client = _create_client(cfg)
        try:
            executions, orders, closed = fetch_bybit(client, cfg.exchange.category, start_ms, end_ms)
            fetch_meta["endpoints_used"] = ["/v5/execution/list", "/v5/order/history", "/v5/position/closed-pnl"]
            fetch_meta["pagination_counts"] = {"executions": len(executions), "orders": len(orders), "closed_pnl": len(closed)}
            trades_df = build_trades_df(executions, orders, closed, tz)
            if not trades_df.empty and "order_id" in trades_df.columns:
                matched = trades_df["order_id"].notna().sum()
                fetch_meta["merge_match_rate"] = float(matched) / len(trades_df)
        finally:
            client.close()

    if trades_df.empty:
        fetch_meta["warnings"].append("No trades in window")
        trades_path.write_text("timestamp,symbol,side,qty,price,notional_usd,fee_usd,realized_pnl_usd,order_id,trade_id,reduce_only,order_type,status,source\n")
        daily_path.write_text("date,realized_pnl_usd,fees_usd,net_pnl_usd,trades_count,turnover_usd\n")
        raw_meta_path = out_dir / "raw_fetch_meta.json"
        raw_meta_path.write_text(json.dumps(fetch_meta, indent=2))
        summary = compute_summary_metrics(trades_df, pd.DataFrame(), None, None, window_start, window_end)
        summary["data_sources"] = fetch_meta
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("No trades; wrote empty outputs to {}", out_dir)
        return

    # Ensure column names match spec
    col_map = {"fee": "fee_usd", "realized_pnl": "realized_pnl_usd", "notional": "notional_usd"}
    for old, new in col_map.items():
        if old in trades_df.columns and new not in trades_df.columns:
            trades_df = trades_df.rename(columns={old: new})
    required = ["timestamp", "symbol", "side", "qty", "price", "notional_usd", "fee_usd", "realized_pnl_usd", "order_id", "trade_id", "reduce_only", "order_type", "status", "source"]
    for c in required:
        if c not in trades_df.columns:
            trades_df[c] = None
    trades_df = trades_df[[c for c in required if c in trades_df.columns]]
    trades_df.to_csv(trades_path, index=False)

    daily_df = build_daily_df(trades_df, tz)
    daily_df.to_csv(daily_path, index=False)

    raw_meta_path = out_dir / "raw_fetch_meta.json"
    raw_meta_path.write_text(json.dumps(fetch_meta, indent=2))

    # Equity for ROI: try current wallet (end_equity), then reconstruct start
    start_equity = None
    end_equity = None
    if args.source != "local":
        try:
            from dotenv import load_dotenv
            load_dotenv(_REPO_ROOT / ".env")
            load_dotenv(_PACKAGE_DIR / ".env")
        except ImportError:
            pass
        cfg = _load_config(config_path)
        client = _create_client(cfg)
        try:
            end_equity = _fetch_equity_readonly(client)
            if end_equity is not None and not daily_df.empty:
                start_equity = end_equity - float(daily_df["net_pnl_usd"].sum())
        except Exception as e:
            logger.warning("Could not fetch equity for ROI baseline: {}", e)
        finally:
            client.close()
    if start_equity is None:
        # Fallback: use config backtest.initial_equity as assumption
        try:
            cfg = _load_config(config_path)
            start_equity = float(getattr(cfg.backtest, "initial_equity", 10000.0))
            fetch_meta["warnings"].append("start_equity from config.backtest.initial_equity (assumption)")
        except Exception:
            start_equity = 10000.0
            fetch_meta["warnings"].append("start_equity assumed 10000 USDT")

    summary = compute_summary_metrics(trades_df, daily_df, start_equity, end_equity, window_start, window_end)
    summary["data_sources"] = fetch_meta
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote {} trades, daily, raw_fetch_meta, summary to {}", len(trades_df), out_dir)


if __name__ == "__main__":
    main()