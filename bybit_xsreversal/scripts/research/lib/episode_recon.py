"""
Reconstruct position episodes from execution fills (USDT linear perps, one-way mode).

Signed qty convention (one-way):
  - Buy  => +qty (increase long or reduce short)
  - Sell => -qty (reduce long or increase short)
  - position_qty: running sum per symbol; positive = net long, negative = net short.

Episode: period from position 0 -> nonzero -> back to 0.
  - entry_ts = time of first fill that opens the episode
  - exit_ts = time of last fill that closes position to 0
  - entry_vwap = VWAP of fills that open/increase exposure (same sign as final side)
  - exit_vwap = VWAP of fills that close exposure to zero
  - realized_pnl: from closed_pnl when linkable; else from fill PnL (execution closedPnl) or average-price method.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _signed_qty(side: str, qty: float) -> float:
    """One-way: Buy => +qty, Sell => -qty."""
    q = float(qty) if qty is not None else 0.0
    if not q:
        return 0.0
    s = str(side or "").strip().upper()
    if s == "BUY":
        return q
    if s == "SELL":
        return -q
    return 0.0


def trades_to_fills_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Convert trades_enriched-style DataFrame to fills format (timestamp_utc, symbol, side, qty, ..., signed_qty)."""
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["timestamp_utc", "symbol", "side", "qty", "price", "fee_usd", "realized_pnl_usd", "order_id", "trade_id", "signed_qty"])
    need = ["timestamp_utc", "symbol", "side", "qty", "price", "fee_usd", "realized_pnl_usd", "order_id", "trade_id"]
    missing = [c for c in need if c not in trades_df.columns]
    if missing:
        return pd.DataFrame(columns=need + ["signed_qty"])
    df = trades_df[need].copy()
    df["signed_qty"] = df.apply(lambda r: _signed_qty(str(r.get("side") or ""), r.get("qty")), axis=1)
    return df.sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)


def build_fills_df(executions: list[dict]) -> pd.DataFrame:
    """Normalize executions to a dataframe: timestamp_utc, symbol, side, qty, price, fee_usd, realized_pnl_usd, order_id, trade_id, signed_qty."""
    rows = []
    for ex in executions or []:
        ts_ms = int(_safe_float(ex.get("tradeTime") or ex.get("execTime") or 0) or 0)
        if ts_ms <= 0:
            continue
        symbol = str(ex.get("symbol") or "").strip().upper()
        side = str(ex.get("side") or "").strip()
        qty = _safe_float(ex.get("execQty") or ex.get("qty"))
        price = _safe_float(ex.get("execPrice") or ex.get("price"))
        fee = _safe_float(ex.get("execFee") or ex.get("commission"))
        realized = _safe_float(ex.get("closedPnl"))
        order_id = str(ex.get("orderId") or "").strip() or None
        trade_id = str(ex.get("execId") or ex.get("id") or "").strip() or None
        if not symbol or qty is None:
            continue
        signed = _signed_qty(side, qty)
        rows.append({
            "timestamp_utc": datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "fee_usd": fee,
            "realized_pnl_usd": realized,
            "order_id": order_id,
            "trade_id": trade_id,
            "signed_qty": signed,
        })
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "symbol", "side", "qty", "price", "fee_usd", "realized_pnl_usd", "order_id", "trade_id", "signed_qty"])
    df = pd.DataFrame(rows).sort_values(["symbol", "timestamp_utc"]).reset_index(drop=True)
    return df


def reconstruct_episodes(
    fills: pd.DataFrame,
    closed_pnl_by_order: dict[str, float],
    include_open_episodes: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Build episodes per symbol from fills. Optionally link closed_pnl by order_id.
    When include_open_episodes=True, episodes still open at end of data are emitted with closed=False.

    Returns:
      - episodes_df: episode_id, symbol, side (Long/Short), entry_ts, exit_ts, entry_vwap, exit_vwap,
                     fees_usd, realized_pnl_usd, hold_hours, num_fills, max_abs_position,
                     closed_pnl_mapped (bool), pnl_source, closed (bool)
      - episode_fills_df: episode_id, fill index, timestamp_utc, symbol, side, qty, price, fee_usd,
                         position_after (running position after this fill)
      - linkage_audit: list of { episode_id, closed_pnl_order_ids, unmapped_closed_pnl_order_ids }
    """
    base_cols = ["episode_id", "symbol", "side", "entry_ts", "exit_ts", "entry_vwap", "exit_vwap", "fees_usd", "realized_pnl_usd", "hold_hours", "num_fills", "max_abs_position", "closed_pnl_mapped", "pnl_source", "closed"]
    if fills.empty:
        return (
            pd.DataFrame(columns=base_cols),
            pd.DataFrame(columns=["episode_id", "fill_idx", "timestamp_utc", "symbol", "side", "qty", "price", "fee_usd", "position_after"]),
            [],
        )

    episodes_rows = []
    episode_fills_rows = []
    linkage_audit = []
    used_closed_order_ids: set[str] = set()

    for symbol, grp in fills.groupby("symbol"):
        grp = grp.sort_values("timestamp_utc").reset_index(drop=True)
        position = 0.0
        episode_id = None
        episode_start_idx = None
        direction = 0
        entry_sum_qty = 0.0
        entry_sum_pq = 0.0
        exit_sum_qty = 0.0
        exit_sum_pq = 0.0
        episode_fees = 0.0
        episode_pnl_from_fills = 0.0
        episode_fill_order_ids: list[str] = []
        max_abs_pos = 0.0

        for idx in range(len(grp)):
            row = grp.iloc[idx]
            i = grp.index[idx]
            signed = float(row["signed_qty"])
            position_after = position + signed
            fee = float(row["fee_usd"] or 0.0)
            pnl = row.get("realized_pnl_usd")
            oid = row.get("order_id")
            pnl_val = 0.0
            if pnl is not None and not (isinstance(pnl, float) and math.isnan(pnl)):
                try:
                    pnl_val = float(pnl)
                except (TypeError, ValueError):
                    pass

            if episode_id is None:
                if abs(position_after) > 1e-12:
                    episode_id = f"{symbol}_{row['timestamp_utc'].timestamp():.0f}"
                    episode_start_idx = idx
                    direction = 1 if position_after > 0 else -1
                    entry_sum_qty = 0.0
                    entry_sum_pq = 0.0
                    exit_sum_qty = 0.0
                    exit_sum_pq = 0.0
                    episode_fees = 0.0
                    episode_pnl_from_fills = pnl_val
                    episode_fill_order_ids = [oid] if oid else []
                    max_abs_pos = abs(position_after)
                    if signed * direction > 0:
                        entry_sum_qty += abs(signed)
                        entry_sum_pq += abs(signed) * float(row["price"] or 0)
                    else:
                        exit_sum_qty += abs(signed)
                        exit_sum_pq += abs(signed) * float(row["price"] or 0)
                    episode_fees += fee
            else:
                episode_fees += fee
                episode_pnl_from_fills += pnl_val
                if oid:
                    episode_fill_order_ids.append(oid)
                if abs(position_after) > max_abs_pos:
                    max_abs_pos = abs(position_after)
                if signed * direction > 0:
                    entry_sum_qty += abs(signed)
                    entry_sum_pq += abs(signed) * float(row["price"] or 0)
                else:
                    exit_sum_qty += abs(signed)
                    exit_sum_pq += abs(signed) * float(row["price"] or 0)

            if episode_id is not None:
                episode_fills_rows.append({
                    "episode_id": episode_id,
                    "fill_idx": int(i),
                    "timestamp_utc": row["timestamp_utc"],
                    "symbol": symbol,
                    "side": row["side"],
                    "qty": row["qty"],
                    "price": row["price"],
                    "fee_usd": fee,
                    "position_after": position_after,
                })

            if episode_id and abs(position_after) < 1e-12:
                entry_ts = grp.iloc[episode_start_idx]["timestamp_utc"]
                exit_ts = row["timestamp_utc"]
                entry_vwap = entry_sum_pq / entry_sum_qty if entry_sum_qty > 0 else 0.0
                exit_vwap = exit_sum_pq / exit_sum_qty if exit_sum_qty > 0 else 0.0
                delta = pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)
                hold_hours = delta.total_seconds() / 3600.0
                num_fills = idx - episode_start_idx + 1
                side_str = "Long" if direction > 0 else "Short"
                pnl_from_closed = 0.0
                for o in episode_fill_order_ids:
                    if o in closed_pnl_by_order:
                        pnl_from_closed += closed_pnl_by_order[o]
                        used_closed_order_ids.add(o)
                if pnl_from_closed != 0:
                    realized_pnl = pnl_from_closed
                    pnl_source = "closed_pnl"
                    closed_pnl_mapped = True
                else:
                    realized_pnl = episode_pnl_from_fills
                    pnl_source = "fills" if episode_pnl_from_fills != 0 else "unmatched"
                    closed_pnl_mapped = False
                episodes_rows.append({
                    "episode_id": episode_id,
                    "symbol": symbol,
                    "side": side_str,
                    "entry_ts": entry_ts,
                    "exit_ts": exit_ts,
                    "entry_vwap": entry_vwap,
                    "exit_vwap": exit_vwap,
                    "fees_usd": episode_fees,
                    "realized_pnl_usd": realized_pnl,
                    "hold_hours": hold_hours,
                    "num_fills": num_fills,
                    "max_abs_position": max_abs_pos,
                    "closed_pnl_mapped": closed_pnl_mapped,
                    "pnl_source": pnl_source,
                    "closed": True,
                })
                linkage_audit.append({"episode_id": episode_id, "order_ids": list(episode_fill_order_ids)})
                episode_id = None

            position = position_after

        # Emit open episode at end of symbol's data if requested
        if include_open_episodes and episode_id is not None:
            last_row = grp.iloc[-1]
            entry_ts = grp.iloc[episode_start_idx]["timestamp_utc"]
            exit_ts = last_row["timestamp_utc"]  # last fill so far
            entry_vwap = entry_sum_pq / entry_sum_qty if entry_sum_qty > 0 else 0.0
            exit_vwap = exit_sum_pq / exit_sum_qty if exit_sum_qty > 0 else float("nan")
            delta = pd.Timestamp(exit_ts) - pd.Timestamp(entry_ts)
            hold_hours = delta.total_seconds() / 3600.0
            num_fills = len(grp) - episode_start_idx
            side_str = "Long" if direction > 0 else "Short"
            pnl_from_closed = 0.0
            for o in episode_fill_order_ids:
                if o in closed_pnl_by_order:
                    pnl_from_closed += closed_pnl_by_order[o]
            if pnl_from_closed != 0:
                realized_pnl = pnl_from_closed
                pnl_source = "closed_pnl"
                closed_pnl_mapped = True
            else:
                realized_pnl = episode_pnl_from_fills
                pnl_source = "fills" if episode_pnl_from_fills != 0 else "unmatched"
                closed_pnl_mapped = False
            episodes_rows.append({
                "episode_id": episode_id,
                "symbol": symbol,
                "side": side_str,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_vwap": entry_vwap,
                "exit_vwap": exit_vwap,
                "fees_usd": episode_fees,
                "realized_pnl_usd": realized_pnl,
                "hold_hours": hold_hours,
                "num_fills": num_fills,
                "max_abs_position": max_abs_pos,
                "closed_pnl_mapped": closed_pnl_mapped,
                "pnl_source": pnl_source,
                "closed": False,
            })
            linkage_audit.append({"episode_id": episode_id, "order_ids": list(episode_fill_order_ids), "open": True})

    episodes_df = pd.DataFrame(episodes_rows)
    episode_fills_df = pd.DataFrame(episode_fills_rows)
    unmapped = [oid for oid in closed_pnl_by_order if oid not in used_closed_order_ids]
    if unmapped:
        linkage_audit.append({"episode_id": None, "unmapped_closed_pnl_order_ids": unmapped})
    return episodes_df, episode_fills_df, linkage_audit


def audit_episodes(episodes_df: pd.DataFrame, episode_fills_df: pd.DataFrame, n: int = 5) -> list[dict]:
    """Print audit for n random episodes: list fills with running position and confirm position returns to 0 at exit."""
    if episodes_df.empty or episode_fills_df.empty:
        return []
    import random
    ids = episodes_df["episode_id"].tolist()
    sample = random.sample(ids, min(n, len(ids)))
    out = []
    for eid in sample:
        fills = episode_fills_df[episode_fills_df["episode_id"] == eid].sort_values("timestamp_utc")
        pos_start = 0.0
        positions = []
        for _, r in fills.iterrows():
            pos_after = r["position_after"]
            positions.append({"ts": str(r["timestamp_utc"]), "side": r["side"], "qty": r["qty"], "position_after": pos_after})
        ep = episodes_df[episodes_df["episode_id"] == eid].iloc[0]
        out.append({
            "episode_id": eid,
            "symbol": ep["symbol"],
            "entry_ts": str(ep["entry_ts"]),
            "exit_ts": str(ep["exit_ts"]),
            "position_returns_to_zero": abs(fills["position_after"].iloc[-1]) < 1e-12 if len(fills) else False,
            "fills": positions,
        })
    return out