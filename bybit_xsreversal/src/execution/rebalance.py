from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from src.config import BotConfig
from src.data.bybit_client import BybitClient
from src.data.market_data import MarketData, normalize_symbol
from src.execution.executor import Executor, PlannedOrder


@dataclass(frozen=True)
class Position:
    symbol: str
    size: float  # base qty, signed (long +, short -)
    mark_price: float


@dataclass(frozen=True)
class PositionParseResult:
    positions: dict[str, Position]  # net signed position per symbol (one-way mode)
    hedge_mode_symbols: dict[str, dict[str, Any]]


def _parse_positions(raw_positions: list[dict[str, Any]]) -> PositionParseResult:
    """
    Parse Bybit v5 position list into a net (one-way) position per symbol.

    IMPORTANT: This strategy/executor assumes **one-way** position mode.
    If Bybit account is in hedge-mode (positionIdx=1/2), we detect it and surface it explicitly,
    because correct execution would require sending positionIdx on orders.
    """
    net_by_symbol: dict[str, float] = {}
    mark_by_symbol: dict[str, float] = {}
    idx_seen: dict[str, set[int]] = {}
    nonzero_by_idx: dict[str, dict[int, float]] = {}

    for p in raw_positions:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue
        side = str(p.get("side", ""))
        size = float(p.get("size") or 0.0)
        if abs(size) < 1e-12:
            continue
        mark = float(p.get("markPrice") or p.get("avgPrice") or 0.0)
        try:
            pos_idx = int(p.get("positionIdx") or 0)
        except Exception:
            pos_idx = 0

        signed = size if side == "Buy" else -size
        net_by_symbol[sym] = float(net_by_symbol.get(sym, 0.0) + signed)
        if mark > 0:
            mark_by_symbol[sym] = mark
        idx_seen.setdefault(sym, set()).add(pos_idx)
        nonzero_by_idx.setdefault(sym, {})
        nonzero_by_idx[sym][pos_idx] = float(nonzero_by_idx[sym].get(pos_idx, 0.0) + signed)

    hedge_mode_symbols: dict[str, dict[str, Any]] = {}
    positions: dict[str, Position] = {}

    for sym, net in net_by_symbol.items():
        if abs(net) < 1e-12:
            continue
        pos_idxs = sorted(idx_seen.get(sym, {0}))
        # If we see hedge-mode positionIdxs (1/2) with non-zero exposure, flag it.
        if any(i in (1, 2) for i in pos_idxs):
            hedge_mode_symbols[sym] = {
                "positionIdxs": pos_idxs,
                "net_size": net,
                "by_positionIdx": {str(k): float(v) for k, v in (nonzero_by_idx.get(sym) or {}).items()},
            }
            continue
        positions[sym] = Position(symbol=sym, size=float(net), mark_price=float(mark_by_symbol.get(sym, 0.0)))

    return PositionParseResult(positions=positions, hedge_mode_symbols=hedge_mode_symbols)


def _summarize_reconcile(
    *,
    positions: dict[str, Position],
    target_notionals: dict[str, float],
    md: MarketData,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """
    Create a small, high-signal diff report: current notional vs target notional per symbol,
    sorted by absolute difference.
    """
    rows: list[dict[str, Any]] = []
    symbols = sorted(set(positions) | set(target_notionals))
    for sym in symbols:
        sym = normalize_symbol(sym)
        pos = positions.get(sym)
        cur_qty = float(pos.size) if pos else 0.0
        try:
            ob = md.get_orderbook_stats(sym)
            px = float(ob.mid)
        except Exception:
            px = float(pos.mark_price) if pos and pos.mark_price > 0 else 0.0
        cur_notional = float(cur_qty) * float(px) if px > 0 else 0.0
        tgt_notional = float(target_notionals.get(sym, 0.0))
        diff = tgt_notional - cur_notional
        rows.append(
            {
                "symbol": sym,
                "price_used": px,
                "current_qty": cur_qty,
                "current_notional": cur_notional,
                "target_notional": tgt_notional,
                "diff_notional": diff,
            }
        )
    rows.sort(key=lambda r: abs(float(r.get("diff_notional") or 0.0)), reverse=True)
    return rows[: max(0, int(limit))]


def _wallet_equity_usdt(wallet: dict[str, Any]) -> float:
    # v5 wallet-balance result contains list of coins; prefer USDT equity.
    lst = wallet.get("list") or []
    if not lst:
        return 0.0
    coins = (lst[0].get("coin") or [])
    for c in coins:
        if str(c.get("coin", "")).upper() == "USDT":
            # equity in unified can be "equity" or "walletBalance"
            eq = c.get("equity") or c.get("walletBalance") or c.get("availableToWithdraw")
            try:
                return float(eq)
            except Exception:
                continue
    # fallback: totalEquity
    try:
        return float(lst[0].get("totalEquity") or 0.0)
    except Exception:
        return 0.0


def plan_rebalance_orders(
    *,
    cfg: BotConfig,
    md: MarketData,
    current_positions: dict[str, Position],
    target_notionals: dict[str, float],
) -> list[PlannedOrder]:
    """
    Convert target USD notionals into base qty deltas, with safe handling of sign flips:
    - If a symbol must cross from long->short (or vice versa), first reduce-only to flat, then open.
    """
    symbols = sorted(set(current_positions) | set(target_notionals))
    orders: list[PlannedOrder] = []

    for sym in symbols:
        sym = normalize_symbol(sym)
        # Current
        pos = current_positions.get(sym)
        cur_size = float(pos.size) if pos else 0.0

        # Price for qty conversion (use orderbook mid)
        try:
            ob = md.get_orderbook_stats(sym)
            px = float(ob.mid)
        except Exception as e:
            logger.warning("Skipping {}: cannot fetch orderbook for sizing: {}", sym, e)
            continue
        if px <= 0:
            continue

        tgt_notional = float(target_notionals.get(sym, 0.0))
        tgt_size = tgt_notional / px  # base qty, signed

        # Delta in base qty
        delta = tgt_size - cur_size
        if abs(delta * px) < float(cfg.sizing.min_notional_per_symbol):
            continue

        # Determine if sign flip needed.
        # In ONE-WAY mode (required by this bot), the cleanest approach is to place a single "cross" order
        # that moves from current to target in one trade (avoids duplicate pending orders and reduce-only
        # dust truncation issues).
        if cur_size != 0.0 and np.sign(cur_size) != np.sign(tgt_size) and abs(tgt_size) > 0:
            side: Any = "Buy" if delta > 0 else "Sell"
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    side=side,
                    qty=abs(delta),
                    reduce_only=False,
                    order_type=cfg.execution.order_type,
                    limit_price=None,
                    reason="sign_flip_cross",
                )
            )
            continue

        side: Any = "Buy" if delta > 0 else "Sell"
        # reduce-only if this trade reduces absolute exposure in same direction
        reduce_only = (cur_size > 0 and delta < 0) or (cur_size < 0 and delta > 0)
        orders.append(
            PlannedOrder(
                symbol=sym,
                side=side,
                qty=abs(delta),
                reduce_only=bool(reduce_only),
                order_type=cfg.execution.order_type,
                limit_price=None,
                reason="rebalance_delta",
            )
        )

    return orders


def run_rebalance(
    *,
    cfg: BotConfig,
    client: BybitClient,
    md: MarketData,
    target_notionals: dict[str, float],
    dry_run: bool,
) -> dict[str, Any]:
    """
    Full rebalance cycle:
    - fetch positions
    - plan orders
    - place with maker->market fallback
    """
    # Hygiene: cancel any previous open orders from this bot before planning a new rebalance.
    # Otherwise repeated rebalances can accumulate duplicate pending orders and distort position reconciliation.
    canceled: list[dict[str, Any]] = []
    try:
        open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=None, settle_coin="USDT")
        for o in open_orders:
            try:
                link = str(o.get("orderLinkId") or "")
                if not link.startswith("xsrev-"):
                    continue
                sym = normalize_symbol(str(o.get("symbol") or ""))
                oid = str(o.get("orderId") or "")
                if not sym or not oid:
                    continue
                client.cancel_order(category=cfg.exchange.category, symbol=sym, order_id=oid)
                canceled.append({"symbol": sym, "orderId": oid, "orderLinkId": link})
            except Exception as e:
                logger.warning("Failed to cancel stale open order: {}", e)
        if canceled:
            logger.info("Canceled {} stale open orders from previous rebalances.", len(canceled))
    except Exception as e:
        logger.warning("Open-order cleanup failed (continuing): {}", e)

    raw_pos = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
    parsed = _parse_positions(raw_pos)
    positions = parsed.positions

    if parsed.hedge_mode_symbols:
        logger.error(
            "Detected hedge-mode positions (positionIdx=1/2). This bot currently requires ONE-WAY mode. "
            "Hedge symbols: {}",
            sorted(parsed.hedge_mode_symbols.keys()),
        )
        return {
            "orders": [],
            "positions": {k: v.__dict__ for k, v in positions.items()},
            "hedge_mode_symbols": parsed.hedge_mode_symbols,
            "canceled_open_orders": canceled,
            "summary": {
                "error": "hedge_mode_not_supported",
                "target_symbols": len(target_notionals),
                "current_symbols": len(positions),
                "planned_orders": 0,
            },
        }

    ex = Executor(client=client, md=md, cfg=cfg, dry_run=dry_run)
    orders = plan_rebalance_orders(cfg=cfg, md=md, current_positions=positions, target_notionals=target_notionals)
    reconcile_top = _summarize_reconcile(positions=positions, target_notionals=target_notionals, md=md, limit=12)
    if reconcile_top:
        logger.info("Reconcile (top diffs): {}", reconcile_top)

    if not orders:
        logger.info(
            "No rebalance orders required (targets={}, current_positions={}).",
            len(target_notionals),
            len(positions),
        )
        return {
            "orders": [],
            "positions": {k: v.__dict__ for k, v in positions.items()},
            "reconcile_top": reconcile_top,
            "canceled_open_orders": canceled,
            "summary": {"target_symbols": len(target_notionals), "current_symbols": len(positions), "planned_orders": 0},
        }

    logger.info("Planned {} orders", len(orders))
    for o in orders:
        ex.place_with_fallback(o)

    return {
        "orders": [o.__dict__ for o in orders],
        "positions": {k: v.__dict__ for k, v in positions.items()},
        "reconcile_top": reconcile_top,
        "canceled_open_orders": canceled,
        "summary": {"target_symbols": len(target_notionals), "current_symbols": len(positions), "planned_orders": len(orders)},
    }


def fetch_equity_usdt(*, client: BybitClient) -> float:
    wallet = client.get_wallet_balance(account_type="UNIFIED")
    return _wallet_equity_usdt(wallet)


