from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from src.config import BotConfig
from src.data.bybit_client import BybitClient
from src.data.market_data import MarketData, normalize_symbol
from src.execution.executor import Executor, PlannedOrder
import time


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

        # Check if this symbol is in target universe
        tgt_notional = float(target_notionals.get(sym, 0.0))
        is_outside_universe = (tgt_notional == 0.0 and abs(cur_size) > 1e-8)

        # Price for qty conversion (use orderbook mid)
        # For symbols outside universe, try to use mark price from position as fallback
        px = 0.0
        try:
            ob = md.get_orderbook_stats(sym)
            px = float(ob.mid)
        except Exception as e:
            if is_outside_universe and pos and pos.mark_price > 0:
                # For positions outside universe, use mark price as fallback to allow closing
                px = float(pos.mark_price)
                logger.warning("Cannot fetch orderbook for {} (outside universe), using mark price {:.6f} to close position", sym, px)
            else:
                logger.warning("Skipping {}: cannot fetch orderbook for sizing: {}", sym, e)
                continue
        if px <= 0:
            if is_outside_universe and pos and pos.mark_price > 0:
                px = float(pos.mark_price)
                logger.warning("Orderbook price invalid for {} (outside universe), using mark price {:.6f} to close position", sym, px)
            else:
                continue

        # Instrument constraints
        try:
            meta = md.get_instrument_meta(sym)
        except Exception as e:
            if is_outside_universe:
                # For positions outside universe, create minimal meta to allow closing
                # Use defaults that allow closing positions even if we can't fetch full meta
                from src.data.market_data import InstrumentMeta
                meta = InstrumentMeta(
                    symbol=sym,
                    qty_step=1e-8,  # Small step to allow any quantity
                    min_qty=0.0,  # No minimum to allow closing
                    max_qty=None,  # No maximum
                    tick_size=1e-8,  # Small tick
                    min_notional=None,  # No minimum notional
                )
                logger.warning("Cannot fetch instrument meta for {} (outside universe), using defaults to close position", sym)
            else:
                logger.warning("Skipping {}: cannot fetch instrument meta: {}", sym, e)
                continue
        tgt_size = tgt_notional / px  # base qty, signed

        # If target is non-zero but below exchange minimum size, only "bump" when opening from flat.
        # If we already have a position in the same direction, don't keep adding minimum clips.
        min_qty = float(getattr(meta, "min_qty", 0.0) or 0.0)
        if abs(tgt_size) > 0 and min_qty > 0 and abs(tgt_size) < min_qty:
            if cur_size != 0.0 and np.sign(cur_size) == np.sign(tgt_size):
                logger.info(
                    "Skipping {}: target below minQty (tgt_qty={:.6g} < minQty={:.6g}) but already positioned in same direction (cur_qty={:.6g}).",
                    sym,
                    float(tgt_size),
                    float(min_qty),
                    float(cur_size),
                )
                continue
            # Opening from flat: bump to minQty in target direction.
            tgt_size = float(np.sign(tgt_size) * min_qty)

        # Delta in base qty
        delta = tgt_size - cur_size
        
        # CRITICAL: If there's an existing position in the same direction, check if we should skip
        # This prevents adding to existing positions unless there's a meaningful size change needed
        same_direction = (cur_size > 0 and tgt_size > 0) or (cur_size < 0 and tgt_size < 0)
        has_existing_position = abs(cur_size) > 1e-8
        
        if has_existing_position and same_direction:
            # There's an existing position in the same direction - be more conservative
            abs_delta_notional = abs(delta * px)
            
            # Calculate relative delta (percentage of current position size)
            if abs(cur_size) > 1e-8:
                rel_delta_pct = abs(delta / cur_size)
            else:
                rel_delta_pct = 1.0
            
            # Skip if the adjustment is too small (either absolute or relative)
            # Use a larger tolerance (5%) when there's already a position in the same direction
            # BUT: If the relative change is significant (>10%), allow it even if absolute notional is small
            # This prevents positions from drifting too far from target
            min_meaningful_delta_notional = float(cfg.sizing.min_notional_per_symbol)
            min_meaningful_rel_delta = 0.05  # 5% - only adjust if size change is > 5%
            significant_rel_delta = 0.10  # 10% - if relative change is >10%, always adjust
            
            is_meaningful_adjustment = (
                # Either: absolute notional is above minimum AND relative change is >5%
                (abs_delta_notional >= min_meaningful_delta_notional and rel_delta_pct >= min_meaningful_rel_delta) or
                # Or: relative change is significant (>10%) - allow even if absolute is small
                (rel_delta_pct >= significant_rel_delta)
            )
            
            if not is_meaningful_adjustment:
                logger.info(
                    "Skipping {}: existing position in same direction (cur={:.6g}, tgt={:.6g}, delta={:.6g}, delta_notional=${:.2f}, rel_delta={:.2%}). Adjustment too small.",
                    sym,
                    cur_size,
                    tgt_size,
                    delta,
                    abs_delta_notional,
                    rel_delta_pct,
                )
                continue
        
        # Skip if position is already at target (within tolerance)
        # This prevents unnecessary orders when the position is already correct
        abs_delta_notional = abs(delta * px)
        
        # Calculate relative delta (percentage of target)
        if abs(tgt_size) > 1e-8:
            rel_delta_pct = abs(delta / tgt_size)
        elif abs(cur_size) < 1e-8:
            # Both current and target are zero - already at target
            rel_delta_pct = 0.0
        else:
            # Target is zero but current is not - this is a closing position
            rel_delta_pct = 1.0
        
        # Skip if already at target: either absolute delta is tiny OR relative delta is tiny (< 0.1%)
        # BUT always allow closing positions (even if small) to ensure reconciliation
        is_closing_position = (abs(cur_size) > 1e-8 and abs(tgt_size) < 1e-8)
        is_already_at_target = (
            abs_delta_notional < float(cfg.sizing.min_notional_per_symbol) or 
            (abs(tgt_size) > 1e-8 and rel_delta_pct < 0.001)  # 0.1% tolerance for non-zero targets
        )
        
        if not is_closing_position and is_already_at_target:
            logger.debug(
                "Skipping {}: already at target (cur={:.6g}, tgt={:.6g}, delta={:.6g}, delta_notional=${:.2f}, rel_delta={:.2%})",
                sym,
                cur_size,
                tgt_size,
                delta,
                abs_delta_notional,
                rel_delta_pct,
            )
            continue

        # If this is a reduce-only trim and the delta is below minQty, we can't execute it without over-closing.
        # BUT: Always allow full closes (target = 0) regardless of minQty to ensure reconciliation.
        # This ensures positions outside the universe are always closed, even if very small.
        # Skip to avoid churn (flatten now, reopen next rebalance) only for partial reductions.
        if min_qty > 0:
            reduce_only_candidate = (cur_size > 0 and delta < 0) or (cur_size < 0 and delta > 0)
            is_full_close = abs(tgt_size) < 1e-8  # Target is zero (full close)
            if reduce_only_candidate and not is_full_close:
                # This is a partial reduction (not a full close)
                if abs(delta) < min_qty:
                    logger.info(
                        "Skipping {}: reduce-only trim below minQty (delta_qty={:.6g} < minQty={:.6g}). cur_qty={:.6g} tgt_qty={:.6g}",
                        sym,
                        float(delta),
                        float(min_qty),
                        float(cur_size),
                        float(tgt_size),
                    )
                    continue
                
                # CRITICAL: Check if remaining position size after reduction meets minimum requirements
                # If reducing would leave a position that's > 0 but < minQty, we need to either:
                # 1. Close the entire position (reduce to zero), OR
                # 2. Skip the reduction
                remaining_size = abs(tgt_size)
                if remaining_size > 0 and remaining_size < min_qty:
                    # Remaining position would be below minimum - close entire position instead
                    logger.info(
                        "Adjusting {}: reduction would leave position below minQty (remaining={:.6g} < minQty={:.6g}). Closing entire position instead. cur_qty={:.6g}",
                        sym,
                        remaining_size,
                        float(min_qty),
                        float(cur_size),
                    )
                    # Set target to zero to close the entire position
                    tgt_size = 0.0
                    delta = tgt_size - cur_size
                    # Recalculate abs_delta_notional for the new delta
                    abs_delta_notional = abs(delta * px)
                
                # Also check if target position meets minimum notional requirement
                # If target is non-zero but below min_notional, we should close the position
                if abs(tgt_size) > 0:
                    tgt_notional_check = abs(tgt_size * px)
                    min_notional = float(cfg.sizing.min_notional_per_symbol)
                    if tgt_notional_check < min_notional:
                        logger.info(
                            "Adjusting {}: target position would be below min_notional (${:.2f} < ${:.2f}). Closing entire position instead. cur_qty={:.6g} tgt_qty={:.6g}",
                            sym,
                            tgt_notional_check,
                            min_notional,
                            float(cur_size),
                            float(tgt_size),
                        )
                        # Set target to zero to close the entire position
                        tgt_size = 0.0
                        delta = tgt_size - cur_size
                        abs_delta_notional = abs(delta * px)

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
        
        # Log if this is closing a position outside the universe
        if is_closing_position:
            logger.info(
                "Planning order to close position outside universe: {} {} qty={:.6g} (cur={:.6g}, tgt=0, delta_notional=${:.2f}, reduceOnly={})",
                sym,
                side,
                abs(delta),
                cur_size,
                abs_delta_notional,
                bool(reduce_only),
            )
        
        orders.append(
            PlannedOrder(
                symbol=sym,
                side=side,
                qty=abs(delta),
                reduce_only=bool(reduce_only),
                order_type=cfg.execution.order_type,
                limit_price=None,
                reason="close_outside_universe" if is_closing_position else "rebalance_delta",
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
    # Hygiene: manage existing open orders to avoid duplicate pending orders across rebalances.
    # Modes:
    # - none: don't cancel open orders
    # - bot_only: cancel only orders placed by this bot (orderLinkId starts with xsrev-)
    # - symbols: cancel ALL open orders for symbols in play (current positions or targets)
    # - all: cancel ALL open LIMIT orders in the account (USDT-settled) before rebalancing (use with dedicated subaccounts)
    canceled: list[dict[str, Any]] = []
    mode = str(getattr(cfg.execution, "cancel_open_orders", "bot_only"))

    raw_pos0 = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
    parsed0 = _parse_positions(raw_pos0)
    positions0 = parsed0.positions
    symbols_in_play = {normalize_symbol(s) for s in (set(positions0) | set(target_notionals))}

    try:
        if mode == "bot_only":
            xsrev_open_by_symbol: dict[str, list[dict[str, Any]]] = {}
            open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=None, settle_coin="USDT")
            for o in open_orders:
                link = str(o.get("orderLinkId") or "")
                if not link.startswith("xsrev-"):
                    continue
                sym = normalize_symbol(str(o.get("symbol") or ""))
                oid = str(o.get("orderId") or "")
                if not sym or not oid:
                    continue
                xsrev_open_by_symbol.setdefault(sym, []).append(o)
            for sym, lst in xsrev_open_by_symbol.items():
                for o in lst:
                    try:
                        oid = str(o.get("orderId") or "")
                        link = str(o.get("orderLinkId") or "")
                        if oid:
                            client.cancel_order(category=cfg.exchange.category, symbol=sym, order_id=oid)
                            canceled.append({"symbol": sym, "orderId": oid, "orderLinkId": link})
                    except Exception as e:
                        logger.warning("Failed to cancel stale open order: {}", e)
        elif mode == "symbols":
            # Cancel ALL open orders for symbols we manage.
            for sym in sorted(symbols_in_play):
                try:
                    open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
                except Exception as e:
                    logger.warning("Open-order list failed for {}: {}", sym, e)
                    continue
                for o in open_orders:
                    try:
                        oid = str(o.get("orderId") or "")
                        link = str(o.get("orderLinkId") or "")
                        if oid:
                            client.cancel_order(category=cfg.exchange.category, symbol=sym, order_id=oid)
                            canceled.append({"symbol": sym, "orderId": oid, "orderLinkId": link})
                    except Exception as e:
                        logger.warning("Failed to cancel open order for {}: {}", sym, e)
        elif mode == "all":
            # Cancel *all* open orders for each symbol in play (including conditional/stop orders),
            # then verify and hard-cancel by orderId as a last resort.
            for sym in sorted(symbols_in_play):
                # 1) cancel normal orders
                try:
                    client.cancel_all_orders(category=cfg.exchange.category, symbol=sym, order_filter="Order")
                except Exception as e:
                    logger.warning("Cancel-all(Order) failed for {}: {}", sym, e)
                # 2) cancel conditional/stop orders (if any)
                try:
                    client.cancel_all_orders(category=cfg.exchange.category, symbol=sym, order_filter="StopOrder")
                except Exception:
                    # Not all accounts/categories support this filter consistently; ignore.
                    pass

                # Give the exchange a brief moment to reflect cancellations
                time.sleep(0.15)

                # 3) verify and hard-cancel anything still open
                try:
                    open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
                except Exception as e:
                    logger.warning("Open-order list failed for {} after cancel-all: {}", sym, e)
                    continue
                for o in open_orders:
                    try:
                        oid = str(o.get("orderId") or "")
                        link = str(o.get("orderLinkId") or "")
                        if oid:
                            client.cancel_order(category=cfg.exchange.category, symbol=sym, order_id=oid)
                            canceled.append({"symbol": sym, "orderId": oid, "orderLinkId": link})
                    except Exception as e:
                        logger.warning("Failed to cancel open order for {}: {}", sym, e)
        else:
            # none
            pass
    except Exception as e:
        logger.warning("Open-order cleanup failed (continuing): {}", e)

    if canceled:
        logger.info("Canceled {} open orders (mode={}).", len(canceled), mode)
        # Brief pause to allow exchange to reflect cancellations before re-fetching positions
        time.sleep(0.2)

    # Re-fetch positions after canceling orders to get current state (in case any pending orders filled)
    raw_pos = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
    logger.info("Fetched {} raw position records from exchange", len(raw_pos))
    parsed = _parse_positions(raw_pos)
    positions = parsed.positions
    hedge_mode_symbols = parsed.hedge_mode_symbols
    
    # Log all positions including those filtered out
    if hedge_mode_symbols:
        logger.warning(
            "Found {} positions in HEDGE MODE (positionIdx=1/2) - these are excluded from one-way mode: {}",
            len(hedge_mode_symbols),
            ", ".join(sorted(hedge_mode_symbols.keys())),
        )
        for sym, info in sorted(hedge_mode_symbols.items()):
            logger.warning(
                "  - {}: net_size={:.6g}, positionIdxs={} (CANNOT BE CLOSED by one-way mode bot)",
                sym,
                info.get("net_size", 0.0),
                info.get("positionIdxs", []),
            )
    
    # Check for positions that might have been filtered (very small sizes)
    all_symbols_from_raw = set()
    filtered_small_positions: list[tuple[str, float]] = []
    for p in raw_pos:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue
        all_symbols_from_raw.add(sym)
        size = float(p.get("size") or 0.0)
        if abs(size) > 1e-12 and sym not in positions and sym not in hedge_mode_symbols:
            # Position exists but was filtered - likely net to zero or very small
            filtered_small_positions.append((sym, size))
    
    if filtered_small_positions:
        logger.info(
            "Found {} positions filtered out (net size < 1e-12 or net to zero): {}",
            len(filtered_small_positions),
            ", ".join(sorted([s for s, _ in filtered_small_positions])),
        )
    
    logger.info(
        "Position parsing summary: {} raw records -> {} parsed positions ({} hedge mode, {} filtered as zero)",
        len(raw_pos),
        len(positions),
        len(hedge_mode_symbols),
        len(filtered_small_positions),
    )
    
    # Early check: Identify positions outside universe from raw exchange positions (before adjustments)
    # This gives us the true picture of what's actually open on the exchange
    positions_outside_universe_raw: list[tuple[str, float]] = []
    for sym, pos in positions.items():
        tgt_notional = target_notionals.get(sym, 0.0)
        if abs(tgt_notional) < 1e-8 and abs(pos.size) > 1e-8:
            positions_outside_universe_raw.append((sym, pos.size))
    
    if positions_outside_universe_raw:
        outside_symbols = sorted([s for s, _ in positions_outside_universe_raw])
        logger.warning(
            "Found {} positions outside target universe (should be closed) - RAW exchange positions: {}",
            len(positions_outside_universe_raw),
            ", ".join(outside_symbols),
        )
        for sym, size in sorted(positions_outside_universe_raw, key=lambda x: abs(x[1]), reverse=True):
            pos = positions.get(sym)
            try:
                ob = md.get_orderbook_stats(sym)
                px = float(ob.mid)
                notional = abs(size * px)
            except Exception:
                px = float(pos.mark_price) if pos and pos.mark_price > 0 else 0.0
                notional = abs(size * px) if px > 0 else 0.0
            logger.warning(
                "  - {}: qty={:.6g}, notional=${:.2f} (target=0, should close)",
                sym,
                size,
                notional,
            )
    else:
        logger.info("All parsed exchange positions match the target universe ({} symbols)", len(positions))

    # Adjust positions to account for any remaining open orders (pending fills)
    # This prevents over-sizing when orders from previous rebalances are still pending
    symbols_in_play_adjusted = {normalize_symbol(s) for s in (set(positions) | set(target_notionals))}
    pending_order_adjustments: dict[str, float] = {}
    for sym in symbols_in_play_adjusted:
        try:
            open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
            if not open_orders:
                continue
            # Sum pending order quantities (positive for buys, negative for sells)
            pending_qty = 0.0
            for o in open_orders:
                side = str(o.get("side", "")).upper()
                qty = float(o.get("qty") or o.get("orderQty") or 0.0)
                if side == "BUY":
                    pending_qty += qty
                elif side == "SELL":
                    pending_qty -= qty
            if abs(pending_qty) > 1e-8:
                pending_order_adjustments[sym] = pending_qty
                logger.debug(
                    "{} has pending orders: {:.6g} qty (will adjust position from {:.6g})",
                    sym,
                    pending_qty,
                    float(positions.get(sym).size) if sym in positions else 0.0,
                )
        except Exception as e:
            logger.warning("Failed to check open orders for {}: {}", sym, e)

    # Apply pending order adjustments to positions
    if pending_order_adjustments:
        adjusted_positions: dict[str, Position] = {}
        for sym, pos in positions.items():
            adj = pending_order_adjustments.get(sym, 0.0)
            adjusted_positions[sym] = Position(symbol=sym, size=float(pos.size) + adj, mark_price=pos.mark_price)
        # Also create entries for symbols with pending orders but no current position
        for sym, adj in pending_order_adjustments.items():
            if sym not in adjusted_positions:
                # Need mark price for new position - use orderbook or 0
                try:
                    ob = md.get_orderbook_stats(sym)
                    mark = float(ob.mid)
                except Exception:
                    mark = 0.0
                adjusted_positions[sym] = Position(symbol=sym, size=adj, mark_price=mark)
        positions = adjusted_positions
        logger.info(
            "Adjusted positions for {} symbols with pending orders: {}",
            len(pending_order_adjustments),
            sorted(pending_order_adjustments.keys()),
        )

    # Check for hedge mode (check initial parse, will check final parse later)
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
                "current_symbols": len(positions_final),
                "planned_orders": 0,
            },
        }

    # Provide equity to the executor so it can apply "bump_to_min_qty" safely within per-symbol caps.
    try:
        equity_exec = fetch_equity_usdt(client=client)
    except Exception:
        equity_exec = None
    ex = Executor(client=client, md=md, cfg=cfg, dry_run=dry_run, equity_usdt=equity_exec)
    # Final position re-fetch right before planning to catch any fills that happened during cancellation/adjustment
    # This ensures we're using the absolute latest position data
    raw_pos_final = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
    parsed_final = _parse_positions(raw_pos_final)
    positions_final = parsed_final.positions
    
    # Check for hedge mode in final fetch as well
    if parsed_final.hedge_mode_symbols:
        logger.error(
            "Detected hedge-mode positions (positionIdx=1/2) in final position fetch. This bot currently requires ONE-WAY mode. "
            "Hedge symbols: {}",
            sorted(parsed_final.hedge_mode_symbols.keys()),
        )
        return {
            "orders": [],
            "positions": {k: v.__dict__ for k, v in positions_final.items()},
            "hedge_mode_symbols": parsed_final.hedge_mode_symbols,
            "canceled_open_orders": canceled,
            "summary": {
                "error": "hedge_mode_not_supported",
                "target_symbols": len(target_notionals),
                "current_symbols": len(positions_final),
                "planned_orders": 0,
            },
        }
    
    # Log if positions changed between adjustment and final fetch (indicates rapid fills)
    if pending_order_adjustments:
        for sym in set(positions) | set(positions_final):
            old_size = float(positions.get(sym).size) if sym in positions else 0.0
            new_size = float(positions_final.get(sym).size) if sym in positions_final else 0.0
            if abs(new_size - old_size) > 1e-6:
                logger.info(
                    "Position changed for {}: {:.6g} -> {:.6g} (likely filled during cancellation/adjustment)",
                    sym,
                    old_size,
                    new_size,
                )
    
    # Verify no open orders remain for symbols we're about to trade (safety check)
    # If orders remain, retry cancellation and re-fetch positions to account for any fills
    if mode in ("symbols", "all"):
        symbols_to_check = {normalize_symbol(s) for s in (set(positions_final) | set(target_notionals))}
        remaining_orders: list[dict[str, Any]] = []
        for sym in symbols_to_check:
            try:
                open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
                if open_orders:
                    remaining_orders.extend(open_orders)
            except Exception:
                pass
        
        if remaining_orders:
            logger.warning(
                "Found {} remaining open orders after cancellation (mode={}). Retrying cancellation. Symbols: {}",
                len(remaining_orders),
                mode,
                sorted({normalize_symbol(str(o.get("symbol", ""))) for o in remaining_orders}),
            )
            # Retry cancellation for remaining orders
            for o in remaining_orders:
                try:
                    sym = normalize_symbol(str(o.get("symbol", "")))
                    oid = str(o.get("orderId") or "")
                    if sym and oid:
                        client.cancel_order(category=cfg.exchange.category, symbol=sym, order_id=oid)
                        canceled.append({"symbol": sym, "orderId": oid, "orderLinkId": str(o.get("orderLinkId", ""))})
                except Exception as e:
                    logger.warning("Failed to cancel remaining order {}: {}", o.get("orderId"), e)
            
            # Wait a moment for cancellations to process
            time.sleep(0.3)
            
            # Re-fetch positions again to account for any orders that filled during the retry
            raw_pos_final = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
            parsed_final = _parse_positions(raw_pos_final)
            positions_final = parsed_final.positions
            
            # Check one more time for any still-remaining orders
            # Filter out orders that are filled or partially filled - only count truly open orders
            still_remaining: list[dict[str, Any]] = []
            for sym in symbols_to_check:
                try:
                    open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
                    if open_orders:
                        # Filter out filled/partially filled orders - only count orders that are still open
                        for o in open_orders:
                            order_status = str(o.get("orderStatus", "")).upper()
                            # Only count orders that are actually still open (not filled, cancelled, etc.)
                            if order_status in ("NEW", "PARTIALLYFILLED"):
                                still_remaining.append(o)
                            # If order is filled, it's fine - it will be reflected in positions
                except Exception:
                    pass
            
            if still_remaining:
                # Log details about remaining orders
                order_details = {}
                for o in still_remaining:
                    sym = normalize_symbol(str(o.get("symbol", "")))
                    status = str(o.get("orderStatus", ""))
                    qty = float(o.get("qty") or o.get("orderQty") or 0.0)
                    filled = float(o.get("cumExecQty") or 0.0)
                    order_details.setdefault(sym, []).append({
                        "status": status,
                        "qty": qty,
                        "filled": filled,
                        "remaining": qty - filled,
                    })
                
                logger.warning(
                    "{} orders still open after retry cancellation (some may be filling). Symbols: {}. Order details: {}. Continuing with rebalance but positions may be adjusted.",
                    len(still_remaining),
                    sorted({normalize_symbol(str(o.get("symbol", ""))) for o in still_remaining}),
                    order_details,
                )
                # Don't abort - these orders may fill, and we'll account for them in position calculations
            else:
                logger.info("Successfully canceled all remaining orders on retry.")

    # Final check: account for any pending orders in position calculation (defensive)
    # This ensures we don't over-size if orders are still pending
    symbols_final_check = {normalize_symbol(s) for s in (set(positions_final) | set(target_notionals))}
    pending_qty_by_symbol: dict[str, float] = {}
    for sym in symbols_final_check:
        try:
            open_orders = client.get_open_orders(category=cfg.exchange.category, symbol=sym)
            if not open_orders:
                continue
            pending_qty = 0.0
            for o in open_orders:
                side = str(o.get("side", "")).upper()
                qty = float(o.get("qty") or o.get("orderQty") or 0.0)
                if side == "BUY":
                    pending_qty += qty
                elif side == "SELL":
                    pending_qty -= qty
            if abs(pending_qty) > 1e-8:
                pending_qty_by_symbol[sym] = pending_qty
        except Exception:
            pass
    
    # Adjust positions to account for pending orders
    if pending_qty_by_symbol:
        logger.warning(
            "Found pending orders for {} symbols after final check. Adjusting position calculations. Symbols: {}",
            len(pending_qty_by_symbol),
            sorted(pending_qty_by_symbol.keys()),
        )
        adjusted_positions_final: dict[str, Position] = {}
        for sym, pos in positions_final.items():
            adj = pending_qty_by_symbol.get(sym, 0.0)
            adjusted_positions_final[sym] = Position(symbol=sym, size=float(pos.size) + adj, mark_price=pos.mark_price)
        # Also create entries for symbols with pending orders but no current position
        for sym, adj in pending_qty_by_symbol.items():
            if sym not in adjusted_positions_final:
                try:
                    ob = md.get_orderbook_stats(sym)
                    mark = float(ob.mid)
                except Exception:
                    mark = 0.0
                adjusted_positions_final[sym] = Position(symbol=sym, size=adj, mark_price=mark)
        positions_final = adjusted_positions_final

    # Identify positions outside the target universe (should be closed)
    positions_outside_universe: list[tuple[str, float]] = []
    for sym, pos in positions_final.items():
        tgt_notional = target_notionals.get(sym, 0.0)
        if abs(tgt_notional) < 1e-8 and abs(pos.size) > 1e-8:
            # Position exists but target is zero (outside universe)
            positions_outside_universe.append((sym, pos.size))
    
    if positions_outside_universe:
        outside_symbols = sorted([s for s, _ in positions_outside_universe])
        logger.warning(
            "Found {} positions outside target universe (should be closed): {}",
            len(positions_outside_universe),
            ", ".join(outside_symbols),
        )
        for sym, size in sorted(positions_outside_universe, key=lambda x: abs(x[1]), reverse=True):
            pos = positions_final.get(sym)
            try:
                ob = md.get_orderbook_stats(sym)
                px = float(ob.mid)
                notional = abs(size * px)
            except Exception:
                px = float(pos.mark_price) if pos and pos.mark_price > 0 else 0.0
                notional = abs(size * px) if px > 0 else 0.0
            logger.warning(
                "  - {}: qty={:.6g}, notional=${:.2f} (target=0, should close)",
                sym,
                size,
                notional,
            )
    else:
        logger.info("All open positions match the target universe ({} symbols)", len(positions_final))

    orders = plan_rebalance_orders(cfg=cfg, md=md, current_positions=positions_final, target_notionals=target_notionals)
    reconcile_top = _summarize_reconcile(positions=positions_final, target_notionals=target_notionals, md=md, limit=12)
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
            "positions": {k: v.__dict__ for k, v in positions_final.items()},
            "reconcile_top": reconcile_top,
            "canceled_open_orders": canceled,
            "summary": {"target_symbols": len(target_notionals), "current_symbols": len(positions_final), "planned_orders": 0},
        }

    logger.info("Planned {} orders", len(orders))
    for o in orders:
        ex.place_with_fallback(o)

    # Summary of position reconciliation
    positions_in_universe = [s for s in positions_final.keys() if abs(target_notionals.get(s, 0.0)) > 1e-8]
    positions_outside = [s for s in positions_final.keys() if abs(target_notionals.get(s, 0.0)) < 1e-8]
    targets_not_open = [s for s in target_notionals.keys() if s not in positions_final or abs(positions_final.get(s, Position(symbol=s, size=0.0, mark_price=0.0)).size) < 1e-8]
    
    logger.info(
        "Position reconciliation summary: {} total open positions ({} in universe, {} outside), {} target symbols ({} not yet open)",
        len(positions_final),
        len(positions_in_universe),
        len(positions_outside),
        len(target_notionals),
        len(targets_not_open),
    )

    return {
        "orders": [o.__dict__ for o in orders],
        "positions": {k: v.__dict__ for k, v in positions_final.items()},
        "reconcile_top": reconcile_top,
        "canceled_open_orders": canceled,
        "summary": {
            "target_symbols": len(target_notionals),
            "current_symbols": len(positions_final),
            "positions_in_universe": len(positions_in_universe),
            "positions_outside_universe": len(positions_outside),
            "targets_not_open": len(targets_not_open),
            "planned_orders": len(orders),
        },
    }


def fetch_equity_usdt(*, client: BybitClient) -> float:
    wallet = client.get_wallet_balance(account_type="UNIFIED")
    return _wallet_equity_usdt(wallet)


