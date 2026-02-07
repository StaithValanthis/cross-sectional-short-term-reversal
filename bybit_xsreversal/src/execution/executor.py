from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Literal

import numpy as np
from loguru import logger

from src.config import BotConfig
from src.data.bybit_client import BybitAPIError
from src.data.bybit_client import BybitClient
from src.data.market_data import InstrumentMeta, MarketData


Side = Literal["Buy", "Sell"]


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return float(np.floor(x / step) * step)


def _round_tick(px: float, tick: float) -> float:
    if tick <= 0:
        return px
    return float(np.round(px / tick) * tick)


@dataclass(frozen=True)
class PlannedOrder:
    symbol: str
    side: Side
    qty: float
    reduce_only: bool
    order_type: Literal["limit", "market"]
    limit_price: float | None
    reason: str
    # Optional orderLinkId prefix (used to tag specific subsystems like intraday exits)
    order_link_id_prefix: str | None = None


class Executor:
    def __init__(self, *, client: BybitClient, md: MarketData, cfg: BotConfig, dry_run: bool, equity_usdt: float | None = None) -> None:
        self.client = client
        self.md = md
        self.cfg = cfg
        self.dry_run = dry_run
        self.equity_usdt = float(equity_usdt) if equity_usdt is not None else None
        self._cancel_replace_counts: dict[str, int] = {}
        self._quality: dict[str, Any] = {
            "placed_limit": 0,
            "placed_market": 0,
            "fallback_market": 0,
            "canceled_stale": 0,
            "cancel_failed": 0,
            "poll_fail": 0,
            "min_value_market_fallback": 0,
            "forced_close_direct_market": 0,
            "avg_cancel_age_seconds": None,
        }

    def _limit_price(self, symbol: str, side: Side) -> float:
        stats = self.md.get_orderbook_stats(symbol)
        off = float(self.cfg.execution.price_offset_bps) / 10_000.0

        if side == "Buy":
            # maker-biased: sit at/below best bid
            px = stats.best_bid * (1.0 - off) if self.cfg.execution.maker_bias else stats.mid
            px = min(px, stats.best_ask)  # avoid crossing too hard
        else:
            px = stats.best_ask * (1.0 + off) if self.cfg.execution.maker_bias else stats.mid
            px = max(px, stats.best_bid)

        meta = self.md.get_instrument_meta(symbol)
        return _round_tick(px, meta.tick_size)

    def _format_qty(self, qty: float, meta: InstrumentMeta) -> float:
        q = _round_step(abs(qty), meta.qty_step)
        if q < meta.min_qty:
            return 0.0
        if meta.max_qty is not None:
            q = min(q, meta.max_qty)
        return float(q)

    def _format_qty_str(self, qty: float, meta: InstrumentMeta) -> str:
        """
        Format qty for Bybit:
        - round DOWN to the instrument qty_step
        - enforce min_qty/max_qty
        - emit fixed-point string (no scientific notation, no float artifacts)
        """
        step = Decimal(str(meta.qty_step))
        q = Decimal(str(abs(qty)))
        if step > 0:
            k = (q / step).to_integral_value(rounding=ROUND_DOWN)
            q = (k * step).quantize(step)
        if q <= 0:
            return ""
        # Enforce bounds
        min_q = Decimal(str(meta.min_qty))
        if q < min_q:
            return ""
        if meta.max_qty is not None:
            max_q = Decimal(str(meta.max_qty))
            q = min(q, max_q)
            if step > 0:
                k = (q / step).to_integral_value(rounding=ROUND_DOWN)
                q = (k * step).quantize(step)
        # fixed-point string
        s = format(q, "f")
        return s

    def _ceil_qty_str(self, qty: float, meta: InstrumentMeta) -> str:
        """
        Round UP to the next qty_step (used when we must satisfy minQty/minNotional).
        """
        step = Decimal(str(meta.qty_step))
        q = Decimal(str(abs(qty)))
        if step > 0:
            k = (q / step).to_integral_value(rounding=ROUND_UP)
            q = (k * step).quantize(step)
        if q <= 0:
            return ""
        # enforce max qty if present
        if meta.max_qty is not None:
            max_q = Decimal(str(meta.max_qty))
            q = min(q, max_q)
        return format(q, "f")

    def place(self, order: PlannedOrder) -> dict[str, Any] | None:
        meta = self.md.get_instrument_meta(order.symbol)
        qty_str = self._format_qty_str(order.qty, meta)
        reason_s = str(order.reason)
        # Forced closes must prioritize closure over minQty/minNotional edge-cases.
        # Treat outside-universe closes and risk-triggered closes the same.
        is_forced_close = bool(order.reduce_only) and (
            ("close_outside_universe" in reason_s) or reason_s.startswith("risk_") or ("|risk_" in reason_s)
        )
        
        if not qty_str:
            # For positions outside the universe that must be closed, try market order even if below minQty
            # The exchange may accept market orders for very small positions even if limit orders are rejected
            if is_forced_close:
                logger.warning(
                    "Order qty below minQty for {} {} (forced close). Attempting market order with minQty.",
                    order.symbol,
                    order.side,
                )
                # Use minQty to ensure we can close the position
                if float(meta.min_qty) > 0:
                    qty_str = self._ceil_qty_str(float(meta.min_qty), meta)
                    if qty_str:
                        # Force market order for closing positions outside universe when qty is too small
                        logger.info("Using market order with minQty={} for {} {} (forced close)", qty_str, order.symbol, order.side)
                        # Create a new order with market type
                        market_order = PlannedOrder(
                            symbol=order.symbol,
                            side=order.side,
                            qty=float(qty_str),
                            reduce_only=order.reduce_only,
                            order_type="market",
                            limit_price=None,
                            reason=f"{order.reason}|forced_market",
                        )
                        # Recursively call place with the market order
                        return self.place(market_order)
                    else:
                        logger.error("Cannot format minQty for {} {} - cannot force-close position", order.symbol, order.side)
                        return None
                else:
                    logger.error("No minQty available for {} {} - cannot force-close position", order.symbol, order.side)
                    return None
            
            # Optional: bump tiny orders up to minQty when possible.
            # This is only safe-ish when we can ensure it doesn't exceed per-symbol notional caps.
            if not qty_str and (
                bool(getattr(self.cfg.execution, "bump_to_min_qty", False))
                and float(meta.min_qty) > 0
            ):
                bump_mode = str(getattr(self.cfg.execution, "bump_mode", "respect_cap"))
                # By default we avoid bumping reduce-only partial trims (can over-close/flatten and create churn).
                # If you really want "always trade min size", set bump_mode=force_exchange_min.
                if bool(order.reduce_only) and bump_mode != "force_exchange_min":
                    logger.info(
                        "Skipping {} {}: qty below min step (raw_qty={} qtyStep={} minQty={}); reduceOnly trim not bumped in mode={}.",
                        order.symbol,
                        order.side,
                        float(order.qty),
                        float(meta.qty_step),
                        float(meta.min_qty),
                        bump_mode,
                    )
                    return None
                try:
                    stats = self.md.get_orderbook_stats(order.symbol)
                    px_mid = float(stats.mid)
                except Exception:
                    px_mid = 0.0

                min_qty_str = self._ceil_qty_str(float(meta.min_qty), meta)
                min_notional = float(meta.min_qty) * float(px_mid) if px_mid > 0 else None

                cap = None
                if self.equity_usdt is not None:
                    cap = min(
                        float(self.cfg.sizing.max_notional_per_symbol),
                        float(self.cfg.sizing.max_leverage_per_symbol) * float(self.equity_usdt),
                    )

                allow_bump = False
                if bump_mode == "force_exchange_min":
                    allow_bump = bool(min_qty_str) and (min_notional is not None)
                else:
                    allow_bump = bool(min_qty_str) and (min_notional is not None) and (cap is not None) and (min_notional <= cap)

                if allow_bump and min_qty_str and min_notional is not None:
                    logger.warning(
                        "Bumping {} {} qty up to minQty={} (~{:.2f} USDT) because bump_to_min_qty=true (mode={} cap~{} USDT).",
                        order.symbol,
                        order.side,
                        min_qty_str,
                        float(min_notional),
                        bump_mode,
                        "n/a" if cap is None else f"{cap:.2f}",
                    )
                    qty_str = min_qty_str
                else:
                    if min_notional is not None and cap is not None:
                        logger.info(
                            "Skipping {} {}: qty below min step (raw_qty={} qtyStep={} minQty={}); cannot bump (minNotional~{:.2f} > cap~{:.2f} or missing price).",
                            order.symbol,
                            order.side,
                            float(order.qty),
                            float(meta.qty_step),
                            float(meta.min_qty),
                            float(min_notional),
                            float(cap),
                        )
                        return None

            if not qty_str:
                logger.info(
                    "Skipping {} {}: qty below min step (raw_qty={} qtyStep={} minQty={})",
                    order.symbol,
                    order.side,
                    float(order.qty),
                    float(meta.qty_step),
                    float(meta.min_qty),
                )
                return None

        # If we have a qty, ensure the order also satisfies minimum order VALUE (minNotional).
        if bool(getattr(self.cfg.execution, "bump_to_min_qty", False)):
            bump_mode = str(getattr(self.cfg.execution, "bump_mode", "respect_cap"))
            allow_reduce_only_bump = (not bool(order.reduce_only)) or (bump_mode == "force_exchange_min")
            if allow_reduce_only_bump:
                try:
                    px_for_notional = float(order.limit_price) if order.limit_price is not None else self._limit_price(order.symbol, order.side)
                except Exception:
                    px_for_notional = 0.0
                if px_for_notional > 0:
                    min_val = getattr(meta, "min_notional", None)
                    if min_val is None:
                        min_val = getattr(self.cfg.execution, "min_order_value_usdt", None)
                    try:
                        min_val_f = float(min_val) if min_val is not None else None
                    except Exception:
                        min_val_f = None
                    if min_val_f is not None and min_val_f > 0:
                        cur_qty_f = float(qty_str)
                        cur_notional = cur_qty_f * px_for_notional
                        if cur_notional + 1e-9 < min_val_f:
                            # Bump qty up to satisfy min order value
                            need_qty = min_val_f / px_for_notional
                            bumped_qty_str = self._ceil_qty_str(need_qty, meta)
                            cap = None
                            if self.equity_usdt is not None:
                                cap = min(
                                    float(self.cfg.sizing.max_notional_per_symbol),
                                    float(self.cfg.sizing.max_leverage_per_symbol) * float(self.equity_usdt),
                                )
                            bumped_notional = float(bumped_qty_str) * px_for_notional if bumped_qty_str else None
                            allow_bump_val = False
                            if bump_mode == "force_exchange_min":
                                allow_bump_val = bool(bumped_qty_str) and (bumped_notional is not None)
                            else:
                                allow_bump_val = bool(bumped_qty_str) and (bumped_notional is not None) and (cap is not None) and (bumped_notional <= cap)

                            if allow_bump_val and bumped_qty_str and bumped_notional is not None:
                                logger.warning(
                                    "Bumping {} {} qty up to {} to satisfy min order value {:.2f} USDT (was {:.2f}; mode={} cap~{}).",
                                    order.symbol,
                                    order.side,
                                    bumped_qty_str,
                                    float(min_val_f),
                                    float(cur_notional),
                                    bump_mode,
                                    "n/a" if cap is None else f"{cap:.2f}",
                                )
                                qty_str = bumped_qty_str
                            elif bumped_notional is not None and cap is not None:
                                logger.info(
                                    "Skipping {} {}: would violate min order value {:.2f} USDT (current {:.2f}); bump would be {:.2f} > cap~{:.2f}.",
                                    order.symbol,
                                    order.side,
                                    float(min_val_f),
                                    float(cur_notional),
                                    float(bumped_notional),
                                    float(cap),
                                )
                                return None

        if self.dry_run:
            logger.info("[DRY RUN] {} {} qty={} reduceOnly={} type={} px={} reason={}",
                        order.symbol, order.side, qty_str, order.reduce_only, order.order_type, order.limit_price, order.reason)
            return None

        prefix = str(order.order_link_id_prefix or "xsrev-")
        link_id = f"{prefix}{uuid.uuid4().hex[:16]}"
        if order.order_type == "market":
            self._quality["placed_market"] = int(self._quality["placed_market"]) + 1
            payload = {
                "symbol": order.symbol,
                "side": order.side,
                "orderType": "Market",
                "qty": qty_str,
                "timeInForce": "IOC",
                "reduceOnly": bool(order.reduce_only),
                "orderLinkId": link_id,
            }
        else:
            # For forced closes, do NOT use post-only / maker bias. Prefer IOC so we close now.
            if is_forced_close:
                stats = self.md.get_orderbook_stats(order.symbol)
                px = float(order.limit_price) if order.limit_price is not None else (float(stats.best_ask) if order.side == "Buy" else float(stats.best_bid))
                tif = "IOC"
            else:
                px = float(order.limit_price) if order.limit_price is not None else self._limit_price(order.symbol, order.side)
                tif = "PostOnly" if self.cfg.execution.post_only else "GTC"
            self._quality["placed_limit"] = int(self._quality["placed_limit"]) + 1
            payload = {
                "symbol": order.symbol,
                "side": order.side,
                "orderType": "Limit",
                "qty": qty_str,
                "price": str(px),
                "timeInForce": tif,
                "reduceOnly": bool(order.reduce_only),
                "orderLinkId": link_id,
            }

        try:
            res = self.client.create_order(category=self.cfg.exchange.category, order=payload)
            logger.info(
                "Order placed: {} {} qty={} type={} reduceOnly={} reason={} id={}",
                order.symbol,
                order.side,
                qty_str,
                order.order_type,
                bool(order.reduce_only),
                order.reason,
                res.get("orderId"),
            )
            return res
        except BybitAPIError as e:
            # If limit order fails due to minimum order value (110094) and this is a reduce-only order (closing position),
            # automatically retry with market order to ensure the position is closed
            # This is critical for positions outside the universe that must be closed
            is_min_order_value_error = (str(e.ret_code) == "110094" and "minimum order value" in str(e.ret_msg).lower())
            is_closing_position = bool(order.reduce_only)  # reduce_only means we're closing/reducing a position
            
            if is_min_order_value_error and is_closing_position and order.order_type == "limit":
                logger.warning(
                    "Limit order rejected due to minimum order value for {} {} (closing position). Retrying with market order to ensure position is closed.",
                    order.symbol,
                    order.side,
                )
                self._quality["min_value_market_fallback"] = int(self._quality["min_value_market_fallback"]) + 1
                # Retry with market order - for closing positions, we prioritize execution over maker fees
                market_payload = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "orderType": "Market",
                    "qty": qty_str,
                    "timeInForce": "IOC",
                    "reduceOnly": bool(order.reduce_only),
                    "orderLinkId": f"{prefix}{uuid.uuid4().hex[:16]}",
                }
                try:
                    self._quality["placed_market"] = int(self._quality["placed_market"]) + 1
                    res = self.client.create_order(category=self.cfg.exchange.category, order=market_payload)
                    logger.info(
                        "Market order placed (fallback for min value): {} {} qty={} reduceOnly={} reason={} id={}",
                        order.symbol,
                        order.side,
                        qty_str,
                        bool(order.reduce_only),
                        f"{order.reason}|market_fallback",
                        res.get("orderId"),
                    )
                    return res
                except BybitAPIError as e2:
                    logger.error(
                        "Market order also rejected for {} {}: retCode={} retMsg={}",
                        order.symbol,
                        order.side,
                        e2.ret_code,
                        e2.ret_msg,
                    )
                    return None
            
            logger.error(
                "Order rejected by Bybit: {} {} qty={} type={} reduceOnly={} reason={} retCode={} retMsg={}",
                order.symbol,
                order.side,
                qty_str,
                order.order_type,
                bool(order.reduce_only),
                order.reason,
                e.ret_code,
                e.ret_msg,
            )
            return None

    def place_with_fallback(self, order: PlannedOrder) -> None:
        """
        Place a maker-biased limit order and wait briefly. If still open, cancel and fall back to market/IOC.
        """
        reason_s = str(order.reason)
        is_forced_close = bool(order.reduce_only) and (
            ("close_outside_universe" in reason_s) or reason_s.startswith("risk_") or ("|risk_" in reason_s)
        )

        # Churn reduction: after N cancel/replace attempts per symbol in this cycle, use market directly for closes.
        max_cr = int(getattr(self.cfg.execution, "max_cancel_replace_per_symbol", 3) or 0)
        n_cr = int(self._cancel_replace_counts.get(order.symbol, 0))
        if is_forced_close and max_cr > 0 and n_cr >= max_cr:
            self._quality["forced_close_direct_market"] = int(self._quality["forced_close_direct_market"]) + 1
            fallback = PlannedOrder(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                reduce_only=order.reduce_only,
                order_type="market",
                limit_price=None,
                reason=f"{order.reason}|direct_market_after_cancel_replace_cap",
                order_link_id_prefix=order.order_link_id_prefix,
            )
            self.place(fallback)
            return

        if order.order_type == "market" or not self.cfg.execution.ioc_fallback:
            self.place(order)
            return

        placed = self.place(order)
        if self.dry_run or placed is None:
            return

        oid = str(placed.get("orderId", ""))
        if not oid:
            return

        deadline = time.time() + float(self.cfg.execution.max_order_age_seconds)
        while time.time() < deadline:
            try:
                open_orders = self.client.get_open_orders(category=self.cfg.exchange.category, symbol=order.symbol)
                if not any(str(o.get("orderId")) == oid for o in open_orders):
                    return  # filled or closed
            except Exception as e:
                # Don't crash the whole rebalance if an order status poll fails (rate limits, temporary API issues).
                logger.warning("Open-order poll failed for {} (orderId={}): {}", order.symbol, oid, e)
                self._quality["poll_fail"] = int(self._quality["poll_fail"]) + 1
            time.sleep(0.5)

        # Cancel and fallback
        try:
            self.client.cancel_order(category=self.cfg.exchange.category, symbol=order.symbol, order_id=oid)
            logger.info("Canceled stale order {}", oid)
            self._quality["canceled_stale"] = int(self._quality["canceled_stale"]) + 1
            self._cancel_replace_counts[order.symbol] = int(self._cancel_replace_counts.get(order.symbol, 0)) + 1
        except Exception as e:
            logger.warning("Cancel failed (maybe already filled): {}: {}", oid, e)
            self._quality["cancel_failed"] = int(self._quality["cancel_failed"]) + 1

        fallback = PlannedOrder(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            reduce_only=order.reduce_only,
            order_type="market",
            limit_price=None,
            reason=f"{order.reason}|fallback_market",
            order_link_id_prefix=order.order_link_id_prefix,
        )
        self._quality["fallback_market"] = int(self._quality["fallback_market"]) + 1
        self.place(fallback)

    def execution_quality(self) -> dict[str, Any]:
        # Provide a stable snapshot for logging / returning in run_rebalance
        q = dict(self._quality)
        return q


