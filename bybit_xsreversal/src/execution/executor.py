from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from loguru import logger

from src.config import BotConfig
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


class Executor:
    def __init__(self, *, client: BybitClient, md: MarketData, cfg: BotConfig, dry_run: bool) -> None:
        self.client = client
        self.md = md
        self.cfg = cfg
        self.dry_run = dry_run

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

    def place(self, order: PlannedOrder) -> dict[str, Any] | None:
        meta = self.md.get_instrument_meta(order.symbol)
        qty = self._format_qty(order.qty, meta)
        if qty <= 0:
            logger.info("Skipping {} {}: qty below min step", order.symbol, order.side)
            return None

        if self.dry_run:
            logger.info("[DRY RUN] {} {} qty={} reduceOnly={} type={} px={} reason={}",
                        order.symbol, order.side, qty, order.reduce_only, order.order_type, order.limit_price, order.reason)
            return None

        link_id = f"xsrev-{uuid.uuid4().hex[:16]}"
        if order.order_type == "market":
            payload = {
                "symbol": order.symbol,
                "side": order.side,
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",
                "reduceOnly": bool(order.reduce_only),
                "orderLinkId": link_id,
            }
        else:
            px = float(order.limit_price) if order.limit_price is not None else self._limit_price(order.symbol, order.side)
            tif = "PostOnly" if self.cfg.execution.post_only else "GTC"
            payload = {
                "symbol": order.symbol,
                "side": order.side,
                "orderType": "Limit",
                "qty": str(qty),
                "price": str(px),
                "timeInForce": tif,
                "reduceOnly": bool(order.reduce_only),
                "orderLinkId": link_id,
            }

        res = self.client.create_order(category=self.cfg.exchange.category, order=payload)
        logger.info("Order placed: {} {} qty={} type={} id={}", order.symbol, order.side, qty, order.order_type, res.get("orderId"))
        return res

    def place_with_fallback(self, order: PlannedOrder) -> None:
        """
        Place a maker-biased limit order and wait briefly. If still open, cancel and fall back to market/IOC.
        """
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
            time.sleep(0.5)

        # Cancel and fallback
        try:
            self.client.cancel_order(category=self.cfg.exchange.category, symbol=order.symbol, order_id=oid)
            logger.info("Canceled stale order {}", oid)
        except Exception as e:
            logger.warning("Cancel failed (maybe already filled): {}: {}", oid, e)

        fallback = PlannedOrder(
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            reduce_only=order.reduce_only,
            order_type="market",
            limit_price=None,
            reason=f"{order.reason}|fallback_market",
        )
        self.place(fallback)


