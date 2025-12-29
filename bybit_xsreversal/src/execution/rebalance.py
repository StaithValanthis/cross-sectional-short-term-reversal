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


def _parse_positions(raw_positions: list[dict[str, Any]]) -> dict[str, Position]:
    out: dict[str, Position] = {}
    for p in raw_positions:
        sym = normalize_symbol(str(p.get("symbol", "")))
        if not sym:
            continue
        side = str(p.get("side", ""))
        size = float(p.get("size") or 0.0)
        mark = float(p.get("markPrice") or p.get("avgPrice") or 0.0)
        # In one-way mode, Bybit returns two rows with side=Buy/Sell size>0; convert to signed.
        signed = size if side == "Buy" else -size
        if abs(signed) < 1e-12:
            continue
        out[sym] = Position(symbol=sym, size=signed, mark_price=mark)
    return out


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

        # Determine if sign flip needed
        if cur_size != 0.0 and np.sign(cur_size) != np.sign(tgt_size) and abs(tgt_size) > 0:
            # 1) reduce-only to flat
            side_flat: Any = "Sell" if cur_size > 0 else "Buy"
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    side=side_flat,
                    qty=abs(cur_size),
                    reduce_only=True,
                    order_type=cfg.execution.order_type,
                    limit_price=None,
                    reason="sign_flip_flatten",
                )
            )
            # 2) open to target from flat
            side_open: Any = "Buy" if tgt_size > 0 else "Sell"
            orders.append(
                PlannedOrder(
                    symbol=sym,
                    side=side_open,
                    qty=abs(tgt_size),
                    reduce_only=False,
                    order_type=cfg.execution.order_type,
                    limit_price=None,
                    reason="sign_flip_open",
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
    raw_pos = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
    positions = _parse_positions(raw_pos)

    ex = Executor(client=client, md=md, cfg=cfg, dry_run=dry_run)
    orders = plan_rebalance_orders(cfg=cfg, md=md, current_positions=positions, target_notionals=target_notionals)

    if not orders:
        logger.info("No rebalance orders required.")
        return {"orders": [], "positions": {k: v.__dict__ for k, v in positions.items()}}

    logger.info("Planned {} orders", len(orders))
    for o in orders:
        ex.place_with_fallback(o)

    return {"orders": [o.__dict__ for o in orders], "positions": {k: v.__dict__ for k, v in positions.items()}}


def fetch_equity_usdt(*, client: BybitClient) -> float:
    wallet = client.get_wallet_balance(account_type="UNIFIED")
    return _wallet_equity_usdt(wallet)


