from __future__ import annotations

import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import BotConfig
from src.data.bybit_client import BybitAuth, BybitClient
from src.data.market_data import MarketData, normalize_symbol
from src.execution.executor import Executor, PlannedOrder
from src.execution.intraday_exits import (
    build_intraday_candle_window,
    ensure_entry_ts_from_position,
    evaluate_intraday_exit_decisions,
    load_intraday_state,
    save_intraday_state,
)
from src.execution.rebalance import fetch_equity_usdt, run_rebalance
from src.execution.risk import RiskManager
from src.strategy.xs_reversal import compute_targets_from_daily_candles
from src.utils.lock import FileLock
from src.utils.time import last_complete_daily_close, next_run_time, now_utc, parse_hhmm, utc_day_start


def _need_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing required env var: {name}")
    return v


def _ts_dir(base: Path) -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _print_targets(console: Console, notionals: dict[str, float]) -> None:
    if not notionals:
        console.print("[bold yellow]No target notionals to trade (empty target book).[/bold yellow]")
        return
    t = Table(title="Target notionals (USDT)", show_lines=False)
    t.add_column("Symbol")
    t.add_column("Notional", justify="right")
    for sym, n in sorted(notionals.items(), key=lambda kv: abs(kv[1]), reverse=True):
        t.add_row(sym, f"{n:,.2f}")
    console.print(t)


def _apply_leverage_on_startup(*, cfg: BotConfig, client: BybitClient, md: MarketData) -> None:
    ex = cfg.exchange
    if not bool(getattr(ex, "set_leverage_on_startup", False)):
        return

    state_path = Path(str(getattr(ex, "leverage_state_path", "outputs/state/leverage_state.json")))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    today = datetime.now(tz=UTC).date().isoformat()
    leverage = str(getattr(ex, "leverage", "5"))
    mode = str(getattr(ex, "leverage_apply_mode", "universe"))

    # Idempotency: don't spam on restart.
    if state_path.exists():
        try:
            raw = orjson.loads(state_path.read_bytes())
            if (
                str(raw.get("day") or "") == today
                and str(raw.get("leverage") or "") == leverage
                and str(raw.get("mode") or "") == mode
            ):
                logger.info("Leverage already applied today (mode={}, leverage={}); skipping set-leverage calls.", mode, leverage)
                return
        except Exception:
            pass

    symbols: list[str] = []
    if mode == "positions":
        pos = client.get_positions(category="linear", settle_coin="USDT")
        for p in pos:
            sym = normalize_symbol(str(p.get("symbol", "")))
            if not sym:
                continue
            size = float(p.get("size") or 0.0)
            if abs(size) <= 1e-12:
                continue
            symbols.append(sym)
    else:
        symbols = [normalize_symbol(s) for s in md.get_liquidity_ranked_symbols()]

    symbols = sorted(set(symbols))[: int(getattr(ex, "leverage_symbols_max", 200))]
    if not symbols:
        logger.info("Leverage-on-startup enabled but no symbols selected (mode={}).", mode)
        return

    ok = 0
    for sym in symbols:
        try:
            client.set_leverage(category="linear", symbol=sym, leverage=leverage)
            ok += 1
        except Exception as e:
            logger.warning("Failed to set leverage for {}: {}", sym, e)

    payload = {"day": today, "ts_utc": datetime.now(tz=UTC).isoformat(), "mode": mode, "leverage": leverage, "symbols": symbols, "ok": ok}
    state_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    logger.info("Set leverage on startup: ok={} / {} (mode={}, leverage={})", ok, len(symbols), mode, leverage)


def _has_hedge_mode_positions(raw_positions: list[dict[str, Any]]) -> bool:
    for p in raw_positions or []:
        try:
            pos_idx = int(p.get("positionIdx") or 0)
        except Exception:
            pos_idx = 0
        size = float(p.get("size") or 0.0)
        if pos_idx in (1, 2) and abs(size) > 1e-12:
            return True
    return False


def _get_last_prices_by_symbol(client: BybitClient, symbols: list[str]) -> dict[str, float]:
    want = set(map(normalize_symbol, symbols))
    out: dict[str, float] = {}
    for t in client.get_tickers(category="linear"):
        sym = normalize_symbol(str(t.get("symbol", "")))
        if sym not in want:
            continue
        lp = t.get("lastPrice") or t.get("markPrice") or t.get("indexPrice")
        try:
            out[sym] = float(lp)
        except Exception:
            continue
    return out


def _cancel_existing_exit_orders(*, client: BybitClient, symbols: list[str]) -> int:
    """
    Cancel only intraday-exit bot orders (identified by orderLinkId prefix).
    """
    canceled = 0
    for sym in symbols:
        try:
            open_orders = client.get_open_orders(category="linear", symbol=sym)
        except Exception:
            continue
        for o in open_orders:
            link = str(o.get("orderLinkId") or "")
            oid = str(o.get("orderId") or "")
            if not oid:
                continue
            if not link.startswith("xsrev-exit-"):
                continue
            try:
                client.cancel_order(category="linear", symbol=sym, order_id=oid)
                canceled += 1
            except Exception:
                pass
    return canceled


def run_live(cfg: BotConfig, *, dry_run: bool, run_once: bool = False, force: bool = False) -> None:
    console = Console()

    # Optional env override for testnet.
    env_testnet = os.getenv("BYBIT_TESTNET", "").strip().lower()
    if env_testnet in ("1", "true", "yes", "y"):
        if cfg.exchange.testnet is not True:
            logger.warning("Overriding config exchange.testnet={} with env BYBIT_TESTNET=true", cfg.exchange.testnet)
        cfg.exchange.testnet = True
    elif env_testnet in ("0", "false", "no", "n"):
        if cfg.exchange.testnet is not False:
            logger.warning("Overriding config exchange.testnet={} with env BYBIT_TESTNET=false", cfg.exchange.testnet)
        cfg.exchange.testnet = False

    auth = BybitAuth(api_key=_need_env(cfg.exchange.api_key_env), api_secret=_need_env(cfg.exchange.api_secret_env))
    client = BybitClient(auth=auth, testnet=cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    risk = RiskManager(cfg=cfg.risk, state_dir=Path("outputs") / "state")
    rb_state_path = Path("outputs") / "state" / "rebalance_state.json"

    intraday_cfg = cfg.intraday_exits
    intraday_state = load_intraday_state(intraday_cfg.state_path) if intraday_cfg.enabled else None
    next_intraday_ts: datetime | None = None

    try:
        logger.info(
            "Live trader starting: testnet={} base_url={} category={} dry_run={} rebalance_time_utc={} intraday_exits={}",
            bool(cfg.exchange.testnet),
            getattr(client, "_base_url", "unknown"),
            str(cfg.exchange.category),
            bool(dry_run),
            str(cfg.rebalance.time_utc),
            bool(intraday_cfg.enabled),
        )

        _apply_leverage_on_startup(cfg=cfg, client=client, md=md)

        def _rebalance_once() -> None:
            lock = FileLock(Path("outputs") / "state" / "rebalance.lock")
            if not lock.acquire():
                logger.warning("Rebalance lock is held by another process; skipping to avoid duplicate orders.")
                return
            try:
                # Delay to ensure daily candle is finalized
                time.sleep(max(0, int(cfg.rebalance.candle_close_delay_seconds)))

                rb_now = now_utc()
                close_ts = last_complete_daily_close(rb_now, cfg.rebalance.candle_close_delay_seconds)
                asof_bar = close_ts - timedelta(days=1)
                logger.info("Rebalance now={} using last complete daily bar start={}", rb_now.isoformat(), asof_bar.isoformat())

                # Interval control (stateful)
                interval_days = max(1, int(cfg.rebalance.interval_days))
                if not force and interval_days > 1 and rb_state_path.exists():
                    try:
                        st = orjson.loads(rb_state_path.read_bytes())
                        last = st.get("last_rebalance_day")
                        if last:
                            last_dt = datetime.fromisoformat(last).replace(tzinfo=UTC)
                            days_since_last = (rb_now.date() - last_dt.date()).days
                            logger.debug(
                                "Interval check: last_rebalance_day={}, today={}, days_since={}, interval_days={}, will_rebalance={}",
                                last_dt.date().isoformat(),
                                rb_now.date().isoformat(),
                                days_since_last,
                                interval_days,
                                days_since_last >= interval_days,
                            )
                            if days_since_last < interval_days:
                                logger.info(
                                    "Skipping rebalance due to interval_days={} (last={}, today={}, days_since={})",
                                    interval_days,
                                    last_dt.date().isoformat(),
                                    rb_now.date().isoformat(),
                                    days_since_last,
                                )
                                return
                    except Exception as e:
                        logger.warning("Failed to parse rebalance_state.json (continuing): {}", e)

                # Universe + microstructure filtering
                symbols = [normalize_symbol(s) for s in md.get_liquidity_ranked_symbols()]
                passed: list[str] = []
                micro_meta: dict[str, Any] = {}
                for s in symbols:
                    try:
                        ok_ms, info = md.passes_microstructure_filters(s)
                        micro_meta[s] = info
                        if ok_ms:
                            passed.append(s)
                    except Exception as e:
                        micro_meta[s] = {"error": str(e)}

                if len(passed) < 5:
                    logger.warning("Universe too small after spread/depth filters: {}", len(passed))
                    return

                logger.info(
                    "Universe: {} symbols (top_n_by_volume={}), passed microstructure filters: {}",
                    len(symbols),
                    int(cfg.universe.top_n_by_volume),
                    len(passed),
                )

                # Funding filter (optional)
                funding_meta: dict[str, Any] = {}
                if cfg.funding.filter.enabled:
                    force_mainnet = bool(cfg.funding.filter.use_mainnet_data_even_on_testnet and cfg.exchange.testnet)
                    max_abs = float(cfg.funding.filter.max_abs_daily_funding_rate)
                    kept: list[str] = []
                    for s in passed:
                        try:
                            fr = md.get_latest_daily_funding_rate(s, force_mainnet=force_mainnet)
                            funding_meta[s] = {"daily_funding_rate": fr}
                            if fr is None or abs(float(fr)) <= max_abs:
                                kept.append(s)
                        except Exception as e:
                            funding_meta[s] = {"error": str(e)}
                    passed = kept

                if len(passed) < 5:
                    logger.warning("Universe too small after funding filter: {}", len(passed))
                    return

                # Equity + risk check
                equity = fetch_equity_usdt(client=client)
                logger.info("Equity: {:.4f} USDT (min_notional_per_symbol={} USDT)", float(equity), float(cfg.sizing.min_notional_per_symbol))
                if float(equity) <= 0.0:
                    logger.error("Equity is <= 0 USDT; cannot size trades. Skipping rebalance.")
                    return
                ok_risk, risk_info = risk.check(equity)
                if not ok_risk:
                    logger.error("Risk kill-switch active; skipping trades: {}", risk_info)
                    return

                # Current weights from positions (for turnover controls inside target computation)
                current_weights: dict[str, float] = {}
                try:
                    positions = client.get_positions(category="linear", settle_coin="USDT")
                    for p in positions:
                        sym = normalize_symbol(str(p.get("symbol", "")))
                        side = str(p.get("side", ""))
                        size = float(p.get("size") or 0.0)
                        mark = float(p.get("markPrice") or p.get("avgPrice") or 0.0)
                        if mark <= 0 or size == 0:
                            continue
                        signed = size if side == "Buy" else -size
                        notional = signed * mark
                        current_weights[sym] = float(notional) / float(equity) if equity > 0 else 0.0
                except Exception as e:
                    logger.warning("Failed to compute current weights from positions: {}", e)

                # Fetch daily candles
                buffer_days = int(
                    max(
                        cfg.universe.min_history_days + 10,
                        cfg.sizing.vol_lookback_days + 10,
                        cfg.signal.lookback_days + 5,
                        cfg.filters.regime_filter.ema_slow + 20,
                        40,
                    )
                )
                start = asof_bar - timedelta(days=buffer_days)
                end = asof_bar + timedelta(days=1)
                logger.info(
                    "Candle window: start={} end={} (buffer_days={}, min_history_days={})",
                    start.date().isoformat(),
                    end.date().isoformat(),
                    buffer_days,
                    int(cfg.universe.min_history_days),
                )

                candles: dict[str, Any] = {}
                candle_skip: dict[str, int] = {"load_error": 0, "missing_asof_bar": 0}
                for s in passed:
                    try:
                        df = md.get_daily_candles(s, start, end, use_cache=True, cache_write=True)
                        if asof_bar in df.index:
                            candles[s] = df
                        else:
                            candle_skip["missing_asof_bar"] += 1
                    except Exception as e:
                        logger.warning("Skipping {}: candle load failed: {}", s, e)
                        candle_skip["load_error"] += 1

                logger.info(
                    "Candles loaded: {} / {} symbols have required asof_bar={}",
                    len(candles),
                    len(passed),
                    asof_bar.date().isoformat(),
                )
                if candle_skip["missing_asof_bar"] or candle_skip["load_error"]:
                    logger.info(
                        "Candle skips: missing_asof_bar={}, load_error={}",
                        int(candle_skip["missing_asof_bar"]),
                        int(candle_skip["load_error"]),
                    )

                market_df = None
                if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime:
                    proxy = cfg.filters.regime_filter.market_proxy_symbol
                    try:
                        market_df = md.get_daily_candles(proxy, start, end, use_cache=True, cache_write=True)
                    except Exception as e:
                        logger.warning("Market proxy candles unavailable: {}", e)

                try:
                    targets, snapshot = compute_targets_from_daily_candles(
                        candles=candles,
                        config=cfg,
                        equity_usd=equity,
                        asof=asof_bar,
                        market_proxy_candles=market_df,
                        current_weights=current_weights,
                    )
                except ValueError as e:
                    logger.warning("Skipping rebalance: {}", str(e))
                    return
                except Exception as e:
                    logger.exception("Unexpected error during target computation; skipping rebalance: {}", e)
                    return

                gross_w = float(sum(abs(w) for w in targets.weights.values()))
                logger.info(
                    "Targets: {} symbols (longs={}, shorts={}), gross_weight={:.3f}, signal_mode={}",
                    len(targets.notionals_usd),
                    len(snapshot.selected_longs),
                    len(snapshot.selected_shorts),
                    gross_w,
                    str(snapshot.filters.get("signal_mode")),
                )
                _print_targets(console, targets.notionals_usd)

                out_dir = _ts_dir(Path("outputs") / "live")
                snap_path = out_dir / "rebalance_snapshot.json"
                snap_payload = {
                    "asof_bar": asof_bar.isoformat(),
                    "equity_usd": equity,
                    "targets": targets.notionals_usd,
                    "snapshot": snapshot.__dict__,
                    "microstructure": micro_meta,
                    "funding": funding_meta,
                    "risk": risk_info,
                    "config": cfg.model_dump(),
                }
                snap_path.write_bytes(orjson.dumps(snap_payload, option=orjson.OPT_INDENT_2))
                logger.info("Saved rebalance snapshot: {}", snap_path.resolve())

                if not targets.notionals_usd:
                    logger.warning(
                        "No target notionals produced (empty target book). Will still reconcile existing positions. See snapshot: {}",
                        snap_path.resolve(),
                    )

                res = run_rebalance(cfg=cfg, client=client, md=md, target_notionals=targets.notionals_usd, dry_run=dry_run)
                (out_dir / "execution_result.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
                logger.info("Rebalance done. Result saved to {}", (out_dir / "execution_result.json").resolve())

                # Enrich snapshot with risk-exit details computed during execution
                try:
                    snap_obj = orjson.loads(snap_path.read_bytes())
                    snap_obj["risk_exits"] = res.get("risk_exits") or {}
                    snap_obj["targets_effective"] = res.get("targets_effective") or snap_obj.get("targets")
                    snap_path.write_bytes(orjson.dumps(snap_obj, option=orjson.OPT_INDENT_2))
                except Exception as e:
                    logger.warning("Failed to enrich rebalance snapshot with risk exits: {}", e)

                # Update rebalance state
                try:
                    rb_state_path.parent.mkdir(parents=True, exist_ok=True)
                    saved_date = rb_now.date().isoformat()
                    rb_state_path.write_bytes(orjson.dumps({"last_rebalance_day": saved_date}, option=orjson.OPT_INDENT_2))
                    logger.info("Saved rebalance state: last_rebalance_day={} (interval_days={})", saved_date, interval_days)
                except Exception as e:
                    logger.warning("Failed to write rebalance_state.json: {}", e)
            except Exception as e:
                logger.exception("Rebalance execution failed (continuing scheduler): {}", e)
            finally:
                lock.release()

        def _intraday_exits_cycle() -> None:
            nonlocal intraday_state
            if not intraday_cfg.enabled or intraday_state is None:
                return

            cycle0 = time.time()
            counters = {
                "positions_checked": 0,
                "decisions_triggered": 0,
                "exits_placed": 0,
                "skipped_hedge_mode": 0,
                "skipped_kill_switch": 0,
                "missing_candles": 0,
                "missing_entry_price": 0,
            }

            try:
                equity = fetch_equity_usdt(client=client)
                ok_risk, risk_info = risk.check(equity)
                if not ok_risk:
                    counters["skipped_kill_switch"] += 1
                    logger.warning("Intraday exits: kill-switch active; skipping cycle: {}", risk_info)
                    return

                raw_pos = client.get_positions(category="linear", settle_coin="USDT")
                if _has_hedge_mode_positions(raw_pos):
                    counters["skipped_hedge_mode"] += 1
                    logger.warning("Intraday exits: hedge-mode positions detected (positionIdx=1/2). Skipping intraday cycle.")
                    return

                # Only consider non-zero positions
                active = [p for p in raw_pos if abs(float(p.get("size") or 0.0)) > 1e-12]
                counters["positions_checked"] = len(active)
                if not active:
                    return

                ensure_entry_ts_from_position(st=intraday_state, positions=active)

                start, end = build_intraday_candle_window(intraday_cfg)
                symbols = sorted({normalize_symbol(str(p.get("symbol", ""))) for p in active if p.get("symbol")})

                c1: dict[str, Any] = {}
                c4: dict[str, Any] = {}
                for sym in symbols:
                    try:
                        c1[sym] = md.get_candles(sym, intraday_cfg.candle_interval_trigger, start, end, use_cache=True, cache_write=True)
                        c4[sym] = md.get_candles(sym, intraday_cfg.atr_interval, start, end, use_cache=True, cache_write=True)
                    except Exception as e:
                        counters["missing_candles"] += 1
                        logger.warning("Intraday exits: candle fetch failed for {}: {}", sym, e)

                last_px = None
                if intraday_cfg.use_last_price_trigger:
                    last_px = _get_last_prices_by_symbol(client, symbols)

                decisions = evaluate_intraday_exit_decisions(
                    cfg=intraday_cfg,
                    positions=active,
                    candles_1h_by_symbol=c1,
                    candles_4h_by_symbol=c4,
                    last_prices_by_symbol=last_px,
                    state=intraday_state,
                )
                counters["decisions_triggered"] = len(decisions)

                if not decisions:
                    return

                if intraday_cfg.dry_run:
                    for d in decisions:
                        logger.info(
                            "Intraday exits [DRY RUN] would_exit {} {} qty={} reason={} stop_type={} stop={} trigger={} atr={} entry={}",
                            d.symbol,
                            d.side,
                            d.qty_to_close,
                            d.reason,
                            d.stop_type,
                            d.stop_price,
                            d.trigger_price,
                            d.atr,
                            d.entry_price,
                        )
                    return

                # Cancel previous exit orders (optional)
                if intraday_cfg.cancel_existing_exit_orders:
                    _cancel_existing_exit_orders(client=client, symbols=[d.symbol for d in decisions])

                ex = Executor(client=client, md=md, cfg=cfg, dry_run=False, equity_usdt=equity)
                for d in decisions:
                    order_type = "market" if intraday_cfg.exit_order_type == "market" else "limit"
                    limit_px = None
                    if order_type == "limit":
                        off = float(intraday_cfg.exit_price_offset_bps) / 10_000.0
                        stats = md.get_orderbook_stats(d.symbol)
                        if d.side == "Sell":
                            limit_px = float(stats.best_bid) * (1.0 - off)
                        else:
                            limit_px = float(stats.best_ask) * (1.0 + off)
                    po = PlannedOrder(
                        symbol=d.symbol,
                        side=d.side,
                        qty=d.qty_to_close,
                        reduce_only=True,
                        order_type=order_type,
                        limit_price=limit_px,
                        reason=d.reason,
                        order_link_id_prefix="xsrev-exit-",
                    )
                    ex.place_with_fallback(po)
                    counters["exits_placed"] += 1

            finally:
                if intraday_state is not None:
                    save_intraday_state(intraday_cfg.state_path, intraday_state)
                dt = time.time() - cycle0
                logger.info(
                    "Intraday exits cycle done in {:.2f}s: {}",
                    dt,
                    counters,
                )

        if run_once:
            logger.info("run_once enabled: executing a single rebalance immediately.")
            _rebalance_once()
            return

        # Startup catch-up: if started after today's scheduled time and we haven't rebalanced today, run once.
        try:
            now0 = now_utc()
            t = parse_hhmm(cfg.rebalance.time_utc)
            today0 = utc_day_start(now0)
            scheduled_today = datetime.combine(today0.date(), t, tzinfo=UTC)
            if now0 >= scheduled_today:
                last_done = None
                if rb_state_path.exists():
                    try:
                        st = orjson.loads(rb_state_path.read_bytes())
                        last_done = st.get("last_rebalance_day")
                    except Exception:
                        last_done = None
                if last_done != now0.date().isoformat():
                    logger.warning(
                        "Missed scheduled rebalance time earlier today (scheduled_today={} now={}). Catching up immediately.",
                        scheduled_today.isoformat(),
                        now0.isoformat(),
                    )
                    _rebalance_once()
        except Exception as e:
            logger.warning("Startup catch-up check failed (continuing with normal scheduler): {}", e)

        # Scheduler: daily rebalance + optional intraday exit-only loop between rebalances.
        while True:
            now = now_utc()
            nxt_rb = next_run_time(now, cfg.rebalance.time_utc, grace_seconds=int(cfg.rebalance.startup_grace_seconds))

            if intraday_cfg.enabled and next_intraday_ts is None:
                next_intraday_ts = now + timedelta(minutes=int(intraday_cfg.interval_minutes))

            # If we're within the daily rebalance window, run rebalance.
            if now >= nxt_rb:
                _rebalance_once()
                # Reset intraday schedule after a rebalance to avoid clustering.
                if intraday_cfg.enabled:
                    next_intraday_ts = now_utc() + timedelta(minutes=int(intraday_cfg.interval_minutes))
                continue

            # Intraday exits between rebalances
            if intraday_cfg.enabled and next_intraday_ts is not None and now >= next_intraday_ts:
                _intraday_exits_cycle()
                next_intraday_ts = now_utc() + timedelta(minutes=int(intraday_cfg.interval_minutes))
                continue

            # Sleep until the next event
            next_events: list[datetime] = [nxt_rb]
            if intraday_cfg.enabled and next_intraday_ts is not None:
                next_events.append(next_intraday_ts)
            wake = min(next_events)
            sleep_s = max(1.0, (wake - now).total_seconds())
            logger.info("Next event scheduled at {} (sleep {:.1f}s)", wake.isoformat(), sleep_s)
            time.sleep(sleep_s)
    finally:
        client.close()

