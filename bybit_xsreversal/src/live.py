from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import orjson
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import BotConfig
from src.data.bybit_client import BybitAuth, BybitClient
from src.data.market_data import MarketData, normalize_symbol
from src.execution.rebalance import fetch_equity_usdt, run_rebalance
from src.execution.risk import RiskManager
from src.strategy.xs_reversal import compute_targets_from_daily_candles
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


def run_live(cfg: BotConfig, *, dry_run: bool, run_once: bool = False, force: bool = False) -> None:
    console = Console()

    # Optional env override for testnet (installer writes this, and many operators expect it to control the environment).
    # Config still remains the source of truth if env var is absent.
    env_testnet = os.getenv("BYBIT_TESTNET", "").strip().lower()
    if env_testnet in ("1", "true", "yes", "y"):
        if cfg.exchange.testnet is not True:
            logger.warning("Overriding config exchange.testnet={} with env BYBIT_TESTNET=true", cfg.exchange.testnet)
        cfg.exchange.testnet = True
    elif env_testnet in ("0", "false", "no", "n"):
        if cfg.exchange.testnet is not False:
            logger.warning("Overriding config exchange.testnet={} with env BYBIT_TESTNET=false", cfg.exchange.testnet)
        cfg.exchange.testnet = False

    auth = BybitAuth(
        api_key=_need_env(cfg.exchange.api_key_env),
        api_secret=_need_env(cfg.exchange.api_secret_env),
    )
    client = BybitClient(auth=auth, testnet=cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    risk = RiskManager(cfg=cfg.risk, state_dir=Path("outputs") / "state")
    state_path = Path("outputs") / "state" / "rebalance_state.json"

    try:
        logger.info(
            "Live trader starting: testnet={} base_url={} category={} dry_run={} rebalance_time_utc={}",
            bool(cfg.exchange.testnet),
            getattr(client, "_base_url", "unknown"),
            str(cfg.exchange.category),
            bool(dry_run),
            str(cfg.rebalance.time_utc),
        )

        def _rebalance_once() -> None:
            # small delay to ensure daily candle is finalized
            time.sleep(max(0, int(cfg.rebalance.candle_close_delay_seconds)))

            rb_now = now_utc()  # use current wall time after delay
            close_ts = last_complete_daily_close(rb_now, cfg.rebalance.candle_close_delay_seconds)
            asof_bar = close_ts - timedelta(days=1)
            logger.info("Rebalance now={} using last complete daily bar start={}", rb_now.isoformat(), asof_bar.isoformat())

            # Rebalance interval control (stateful)
            interval_days = max(1, int(cfg.rebalance.interval_days))
            if not force and interval_days > 1 and state_path.exists():
                try:
                    st = orjson.loads(state_path.read_bytes())
                    last = st.get("last_rebalance_day")
                    if last:
                        last_dt = datetime.fromisoformat(last).replace(tzinfo=UTC)
                        if (asof_bar.date() - last_dt.date()).days < interval_days:
                            logger.info("Skipping rebalance due to interval_days={} (last={})", interval_days, last_dt.date().isoformat())
                            return
                except Exception as e:
                    logger.warning("Failed to parse rebalance_state.json: {}", e)

            # Universe + microstructure filtering
            symbols = [normalize_symbol(s) for s in md.get_liquidity_ranked_symbols()]
            passed: list[str] = []
            micro_meta: dict[str, Any] = {}
            for s in symbols:
                try:
                    ok, info = md.passes_microstructure_filters(s)
                    micro_meta[s] = info
                    if ok:
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
            logger.info(
                "Equity: {:.4f} USDT (min_notional_per_symbol={} USDT)",
                float(equity),
                float(cfg.sizing.min_notional_per_symbol),
            )
            if float(equity) <= 0.0:
                logger.error("Equity is <= 0 USDT; cannot size trades. Skipping rebalance.")
                return
            ok, risk_info = risk.check(equity)
            if not ok:
                logger.error("Risk kill-switch active; skipping trades: {}", risk_info)
                return

            # Current weights from positions (for turnover controls)
            current_weights: dict[str, float] = {}
            try:
                positions = client.get_positions(category=cfg.exchange.category, settle_coin="USDT")
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

            # Fetch candles needed for indicators/signal.
            # Must cover min_history_days; otherwise all symbols can be filtered out as "insufficient_history".
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
                # Common: universe too small (new listings, missing candles, insufficient history)
                logger.warning("Skipping rebalance: {}", str(e))
                return
            except Exception as e:
                logger.exception("Unexpected error during target computation; skipping rebalance: {}", e)
                return

            # High-signal rebalance summary for journald visibility.
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

            # Safety: if the strategy produced an empty target book, do NOT interpret that as
            # "flatten everything" by default. This usually means filters/thresholds removed all targets.
            if not targets.notionals_usd:
                if bool(getattr(cfg.rebalance, "flatten_on_empty_targets", False)):
                    logger.warning(
                        "Empty target book and flatten_on_empty_targets=true: flattening all positions. See snapshot: {}",
                        snap_path.resolve(),
                    )
                    res = run_rebalance(cfg=cfg, client=client, md=md, target_notionals={}, dry_run=dry_run)
                    (out_dir / "execution_result.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
                    logger.info("Flatten done. Result saved to {}", (out_dir / "execution_result.json").resolve())
                else:
                    logger.warning(
                        "No target notionals produced (empty target book). Skipping execution. See snapshot: {}",
                        snap_path.resolve(),
                    )
                return

            res = run_rebalance(cfg=cfg, client=client, md=md, target_notionals=targets.notionals_usd, dry_run=dry_run)
            (out_dir / "execution_result.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
            logger.info("Rebalance done. Result saved to {}", (out_dir / "execution_result.json").resolve())

            # Update rebalance state
            try:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_bytes(orjson.dumps({"last_rebalance_day": asof_bar.date().isoformat()}, option=orjson.OPT_INDENT_2))
            except Exception as e:
                logger.warning("Failed to write rebalance_state.json: {}", e)

        if run_once:
            logger.info("run_once enabled: executing a single rebalance immediately.")
            _rebalance_once()
            return

        # Catch-up behavior:
        # If the process starts AFTER today's scheduled time (or grace window) and we haven't rebalanced for the
        # current asof_bar yet, run once immediately instead of silently skipping a whole day.
        try:
            now0 = now_utc()
            t = parse_hhmm(cfg.rebalance.time_utc)
            today = utc_day_start(now0)
            scheduled_today = datetime.combine(today.date(), t, tzinfo=UTC)
            if now0 >= scheduled_today:
                close_ts0 = last_complete_daily_close(now0, int(cfg.rebalance.candle_close_delay_seconds))
                asof0 = close_ts0 - timedelta(days=1)
                last_done = None
                if state_path.exists():
                    try:
                        st = orjson.loads(state_path.read_bytes())
                        last_done = st.get("last_rebalance_day")
                    except Exception:
                        last_done = None
                if last_done != asof0.date().isoformat():
                    logger.warning(
                        "Missed scheduled rebalance time earlier today (scheduled_today={} now={}). "
                        "Catching up immediately for asof_bar={}.",
                        scheduled_today.isoformat(),
                        now0.isoformat(),
                        asof0.date().isoformat(),
                    )
                    _rebalance_once()
        except Exception as e:
            logger.warning("Startup catch-up check failed (continuing with normal scheduler): {}", e)

        while True:
            now = now_utc()
            nxt = next_run_time(now, cfg.rebalance.time_utc, grace_seconds=int(cfg.rebalance.startup_grace_seconds))
            sleep_s = max(1.0, (nxt - now).total_seconds())
            logger.info("Next rebalance scheduled at {} (sleep {:.1f}s)", nxt.isoformat(), sleep_s)
            time.sleep(sleep_s)
            _rebalance_once()
    finally:
        client.close()


