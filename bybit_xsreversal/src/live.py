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
from src.utils.time import last_complete_daily_close, next_run_time, now_utc


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
    t = Table(title="Target notionals (USDT)", show_lines=False)
    t.add_column("Symbol")
    t.add_column("Notional", justify="right")
    for sym, n in sorted(notionals.items(), key=lambda kv: abs(kv[1]), reverse=True):
        t.add_row(sym, f"{n:,.2f}")
    console.print(t)


def run_live(cfg: BotConfig, *, dry_run: bool) -> None:
    console = Console()

    auth = BybitAuth(
        api_key=_need_env(cfg.exchange.api_key_env),
        api_secret=_need_env(cfg.exchange.api_secret_env),
    )
    client = BybitClient(auth=auth, testnet=cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    risk = RiskManager(cfg=cfg.risk, state_dir=Path("outputs") / "state")

    try:
        while True:
            now = now_utc()
            nxt = next_run_time(now, cfg.rebalance.time_utc)
            sleep_s = max(1.0, (nxt - now).total_seconds())
            logger.info("Next rebalance scheduled at {} (sleep {:.1f}s)", nxt.isoformat(), sleep_s)
            time.sleep(sleep_s)

            # small delay to ensure daily candle is finalized
            time.sleep(max(0, int(cfg.rebalance.candle_close_delay_seconds)))

            rb_now = now_utc()
            close_ts = last_complete_daily_close(rb_now, cfg.rebalance.candle_close_delay_seconds)
            asof_bar = close_ts - timedelta(days=1)
            logger.info("Rebalance now={} using last complete daily bar start={}", rb_now.isoformat(), asof_bar.isoformat())

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
                continue

            # Equity + risk check
            equity = fetch_equity_usdt(client=client)
            ok, risk_info = risk.check(equity)
            if not ok:
                logger.error("Risk kill-switch active; skipping trades: {}", risk_info)
                continue

            # Fetch candles needed for indicators/signal
            buffer_days = int(max(cfg.sizing.vol_lookback_days + 10, cfg.signal.lookback_days + 5, cfg.filters.regime_filter.ema_slow + 20, 40))
            start = asof_bar - timedelta(days=buffer_days)
            end = asof_bar + timedelta(days=1)

            candles: dict[str, Any] = {}
            for s in passed:
                try:
                    df = md.get_daily_candles(s, start, end, use_cache=True, cache_write=True)
                    if asof_bar in df.index:
                        candles[s] = df
                except Exception as e:
                    logger.warning("Skipping {}: candle load failed: {}", s, e)

            market_df = None
            if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime:
                proxy = cfg.filters.regime_filter.market_proxy_symbol
                try:
                    market_df = md.get_daily_candles(proxy, start, end, use_cache=True, cache_write=True)
                except Exception as e:
                    logger.warning("Market proxy candles unavailable: {}", e)

            targets, snapshot = compute_targets_from_daily_candles(
                candles=candles,
                config=cfg,
                equity_usd=equity,
                asof=asof_bar,
                market_proxy_candles=market_df,
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
                "risk": risk_info,
                "config": cfg.model_dump(),
            }
            snap_path.write_bytes(orjson.dumps(snap_payload, option=orjson.OPT_INDENT_2))
            logger.info("Saved rebalance snapshot: {}", snap_path.resolve())

            res = run_rebalance(cfg=cfg, client=client, md=md, target_notionals=targets.notionals_usd, dry_run=dry_run)
            (out_dir / "execution_result.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
            logger.info("Rebalance done. Result saved to {}", (out_dir / "execution_result.json").resolve())
    finally:
        client.close()


