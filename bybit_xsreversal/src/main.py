from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.backtest.backtester import run_backtest
from src.config import load_config
from src.data.bybit_client import BybitAuth, BybitClient
from src.data.market_data import MarketData
from src.live import run_live
from src.utils.logging import setup_logging


def _default_config_path() -> str:
    return os.getenv("BOT_CONFIG_PATH", "config/config.yaml")


def _ts_dir(base: Path) -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_market_data(cfg) -> MarketData:
    # For backtests, auth is optional. For live, run_live constructs its own authenticated client.
    auth = None
    key = os.getenv(cfg.exchange.api_key_env)
    sec = os.getenv(cfg.exchange.api_secret_env)
    if key and sec:
        auth = BybitAuth(api_key=key, api_secret=sec)
    client = BybitClient(auth=auth, testnet=cfg.exchange.testnet)
    return MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(prog="bybit-xsreversal")
    p.add_argument("--config", default=_default_config_path(), help="Path to config YAML")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run backtest")
    bt.add_argument("--output-dir", default=None, help="Output directory (default outputs/backtest/<ts>)")

    lv = sub.add_parser("live", help="Run live trader (scheduler)")
    lv.add_argument("--dry-run", action="store_true", help="Print intended orders without placing them")

    args = p.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path("outputs") / "logs", level=args.log_level)
    logger.info("Loaded config from {}", Path(args.config).resolve())

    if args.cmd == "backtest":
        out_dir = Path(args.output_dir) if args.output_dir else _ts_dir(Path("outputs") / "backtest")
        md = _build_market_data(cfg)
        try:
            run_backtest(cfg, md, out_dir)
        finally:
            md.client.close()
        return

    if args.cmd == "live":
        run_live(cfg, dry_run=bool(args.dry_run))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()


