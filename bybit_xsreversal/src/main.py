from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.backtest.backtester import run_backtest
from src.backtest.optimizer import optimize_config
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

def _normalize_argv(argv: list[str]) -> list[str]:
    """
    Argparse requires global options (like --config) to appear before the subcommand.
    Users often run: `bybit-xsreversal optimize --config config/config.yaml`.
    Make this ergonomic by moving `--config PATH` (and `--log-level LEVEL`) to the front if needed.
    """
    if not argv:
        return argv

    subcmds = {"backtest", "optimize", "live"}
    if argv[0] not in subcmds:
        return argv

    out = list(argv)
    for flag in ("--config", "--log-level"):
        if flag in out:
            i = out.index(flag)
            # Need a value after the flag
            if i + 1 >= len(out):
                continue
            if i > 0:
                val = out[i + 1]
                # remove flag+val
                del out[i : i + 2]
                # insert at front
                out = [flag, val] + out
    return out


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(prog="bybit-xsreversal")
    p.add_argument("--config", default=_default_config_path(), help="Path to config YAML")
    p.add_argument("--log-level", default="INFO", help="Logging level (INFO/DEBUG/...)")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest", help="Run backtest")
    bt.add_argument("--output-dir", default=None, help="Output directory (default outputs/backtest/<ts>)")

    opt = sub.add_parser("optimize", help="Optimize strategy parameters (small grid search) and write best back to config.yaml")
    opt.add_argument("--output-dir", default=None, help="Output directory (default outputs/optimize/<ts>)")
    opt.add_argument("--level", choices=["quick", "standard", "deep"], default="standard", help="Optimization depth preset")
    opt.add_argument("--candidates", type=int, default=None, help="Override number of candidates to evaluate")
    opt.add_argument("--method", choices=["random", "grid"], default="random", help="Candidate generation method")
    opt.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    opt.add_argument("--stage2-topk", type=int, default=None, help="How many top stage-1 candidates to re-evaluate with full backtest logic")
    opt.add_argument("--no-progress", action="store_true", help="Disable progress bar/ETA output")

    lv = sub.add_parser("live", help="Run live trader (scheduler)")
    lv.add_argument("--dry-run", action="store_true", help="Print intended orders without placing them")
    lv.add_argument("--run-once", action="store_true", help="Run a single rebalance immediately and exit")
    lv.add_argument("--force", action="store_true", help="Ignore interval_days state and force a rebalance (still respects risk checks)")

    argv = _normalize_argv(sys.argv[1:])
    args = p.parse_args(argv)

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

    if args.cmd == "optimize":
        out_dir = Path(args.output_dir) if args.output_dir else _ts_dir(Path("outputs") / "optimize")
        optimize_config(
            config_path=args.config,
            output_dir=out_dir,
            level=str(args.level),
            candidates=args.candidates,
            method=str(args.method),
            seed=int(args.seed),
            stage2_topk=args.stage2_topk,
            show_progress=not bool(args.no_progress),
        )
        logger.info("Config updated: {}", Path(args.config).resolve())
        return

    if args.cmd == "live":
        run_live(cfg, dry_run=bool(args.dry_run), run_once=bool(args.run_once), force=bool(args.force))
        return

    raise SystemExit(2)


if __name__ == "__main__":
    main()


