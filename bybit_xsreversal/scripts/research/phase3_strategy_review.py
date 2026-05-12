from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src.config import load_config
from src.data.bybit_client import BybitClient
from src.data.market_data import MarketData
from src.research.phase3 import run_phase3_review
from src.utils.logging import setup_logging


def _ts_dir(base: Path) -> Path:
    ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Phase 3 strategy robustness review (research only).")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant names to run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else _ts_dir(Path("outputs") / "research" / "phase3")
    setup_logging(Path("outputs") / "logs", level="INFO")

    client = BybitClient(auth=None, testnet=cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    try:
        outputs = run_phase3_review(cfg=cfg, md=md, output_dir=out_dir, variant_names=args.variants)
    finally:
        client.close()

    logger.info("Phase 3 research outputs written to {}", out_dir.resolve())
    for name, path in outputs.items():
        logger.info("  {} -> {}", name, path.resolve())


if __name__ == "__main__":
    main()
