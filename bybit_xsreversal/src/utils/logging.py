from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_dir: str | Path, level: str = "INFO") -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, enqueue=True, backtrace=False, diagnose=False)
    logger.add(
        str(Path(log_dir) / "bot.log"),
        level=level,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        rotation="20 MB",
        retention="10 days",
    )


