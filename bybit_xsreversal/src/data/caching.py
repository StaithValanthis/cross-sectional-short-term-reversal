from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_bytes(orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def read_json(path: str | Path) -> Any:
    p = Path(path)
    return orjson.loads(p.read_bytes())


def candles_cache_path(cache_dir: str | Path, symbol: str, interval: str) -> Path:
    base = ensure_dir(Path(cache_dir) / "klines")
    return base / f"{symbol}_{interval}.csv"


def load_cached_candles(cache_dir: str | Path, symbol: str, interval: str) -> pd.DataFrame | None:
    p = candles_cache_path(cache_dir, symbol, interval)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def save_cached_candles(cache_dir: str | Path, symbol: str, interval: str, df: pd.DataFrame) -> None:
    p = candles_cache_path(cache_dir, symbol, interval)
    ensure_dir(p.parent)
    out = df.copy()
    out = out.sort_index()
    out = out.reset_index().rename(columns={"index": "ts"})
    out.to_csv(p, index=False)


def funding_cache_path(cache_dir: str | Path, symbol: str) -> Path:
    base = ensure_dir(Path(cache_dir) / "funding")
    return base / f"{symbol}_funding.csv"


def load_cached_funding(cache_dir: str | Path, symbol: str) -> pd.DataFrame | None:
    p = funding_cache_path(cache_dir, symbol)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.set_index("ts").sort_index()
    return df


def save_cached_funding(cache_dir: str | Path, symbol: str, df: pd.DataFrame) -> None:
    p = funding_cache_path(cache_dir, symbol)
    ensure_dir(p.parent)
    out = df.sort_index().reset_index().rename(columns={"index": "ts"})
    out.to_csv(p, index=False)


