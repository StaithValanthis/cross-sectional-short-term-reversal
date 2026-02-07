from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import BotConfig
from src.data.bybit_client import BybitClient
from src.data.caching import load_cached_candles, save_cached_candles, write_json, load_cached_funding, save_cached_funding


def normalize_symbol(sym: str) -> str:
    return sym.strip().upper()


@dataclass(frozen=True)
class OrderBookStats:
    symbol: str
    mid: float
    best_bid: float
    best_ask: float
    spread_bps: float
    depth_usd: float


@dataclass(frozen=True)
class InstrumentMeta:
    symbol: str
    qty_step: float
    min_qty: float
    max_qty: float | None
    tick_size: float
    min_notional: float | None  # minimum order value in USDT, if provided by exchange metadata


class MarketData:
    def __init__(self, *, client: BybitClient, config: BotConfig, cache_dir: str | Path) -> None:
        self.client = client
        self.config = config
        self.cache_dir = Path(cache_dir)
        self._instrument_cache: dict[str, InstrumentMeta] = {}

    # -------- Universe / liquidity --------
    def get_liquidity_ranked_symbols(self) -> list[str]:
        ucfg = self.config.universe
        if ucfg.include_symbols:
            syms = [normalize_symbol(s) for s in ucfg.include_symbols]
            syms = [s for s in syms if s and s not in set(map(normalize_symbol, ucfg.exclude_symbols))]
            return syms

        tickers = self.client.get_tickers(category=self.config.exchange.category)
        rows: list[tuple[str, float, float]] = []
        for t in tickers:
            sym = normalize_symbol(str(t.get("symbol", "")))
            if not sym.endswith("USDT"):
                continue
            if sym in set(map(normalize_symbol, ucfg.exclude_symbols)):
                continue
            try:
                quote_vol = float(t.get("turnover24h") or t.get("quoteVolume24h") or 0.0)
                oi = float(t.get("openInterest") or 0.0)
            except Exception:
                continue
            if quote_vol < ucfg.min_24h_quote_volume:
                continue
            if oi < ucfg.min_open_interest:
                continue
            rows.append((sym, quote_vol, oi))

        # Primary: quote volume, Secondary: open interest
        rows.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Liquidity bucket filter (rank-based)
        if ucfg.liquidity_bucket.enabled:
            lb = ucfg.liquidity_bucket
            filtered: list[tuple[str, float, float]] = []
            for i, r in enumerate(rows, start=1):
                if i <= lb.exclude_top_k:
                    continue
                if i < lb.min_rank or i > lb.max_rank:
                    continue
                filtered.append(r)
            rows = filtered

        ranked = [r[0] for r in rows[: ucfg.top_n_by_volume]]
        return ranked

    # -------- Orderbook filters --------
    def get_orderbook_stats(self, symbol: str) -> OrderBookStats:
        symbol = normalize_symbol(symbol)
        ob = self.client.get_orderbook(category=self.config.exchange.category, symbol=symbol, limit=50)
        bids = ob.get("b") or []
        asks = ob.get("a") or []
        if not bids or not asks:
            raise ValueError(f"Empty orderbook for {symbol}")
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = 0.5 * (best_bid + best_ask)
        spread_bps = (best_ask - best_bid) / mid * 10_000.0

        band_bps = float(self.config.filters.depth_band_bps)
        lo = mid * (1.0 - band_bps / 10_000.0)
        hi = mid * (1.0 + band_bps / 10_000.0)

        bid_depth = 0.0
        for px, qty in bids:
            pxf = float(px)
            if pxf < lo:
                break
            bid_depth += pxf * float(qty)

        ask_depth = 0.0
        for px, qty in asks:
            pxf = float(px)
            if pxf > hi:
                break
            ask_depth += pxf * float(qty)

        depth_usd = bid_depth + ask_depth
        return OrderBookStats(symbol=symbol, mid=mid, best_bid=best_bid, best_ask=best_ask, spread_bps=spread_bps, depth_usd=depth_usd)

    def passes_microstructure_filters(self, symbol: str) -> tuple[bool, dict[str, Any]]:
        cfg = self.config.filters
        stats = self.get_orderbook_stats(symbol)
        ok_spread = stats.spread_bps <= cfg.max_spread_bps
        ok_depth = stats.depth_usd >= cfg.min_orderbook_depth_usd
        info = {
            "spread_bps": stats.spread_bps,
            "depth_usd": stats.depth_usd,
            "mid": stats.mid,
            "best_bid": stats.best_bid,
            "best_ask": stats.best_ask,
        }
        return (ok_spread and ok_depth), info

    # -------- Instrument metadata (qty steps, tick sizes) --------
    def get_instrument_meta(self, symbol: str) -> InstrumentMeta:
        symbol = normalize_symbol(symbol)
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        infos = self.client.get_instruments_info(category=self.config.exchange.category, symbol=symbol)
        if not infos:
            raise ValueError(f"No instrument info for {symbol}")
        info = infos[0]

        lot = info.get("lotSizeFilter") or {}
        price_f = info.get("priceFilter") or {}

        qty_step = float(lot.get("qtyStep") or 0.0)
        min_qty = float(lot.get("minOrderQty") or 0.0)
        max_qty = lot.get("maxOrderQty")
        # Different Bybit instruments can expose min order value under different keys.
        # We store it as "min_notional" in USDT when available.
        min_notional_raw = (
            lot.get("minOrderAmt")
            or lot.get("minOrderValue")
            or lot.get("minNotionalValue")
            or lot.get("minNotional")
            or lot.get("minOrderValueUSDT")
        )
        try:
            min_notional = float(min_notional_raw) if min_notional_raw not in (None, "", "0") else None
        except Exception:
            min_notional = None
        tick_size = float(price_f.get("tickSize") or 0.0)
        meta = InstrumentMeta(
            symbol=symbol,
            qty_step=qty_step if qty_step > 0 else 1e-8,
            min_qty=min_qty if min_qty > 0 else 0.0,
            max_qty=float(max_qty) if max_qty not in (None, "", "0") else None,
            tick_size=tick_size if tick_size > 0 else 1e-8,
            min_notional=min_notional if (min_notional is not None and min_notional > 0) else None,
        )
        self._instrument_cache[symbol] = meta
        return meta

    # -------- Candles (any interval) with caching --------
    def get_candles(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        use_cache: bool = True,
        cache_write: bool = True,
    ) -> pd.DataFrame:
        """
        Returns klines for [start, end] (inclusive-ish), indexed by UTC timestamp.

        Intervals follow Bybit v5 kline intervals, e.g. "D", "60" (1H), "240" (4H).
        """
        symbol = normalize_symbol(symbol)
        interval = str(interval).strip()
        start = start.astimezone(UTC)
        end = end.astimezone(UTC)

        cached = load_cached_candles(self.cache_dir, symbol, interval) if use_cache else None
        if cached is not None and not cached.empty:
            have_start = cached.index.min()
            have_end = cached.index.max()
            if have_start <= start and have_end >= end:
                return cached.loc[start:end].copy()

        df = self._fetch_candles_remote(symbol=symbol, interval=interval, start=start, end=end)
        # If remote returned nothing (API gap/outage), fall back to cache to keep behavior stable.
        if (df is None) or df.empty:
            if cached is not None and not cached.empty:
                df = cached
        if cached is not None and not cached.empty:
            df = pd.concat([cached, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        if cache_write:
            save_cached_candles(self.cache_dir, symbol, interval, df)
        return df.loc[start:end].copy()

    def get_daily_candles(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        use_cache: bool = True,
        cache_write: bool = True,
    ) -> pd.DataFrame:
        # Backwards compatible wrapper.
        # Align daily windows to day starts to match prior behavior.
        start = start.astimezone(UTC)
        end = end.astimezone(UTC)
        start_day = datetime(start.year, start.month, start.day, tzinfo=UTC)
        end_day = datetime(end.year, end.month, end.day, tzinfo=UTC)
        return self.get_candles(symbol, "D", start_day, end_day, use_cache=use_cache, cache_write=cache_write)

    def _fetch_candles_remote(self, *, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
        # Bybit kline end is exclusive-ish; request with small buffer.
        start_ms = int(start.timestamp() * 1000)
        # Add 1 interval buffer for non-daily; daily handled similarly.
        end_ms = int((end + timedelta(days=1)).timestamp() * 1000) if interval == "D" else int((end.timestamp() * 1000) + 1)

        all_rows: list[list[str]] = []
        cursor_end = end_ms
        # Bybit returns newest-first, so we page backwards by end timestamp.
        for _ in range(40):
            chunk = self.client.get_kline(
                category=self.config.exchange.category,
                symbol=symbol,
                interval=interval,
                start_ms=start_ms,
                end_ms=cursor_end,
                limit=1000,
            )
            if not chunk:
                break
            all_rows.extend(chunk)
            oldest_ts = int(chunk[-1][0])
            if oldest_ts <= start_ms:
                break
            cursor_end = oldest_ts - 1

        # Fallback: some instruments can return empty when a very old start_ms is provided.
        # Fetch most recent candles and slice locally.
        if not all_rows:
            cursor_end = end_ms
            for _ in range(40):
                chunk = self.client.get_kline(
                    category=self.config.exchange.category,
                    symbol=symbol,
                    interval=interval,
                    start_ms=None,
                    end_ms=cursor_end,
                    limit=1000,
                )
                if not chunk:
                    break
                all_rows.extend(chunk)
                oldest_ts = int(chunk[-1][0])
                if oldest_ts <= start_ms:
                    break
                cursor_end = oldest_ts - 1

        if not all_rows:
            raise ValueError(f"No candles returned for {symbol} interval={interval}")

        rows = []
        for r in all_rows:
            try:
                ts = datetime.fromtimestamp(int(r[0]) / 1000, tz=UTC)
                rows.append(
                    {
                        "ts": ts,
                        "open": float(r[1]),
                        "high": float(r[2]),
                        "low": float(r[3]),
                        "close": float(r[4]),
                        "volume": float(r[5]),
                        "turnover": float(r[6]) if len(r) > 6 else np.nan,
                    }
                )
            except Exception:
                continue
        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
        df = df.set_index("ts").sort_index()
        return df

    def cache_universe_snapshot(self, *, symbols: list[str], meta: dict[str, Any]) -> None:
        out = {"ts_utc": datetime.now(tz=UTC).isoformat(), "symbols": symbols, "meta": meta}
        write_json(self.cache_dir / "universe_snapshot.json", out)
        logger.info("Saved universe snapshot: {}", self.cache_dir / "universe_snapshot.json")

    # -------- Funding rates (perps) --------
    def get_funding_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        use_cache: bool = True,
        cache_write: bool = True,
        force_mainnet: bool | None = None,
    ) -> pd.DataFrame:
        """
        Returns funding events indexed by UTC timestamp with column funding_rate.
        """
        symbol = normalize_symbol(symbol)
        start = start.astimezone(UTC)
        end = end.astimezone(UTC)

        cached = load_cached_funding(self.cache_dir, symbol) if use_cache else None
        if cached is not None and not cached.empty:
            have_start = cached.index.min()
            have_end = cached.index.max()
            if have_start <= start and have_end >= end:
                return cached.loc[start:end].copy()

        df = self._fetch_funding_remote(symbol=symbol, start=start, end=end, force_mainnet=force_mainnet)
        # If remote returned nothing, fall back to cache to avoid noisy warnings and keep behavior stable.
        if (df is None) or df.empty:
            if cached is not None and not cached.empty:
                df = cached
        if cached is not None and not cached.empty:
            # Avoid FutureWarning when concatenating with an empty frame by guarding above.
            df = pd.concat([cached, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        if cache_write:
            save_cached_funding(self.cache_dir, symbol, df)
        return df.loc[start:end].copy()

    def get_daily_funding_rate(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        force_mainnet: bool | None = None,
    ) -> pd.Series:
        """
        Aggregate funding events into a daily total funding rate per UTC day (sum of 8h rates).
        Returned index is UTC day start.
        """
        df = self.get_funding_history(symbol, start, end, force_mainnet=force_mainnet)
        if df.empty:
            return pd.Series(dtype=float)
        s = df["funding_rate"].astype(float)
        day = s.index.floor("D")
        daily = s.groupby(day).sum().sort_index()
        return daily

    def get_latest_daily_funding_rate(self, symbol: str, *, lookback_days: int = 3, force_mainnet: bool | None = None) -> float | None:
        now = datetime.now(tz=UTC)
        start = now - timedelta(days=max(5, lookback_days * 3))
        end = now + timedelta(days=1)
        daily = self.get_daily_funding_rate(symbol, start, end, force_mainnet=force_mainnet)
        if daily.empty:
            return None
        return float(daily.iloc[-1])

    def _fetch_funding_remote(self, *, symbol: str, start: datetime, end: datetime, force_mainnet: bool | None) -> pd.DataFrame:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        # Optionally override data source environment.
        client = self.client
        if force_mainnet is True and "api-testnet" in getattr(self.client, "_base_url", ""):
            # Create a temporary mainnet client without auth
            from src.data.bybit_client import BybitClient as _C

            client = _C(auth=None, testnet=False)
        elif force_mainnet is False and "api.bybit.com" in getattr(self.client, "_base_url", ""):
            from src.data.bybit_client import BybitClient as _C

            client = _C(auth=None, testnet=True)

        rows: list[dict[str, Any]] = []
        cursor_end = end_ms
        for _ in range(40):
            chunk = client.get_funding_history(category=self.config.exchange.category, symbol=symbol, start_ms=start_ms, end_ms=cursor_end, limit=200)
            if not chunk:
                break
            rows.extend(chunk)
            # Funding list is typically newest-first
            oldest = chunk[-1]
            try:
                oldest_ts = int(oldest.get("fundingRateTimestamp") or oldest.get("fundingRateTime") or 0)
            except Exception:
                oldest_ts = 0
            if oldest_ts <= start_ms or oldest_ts == 0:
                break
            cursor_end = oldest_ts - 1

        if client is not self.client:
            try:
                client.close()
            except Exception:
                pass

        if not rows:
            return pd.DataFrame(columns=["funding_rate"]).astype({"funding_rate": float})

        out_rows = []
        for r in rows:
            ts_ms = r.get("fundingRateTimestamp") or r.get("fundingRateTime") or r.get("fundingRateTimestamp")
            fr = r.get("fundingRate")
            try:
                ts = datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC)
                out_rows.append({"ts": ts, "funding_rate": float(fr)})
            except Exception:
                continue
        df = pd.DataFrame(out_rows).drop_duplicates(subset=["ts"]).sort_values("ts")
        if df.empty:
            return pd.DataFrame(columns=["funding_rate"]).astype({"funding_rate": float})
        return df.set_index("ts").sort_index()


