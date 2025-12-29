from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.backtest.metrics import compute_metrics
from src.config import BotConfig, load_config, load_yaml_config
from src.data.bybit_client import BybitClient
from src.data.market_data import MarketData
from src.strategy.xs_reversal import compute_targets_from_daily_candles


@dataclass(frozen=True)
class Candidate:
    lookback_days: int
    long_quantile: float
    short_quantile: float
    target_gross_leverage: float
    rebalance_time_utc: str
    vol_lookback_days: int


def _parse_date(d: str) -> datetime:
    dt = datetime.fromisoformat(d)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _default_opt_window_days() -> int:
    return 365


def _ensure_backtest_range(cfg: BotConfig) -> tuple[datetime, datetime]:
    # Optimizer should be robust on fresh installs; prefer a recent rolling window
    # instead of relying on whatever backtest start/end happen to be set to.
    # Allow override via env: BYBIT_OPT_WINDOW_DAYS (int).
    win_env = os.getenv("BYBIT_OPT_WINDOW_DAYS", "").strip()
    try:
        win_days = int(win_env) if win_env else _default_opt_window_days()
    except Exception:
        win_days = _default_opt_window_days()

    end = datetime.now(tz=UTC).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    start = end - timedelta(days=win_days)
    return start, end


def _buffer_days(cfg: BotConfig) -> int:
    rf = cfg.filters.regime_filter
    return int(max(cfg.sizing.vol_lookback_days + 10, cfg.signal.lookback_days + 5, rf.ema_slow + 20, 40))


def _calendar_from_any(candles: dict[str, pd.DataFrame], start: datetime, end: datetime) -> pd.DatetimeIndex:
    """
    Build a robust daily calendar.
    Avoid using full intersection across all symbols (too brittle when some symbols have sparse history).
    Prefer the longest available history among the downloaded candle sets.
    """
    non_empty = [df for df in candles.values() if df is not None and not df.empty]
    if not non_empty:
        return pd.DatetimeIndex([])
    df_best = max(non_empty, key=lambda d: len(d.index))
    cal = pd.DatetimeIndex(df_best.index)
    cal = cal[(cal >= start) & (cal <= end)].sort_values()
    return cal


def _simulate(
    *,
    cfg: BotConfig,
    candles: dict[str, pd.DataFrame],
    market_df: pd.DataFrame | None,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fee_bps = cfg.backtest.maker_fee_bps if (cfg.execution.order_type == "limit" and cfg.execution.post_only) else cfg.backtest.taker_fee_bps
    tc_bps = float(fee_bps) + float(cfg.backtest.slippage_bps)

    equity = float(cfg.backtest.initial_equity)
    prev_w: dict[str, float] = {}
    eq_curve: list[tuple[datetime, float]] = []
    dr: list[tuple[datetime, float]] = []
    to: list[tuple[datetime, float]] = []

    for i in range(0, len(calendar) - 1):
        asof = calendar[i]
        nxt = calendar[i + 1]

        day_candles = {s: df.loc[:asof] for s, df in candles.items() if asof in df.index}
        if len(day_candles) < 5:
            continue

        targets, _ = compute_targets_from_daily_candles(
            candles=day_candles,
            config=cfg,
            equity_usd=equity,
            asof=asof.to_pydatetime(),
            market_proxy_candles=market_df.loc[:asof] if market_df is not None else None,
        )
        w = targets.weights
        syms = set(prev_w) | set(w)
        turnover = float(sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in syms))
        traded_notional = turnover * equity
        tc_cost = traded_notional * (tc_bps / 10_000.0)

        port_ret = 0.0
        for s, ws in w.items():
            df = candles.get(s)
            if df is None or nxt not in df.index or asof not in df.index:
                continue
            px0 = float(df.loc[asof, "close"])
            px1 = float(df.loc[nxt, "close"])
            r = px1 / px0 - 1.0
            port_ret += float(ws) * float(r)

        equity = equity * (1.0 + port_ret) - tc_cost
        eq_curve.append((nxt.to_pydatetime(), equity))
        dr.append((nxt.to_pydatetime(), port_ret - (tc_cost / max(1e-12, equity))))
        to.append((asof.to_pydatetime(), turnover))
        prev_w = w

    eq = pd.Series({d: v for d, v in eq_curve}).sort_index()
    daily_ret = pd.Series({d: v for d, v in dr}).sort_index()
    daily_turn = pd.Series({d: v for d, v in to}).sort_index()
    return eq, daily_ret, daily_turn


def _level_to_budget(level: str) -> int:
    level = (level or "").strip().lower()
    if level == "quick":
        return 60
    if level == "deep":
        return 800
    return 250  # standard


def _random_candidates(
    *,
    n: int,
    default_time_utc: str,
    seed: int,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    lookbacks = np.array([1, 2, 3, 5], dtype=int)
    vol_lbs = np.array([14, 21, 30], dtype=int)
    # Typical quantiles for decile-ish reversal; allow some exploration
    q_low, q_high = 0.05, 0.25
    # Leverage exploration (gross)
    lev_low, lev_high = 0.5, 1.75

    out: list[Candidate] = []
    for _ in range(n):
        lb = int(rng.choice(lookbacks))
        vlb = int(rng.choice(vol_lbs))
        q = float(rng.uniform(q_low, q_high))
        # discretize to 0.01 to keep config readable
        q = round(q, 2)
        lev = float(rng.uniform(lev_low, lev_high))
        lev = round(lev, 2)
        out.append(
            Candidate(
                lookback_days=lb,
                long_quantile=q,
                short_quantile=q,
                target_gross_leverage=lev,
                rebalance_time_utc=str(default_time_utc),
                vol_lookback_days=vlb,
            )
        )
    # de-dup
    uniq = {(c.lookback_days, c.long_quantile, c.target_gross_leverage, c.vol_lookback_days): c for c in out}
    return list(uniq.values())


def _grid_candidates(*, default_time_utc: str) -> list[Candidate]:
    lookbacks = [1, 2, 3, 5]
    qs = [0.05, 0.10, 0.15, 0.20]
    levs = [0.5, 0.75, 1.0, 1.25, 1.5]
    vol_lbs = [14, 30]
    out: list[Candidate] = []
    for lb, q, lev, vlb in itertools.product(lookbacks, qs, levs, vol_lbs):
        out.append(
            Candidate(
                lookback_days=int(lb),
                long_quantile=float(q),
                short_quantile=float(q),
                target_gross_leverage=float(lev),
                rebalance_time_utc=str(default_time_utc),
                vol_lookback_days=int(vlb),
            )
        )
    return out


def _prepare_close_matrix(
    candles: dict[str, pd.DataFrame],
    *,
    start: datetime,
    end: datetime,
    min_history_days: int,
    max_symbols: int = 60,
) -> pd.DataFrame:
    """
    Build a close price matrix (date x symbol) using symbols with enough history,
    preferring the longest-history symbols to improve optimization stability.
    """
    rows: list[tuple[str, pd.Series]] = []
    for sym, df in candles.items():
        if df is None or df.empty:
            continue
        sub = df.loc[(df.index >= start) & (df.index <= end), "close"].dropna().astype(float)
        if len(sub) < min_history_days:
            continue
        rows.append((sym, sub))
    # Prefer longest history
    rows.sort(key=lambda x: len(x[1]), reverse=True)
    rows = rows[:max_symbols]
    if not rows:
        return pd.DataFrame()
    # Align on union of dates, then later we will drop days with too many NaNs.
    close = pd.concat({sym: s for sym, s in rows}, axis=1).sort_index()
    return close.loc[(close.index >= start) & (close.index <= end)]


def _simulate_candidate_vectorized(
    *,
    close: pd.DataFrame,
    lookback_days: int,
    vol_lookback_days: int,
    q: float,
    gross_leverage: float,
    maker_fee_bps: float,
    taker_fee_bps: float,
    slippage_bps: float,
    use_maker: bool,
    initial_equity: float,
    max_dd_limit: float,
    max_turnover: float,
    min_symbols: int = 10,
) -> dict[str, float] | None:
    """
    Fast daily rebalance simulation for candidate selection:
    - cross-sectional rank on lookback return
    - inverse-vol weights
    - dollar-neutral LS (long losers, short winners)
    - turnover/fees modeled in weight space
    Returns metrics dict or None if not feasible.
    """
    if close.empty:
        return None
    px = close.copy()
    # Require at least min_symbols available per day
    valid_counts = px.notna().sum(axis=1)
    px = px.loc[valid_counts >= min_symbols]
    if len(px) < (vol_lookback_days + lookback_days + 20):
        return None

    # daily returns and lookback returns
    r1 = px.pct_change()
    r_lb = px / px.shift(lookback_days) - 1.0
    vol = r1.rolling(vol_lookback_days, min_periods=vol_lookback_days).std(ddof=0)

    # Weights matrix (T x N)
    weights = pd.DataFrame(0.0, index=px.index, columns=px.columns)

    q = float(q)
    for t in range(len(px.index)):
        dt = px.index[t]
        ret_row = r_lb.loc[dt]
        vol_row = vol.loc[dt]
        # eligible symbols
        mask = ret_row.notna() & vol_row.notna() & (vol_row > 0)
        if mask.sum() < min_symbols:
            continue
        rets = ret_row[mask].astype(float)
        vols = vol_row[mask].astype(float)

        # thresholds
        lo_thr = np.nanquantile(rets.values, q)
        hi_thr = np.nanquantile(rets.values, 1.0 - q)
        longs = rets[rets <= lo_thr].index
        shorts = rets[rets >= hi_thr].index
        if len(longs) == 0 or len(shorts) == 0:
            continue

        invv_l = (1.0 / vols.loc[longs]).replace([np.inf, -np.inf], np.nan).dropna()
        invv_s = (1.0 / vols.loc[shorts]).replace([np.inf, -np.inf], np.nan).dropna()
        if invv_l.empty or invv_s.empty:
            continue

        w_l = invv_l / invv_l.abs().sum()
        w_s = invv_s / invv_s.abs().sum()
        w_l = w_l * (gross_leverage / 2.0)
        w_s = w_s * (gross_leverage / 2.0)
        w = pd.Series(0.0, index=px.columns)
        w.loc[w_l.index] = w_l.values
        w.loc[w_s.index] = -w_s.values
        weights.loc[dt] = w

    weights = weights.dropna(how="all")
    if weights.empty or len(weights) < 30:
        return None

    # Turnover / costs
    dw = weights.diff().abs().sum(axis=1).fillna(0.0)
    avg_turnover = float(dw.mean())
    if avg_turnover > float(max_turnover):
        return None

    fee_bps = float(maker_fee_bps) if use_maker else float(taker_fee_bps)
    tc_bps = fee_bps + float(slippage_bps)

    # Portfolio daily return: sum(w_{t} * r1_{t+1})
    aligned = r1.loc[weights.index].shift(-1)
    port_ret = (weights * aligned).sum(axis=1).dropna()
    if port_ret.empty:
        return None

    # apply transaction costs on traded notional in equity terms
    tc = dw.loc[port_ret.index] * (tc_bps / 10_000.0)
    net_ret = port_ret - tc

    equity = initial_equity * (1.0 + net_ret).cumprod()
    if float(equity.min()) <= 0.0:
        return None

    # drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    if abs(max_dd) > float(max_dd_limit):
        return None

    # sharpe
    if net_ret.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(365) * net_ret.mean() / net_ret.std(ddof=0))

    # CAGR on window length
    years = max(1e-9, (len(equity) - 1) / 365.0)
    cagr = float((float(equity.iloc[-1]) / float(equity.iloc[0])) ** (1.0 / years) - 1.0)
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "avg_daily_turnover": avg_turnover}


def optimize_config(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    level: str = "standard",
    candidates: int | None = None,
    method: Literal["random", "grid"] = "random",
    seed: int = 42,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Optimize a small parameter grid and write best params back to config.yaml.
    Returns summary dict.
    """
    cfg = load_config(config_path)
    start, end = _ensure_backtest_range(cfg)
    buf = _buffer_days(cfg)
    fetch_start = start - timedelta(days=buf)
    fetch_end = end + timedelta(days=2)

    # Optimization should generally use MAINNET historical data even if you trade on testnet.
    # Some testnet instruments have sparse/absent history, which makes optimization unreliable.
    opt_testnet_env = os.getenv("BYBIT_OPT_TESTNET", "").strip().lower()
    if opt_testnet_env in ("1", "true", "yes", "y"):
        data_testnet = True
    elif opt_testnet_env in ("0", "false", "no", "n"):
        data_testnet = False
    else:
        data_testnet = False
    client = BybitClient(auth=None, testnet=data_testnet)
    if data_testnet != cfg.exchange.testnet:
        logger.warning("Optimizer using testnet={} for market data (trading testnet={})", data_testnet, cfg.exchange.testnet)
    md = MarketData(client=client, config=cfg, cache_dir=cfg.backtest.cache_dir)
    try:
        symbols = md.get_liquidity_ranked_symbols()
        candles: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                candles[s] = md.get_daily_candles(s, fetch_start, fetch_end, use_cache=True, cache_write=True)
            except Exception as e:
                logger.warning("Skipping {}: {}", s, e)

        market_df = None
        if cfg.filters.regime_filter.enabled and cfg.filters.regime_filter.use_market_regime:
            proxy = cfg.filters.regime_filter.market_proxy_symbol
            try:
                market_df = md.get_daily_candles(proxy, fetch_start, fetch_end, use_cache=True, cache_write=True)
            except Exception as e:
                logger.warning("Market proxy unavailable: {}", e)

        if market_df is not None and not market_df.empty:
            cal = market_df.index
            cal = cal[(cal >= start) & (cal <= end)].sort_values()
        else:
            cal = _calendar_from_any(candles, start, end)
        if len(cal) < 30:
            raise ValueError("Not enough aligned daily bars for optimization window.")

        # Prepare data matrix once for fast candidate evaluation
        # Use candidate-independent history requirement based on max LB
        min_hist = int(max(80, 2 * max(cfg.sizing.vol_lookback_days, 30)))
        close = _prepare_close_matrix(candles, start=start, end=end, min_history_days=min_hist, max_symbols=60)

        best = None
        best_key: tuple[float, float, float, float] | None = None
        rows: list[dict[str, Any]] = []

        budget = int(candidates) if candidates is not None else _level_to_budget(level)
        if method == "grid":
            cand_list = _grid_candidates(default_time_utc=cfg.rebalance.time_utc)
        else:
            cand_list = _random_candidates(n=budget, default_time_utc=cfg.rebalance.time_utc, seed=seed)
        # If grid, respect budget if provided
        if candidates is not None:
            cand_list = cand_list[: int(candidates)]
        # If random produced fewer uniques than budget, that's OK.
        candidates_list = cand_list

        # Progress bar + ETA
        progress_ctx = (
            Progress(
                SpinnerColumn(),
                TextColumn("[bold]optimize[/bold]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("best_sharpe={task.fields[best_sharpe]}"),
                TextColumn("best_dd={task.fields[best_dd]}"),
                TextColumn("best_cagr={task.fields[best_cagr]}"),
                transient=False,
            )
            if show_progress
            else None
        )

        def fmt_or_na(x: float | None, fmt: str) -> str:
            if x is None or not np.isfinite(x):
                return "n/a"
            return format(float(x), fmt)

        if progress_ctx is None:
            task_id = None
        else:
            progress_ctx.start()
            task_id = progress_ctx.add_task(
                "optimize",
                total=len(candidates_list),
                best_sharpe="n/a",
                best_dd="n/a",
                best_cagr="n/a",
            )

        try:
            for cand in candidates_list:
                # Evaluate candidate quickly (vectorized)
                use_maker = bool(cfg.execution.order_type == "limit" and cfg.execution.post_only)
                m = _simulate_candidate_vectorized(
                    close=close,
                    lookback_days=int(cand.lookback_days),
                    vol_lookback_days=int(cand.vol_lookback_days),
                    q=float(cand.long_quantile),
                    gross_leverage=float(cand.target_gross_leverage),
                    maker_fee_bps=float(cfg.backtest.maker_fee_bps),
                    taker_fee_bps=float(cfg.backtest.taker_fee_bps),
                    slippage_bps=float(cfg.backtest.slippage_bps),
                    use_maker=use_maker,
                    initial_equity=float(cfg.backtest.initial_equity),
                    max_dd_limit=float(cfg.risk.max_drawdown_pct) / 100.0,
                    max_turnover=float(cfg.risk.max_turnover),
                )

                # Record candidate metrics even if rejected (for debugging)
                row = {
                    "candidate": cand.__dict__,
                    "sharpe": (float(m["sharpe"]) if m is not None else float("nan")),
                    "cagr": (float(m["cagr"]) if m is not None else float("nan")),
                    "max_drawdown": (float(m["max_drawdown"]) if m is not None else float("nan")),
                    "avg_daily_turnover": (float(m["avg_daily_turnover"]) if m is not None else float("nan")),
                    "rejected": False,
                    "reject_reason": None,
                }

                if m is None:
                    row["rejected"] = True
                    row["reject_reason"] = "infeasible_or_insufficient_data"
                    rows.append(row)
                    if progress_ctx is not None and task_id is not None:
                        progress_ctx.advance(task_id, 1)
                    continue

                # Objective (lexicographic):
                #   1) maximize Sharpe
                #   2) maximize CAGR
                #   3) minimize drawdown magnitude
                #   4) minimize turnover
                # We'll store as a key where smaller is better.
                key = (-float(row["sharpe"]), -float(row["cagr"]), abs(float(row["max_drawdown"])), float(row["avg_daily_turnover"]))
                # Also keep a scalar score for reporting/debugging
                score = float(row["sharpe"]) + 0.25 * float(row["cagr"]) - 0.05 * float(row["avg_daily_turnover"])
                row["score"] = score
                rows.append(row)

                if best_key is None or key < best_key:
                    best_key = key
                    best = (cand, row)
                    if progress_ctx is not None and task_id is not None:
                        progress_ctx.update(
                            task_id,
                            best_sharpe=fmt_or_na(float(row.get("sharpe", float("nan"))), ".3f"),
                            best_dd=fmt_or_na(float(row.get("max_drawdown", float("nan"))), ".2%"),
                            best_cagr=fmt_or_na(float(row.get("cagr", float("nan"))), ".2%"),
                        )

                if progress_ctx is not None and task_id is not None:
                    progress_ctx.advance(task_id, 1)
        finally:
            if progress_ctx is not None:
                progress_ctx.stop()

        if best is None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "optimization_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            logger.error(
                "Optimization found no feasible candidates. Leaving config unchanged. Results: {}",
                (out / "optimization_results.json").resolve(),
            )
            return {
                "status": "no_feasible_candidate",
                "output_dir": str(out.resolve()),
                "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                "universe_size": len(symbols),
                "candidates": len(rows),
            }

        best_cand, best_row = best

        # Helpful transparency: show top-by-sharpe and top-by-objective in logs.
        try:
            df = pd.DataFrame([r for r in rows if not r.get("rejected")])
            if not df.empty:
                top_sh = df.sort_values("sharpe", ascending=False).head(5)[
                    ["sharpe", "cagr", "max_drawdown", "avg_daily_turnover", "candidate"]
                ]
                top_sc = df.sort_values("score", ascending=False).head(5)[
                    ["score", "sharpe", "cagr", "max_drawdown", "avg_daily_turnover", "candidate"]
                ]
                logger.info("Top 5 by Sharpe:\n{}", top_sh.to_string(index=False))
                logger.info("Top 5 by score:\n{}", top_sc.to_string(index=False))
        except Exception:
            pass

        # Guardrail: don't write params that are statistically/financially "worse than nothing".
        # Default: reject negative Sharpe (BYBIT_OPT_MIN_SHARPE=0.0).
        min_sharpe_env = os.getenv("BYBIT_OPT_MIN_SHARPE", "").strip()
        try:
            min_sharpe = float(min_sharpe_env) if min_sharpe_env else 0.0
        except Exception:
            min_sharpe = 0.0
        best_sharpe = float(best_row.get("sharpe", 0.0))
        if not np.isfinite(best_sharpe) or best_sharpe < min_sharpe:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "optimization_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
            (out / "best.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")
            logger.warning(
                "Optimizer best Sharpe {:.3f} < min {:.3f}; leaving config unchanged. Best: {}",
                best_sharpe,
                min_sharpe,
                (out / "best.json").resolve(),
            )
            return {
                "status": "rejected_by_threshold",
                "min_sharpe": min_sharpe,
                "best": best_row,
                "output_dir": str(out.resolve()),
                "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
                "universe_size": len(symbols),
                "candidates": len(rows),
            }

        # Patch config.yaml (deep merge)
        raw = load_yaml_config(config_path)
        raw.setdefault("signal", {})
        raw.setdefault("rebalance", {})
        raw.setdefault("sizing", {})
        raw.setdefault("sizing", {})

        raw["signal"]["lookback_days"] = int(best_cand.lookback_days)
        raw["signal"]["long_quantile"] = float(best_cand.long_quantile)
        raw["signal"]["short_quantile"] = float(best_cand.short_quantile)
        raw["rebalance"]["time_utc"] = str(best_cand.rebalance_time_utc)
        raw["sizing"]["target_gross_leverage"] = float(best_cand.target_gross_leverage)
        raw.setdefault("sizing", {})
        raw["sizing"]["vol_lookback_days"] = int(best_cand.vol_lookback_days)

        import yaml

        Path(config_path).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "optimization_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        (out / "best.json").write_text(json.dumps(best_row, indent=2, sort_keys=True), encoding="utf-8")

        logger.info(
            "Optimization complete. Best sharpe={:.3f} cagr={:.2%} maxDD={:.2%} turnover={:.3f} params={}",
            float(best_row.get("sharpe", 0.0)),
            float(best_row.get("cagr", 0.0)),
            float(best_row.get("max_drawdown", 0.0)),
            float(best_row.get("avg_daily_turnover", 0.0)),
            best_cand.__dict__,
        )
        return {
            "best": best_row,
            "output_dir": str(out.resolve()),
            "window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
            "universe_size": len(symbols),
            "candidates": len(rows),
            "evaluated": len(candidates_list),
        }
    finally:
        client.close()


