from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.backtest.backtester import BacktestResult, run_backtest_with_target_transform
from src.config import BotConfig
from src.data.market_data import MarketData
from src.strategy.portfolio import PortfolioTargets, weights_to_notionals


TargetTransform = Callable[..., PortfolioTargets]


@dataclass(frozen=True)
class ResearchVariant:
    name: str
    description: str
    cfg: BotConfig
    target_transform: TargetTransform | None = None
    tags: tuple[str, ...] = ()


def describe_strategy_components(cfg: BotConfig) -> dict[str, Any]:
    rf = cfg.filters.regime_filter
    return {
        "reversal_signal": "Cross-sectional short-term reversal: long recent losers, short recent winners.",
        "lookback_period_days": int(cfg.signal.lookback_days),
        "long_quantile": float(cfg.signal.long_quantile),
        "short_quantile": float(cfg.signal.short_quantile),
        "confirmation_logic": {
            "enabled": (cfg.signal.ret_1d_long_max is not None or cfg.signal.ret_1d_short_min is not None),
            "long_requires_1d_return_lte": cfg.signal.ret_1d_long_max,
            "short_requires_1d_return_gte": cfg.signal.ret_1d_short_min,
        },
        "regime_filter": {
            "enabled": bool(rf.enabled),
            "use_market_regime": bool(rf.use_market_regime),
            "market_proxy_symbol": str(rf.market_proxy_symbol),
            "market_action": str(rf.action),
            "market_adx_threshold": float(rf.market_adx_threshold),
            "symbol_adx_threshold": float(rf.symbol_adx_threshold),
            "ema_fast": int(rf.ema_fast),
            "ema_slow": int(rf.ema_slow),
            "scale_factor": float(rf.scale_factor),
        },
        "market_momentum_switch": bool(rf.enabled and rf.use_market_regime and str(rf.action) == "switch_to_momentum"),
        "funding_filter": {
            "enabled": bool(cfg.funding.filter.enabled),
            "max_abs_daily_funding_rate": float(cfg.funding.filter.max_abs_daily_funding_rate),
        },
        "liquidity_filter": {
            "top_n_by_volume": int(cfg.universe.top_n_by_volume),
            "min_24h_quote_volume": float(cfg.universe.min_24h_quote_volume),
            "min_open_interest": float(cfg.universe.min_open_interest),
            "min_history_days": int(cfg.universe.min_history_days),
            "max_spread_bps": float(cfg.filters.max_spread_bps),
            "min_orderbook_depth_usd": float(cfg.filters.min_orderbook_depth_usd),
        },
        "volatility_risk_weighting": {
            "inverse_vol": True,
            "vol_lookback_days": int(cfg.sizing.vol_lookback_days),
            "target_gross_leverage": float(cfg.sizing.target_gross_leverage),
            "max_weight_per_symbol": float(cfg.sizing.max_leverage_per_symbol),
            "max_notional_per_symbol": float(cfg.sizing.max_notional_per_symbol),
            "min_notional_per_symbol": float(cfg.sizing.min_notional_per_symbol),
        },
        "turnover_dampening": {
            "rebalance_fraction": float(cfg.rebalance.rebalance_fraction),
            "min_weight_change_bps": float(cfg.rebalance.min_weight_change_bps),
        },
        "position_side_mode": "long_only" if bool(cfg.signal.long_only) else "long_short",
        "backtest_execution_scenario": str(getattr(cfg.backtest, "execution_scenario", "mixed")),
    }


def _clone_with_updates(cfg: BotConfig, updater: Callable[[BotConfig], None]) -> BotConfig:
    clone = cfg.model_copy(deep=True)
    updater(clone)
    return clone


def _normalize_side_only(weights: dict[str, float], *, side: str, target_gross: float) -> dict[str, float]:
    if side not in {"long", "short"}:
        raise ValueError(f"Unsupported side mode: {side}")
    filtered = {k: float(v) for k, v in weights.items() if (float(v) > 0 if side == "long" else float(v) < 0)}
    gross = sum(abs(v) for v in filtered.values())
    if gross <= 0:
        return {}
    scale = float(target_gross) / gross
    return {k: float(v) * scale for k, v in filtered.items()}


def _make_side_transform(side: str) -> TargetTransform:
    def _transform(*, weights: dict[str, float], notionals: dict[str, float], equity_usd: float, cfg: BotConfig, asof: datetime, meta: dict[str, Any]) -> PortfolioTargets:
        w = _normalize_side_only(weights, side=side, target_gross=float(cfg.sizing.target_gross_leverage))
        notionals_usd = weights_to_notionals(
            w,
            equity_usd=float(equity_usd),
            max_notional_per_symbol=float(cfg.sizing.max_notional_per_symbol),
            min_notional_per_symbol=float(cfg.sizing.min_notional_per_symbol),
        )
        return PortfolioTargets(weights={k: w[k] for k in notionals_usd.keys()}, notionals_usd=notionals_usd, meta=meta)

    return _transform


def build_phase3_variants(cfg: BotConfig) -> list[ResearchVariant]:
    variants: list[ResearchVariant] = []
    variants.append(ResearchVariant(name="baseline", description="Current production baseline, unchanged.", cfg=cfg.model_copy(deep=True), tags=("baseline",)))
    variants.append(
        ResearchVariant(
            name="pure_reversal_only",
            description="Disable confirmation, regime filter, and funding filter.",
            cfg=_clone_with_updates(
                cfg,
                lambda c: (
                    setattr(c.signal, "ret_1d_long_max", None),
                    setattr(c.signal, "ret_1d_short_min", None),
                    setattr(c.filters.regime_filter, "enabled", False),
                    setattr(c.funding.filter, "enabled", False),
                ),
            ),
            tags=("ablation", "pure_reversal"),
        )
    )
    variants.append(
        ResearchVariant(
            name="no_1d_confirmation",
            description="Disable the 1-day confirmation filter.",
            cfg=_clone_with_updates(
                cfg,
                lambda c: (
                    setattr(c.signal, "ret_1d_long_max", None),
                    setattr(c.signal, "ret_1d_short_min", None),
                ),
            ),
            tags=("ablation", "confirmation"),
        )
    )
    variants.append(
        ResearchVariant(
            name="no_regime_scaling",
            description="Disable regime filter and regime-driven scaling/switching.",
            cfg=_clone_with_updates(cfg, lambda c: setattr(c.filters.regime_filter, "enabled", False)),
            tags=("ablation", "regime"),
        )
    )
    variants.append(
        ResearchVariant(
            name="no_market_momentum_switch",
            description="Keep regime filter but disable switch-to-momentum behavior.",
            cfg=_clone_with_updates(cfg, lambda c: setattr(c.filters.regime_filter, "action", "scale_down")),
            tags=("ablation", "regime", "momentum_switch"),
        )
    )
    variants.append(
        ResearchVariant(
            name="no_funding_filter",
            description="Disable the funding-rate filter.",
            cfg=_clone_with_updates(cfg, lambda c: setattr(c.funding.filter, "enabled", False)),
            tags=("ablation", "funding"),
        )
    )
    variants.append(
        ResearchVariant(
            name="no_turnover_dampening",
            description="Disable partial rebalance and minimum weight change threshold.",
            cfg=_clone_with_updates(
                cfg,
                lambda c: (
                    setattr(c.rebalance, "rebalance_fraction", 1.0),
                    setattr(c.rebalance, "min_weight_change_bps", 0.0),
                ),
            ),
            tags=("ablation", "turnover"),
        )
    )
    variants.append(
        ResearchVariant(
            name="long_only",
            description="Research-only long-only version.",
            cfg=_clone_with_updates(cfg, lambda c: setattr(c.signal, "long_only", True)),
            tags=("side", "long_only"),
        )
    )
    variants.append(
        ResearchVariant(
            name="short_only",
            description="Research-only short-only version.",
            cfg=cfg.model_copy(deep=True),
            target_transform=_make_side_transform("short"),
            tags=("side", "short_only"),
        )
    )
    for scenario in ("optimistic_maker", "mixed", "conservative_taker"):
        variants.append(
            ResearchVariant(
                name=f"execution_{scenario}",
                description=f"Baseline strategy under {scenario} execution-cost assumptions.",
                cfg=_clone_with_updates(cfg, lambda c, s=scenario: setattr(c.backtest, "execution_scenario", s)),
                tags=("execution", scenario),
            )
        )
    return variants


def _variant_by_name(variants: list[ResearchVariant], name: str) -> ResearchVariant:
    for variant in variants:
        if variant.name == name:
            return variant
    raise ValueError(f"Unknown research variant: {name}")


def _worst_month(daily_returns: pd.Series) -> tuple[str | None, float]:
    if daily_returns.empty:
        return None, 0.0
    monthly = daily_returns.groupby(pd.Grouper(freq="ME")).apply(lambda s: float((1.0 + s).prod() - 1.0))
    if monthly.empty:
        return None, 0.0
    idx = monthly.idxmin()
    return idx.strftime("%Y-%m"), float(monthly.loc[idx])


def _worst_drawdown_period(equity: pd.Series) -> tuple[str | None, str | None, float]:
    if equity.empty:
        return None, None, 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    trough_ts = dd.idxmin()
    trough_val = float(dd.loc[trough_ts])
    peak_ts = equity.loc[:trough_ts].idxmax()
    return peak_ts.strftime("%Y-%m-%d"), trough_ts.strftime("%Y-%m-%d"), trough_val


def _top_symbol_concentration(symbol_contrib: dict[str, float]) -> float:
    vals = [abs(float(v)) for v in symbol_contrib.values()]
    total = sum(vals)
    if total <= 0:
        return 0.0
    return max(vals) / total


def summarize_backtest_result(variant: ResearchVariant, result: BacktestResult) -> dict[str, Any]:
    meta = dict(result.meta)
    cost_summary = dict(meta.get("cost_summary", {}))
    top_symbol_contrib = dict(meta.get("top_symbol_contributions_usd", {}))
    worst_month_label, worst_month_return = _worst_month(result.daily_returns)
    dd_start, dd_end, dd_val = _worst_drawdown_period(result.equity)
    return {
        "variant": variant.name,
        "description": variant.description,
        "execution_scenario": meta.get("execution_scenario"),
        "net_return": float(meta.get("net_return_total", 0.0)),
        "gross_return": float(meta.get("gross_return_total", 0.0)),
        "max_drawdown": float(result.metrics.max_drawdown),
        "sharpe": float(result.metrics.sharpe),
        "sortino": float(result.metrics.sortino),
        "calmar": float(result.metrics.calmar),
        "profit_factor": float(result.metrics.profit_factor),
        "turnover": float(result.metrics.avg_daily_turnover),
        "fee_slippage_drag_usd": float(cost_summary.get("trading_cost_total_usd", 0.0)),
        "funding_drag_usd": float(cost_summary.get("funding_pnl_total_usd", 0.0)),
        "borrow_drag_usd": float(cost_summary.get("borrow_cost_total_usd", 0.0)),
        "rebalance_events": int(meta.get("rebalance_count", 0)),
        "trade_days": int(meta.get("trade_days", 0)),
        "long_contribution_usd": float(meta.get("long_pnl_total_usd", 0.0)),
        "short_contribution_usd": float(meta.get("short_pnl_total_usd", 0.0)),
        "top_symbol_concentration": float(_top_symbol_concentration(top_symbol_contrib)),
        "worst_month": worst_month_label,
        "worst_month_return": float(worst_month_return),
        "worst_drawdown_start": dd_start,
        "worst_drawdown_end": dd_end,
        "worst_drawdown_value": float(dd_val),
        "warnings": " | ".join(meta.get("warnings", [])),
    }


def _run_variant(variant: ResearchVariant, *, md: MarketData, output_dir: Path) -> BacktestResult:
    return run_backtest_with_target_transform(
        cfg=variant.cfg,
        md=md,
        outputs_dir=output_dir,
        target_transform=variant.target_transform,
    )


def _variant_output_dir(base: Path, name: str) -> Path:
    out = base / "variants" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _regime_masks(btc_returns: pd.Series, btc_vol: pd.Series, funding_proxy: pd.Series | None = None) -> dict[str, pd.Index]:
    idx = btc_returns.index
    vol_med = float(btc_vol.median()) if not btc_vol.dropna().empty else 0.0
    regimes: dict[str, pd.Index] = {
        "bull": idx[btc_returns > 0],
        "bear": idx[btc_returns < 0],
        "sideways": idx[btc_returns.abs() <= btc_returns.abs().median()],
        "high_vol": idx[btc_vol >= vol_med],
        "low_vol": idx[btc_vol < vol_med],
        "btc_uptrend": idx[btc_returns > 0],
        "btc_downtrend": idx[btc_returns < 0],
    }
    if funding_proxy is not None and not funding_proxy.dropna().empty:
        thresh = float(funding_proxy.abs().median())
        regimes["high_funding"] = funding_proxy.index[funding_proxy.abs() >= thresh]
        regimes["normal_funding"] = funding_proxy.index[funding_proxy.abs() < thresh]
    return regimes


def _subset_metrics(result: BacktestResult, dates: pd.Index) -> dict[str, Any]:
    dr = result.daily_returns[result.daily_returns.index.isin(dates)]
    if dr.empty:
        return {"days": 0, "net_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    eq = result.equity[result.equity.index.isin(dates)]
    if eq.empty:
        eq = result.equity.loc[: dr.index.max()]
    net_return = float((1.0 + dr).prod() - 1.0)
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min()) if not eq.empty else 0.0
    std = float(dr.std(ddof=0)) if len(dr) else 0.0
    sharpe = float((dr.mean() / std) * (365.0 ** 0.5)) if std > 0 else 0.0
    return {"days": int(len(dr)), "net_return": net_return, "sharpe": sharpe, "max_drawdown": dd}


def _run_regime_breakdown(*, cfg: BotConfig, md: MarketData, results: dict[str, BacktestResult]) -> pd.DataFrame:
    baseline = results.get("baseline")
    if baseline is None or baseline.daily_returns.empty:
        return pd.DataFrame()

    proxy = str(cfg.filters.regime_filter.market_proxy_symbol)
    proxy_df = md.get_daily_candles(proxy, baseline.daily_returns.index.min(), baseline.daily_returns.index.max(), use_cache=True, cache_write=False)
    close = proxy_df["close"].astype(float).reindex(baseline.daily_returns.index).dropna()
    btc_returns = close.pct_change().dropna()
    btc_vol = btc_returns.rolling(14, min_periods=5).std(ddof=0).bfill()

    funding_proxy = None
    try:
        funding_proxy = md.get_daily_funding_rate(proxy, baseline.daily_returns.index.min(), baseline.daily_returns.index.max(), force_mainnet=True).reindex(btc_returns.index).fillna(0.0)
    except Exception:
        funding_proxy = None

    masks = _regime_masks(btc_returns, btc_vol, funding_proxy)
    rows: list[dict[str, Any]] = []
    for variant_name, result in results.items():
        for regime_name, dates in masks.items():
            stats = _subset_metrics(result, dates)
            rows.append({"variant": variant_name, "regime": regime_name, **stats})
    return pd.DataFrame(rows)


def _sensitivity_grid(cfg: BotConfig) -> list[tuple[str, dict[str, Any]]]:
    q_long = float(cfg.signal.long_quantile)
    q_short = float(cfg.signal.short_quantile)
    frac = float(cfg.rebalance.rebalance_fraction)
    min_bps = float(cfg.rebalance.min_weight_change_bps)
    top_n = int(cfg.universe.top_n_by_volume)
    max_funding = float(cfg.funding.filter.max_abs_daily_funding_rate)
    market_adx = float(cfg.filters.regime_filter.market_adx_threshold)
    symbol_adx = float(cfg.filters.regime_filter.symbol_adx_threshold)
    return [
        ("lookback_days", {"signal": {"lookback_days": v}}) for v in sorted({1, 2, 3, 4, 5, int(cfg.signal.lookback_days)})
    ] + [
        ("long_quantile", {"signal": {"long_quantile": max(0.05, round(v, 4))}}) for v in sorted({q_long * 0.75, q_long, min(0.45, q_long * 1.25)})
    ] + [
        ("short_quantile", {"signal": {"short_quantile": max(0.05, round(v, 4))}}) for v in sorted({q_short * 0.75, q_short, min(0.45, q_short * 1.25)})
    ] + [
        ("rebalance_fraction", {"rebalance": {"rebalance_fraction": max(0.1, min(1.0, round(v, 4)))}}) for v in sorted({frac * 0.75, frac, min(1.0, frac * 1.25 if frac > 0 else 1.0)})
    ] + [
        ("min_weight_change_bps", {"rebalance": {"min_weight_change_bps": max(0.0, round(v, 4))}}) for v in sorted({max(0.0, min_bps * 0.5), min_bps, min_bps * 1.5})
    ] + [
        ("top_n_by_volume", {"universe": {"top_n_by_volume": max(10, int(v))}}) for v in sorted({max(10, int(top_n * 0.75)), top_n, int(top_n * 1.25)})
    ] + [
        ("funding_threshold", {"funding": {"filter": {"max_abs_daily_funding_rate": max(0.0005, round(v, 6))}}}) for v in sorted({max_funding * 0.75, max_funding, max_funding * 1.25})
    ] + [
        ("market_adx_threshold", {"filters": {"regime_filter": {"market_adx_threshold": max(10.0, round(v, 4))}}}) for v in sorted({market_adx - 5.0, market_adx, market_adx + 5.0})
    ] + [
        ("symbol_adx_threshold", {"filters": {"regime_filter": {"symbol_adx_threshold": max(10.0, round(v, 4))}}}) for v in sorted({symbol_adx - 5.0, symbol_adx, symbol_adx + 5.0})
    ]


def _deep_apply_updates(model: BotConfig, updates: dict[str, Any]) -> BotConfig:
    data = model.model_dump(mode="python")

    def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _merge(dst[k], v)
            else:
                dst[k] = v

    _merge(data, updates)
    return BotConfig.model_validate(data)


def run_phase3_review(
    *,
    cfg: BotConfig,
    md: MarketData,
    output_dir: str | Path,
    variant_names: list[str] | None = None,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = build_phase3_variants(cfg)
    if variant_names:
        variants = [_variant_by_name(variants, n) for n in variant_names]

    strategy_map = describe_strategy_components(cfg)
    results: dict[str, BacktestResult] = {}
    ablation_rows: list[dict[str, Any]] = []
    for variant in variants:
        res = _run_variant(variant, md=md, output_dir=_variant_output_dir(out_dir, variant.name))
        results[variant.name] = res
        ablation_rows.append(summarize_backtest_result(variant, res))

    ablation_df = pd.DataFrame(ablation_rows).sort_values(["variant"])
    ablation_path = out_dir / "phase3_ablation_results.csv"
    ablation_df.to_csv(ablation_path, index=False)

    sensitivity_rows: list[dict[str, Any]] = []
    for param_name, updates in _sensitivity_grid(cfg):
        sens_cfg = _deep_apply_updates(cfg, updates)
        sens_variant = ResearchVariant(name=f"sensitivity_{param_name}", description=param_name, cfg=sens_cfg)
        res = _run_variant(sens_variant, md=md, output_dir=_variant_output_dir(out_dir, f"sensitivity_{param_name}_{len(sensitivity_rows)}"))
        row = summarize_backtest_result(sens_variant, res)
        row["parameter"] = param_name
        row["override"] = str(updates)
        sensitivity_rows.append(row)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_path = out_dir / "phase3_sensitivity_results.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)

    regime_df = _run_regime_breakdown(cfg=cfg, md=md, results=results)
    regime_path = out_dir / "phase3_regime_breakdown.csv"
    regime_df.to_csv(regime_path, index=False)

    review_md_path = out_dir / "phase3_strategy_review.md"
    baseline_row = ablation_df[ablation_df["variant"] == "baseline"].iloc[0] if not ablation_df.empty and "baseline" in set(ablation_df["variant"]) else None
    helpful = []
    harmful = []
    if baseline_row is not None:
        for _, row in ablation_df.iterrows():
            if row["variant"] == "baseline":
                continue
            if float(row["sharpe"]) > float(baseline_row["sharpe"]) and float(row["max_drawdown"]) >= float(baseline_row["max_drawdown"]):
                helpful.append(f"- `{row['variant']}` improved Sharpe to {row['sharpe']:.3f} without a better drawdown profile.")
            if float(row["sharpe"]) < float(baseline_row["sharpe"]) and float(row["turnover"]) > float(baseline_row["turnover"]):
                harmful.append(f"- `{row['variant']}` reduced Sharpe and increased turnover.")
    review_md_path.write_text(
        "\n".join(
            [
                "# Phase 3 Strategy Robustness Review",
                "",
                "## Strategy Map",
                f"- Reversal signal: {strategy_map['reversal_signal']}",
                f"- Lookback: `{strategy_map['lookback_period_days']}` days",
                f"- Long quantile: `{strategy_map['long_quantile']}`",
                f"- Short quantile: `{strategy_map['short_quantile']}`",
                f"- Confirmation enabled: `{strategy_map['confirmation_logic']['enabled']}`",
                f"- Market momentum switch enabled: `{strategy_map['market_momentum_switch']}`",
                f"- Funding filter enabled: `{strategy_map['funding_filter']['enabled']}`",
                f"- Rebalance fraction: `{strategy_map['turnover_dampening']['rebalance_fraction']}`",
                f"- Min weight change bps: `{strategy_map['turnover_dampening']['min_weight_change_bps']}`",
                "",
                "## Helpful Signals",
                *(helpful or ["- No clear helper identified mechanically; inspect CSV outputs."]),
                "",
                "## Harmful Or Redundant Signals",
                *(harmful or ["- No clearly harmful component identified mechanically; inspect CSV outputs."]),
                "",
                "## Output Files",
                f"- `{ablation_path.name}`",
                f"- `{sensitivity_path.name}`",
                f"- `{regime_path.name}`",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "ablation": ablation_path,
        "sensitivity": sensitivity_path,
        "regime": regime_path,
        "review": review_md_path,
    }
