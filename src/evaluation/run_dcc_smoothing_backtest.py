"""
Practical DCC-GJR-GARCH hedge-ratio refinement benchmark for Hedra.
Tests EMA smoothing and no-trade bands on the leakage-safe annual OOS DCC ratio.

"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.run_hedge_performance_backtest import (
    BACKTEST_END,
    BACKTEST_START,
    CONFIRMATORY_START,
    DEFAULT_COST_BPS,
    SELECTION_END,
    STATIC_VECM_REFERENCE,
    apply_strategy,
    build_aligned_returns,
    estimate_static_minimum_variance_ratio,
    evaluate_strategy,
    rolling_minimum_variance_ratio,
)
from src.evaluation.run_dcc_garch_backtest import build_annual_oos_dcc_series

REPORT_DIR = ROOT / "outputs" / "reports" / "dcc_smoothing"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EMA_SPANS = [1, 3, 5, 10, 20]
NO_TRADE_BANDS = [0.00, 0.02, 0.05, 0.10]
TOP_N = 15


def format_band(band: float) -> str:
    return f"{band:.2f}".replace(".", "p")


def ema_smooth_ratio(ratio: pd.Series, span: int) -> pd.Series:
    clean = ratio.astype(float).copy()
    if span == 1:
        return clean
    return clean.ewm(span=span, adjust=False, min_periods=1).mean()


def apply_no_trade_band(target_ratio: pd.Series, band: float) -> pd.Series:
    if band < 0:
        raise ValueError("No-trade band must be non-negative")

    executed = pd.Series(index=target_ratio.index, dtype=float, name=target_ratio.name)
    current_position = np.nan

    for date, target in target_ratio.items():
        if pd.isna(target):
            executed.loc[date] = np.nan
            continue

        target_value = float(target)
        if np.isnan(current_position):
            current_position = target_value
        elif abs(target_value - current_position) >= band:
            current_position = target_value

        executed.loc[date] = current_position

    return executed


def build_dcc_execution_candidates(raw_dcc_ratio: pd.Series) -> pd.DataFrame:
    candidates = pd.DataFrame(index=raw_dcc_ratio.index)

    for span in EMA_SPANS:
        smoothed = ema_smooth_ratio(raw_dcc_ratio, span)
        for band in NO_TRADE_BANDS:
            name = f"dcc_ema{span}_band{format_band(band)}"
            candidates[name] = apply_no_trade_band(smoothed, band)

    return candidates


def build_reference_ratios(returns: pd.DataFrame, raw_dcc_ratio: pd.Series) -> pd.DataFrame:
    ratios = pd.DataFrame(index=returns.index)
    ratios["no_hedge"] = 0.0
    ratios["static_50"] = 0.50
    ratios["literature_vecm_static"] = STATIC_VECM_REFERENCE
    ratios["portfolio_static_minvar"] = estimate_static_minimum_variance_ratio(returns)
    ratios["rolling_min_variance"] = rolling_minimum_variance_ratio(
        portfolio_returns=returns["portfolio"],
        futures_returns=returns["futures"],
    )
    ratios["dcc_raw"] = raw_dcc_ratio.reindex(returns.index)
    return ratios


def evaluate_ratio_set(
    returns: pd.DataFrame,
    ratios: pd.DataFrame,
    start: str,
    end: str,
    period_label: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    period_returns = returns.loc[start:end]

    for strategy_name in ratios.columns:
        ratio = ratios[strategy_name].loc[start:end]
        frame = apply_strategy(period_returns, ratio, cost_bps)
        if frame.empty:
            raise ValueError(f"No rows for strategy={strategy_name}, period={period_label}")

        summary_rows.append(
            evaluate_strategy(
                strategy_name=strategy_name,
                strategy_frame=frame,
                period_label=period_label,
                cost_bps=cost_bps,
            )
        )

        details = frame.reset_index().rename(columns={"index": "date"})
        details.insert(1, "strategy", strategy_name)
        details.insert(2, "period", period_label)
        details.insert(3, "cost_bps", cost_bps)
        detail_frames.append(details)

    return pd.DataFrame(summary_rows), pd.concat(detail_frames, ignore_index=True)


def add_selection_ranks(selection_summary: pd.DataFrame) -> pd.DataFrame:
    ranked = selection_summary.copy()
    ranked["volatility_rank"] = ranked["volatility_pct"].rank(method="min", ascending=True)
    ranked["max_drawdown_rank"] = ranked["max_drawdown_pct"].abs().rank(method="min", ascending=True)
    ranked["cvar_rank"] = ranked["cvar_95_pct"].abs().rank(method="min", ascending=True)
    ranked["turnover_rank"] = ranked["total_turnover"].rank(method="min", ascending=True)
    ranked["balanced_rank_score"] = ranked[
        ["volatility_rank", "max_drawdown_rank", "cvar_rank", "turnover_rank"]
    ].mean(axis=1).round(4)
    return ranked


def pareto_frontier(dcc_summary: pd.DataFrame) -> pd.DataFrame:
    frontier_rows = []
    for index, row in dcc_summary.iterrows():
        dominated = False
        for other_index, other in dcc_summary.iterrows():
            if index == other_index:
                continue
            no_worse = (
                other["volatility_pct"] <= row["volatility_pct"]
                and other["total_turnover"] <= row["total_turnover"]
            )
            strictly_better = (
                other["volatility_pct"] < row["volatility_pct"]
                or other["total_turnover"] < row["total_turnover"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier_rows.append(row)

    if not frontier_rows:
        return pd.DataFrame(columns=dcc_summary.columns)
    return pd.DataFrame(frontier_rows).sort_values(["volatility_pct", "total_turnover"])


def main() -> None:
    print("Building aligned portfolio and futures returns ...")
    returns = build_aligned_returns()
    print(f"  Return coverage: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  Return rows    : {len(returns)}")

    print("\nRebuilding leakage-safe annual OOS DCC-GJR-GARCH ratios ...")
    dcc_frame, dcc_refits = build_annual_oos_dcc_series(returns)
    raw_dcc_ratio = dcc_frame["dcc_hedge_ratio"].rename("dcc_raw")
    print(f"\n  Raw DCC coverage: {raw_dcc_ratio.index.min().date()} to {raw_dcc_ratio.index.max().date()}")
    print(f"  Raw DCC rows    : {len(raw_dcc_ratio)}")

    print("\nBuilding EMA-smoothed DCC and no-trade-band candidates ...")
    candidate_ratios = build_dcc_execution_candidates(raw_dcc_ratio)
    reference_ratios = build_reference_ratios(returns, raw_dcc_ratio)
    all_ratios = reference_ratios.join(candidate_ratios, how="left")
    print(f"  Reference strategies : {len(reference_ratios.columns)}")
    print(f"  DCC candidates       : {len(candidate_ratios.columns)}")

    print("\nEvaluating selection period: 2020-2024 ...")
    selection_summary, selection_details = evaluate_ratio_set(
        returns, all_ratios, BACKTEST_START, SELECTION_END, "selection_2020_2024", DEFAULT_COST_BPS
    )

    print("\nEvaluating confirmatory period: 2025-Mar 2026 ...")
    confirmatory_summary, confirmatory_details = evaluate_ratio_set(
        returns, all_ratios, CONFIRMATORY_START, BACKTEST_END, "confirmatory_2025_2026", DEFAULT_COST_BPS
    )

    ranked_selection = add_selection_ranks(selection_summary)
    ranked_dcc = ranked_selection[
        ranked_selection["strategy"].str.startswith("dcc_ema")
    ].sort_values(["balanced_rank_score", "volatility_pct", "total_turnover"])

    frontier = pareto_frontier(
        selection_summary[selection_summary["strategy"].str.startswith("dcc_ema")].copy()
    )

    combined_summary = pd.concat([ranked_selection, confirmatory_summary], ignore_index=True, sort=False)
    combined_details = pd.concat([selection_details, confirmatory_details], ignore_index=True)

    all_ratios.to_csv(REPORT_DIR / "dcc_smoothing_ratio_series.csv", index=True)
    combined_summary.to_csv(REPORT_DIR / "dcc_smoothing_performance_summary.csv", index=False)
    ranked_dcc.to_csv(REPORT_DIR / "dcc_smoothing_selection_ranking.csv", index=False)
    frontier.to_csv(REPORT_DIR / "dcc_smoothing_pareto_frontier.csv", index=False)
    combined_details.to_csv(REPORT_DIR / "dcc_smoothing_daily_details.csv", index=False)
    dcc_refits.to_csv(REPORT_DIR / "dcc_smoothing_source_dcc_refits.csv", index=False)

    metadata = {
        "purpose": "Evaluate practical DCC-GJR-GARCH execution variants using EMA smoothing and no-trade bands.",
        "selection_period": f"{BACKTEST_START} to {SELECTION_END}",
        "confirmatory_period": f"{CONFIRMATORY_START} to {BACKTEST_END}",
        "transaction_cost_assumption_bps": DEFAULT_COST_BPS,
        "ema_spans": EMA_SPANS,
        "no_trade_bands": NO_TRADE_BANDS,
        "selection_rule": "Compare selection-period volatility, drawdown, CVaR and turnover. Do not repeatedly retune the grid using confirmatory results.",
        "timing_rule": "Raw DCC ratio for day t is available before return t. Smoothing and band logic use only current and earlier available targets.",
    }
    with open(REPORT_DIR / "dcc_smoothing_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    display_cols = [
        "strategy", "average_hedge_ratio", "volatility_pct", "max_drawdown_pct",
        "cvar_95_pct", "hedging_effectiveness", "total_turnover",
        "total_transaction_cost_pct", "balanced_rank_score",
    ]

    print("\n" + "=" * 122)
    print("TOP DCC EXECUTION CANDIDATES ON SELECTION PERIOD (2020-2024, 2 BPS ONE-WAY TURNOVER COST)")
    print("=" * 122)
    print(ranked_dcc[display_cols].head(TOP_N).to_string(index=False))

    print("\n" + "=" * 122)
    print("REFERENCE STRATEGIES ON SELECTION PERIOD")
    print("=" * 122)
    references = ranked_selection[~ranked_selection["strategy"].str.startswith("dcc_ema")]
    print(references[display_cols].sort_values("volatility_pct").to_string(index=False))

    print("\n" + "=" * 122)
    print("DCC VOLATILITY-TURNOVER PARETO FRONTIER ON SELECTION PERIOD")
    print("=" * 122)
    print(frontier[[
        "strategy", "volatility_pct", "max_drawdown_pct", "cvar_95_pct",
        "hedging_effectiveness", "total_turnover", "total_transaction_cost_pct",
    ]].to_string(index=False))

    print("\nSaved outputs:")
    for filename in [
        "dcc_smoothing_ratio_series.csv",
        "dcc_smoothing_performance_summary.csv",
        "dcc_smoothing_selection_ranking.csv",
        "dcc_smoothing_pareto_frontier.csv",
        "dcc_smoothing_daily_details.csv",
        "dcc_smoothing_source_dcc_refits.csv",
        "dcc_smoothing_metadata.json",
    ]:
        print(f"  {REPORT_DIR / filename}")


if __name__ == "__main__":
    main()
