"""
Leakage-safe hedge-performance benchmark for the system.

Purpose
-------
Evaluate whether the selected 1-day GJR-GARCH volatility forecast improves
real hedging outcomes for the equal-weight 10-stock portfolio after turnover
and transaction costs.

Strategies
----------
1. no_hedge
2. static_50
3. literature_vecm_static
4. rolling_min_variance
5. gjr_regime_dynamic
6. gjr_scaled_minvar

Important timing rule
---------------------
A volatility forecast produced at the end of day t is applied to the futures
position for day t+1 only. This avoids look-ahead bias.

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

from src.evaluation.metrics import summarise
from src.evaluation.portfolio import (
    compute_futures_returns,
    compute_portfolio_returns,
    compute_stock_returns,
    load_futures_prices,
    load_stock_prices,
)
from src.models.regression.run_volatility_regression_walkforward import (
    build_base_features,
    build_horizon_frame,
)

REPORT_DIR = ROOT / "outputs" / "reports" / "hedge_performance"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
BACKTEST_START = "2020-01-01"
SELECTION_END = "2024-12-31"
CONFIRMATORY_START = "2025-01-01"
BACKTEST_END = "2026-03-31"

ROLLING_WINDOW = 60
MIN_EXPANDING_HISTORY = 252
MAX_HEDGE_RATIO = 1.50

DEFAULT_COST_BPS = 2.0
COST_SENSITIVITY_BPS = [0.0, 1.0, 2.0, 5.0]

STATIC_VECM_REFERENCE = 0.8824
STATIC_MINVAR_ESTIMATION_END = "2019-12-31"

REGIME_LEVELS = {
    "low": 0.35,
    "medium": 0.70,
    "high": 1.00,
}


def build_aligned_returns() -> pd.DataFrame:
    """Build aligned portfolio and futures log-return series."""
    stock_returns = compute_stock_returns(load_stock_prices())
    portfolio_returns = compute_portfolio_returns(stock_returns)
    futures_returns = compute_futures_returns(load_futures_prices())

    returns = pd.concat(
        [portfolio_returns, futures_returns],
        axis=1,
        join="inner",
    ).dropna()

    returns = returns.loc[:BACKTEST_END].copy()

    if returns.empty:
        raise ValueError("No aligned portfolio/futures rows were found")

    return returns


def estimate_static_minimum_variance_ratio(
    returns: pd.DataFrame,
) -> float:
    """
    Estimate a portfolio-specific static minimum-variance hedge ratio
    using only data available before the 2020 backtest period.
    """
    estimation = returns.loc[:STATIC_MINVAR_ESTIMATION_END]

    covariance = estimation["portfolio"].cov(estimation["futures"])
    variance = estimation["futures"].var()

    if variance == 0 or np.isnan(variance):
        raise ValueError("Cannot estimate static minimum-variance ratio")

    return float(
        np.clip(
            covariance / variance,
            0.0,
            MAX_HEDGE_RATIO,
        )
    )


def rolling_minimum_variance_ratio(
    portfolio_returns: pd.Series,
    futures_returns: pd.Series,
    window: int = ROLLING_WINDOW,
) -> pd.Series:
    """
    Estimate the classic minimum-variance hedge ratio:
        h_t = Cov(R_portfolio, R_futures) / Var(R_futures)

    The estimate is shifted by one trading day, so return on day t uses only
    data observed through day t-1.
    """
    covariance = portfolio_returns.rolling(window).cov(futures_returns)
    variance = futures_returns.rolling(window).var()

    ratio = (covariance / variance.replace(0, np.nan)).shift(1)
    ratio = ratio.clip(lower=0.0, upper=MAX_HEDGE_RATIO)
    return ratio.rename("rolling_min_variance")


def build_gjr_forecast_signal() -> pd.Series:
    """
    Rebuild the selected 1-day GJR-GARCH forecast series using the same
    leakage-safe annual expanding-refit method as the regression benchmark.

    The series returned here is shifted by one trading day:
        forecast generated at t -> hedge applied at t+1
    """
    base_features, spot_close = build_base_features()
    horizon_frame = build_horizon_frame(
        base_features=base_features,
        spot_close=spot_close,
        horizon=1,
    )

    signal = horizon_frame["gjr_garch_forecast"].copy()
    signal = signal.shift(1)
    signal.name = "gjr_forecast_lagged"
    return signal


def gjr_regime_dynamic_ratio(
    lagged_forecast: pd.Series,
) -> pd.Series:
    """
    Convert lagged GJR-GARCH forecasts into a transparent three-level hedge.

    Expanding q33 and q67 thresholds are shifted by one day. Therefore, the
    risk threshold used on date t is based only on earlier forecasts.

    Hedge levels:
        Low risk    -> 0.35
        Medium risk -> 0.70
        High risk   -> 1.00
    """
    q33 = (
        lagged_forecast
        .expanding(min_periods=MIN_EXPANDING_HISTORY)
        .quantile(1 / 3)
        .shift(1)
    )

    q67 = (
        lagged_forecast
        .expanding(min_periods=MIN_EXPANDING_HISTORY)
        .quantile(2 / 3)
        .shift(1)
    )

    ratio = pd.Series(index=lagged_forecast.index, dtype=float)

    low_mask = lagged_forecast < q33
    high_mask = lagged_forecast >= q67
    medium_mask = ~(low_mask | high_mask)

    ratio.loc[low_mask] = REGIME_LEVELS["low"]
    ratio.loc[medium_mask] = REGIME_LEVELS["medium"]
    ratio.loc[high_mask] = REGIME_LEVELS["high"]

    ratio = ratio.where(q33.notna() & q67.notna())
    return ratio.rename("gjr_regime_dynamic")


def gjr_scaled_minvar_ratio(
    rolling_minvar: pd.Series,
    lagged_forecast: pd.Series,
) -> pd.Series:
    """
    Scale the rolling minimum-variance ratio using the lagged GJR forecast.

    The volatility scaling factor compares the forecast with its expanding
    historical median, calculated from earlier forecasts only.

    Scaling is capped to avoid excessive leverage and unstable turnover.
    """
    historical_median = (
        lagged_forecast
        .expanding(min_periods=MIN_EXPANDING_HISTORY)
        .median()
        .shift(1)
    )

    scale = (
        lagged_forecast
        / historical_median.replace(0, np.nan)
    ).clip(lower=0.75, upper=1.25)

    ratio = (rolling_minvar * scale).clip(
        lower=0.0,
        upper=MAX_HEDGE_RATIO,
    )

    return ratio.rename("gjr_scaled_minvar")


def build_strategy_ratios(returns: pd.DataFrame) -> pd.DataFrame:
    """Build all hedge-ratio strategies on the common return index."""
    ratios = pd.DataFrame(index=returns.index)

    ratios["no_hedge"] = 0.0
    ratios["static_50"] = 0.50
    ratios["literature_vecm_static"] = STATIC_VECM_REFERENCE
    ratios["portfolio_static_minvar"] = estimate_static_minimum_variance_ratio(
        returns
    )

    ratios["rolling_min_variance"] = rolling_minimum_variance_ratio(
        portfolio_returns=returns["portfolio"],
        futures_returns=returns["futures"],
    )

    lagged_forecast = build_gjr_forecast_signal().reindex(returns.index)
    ratios["gjr_regime_dynamic"] = gjr_regime_dynamic_ratio(
        lagged_forecast=lagged_forecast,
    )

    ratios["gjr_scaled_minvar"] = gjr_scaled_minvar_ratio(
        rolling_minvar=ratios["rolling_min_variance"],
        lagged_forecast=lagged_forecast,
    )

    return ratios


def apply_strategy(
    returns: pd.DataFrame,
    hedge_ratio: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    """
    Apply a dynamic hedge ratio with transaction costs.

    turnover[t] = abs(h[t] - h[t-1])
    cost[t]     = turnover[t] * cost_bps / 10,000
    """
    frame = returns.join(hedge_ratio.rename("hedge_ratio"), how="inner")
    frame = frame.dropna(subset=["hedge_ratio"]).copy()

    frame["turnover"] = frame["hedge_ratio"].diff().abs()
    frame.loc[frame.index[0], "turnover"] = abs(frame.loc[frame.index[0], "hedge_ratio"])
    frame["transaction_cost"] = frame["turnover"] * (cost_bps / 10_000.0)

    frame["hedged_return_gross"] = (
        frame["portfolio"]
        - frame["hedge_ratio"] * frame["futures"]
    )

    frame["hedged_return_net"] = (
        frame["hedged_return_gross"]
        - frame["transaction_cost"]
    )

    return frame


def hedging_effectiveness(
    unhedged_returns: pd.Series,
    hedged_returns: pd.Series,
) -> float:
    """
    Variance-reduction hedging effectiveness:
        HE = 1 - Var(hedged) / Var(unhedged)
    """
    unhedged_var = float(unhedged_returns.var())

    if unhedged_var == 0:
        return np.nan

    return float(
        1.0
        - (hedged_returns.var() / unhedged_var)
    )


def evaluate_strategy(
    strategy_name: str,
    strategy_frame: pd.DataFrame,
    period_label: str,
    cost_bps: float,
) -> dict[str, Any]:
    """Calculate risk, return, turnover and cost metrics."""
    net_returns = strategy_frame["hedged_return_net"]
    gross_returns = strategy_frame["hedged_return_gross"]
    unhedged_returns = strategy_frame["portfolio"]

    metrics = summarise(
        returns=net_returns,
        hedge_ratios=strategy_frame["hedge_ratio"],
        label=strategy_name,
    )

    metrics.update(
        {
            "strategy": strategy_name,
            "period": period_label,
            "cost_bps": cost_bps,
            "n_days": len(strategy_frame),
            "average_hedge_ratio": round(
                float(strategy_frame["hedge_ratio"].mean()),
                4,
            ),
            "min_hedge_ratio": round(
                float(strategy_frame["hedge_ratio"].min()),
                4,
            ),
            "max_hedge_ratio": round(
                float(strategy_frame["hedge_ratio"].max()),
                4,
            ),
            "average_daily_turnover": round(
                float(strategy_frame["turnover"].mean()),
                6,
            ),
            "total_turnover": round(
                float(strategy_frame["turnover"].sum()),
                4,
            ),
            "total_transaction_cost_pct": round(
                float(strategy_frame["transaction_cost"].sum() * 100),
                4,
            ),
            "gross_total_return_pct": round(
                float((np.exp(gross_returns.sum()) - 1) * 100),
                2,
            ),
            "hedging_effectiveness": round(
                hedging_effectiveness(
                    unhedged_returns=unhedged_returns,
                    hedged_returns=net_returns,
                ),
                4,
            ),
        }
    )

    return metrics


def evaluate_period(
    returns: pd.DataFrame,
    ratios: pd.DataFrame,
    start: str,
    end: str,
    period_label: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all strategies for one period."""
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []

    period_returns = returns.loc[start:end]

    for strategy_name in ratios.columns:
        strategy_ratio = ratios[strategy_name].loc[start:end]

        frame = apply_strategy(
            returns=period_returns,
            hedge_ratio=strategy_ratio,
            cost_bps=cost_bps,
        )

        if frame.empty:
            raise ValueError(
                f"No available rows for strategy={strategy_name}, "
                f"period={period_label}"
            )

        summary_rows.append(
            evaluate_strategy(
                strategy_name=strategy_name,
                strategy_frame=frame,
                period_label=period_label,
                cost_bps=cost_bps,
            )
        )

        details = frame.reset_index().rename(
            columns={"index": "date"}
        )

        details.insert(1, "strategy", strategy_name)
        details.insert(2, "period", period_label)
        details.insert(3, "cost_bps", cost_bps)

        detail_frames.append(details)

    return (
        pd.DataFrame(summary_rows),
        pd.concat(detail_frames, ignore_index=True),
    )


def run_cost_sensitivity(
    returns: pd.DataFrame,
    ratios: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate full-period results under multiple futures cost assumptions."""
    rows: list[pd.DataFrame] = []

    for cost_bps in COST_SENSITIVITY_BPS:
        summary, _ = evaluate_period(
            returns=returns,
            ratios=ratios,
            start=BACKTEST_START,
            end=BACKTEST_END,
            period_label="full",
            cost_bps=cost_bps,
        )

        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    print("Building aligned portfolio and futures returns ...")
    returns = build_aligned_returns()

    print(
        f"  Return coverage: {returns.index.min().date()} "
        f"to {returns.index.max().date()}"
    )
    print(f"  Return rows    : {len(returns)}")

    print("\nBuilding static and dynamic hedge-ratio strategies ...")
    ratios = build_strategy_ratios(returns)

    print("\nEvaluating selection period: 2020-2024 ...")
    selection_summary, selection_details = evaluate_period(
        returns=returns,
        ratios=ratios,
        start=BACKTEST_START,
        end=SELECTION_END,
        period_label="selection_2020_2024",
        cost_bps=DEFAULT_COST_BPS,
    )

    print("\nEvaluating confirmatory period: 2025-Mar 2026 ...")
    confirmatory_summary, confirmatory_details = evaluate_period(
        returns=returns,
        ratios=ratios,
        start=CONFIRMATORY_START,
        end=BACKTEST_END,
        period_label="confirmatory_2025_2026",
        cost_bps=DEFAULT_COST_BPS,
    )

    print("\nEvaluating full-period transaction-cost sensitivity ...")
    cost_sensitivity = run_cost_sensitivity(
        returns=returns,
        ratios=ratios,
    )

    combined_summary = pd.concat(
        [
            selection_summary,
            confirmatory_summary,
        ],
        ignore_index=True,
    )

    combined_details = pd.concat(
        [
            selection_details,
            confirmatory_details,
        ],
        ignore_index=True,
    )

    ratios.to_csv(
        REPORT_DIR / "hedge_ratio_series.csv",
        index=True,
    )

    combined_summary.to_csv(
        REPORT_DIR / "hedge_performance_summary.csv",
        index=False,
    )

    combined_details.to_csv(
        REPORT_DIR / "hedge_performance_daily_details.csv",
        index=False,
    )

    cost_sensitivity.to_csv(
        REPORT_DIR / "hedge_cost_sensitivity.csv",
        index=False,
    )

    metadata = {
        "portfolio": "equal-weight portfolio of 10 Nifty 50 constituent stocks",
        "hedging_instrument": "Nifty 50 continuous futures",
        "selected_forecast_model": "GJR-GARCH(1,1) with Student-t errors",
        "forecast_horizon": "1 trading day",
        "timing_rule": (
            "Forecast generated at end of day t is shifted and applied "
            "to the futures position for day t+1."
        ),
        "transaction_cost_assumption_bps": DEFAULT_COST_BPS,
        "rolling_minimum_variance_window_days": ROLLING_WINDOW,
        "max_hedge_ratio": MAX_HEDGE_RATIO,
        "gjr_regime_levels": REGIME_LEVELS,
        "selection_period": f"{BACKTEST_START} to {SELECTION_END}",
        "confirmatory_period": f"{CONFIRMATORY_START} to {BACKTEST_END}",
    }

    with open(
        REPORT_DIR / "hedge_performance_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\n" + "=" * 110)
    print("HEDGE PERFORMANCE SUMMARY - 2 BPS ONE-WAY FUTURES TURNOVER COST")
    print("=" * 110)

    display_cols = [
        "period",
        "strategy",
        "n_days",
        "average_hedge_ratio",
        "total_return_pct",
        "volatility_pct",
        "max_drawdown_pct",
        "cvar_95_pct",
        "sharpe",
        "hedging_effectiveness",
        "total_turnover",
        "total_transaction_cost_pct",
    ]

    print(
        combined_summary[display_cols]
        .sort_values(
            ["period", "volatility_pct"]
        )
        .to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"  {REPORT_DIR / 'hedge_ratio_series.csv'}")
    print(f"  {REPORT_DIR / 'hedge_performance_summary.csv'}")
    print(f"  {REPORT_DIR / 'hedge_performance_daily_details.csv'}")
    print(f"  {REPORT_DIR / 'hedge_cost_sensitivity.csv'}")
    print(f"  {REPORT_DIR / 'hedge_performance_metadata.json'}")


if __name__ == "__main__":
    main()



