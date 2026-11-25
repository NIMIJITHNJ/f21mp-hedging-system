"""
Leakage-safe DCC-GJR-GARCH hedge-ratio benchmark for the Hedra system.

"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.run_hedge_performance_backtest import (
    BACKTEST_END,
    BACKTEST_START,
    CONFIRMATORY_START,
    COST_SENSITIVITY_BPS,
    DEFAULT_COST_BPS,
    MAX_HEDGE_RATIO,
    SELECTION_END,
    STATIC_VECM_REFERENCE,
    apply_strategy,
    build_aligned_returns,
    estimate_static_minimum_variance_ratio,
    evaluate_strategy,
    rolling_minimum_variance_ratio,
)

REPORT_DIR = ROOT / "outputs" / "reports" / "dcc_garch"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DCC_START_YEAR = 2020
MIN_ESTIMATION_ROWS = 500
EPS = 1e-10

# Multiple starting points reduce the risk of retaining a weak local optimum.
DCC_STARTING_VALUES = [
    (0.03, 0.94),
    (0.05, 0.90),
    (0.08, 0.85),
    (0.12, 0.75),
]


def gjr_garch_spec() -> dict[str, Any]:
    """GJR-GARCH(1,1) with Student-t innovations."""
    return {
        "mean": "Zero",
        "vol": "GARCH",
        "p": 1,
        "o": 1,
        "q": 1,
        "dist": "t",
        "rescale": False,
    }


def fit_gjr_parameters(returns_pct: pd.Series) -> pd.Series:
    """Fit one univariate GJR-GARCH model on percent returns."""
    fitted = arch_model(
        returns_pct,
        **gjr_garch_spec(),
    ).fit(
        disp="off",
        show_warning=False,
    )
    return fitted.params


def fixed_conditional_sigma(
    returns_pct: pd.Series,
    fitted_params: pd.Series,
) -> pd.Series:
    """
    Recurse conditional sigma through a return series using fixed parameters.

    """
    fixed = arch_model(
        returns_pct,
        **gjr_garch_spec(),
    ).fix(fitted_params)
    sigma = fixed.conditional_volatility.astype(float)
    return sigma.clip(lower=EPS)


def theta_to_dcc(theta: np.ndarray) -> tuple[float, float]:
    """
    Map unconstrained optimiser parameters to valid DCC coefficients.
    """
    exp_theta = np.exp(
        np.clip(
            np.asarray(theta, dtype=float),
            -30.0,
            30.0,
        )
    )
    denominator = 1.0 + exp_theta.sum()
    a = 0.999 * exp_theta[0] / denominator
    b = 0.999 * exp_theta[1] / denominator
    return float(a), float(b)


def dcc_to_theta(a: float, b: float) -> np.ndarray:
    """Convert valid DCC coefficients into unconstrained optimiser values."""
    remainder = max(0.999 - a - b, EPS)
    return np.log(
        np.array(
            [
                a / remainder,
                b / remainder,
            ],
            dtype=float,
        )
    )


def normalise_q(q_matrix: np.ndarray) -> np.ndarray:
    """Convert a positive covariance-like Q matrix into a correlation matrix."""
    diagonal = np.sqrt(
        np.clip(
            np.diag(q_matrix),
            EPS,
            None,
        )
    )
    correlation = q_matrix / np.outer(diagonal, diagonal)
    correlation = (correlation + correlation.T) / 2.0
    correlation[0, 0] = 1.0
    correlation[1, 1] = 1.0
    correlation[0, 1] = np.clip(correlation[0, 1], -0.999, 0.999)
    correlation[1, 0] = correlation[0, 1]
    return correlation


def dcc_negative_log_likelihood(
    theta: np.ndarray,
    standardized_residuals: np.ndarray,
    q_bar: np.ndarray,
) -> float:
    """
    Gaussian QML objective for DCC(1,1).

    """
    a, b = theta_to_dcc(theta)
    q_t = np.array(q_bar, dtype=float, copy=True)
    objective = 0.0

    for row_number in range(len(standardized_residuals)):
        r_t = normalise_q(q_t)
        sign, log_determinant = np.linalg.slogdet(r_t)
        if sign <= 0 or not np.isfinite(log_determinant):
            return 1e12

        try:
            inverse_r = np.linalg.inv(r_t)
        except np.linalg.LinAlgError:
            return 1e12

        z_t = standardized_residuals[row_number]
        contribution = log_determinant + float(z_t.T @ inverse_r @ z_t)

        if not np.isfinite(contribution):
            return 1e12

        objective += contribution

        if row_number < len(standardized_residuals) - 1:
            q_t = (
                (1.0 - a - b) * q_bar
                + a * np.outer(z_t, z_t)
                + b * q_t
            )

    return float(objective)


def estimate_dcc_parameters(
    standardized_residuals: pd.DataFrame,
) -> tuple[float, float, np.ndarray, float]:
    """Estimate DCC(1,1) coefficients using multiple starting points."""
    clean = standardized_residuals.replace([np.inf, -np.inf], np.nan).dropna()
    values = clean.to_numpy(dtype=float)

    if len(values) < MIN_ESTIMATION_ROWS:
        raise ValueError(
            "Insufficient standardized residual rows for DCC estimation: "
            f"{len(values)}"
        )

    q_bar = np.cov(values, rowvar=False, ddof=1)
    q_bar = (q_bar + q_bar.T) / 2.0
    best_result = None

    for starting_a, starting_b in DCC_STARTING_VALUES:
        result = minimize(
            fun=dcc_negative_log_likelihood,
            x0=dcc_to_theta(starting_a, starting_b),
            args=(values, q_bar),
            method="L-BFGS-B",
            options={
                "maxiter": 500,
                "ftol": 1e-10,
            },
        )

        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None or not np.isfinite(best_result.fun):
        raise RuntimeError("DCC optimisation did not produce a finite objective")

    a, b = theta_to_dcc(best_result.x)
    return a, b, q_bar, float(best_result.fun)


def build_conditional_hedge_ratio(
    standardized_residuals: pd.DataFrame,
    portfolio_sigma: pd.Series,
    futures_sigma: pd.Series,
    q_bar: np.ndarray,
    dcc_a: float,
    dcc_b: float,
) -> pd.DataFrame:
    """
    Recurse DCC state and build the hedge ratio available before each return.

    """
    index = standardized_residuals.index
    z_values = standardized_residuals.to_numpy(dtype=float)
    portfolio_sigma_values = portfolio_sigma.loc[index].to_numpy(dtype=float)
    futures_sigma_values = futures_sigma.loc[index].to_numpy(dtype=float)
    q_t = np.array(q_bar, dtype=float, copy=True)
    rows: list[dict[str, float]] = []

    for row_number, date in enumerate(index):
        r_t = normalise_q(q_t)
        conditional_correlation = float(r_t[0, 1])
        sigma_portfolio = float(portfolio_sigma_values[row_number])
        sigma_futures = float(futures_sigma_values[row_number])
        conditional_covariance = conditional_correlation * sigma_portfolio * sigma_futures
        conditional_futures_variance = sigma_futures ** 2
        hedge_ratio = conditional_covariance / max(conditional_futures_variance, EPS)
        hedge_ratio = float(np.clip(hedge_ratio, 0.0, MAX_HEDGE_RATIO))

        rows.append(
            {
                "date": date,
                "dcc_correlation": conditional_correlation,
                "portfolio_sigma_pct": sigma_portfolio,
                "futures_sigma_pct": sigma_futures,
                "conditional_covariance": conditional_covariance,
                "conditional_futures_variance": conditional_futures_variance,
                "dcc_hedge_ratio": hedge_ratio,
            }
        )

        if row_number < len(index) - 1:
            z_t = z_values[row_number]
            q_t = (
                (1.0 - dcc_a - dcc_b) * q_bar
                + dcc_a * np.outer(z_t, z_t)
                + dcc_b * q_t
            )

    return pd.DataFrame(rows).set_index("date")


def build_annual_oos_dcc_series(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate annual out-of-sample DCC hedge ratios from 2020 onward.

    """
    returns_pct = (returns[["portfolio", "futures"]] * 100.0).astype(float)
    final_year = int(min(pd.Timestamp(BACKTEST_END).year, returns_pct.index.max().year))
    ratio_parts: list[pd.DataFrame] = []
    refit_rows: list[dict[str, Any]] = []

    for year in range(DCC_START_YEAR, final_year + 1):
        estimation_end = pd.Timestamp(year=year - 1, month=12, day=31)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)
        estimation = returns_pct.loc[:estimation_end]
        recursion = returns_pct.loc[:year_end]

        if len(estimation) < MIN_ESTIMATION_ROWS:
            raise ValueError(
                f"Insufficient estimation history for {year}: {len(estimation)} rows"
            )

        print(
            f"  Refitting DCC-GJR-GARCH for {year} using {len(estimation)} rows "
            f"through {estimation_end.date()} ..."
        )

        portfolio_params = fit_gjr_parameters(estimation["portfolio"])
        futures_params = fit_gjr_parameters(estimation["futures"])
        portfolio_sigma = fixed_conditional_sigma(recursion["portfolio"], portfolio_params)
        futures_sigma = fixed_conditional_sigma(recursion["futures"], futures_params)

        standardized = pd.DataFrame(
            {
                "portfolio": recursion["portfolio"] / portfolio_sigma,
                "futures": recursion["futures"] / futures_sigma,
            }
        ).replace([np.inf, -np.inf], np.nan).dropna()

        dcc_a, dcc_b, q_bar, objective = estimate_dcc_parameters(
            standardized.loc[:estimation_end]
        )

        print(
            f"    DCC parameters: a={dcc_a:.6f}, b={dcc_b:.6f}, "
            f"a+b={dcc_a + dcc_b:.6f}"
        )

        recursion_frame = build_conditional_hedge_ratio(
            standardized_residuals=standardized,
            portfolio_sigma=portfolio_sigma,
            futures_sigma=futures_sigma,
            q_bar=q_bar,
            dcc_a=dcc_a,
            dcc_b=dcc_b,
        )

        year_frame = recursion_frame[
            (recursion_frame.index >= year_start)
            & (recursion_frame.index <= year_end)
        ].copy()

        year_frame["refit_year"] = year
        year_frame["dcc_a"] = dcc_a
        year_frame["dcc_b"] = dcc_b
        ratio_parts.append(year_frame)

        refit_rows.append(
            {
                "forecast_year": year,
                "estimation_end": estimation_end.date().isoformat(),
                "estimation_rows": len(estimation),
                "dcc_a": dcc_a,
                "dcc_b": dcc_b,
                "dcc_persistence": dcc_a + dcc_b,
                "dcc_objective": objective,
                "portfolio_gjr_parameters": json.dumps(
                    {key: float(value) for key, value in portfolio_params.items()}
                ),
                "futures_gjr_parameters": json.dumps(
                    {key: float(value) for key, value in futures_params.items()}
                ),
            }
        )

    if not ratio_parts:
        raise RuntimeError("No annual DCC ratio frames were generated")

    ratios = pd.concat(ratio_parts).sort_index()
    ratios = ratios[~ratios.index.duplicated(keep="last")]
    return ratios, pd.DataFrame(refit_rows)


def build_comparison_ratios(
    returns: pd.DataFrame,
    dcc_ratio: pd.Series,
) -> pd.DataFrame:
    """Construct DCC and practical benchmark hedge-ratio columns."""
    ratios = pd.DataFrame(index=returns.index)
    ratios["no_hedge"] = 0.0
    ratios["static_50"] = 0.50
    ratios["literature_vecm_static"] = STATIC_VECM_REFERENCE
    ratios["portfolio_static_minvar"] = estimate_static_minimum_variance_ratio(returns)
    ratios["rolling_min_variance"] = rolling_minimum_variance_ratio(
        portfolio_returns=returns["portfolio"],
        futures_returns=returns["futures"],
    )
    ratios["dcc_gjr_garch"] = dcc_ratio.reindex(returns.index)
    return ratios


def evaluate_period(
    returns: pd.DataFrame,
    ratios: pd.DataFrame,
    start: str,
    end: str,
    period_label: str,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every strategy within one date range."""
    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    period_returns = returns.loc[start:end]

    for strategy_name in ratios.columns:
        frame = apply_strategy(
            returns=period_returns,
            hedge_ratio=ratios[strategy_name].loc[start:end],
            cost_bps=cost_bps,
        )

        if frame.empty:
            raise ValueError(
                f"No available rows for strategy={strategy_name}, period={period_label}"
            )

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


def run_cost_sensitivity(
    returns: pd.DataFrame,
    ratios: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate full-period strategy robustness across cost assumptions."""
    frames: list[pd.DataFrame] = []

    for cost_bps in COST_SENSITIVITY_BPS:
        summary, _ = evaluate_period(
            returns=returns,
            ratios=ratios,
            start=BACKTEST_START,
            end=BACKTEST_END,
            period_label="full",
            cost_bps=cost_bps,
        )
        frames.append(summary)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    print("Building aligned portfolio and futures returns ...")
    returns = build_aligned_returns()
    print(f"  Return coverage: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  Return rows    : {len(returns)}")

    print("\nBuilding annual out-of-sample DCC-GJR-GARCH hedge ratios ...")
    dcc_frame, refits = build_annual_oos_dcc_series(returns)
    print(f"\n  DCC ratio coverage: {dcc_frame.index.min().date()} to {dcc_frame.index.max().date()}")
    print(f"  DCC ratio rows    : {len(dcc_frame)}")

    ratios = build_comparison_ratios(
        returns=returns,
        dcc_ratio=dcc_frame["dcc_hedge_ratio"],
    )

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

    print("\nEvaluating transaction-cost sensitivity ...")
    cost_sensitivity = run_cost_sensitivity(returns=returns, ratios=ratios)

    combined_summary = pd.concat([selection_summary, confirmatory_summary], ignore_index=True)
    combined_details = pd.concat([selection_details, confirmatory_details], ignore_index=True)

    dcc_frame.to_csv(REPORT_DIR / "dcc_garch_ratio_series.csv", index=True)
    refits.to_csv(REPORT_DIR / "dcc_garch_annual_refits.csv", index=False)
    combined_summary.to_csv(REPORT_DIR / "dcc_garch_performance_summary.csv", index=False)
    combined_details.to_csv(REPORT_DIR / "dcc_garch_daily_details.csv", index=False)
    cost_sensitivity.to_csv(REPORT_DIR / "dcc_garch_cost_sensitivity.csv", index=False)

    metadata = {
        "portfolio": "equal-weight portfolio of 10 Nifty 50 constituent stocks",
        "hedging_instrument": "Nifty 50 continuous futures",
        "marginal_models": "GJR-GARCH(1,1) with Student-t errors for portfolio and futures returns",
        "dependence_model": "DCC(1,1) estimated using Gaussian QML",
        "hedge_ratio_formula": "conditional covariance(portfolio, futures) / conditional variance(futures)",
        "timing_rule": "The ratio used for return at day t is based on conditional state available before return t is observed.",
        "refit_rule": "Annual expanding refit using data through the previous calendar year-end only.",
        "transaction_cost_assumption_bps": DEFAULT_COST_BPS,
        "selection_period": f"{BACKTEST_START} to {SELECTION_END}",
        "confirmatory_period": f"{CONFIRMATORY_START} to {BACKTEST_END}",
        "max_hedge_ratio": MAX_HEDGE_RATIO,
    }

    with open(REPORT_DIR / "dcc_garch_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    display_columns = [
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

    print("\n" + "=" * 114)
    print("DCC-GJR-GARCH HEDGE PERFORMANCE SUMMARY - 2 BPS ONE-WAY FUTURES TURNOVER COST")
    print("=" * 114)
    print(
        combined_summary[display_columns]
        .sort_values(["period", "volatility_pct"])
        .to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"  {REPORT_DIR / 'dcc_garch_ratio_series.csv'}")
    print(f"  {REPORT_DIR / 'dcc_garch_annual_refits.csv'}")
    print(f"  {REPORT_DIR / 'dcc_garch_performance_summary.csv'}")
    print(f"  {REPORT_DIR / 'dcc_garch_daily_details.csv'}")
    print(f"  {REPORT_DIR / 'dcc_garch_cost_sensitivity.csv'}")
    print(f"  {REPORT_DIR / 'dcc_garch_metadata.json'}")


if __name__ == "__main__":
    main()