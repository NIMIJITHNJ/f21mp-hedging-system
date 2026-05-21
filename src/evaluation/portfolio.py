"""
portfolio.py

Computes hedged portfolio return series from spot returns, futures returns,
and hedge ratio decisions.

Core formula:
    hedged_return_t = spot_return_t - hedge_ratio_t × futures_return_t

A hedge ratio of 0.0 means no hedge (full spot exposure).
A hedge ratio of 1.0 means fully hedged (spot return fully offset by futures).
Intermediate values represent partial hedges.

The hedge ratio used here is the VECM-derived optimal hedge ratio from the
literature (Vasantha & Mallikarjunappa, 2018) as the baseline, overridden
by the PDDL planner output during live operation.

Functions:
    compute_returns()         — load spot and futures log returns
    apply_hedge()             — apply hedge ratio series to compute hedged returns
    compute_portfolio_value() — cumulative portfolio value from return series
"""

import os
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"


def compute_returns(
    spot_csv: str = "NIFTY50_spot.csv",
    futures_csv: str = "NIFTY_futures_continuous.csv",
) -> pd.DataFrame:
    """
    Load spot and futures daily log returns from raw CSVs.

    Returns a DataFrame with columns:
        spot_return    : daily log return of Nifty 50 spot
        futures_return : daily log return of Nifty futures (continuous)

    First row is NaN (no prior close to compute return from) and is dropped.
    """
    spot = pd.read_csv(
        os.path.join(RAW_DIR, spot_csv),
        parse_dates=["date"],
    ).set_index("date").sort_index()

    futures = pd.read_csv(
        os.path.join(RAW_DIR, futures_csv),
        parse_dates=["date"],
    ).set_index("date").sort_index()

    returns = pd.DataFrame(index=spot.index)
    returns["spot_return"]    = np.log(spot["close"] / spot["close"].shift(1))
    returns["futures_return"] = np.log(futures["close"] / futures["close"].shift(1))

    returns = returns.dropna()
    return returns


def apply_hedge(
    returns: pd.DataFrame,
    hedge_ratios: pd.Series,
) -> pd.Series:
    """
    Apply a time-varying hedge ratio to compute the hedged portfolio return.

    Parameters
    ----------
    returns : DataFrame with columns spot_return and futures_return,
              indexed by date.
    hedge_ratios : Series of hedge ratios indexed by date.
                   Values should be between 0.0 (no hedge) and 1.0 (full hedge).
                   Dates not in the index default to 0.0 (no hedge).

    Returns
    -------
    hedged_returns : Series of daily hedged log returns.

    Formula
    -------
    hedged_return_t = spot_return_t - hedge_ratio_t × futures_return_t
    """
    # Align hedge ratios to the return index, fill missing dates with 0
    aligned_ratios = hedge_ratios.reindex(returns.index).fillna(0.0)

    hedged = (
        returns["spot_return"]
        - aligned_ratios * returns["futures_return"]
    )
    hedged.name = "hedged_return"
    return hedged


def compute_portfolio_value(
    returns: pd.Series,
    initial_value: float = 1_000_000.0,
) -> pd.Series:
    """
    Compute cumulative portfolio value from a log return series.

    Parameters
    ----------
    returns       : Series of daily log returns.
    initial_value : Starting portfolio value in ₹ (default ₹10,00,000).

    Returns
    -------
    portfolio_value : Series of daily portfolio values starting from
                      initial_value on the first date.

    Note: log returns are converted to simple returns for compounding:
        value_t = value_{t-1} × exp(log_return_t)
    """
    cumulative    = returns.cumsum().apply(np.exp)
    portfolio_val = cumulative * initial_value
    portfolio_val.name = "portfolio_value"
    return portfolio_val


def build_hedged_series(
    hedge_ratios: pd.Series,
    start_date: str = None,
    end_date: str = None,
    initial_value: float = 1_000_000.0,
) -> pd.DataFrame:
    """
    Convenience function — loads returns, applies hedge, computes portfolio value.

    Parameters
    ----------
    hedge_ratios  : Series of hedge ratios indexed by date.
    start_date    : Optional filter start date (inclusive), e.g. "2025-01-01".
    end_date      : Optional filter end date (inclusive), e.g. "2025-03-28".
    initial_value : Starting portfolio value in ₹.

    Returns
    -------
    DataFrame with columns:
        spot_return, futures_return, hedge_ratio,
        hedged_return, portfolio_value
    """
    returns = compute_returns()

    if start_date:
        returns = returns[returns.index >= start_date]
    if end_date:
        returns = returns[returns.index <= end_date]

    hedged_ret = apply_hedge(returns, hedge_ratios)
    port_val   = compute_portfolio_value(hedged_ret, initial_value)

    result = returns.copy()
    result["hedge_ratio"]     = hedge_ratios.reindex(returns.index).fillna(0.0)
    result["hedged_return"]   = hedged_ret
    result["portfolio_value"] = port_val
    return result