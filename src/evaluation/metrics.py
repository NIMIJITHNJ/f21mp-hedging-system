"""
metrics.py

Evaluation metrics for comparing hedging strategies.
All functions take a Series of daily returns and return scalar values.

Metrics implemented:
    sharpe_ratio()          — annualised risk-adjusted return
    sortino_ratio()         — downside-only Sharpe
    max_drawdown()          — largest peak-to-trough decline
    cvar()                  — Conditional Value at Risk (Expected Shortfall)
    volatility()            — annualised standard deviation of returns
    calmar_ratio()          — annualised return / max drawdown
    hedge_ratio_stability() — std deviation of hedge ratio series
    summarise()             — all metrics in one dict
"""

import numpy as np
import pandas as pd


# Trading days per year — used for annualisation
TRADING_DAYS = 252


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    returns        : Series of daily log returns.
    risk_free_rate : Annual risk-free rate (default 6.5% — approx India 91-day T-bill).

    Formula
    -------
    Sharpe = (mean_daily_return - daily_rf) / std_daily_return × sqrt(252)
    """
    if returns.std() == 0:
        return np.nan
    daily_rf = risk_free_rate / TRADING_DAYS
    excess   = returns - daily_rf
    return float((excess.mean() / returns.std()) * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> float:
    """
    Annualised Sortino ratio — penalises only downside volatility.

    Parameters
    ----------
    returns        : Series of daily log returns.
    risk_free_rate : Annual risk-free rate (default 6.5%).

    Formula
    -------
    Sortino = (mean_daily_return - daily_rf) / downside_std × sqrt(252)
    where downside_std = std of returns below the risk-free rate only.
    """
    daily_rf       = risk_free_rate / TRADING_DAYS
    excess         = returns - daily_rf
    downside       = excess[excess < 0]
    downside_std   = downside.std()

    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return float((excess.mean() / downside_std) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown — largest percentage decline from a peak to a subsequent trough.

    Returns a negative float (e.g. -0.12 means 12% drawdown).

    Formula
    -------
    wealth_index = cumulative product of (1 + simple_return)
    drawdown     = wealth_index / rolling_max(wealth_index) - 1
    max_drawdown = min(drawdown)
    """
    # Convert log returns to simple returns for wealth index
    simple       = np.exp(returns) - 1
    wealth       = (1 + simple).cumprod()
    rolling_max  = wealth.cummax()
    drawdown     = wealth / rolling_max - 1
    return float(drawdown.min())


def cvar(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Conditional Value at Risk (CVaR) / Expected Shortfall at given confidence level.

    CVaR is the mean of losses beyond the VaR threshold — a more informative
    tail risk measure than VaR alone.

    Parameters
    ----------
    returns    : Series of daily log returns.
    confidence : Confidence level (default 0.95 → 95% CVaR).

    Returns a negative float representing the expected loss in the worst
    (1 - confidence) fraction of days.
    """
    threshold = returns.quantile(1 - confidence)
    tail      = returns[returns <= threshold]
    if tail.empty:
        return np.nan
    return float(tail.mean())


def volatility(returns: pd.Series) -> float:
    """
    Annualised volatility — standard deviation of daily log returns × sqrt(252).
    """
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar ratio — annualised return divided by absolute max drawdown.
    Higher is better. Common in hedge fund performance evaluation.
    """
    annual_return = returns.mean() * TRADING_DAYS
    mdd           = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(annual_return / abs(mdd))


def hedge_ratio_stability(hedge_ratios: pd.Series) -> float:
    """
    Stability of the hedge ratio series — measured as standard deviation.
    A lower value indicates a more stable, consistent hedging strategy.
    A higher value indicates frequent large adjustments (turnover cost risk).
    """
    return float(hedge_ratios.std())


def var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Value at Risk at given confidence level.
    Returns the loss threshold below which (1 - confidence) of daily returns fall.
    """
    return float(returns.quantile(1 - confidence))


def summarise(
    returns: pd.Series,
    hedge_ratios: pd.Series = None,
    risk_free_rate: float = 0.065,
    label: str = "Strategy",
) -> dict:
    """
    Compute all metrics and return as a dictionary.

    Parameters
    ----------
    returns        : Series of daily log returns.
    hedge_ratios   : Optional Series of hedge ratios (for stability metric).
    risk_free_rate : Annual risk-free rate for Sharpe/Sortino.
    label          : Strategy name for display.

    Returns
    -------
    Dictionary with keys:
        label, total_return_pct, annual_return_pct, volatility_pct,
        sharpe, sortino, max_drawdown_pct, cvar_95_pct, var_95_pct,
        calmar, hedge_ratio_std (if hedge_ratios provided)
    """
    annual_ret = returns.mean() * TRADING_DAYS
    total_ret  = float(np.exp(returns.sum()) - 1)

    result = {
        "label":              label,
        "total_return_pct":   round(total_ret * 100, 2),
        "annual_return_pct":  round(annual_ret * 100, 2),
        "volatility_pct":     round(volatility(returns) * 100, 2),
        "sharpe":             round(sharpe_ratio(returns, risk_free_rate), 3),
        "sortino":            round(sortino_ratio(returns, risk_free_rate), 3),
        "max_drawdown_pct":   round(max_drawdown(returns) * 100, 2),
        "cvar_95_pct":        round(cvar(returns) * 100, 2),
        "var_95_pct":         round(var(returns) * 100, 2),
        "calmar":             round(calmar_ratio(returns), 3),
    }

    if hedge_ratios is not None:
        result["hedge_ratio_std"] = round(hedge_ratio_stability(hedge_ratios), 4)

    return result


def print_summary(metrics: dict) -> None:
    """Pretty-print a metrics summary dictionary."""
    print(f"\n{'─' * 44}")
    print(f"  {metrics['label']}")
    print(f"{'─' * 44}")
    print(f"  Total Return       : {metrics['total_return_pct']:>8.2f} %")
    print(f"  Annual Return      : {metrics['annual_return_pct']:>8.2f} %")
    print(f"  Volatility         : {metrics['volatility_pct']:>8.2f} %")
    print(f"  Sharpe Ratio       : {metrics['sharpe']:>8.3f}")
    print(f"  Sortino Ratio      : {metrics['sortino']:>8.3f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown_pct']:>8.2f} %")
    print(f"  CVaR (95%)         : {metrics['cvar_95_pct']:>8.2f} %")
    print(f"  VaR  (95%)         : {metrics['var_95_pct']:>8.2f} %")
    print(f"  Calmar Ratio       : {metrics['calmar']:>8.3f}")
    if "hedge_ratio_std" in metrics:
        print(f"  Hedge Ratio Std    : {metrics['hedge_ratio_std']:>8.4f}")
    print(f"{'─' * 44}")