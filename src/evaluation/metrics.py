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
    """
    daily_rf       = risk_free_rate / TRADING_DAYS
    excess         = returns - daily_rf
    downside       = excess[excess < 0]
    downside_std   = downside.std()

    if downside_std == 0 or np.isnan(downside_std):
        return np.nan

    return float((excess.mean() / downside_std) * np.sqrt(TRADING_DAYS))

def downside_deviation(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> float:
    """
    Annualised downside deviation — standard deviation of returns
    below the daily risk-free rate only.
    Used as the denominator in the Sortino ratio.
    """
    daily_rf     = risk_free_rate / TRADING_DAYS
    excess       = returns - daily_rf
    downside     = excess[excess < 0]
    down_std     = downside.std()
    if np.isnan(down_std):
        return np.nan
    return float(down_std * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """
    Maximum drawdown — largest percentage decline from a peak to a subsequent trough.

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

    """
    annual_return = returns.mean() * TRADING_DAYS
    mdd           = max_drawdown(returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return float(annual_return / abs(mdd))


def hedge_ratio_stability(hedge_ratios: pd.Series) -> float:
    """
    Stability of the hedge ratio series — measured as standard deviation.

    """
    return float(hedge_ratios.std())


def var(
    returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """
    Value at Risk at given confidence level.

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
        "downside_deviation_pct": round(downside_deviation(returns, risk_free_rate) * 100, 2),
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
    print(f"  Downside Deviation : {metrics['downside_deviation_pct']:>8.2f} %")
    print(f"  Max Drawdown       : {metrics['max_drawdown_pct']:>8.2f} %")
    print(f"  CVaR (95%)         : {metrics['cvar_95_pct']:>8.2f} %")
    print(f"  VaR  (95%)         : {metrics['var_95_pct']:>8.2f} %")
    print(f"  Calmar Ratio       : {metrics['calmar']:>8.3f}")
    if "hedge_ratio_std" in metrics:
        print(f"  Hedge Ratio Std    : {metrics['hedge_ratio_std']:>8.4f}")
    print(f"{'─' * 44}")
