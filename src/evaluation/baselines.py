"""
baselines.py

Two baseline hedging strategies used as benchmarks in the Hedra evaluation:

1. No Hedge       — hedge ratio fixed at 0.0 throughout (pure spot exposure)
2. Static Hedge   — hedge ratio fixed at 0.5 throughout (standard 50% hedge)
3. VECM Optimal   — hedge ratio fixed at 0.8824 (Vasantha & Mallikarjunappa 2018)

These baselines are compared against the dynamic LSTM-GARCH-PDDL strategy
in Week 9 evaluation. A good dynamic strategy should outperform all three
on risk-adjusted metrics (Sharpe, Sortino, CVaR) while maintaining
competitive returns.

Functions:
    no_hedge_baseline()       — returns metrics for 0% hedge strategy
    static_hedge_baseline()   — returns metrics for 50% hedge strategy
    vecm_hedge_baseline()     — returns metrics for 88.24% hedge strategy
    run_all_baselines()       — runs all three and prints comparison table
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from evaluation.portfolio import compute_returns, apply_hedge, compute_portfolio_value
from evaluation.metrics import summarise, print_summary


# VECM optimal hedge ratio from Vasantha & Mallikarjunappa (2018)
VECM_HEDGE_RATIO = 0.8824

# Test period — used for evaluating out-of-sample performance
TEST_START = "2025-01-01"
TEST_END   = "2025-03-28"


def _make_constant_ratios(returns: pd.DataFrame, ratio: float) -> pd.Series:
    """Create a constant hedge ratio series aligned to the returns index."""
    return pd.Series(ratio, index=returns.index, name="hedge_ratio")


def no_hedge_baseline(
    returns: pd.DataFrame = None,
    period: str = "full",
) -> dict:
    """
    Baseline 1 — No hedge (hedge ratio = 0.0 throughout).
    Equivalent to holding the Nifty 50 spot index with no futures position.

    Parameters
    ----------
    returns : Pre-loaded returns DataFrame. If None, loads from data/raw/.
    period  : 'full' (Jan 2021 - Mar 2025) or 'test' (Jan - Mar 2025).

    Returns
    -------
    dict of performance metrics.
    """
    if returns is None:
        returns = compute_returns()
    if period == "test":
        returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]

    hedge_ratios = _make_constant_ratios(returns, 0.0)
    hedged_ret   = apply_hedge(returns, hedge_ratios)
    return summarise(hedged_ret, hedge_ratios=hedge_ratios, label="No Hedge (ratio=0.00)")


def static_hedge_baseline(
    returns: pd.DataFrame = None,
    period: str = "full",
    ratio: float = 0.5,
) -> dict:
    """
    Baseline 2 — Static 50% hedge (hedge ratio = 0.5 throughout).
    Represents a naive equal-split between hedged and unhedged exposure.

    Parameters
    ----------
    returns : Pre-loaded returns DataFrame. If None, loads from data/raw/.
    period  : 'full' or 'test'.
    ratio   : Fixed hedge ratio (default 0.5).

    Returns
    -------
    dict of performance metrics.
    """
    if returns is None:
        returns = compute_returns()
    if period == "test":
        returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]

    hedge_ratios = _make_constant_ratios(returns, ratio)
    hedged_ret   = apply_hedge(returns, hedge_ratios)
    label        = f"Static Hedge (ratio={ratio:.2f})"
    return summarise(hedged_ret, hedge_ratios=hedge_ratios, label=label)


def vecm_hedge_baseline(
    returns: pd.DataFrame = None,
    period: str = "full",
) -> dict:
    """
    Baseline 3 — VECM optimal hedge (hedge ratio = 0.8824 throughout).
    Uses the cointegration-based optimal hedge ratio from the literature
    (Vasantha & Mallikarjunappa, 2018) as a constant — no dynamic adjustment.

    This is the most important baseline since the Hedra system's dynamic
    strategy should beat it by adjusting the ratio based on predicted risk.

    Parameters
    ----------
    returns : Pre-loaded returns DataFrame. If None, loads from data/raw/.
    period  : 'full' or 'test'.

    Returns
    -------
    dict of performance metrics.
    """
    if returns is None:
        returns = compute_returns()
    if period == "test":
        returns = returns[(returns.index >= TEST_START) & (returns.index <= TEST_END)]

    hedge_ratios = _make_constant_ratios(returns, VECM_HEDGE_RATIO)
    hedged_ret   = apply_hedge(returns, hedge_ratios)
    return summarise(hedged_ret, hedge_ratios=hedge_ratios,
                     label=f"VECM Hedge (ratio={VECM_HEDGE_RATIO})")


def run_all_baselines(period: str = "full") -> pd.DataFrame:
    """
    Run all three baselines and return a comparison DataFrame.

    Parameters
    ----------
    period : 'full' (Jan 2021 - Mar 2025) or 'test' (Jan - Mar 2025).

    Returns
    -------
    DataFrame with one row per strategy and one column per metric.
    """
    returns = compute_returns()

    results = [
        no_hedge_baseline(returns, period),
        static_hedge_baseline(returns, period),
        vecm_hedge_baseline(returns, period),
    ]

    df = pd.DataFrame(results).set_index("label")
    return df


def print_comparison(period: str = "full") -> None:
    """Print a formatted side-by-side comparison of all baselines."""
    label = "Full Period (Jan 2021 – Mar 2025)" if period == "full" \
            else "Test Period (Jan – Mar 2025)"

    print(f"\n{'═' * 66}")
    print(f"  Baseline Comparison — {label}")
    print(f"{'═' * 66}")

    returns = compute_returns()

    baselines = [
        no_hedge_baseline(returns, period),
        static_hedge_baseline(returns, period),
        vecm_hedge_baseline(returns, period),
    ]

    for m in baselines:
        print_summary(m)

    # Summary table
    df = pd.DataFrame(baselines).set_index("label")
    cols = ["total_return_pct", "annual_return_pct", "volatility_pct",
            "sharpe", "sortino", "max_drawdown_pct", "cvar_95_pct"]
    print(f"\n{'─' * 66}")
    print("  Summary Table")
    print(f"{'─' * 66}")
    print(df[cols].to_string())
    print(f"{'─' * 66}\n")


if __name__ == "__main__":
    print_comparison(period="full")
    print_comparison(period="test")