import sys
from pathlib import Path

# Allow running directly from the project root: python src/evaluation/baselines.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.evaluation.portfolio import build_return_series, compute_portfolio_value
from src.evaluation.metrics   import summarise, print_summary

# Hedge ratio constants
RATIO_NO_HEDGE    = 0.00
RATIO_STATIC_50   = 0.50

# baseline for the stock portfolio hedge.
RATIO_VECM        = 0.8824

RISK_FREE_RATE    = 0.065   # India 91-day T-bill rate (annualised)

# Baseline evaluation

def evaluate_baseline(
    hedge_ratio: float,
    label: str,
    split: str = "full",
) -> dict:
    """
    Evaluate a single constant hedge ratio strategy.

    Parameters
    ----------
    hedge_ratio : the constant h applied throughout the period
    label       : human-readable name for this strategy
    split       : 'full' | 'train' | 'val' | 'test'

    Returns
    -------
    dict containing label, hedge_ratio, split, metrics summary, and
    the full return series.
    """
    returns = build_return_series(hedge_ratio=hedge_ratio, split=split)

    # Hedged portfolio returns — this is what we evaluate
    hedged_returns = returns["hedged"]

    # Compute performance metrics
    metrics = summarise(hedged_returns, risk_free_rate=RISK_FREE_RATE)

    # Also compute unhedged and Nifty benchmark metrics for comparison
    portfolio_metrics = summarise(
        returns["portfolio"], risk_free_rate=RISK_FREE_RATE
    )
    nifty_metrics = summarise(
        returns["nifty_spot"], risk_free_rate=RISK_FREE_RATE
    )

    return {
        "label":              label,
        "hedge_ratio":        hedge_ratio,
        "split":              split,
        "metrics":            metrics,
        "portfolio_metrics":  portfolio_metrics,   # unhedged stock portfolio
        "nifty_metrics":      nifty_metrics,        # Nifty benchmark (not hedged)
        "returns":            returns,
    }


def run_all_baselines(split: str = "full") -> list:
    """
    Run all three baselines and return a list of result dicts.
    """
    baselines = [
        (RATIO_NO_HEDGE,  "Baseline 1: No Hedge"),
        (RATIO_STATIC_50, "Baseline 2: Static 50% Hedge"),
        (RATIO_VECM,      "Baseline 3: VECM Static Hedge (Vasantha & Mallikarjunappa)"),
    ]

    results = []
    for ratio, label in baselines:
        result = evaluate_baseline(ratio, label, split=split)
        results.append(result)

    return results


def print_baseline_comparison(results: list) -> None:
    """
    Print a side-by-side comparison of all baseline strategies.
    """
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON — 10-STOCK EQUAL-WEIGHT PORTFOLIO")
    print("Hedging instrument: Nifty 50 futures")
    print("Benchmark:          Nifty 50 spot index (not hedged)")
    print("=" * 70)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"  {r['label']}")
        print(f"  Hedge ratio : {r['hedge_ratio']:.4f}  |  Period: {r['split']}")
        print(f"{'─' * 70}")
        print_summary(r["metrics"])

    # Print unhedged portfolio and Nifty benchmark once (same for all baselines)
    ref = results[0]
    print(f"\n{'─' * 70}")
    print("  REFERENCE: Unhedged stock portfolio")
    print(f"{'─' * 70}")
    print_summary(ref["portfolio_metrics"])

    print(f"\n{'─' * 70}")
    print("  BENCHMARK: Nifty 50 spot index (comparison only — not a position)")
    print(f"{'─' * 70}")
    print_summary(ref["nifty_metrics"])

    print("\n" + "=" * 70)


# Entry point

if __name__ == "__main__":
    print("\nRunning baselines on FULL period (Jan 2021 – Mar 2025) ...")
    full_results = run_all_baselines(split="full")
    print_baseline_comparison(full_results)

    print("\nRunning baselines on TEST period (Jan 2025 – Mar 2025) ...")
    test_results = run_all_baselines(split="test")
    print_baseline_comparison(test_results)