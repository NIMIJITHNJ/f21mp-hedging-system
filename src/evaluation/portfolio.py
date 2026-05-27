"""
Core portfolio and hedging logic for Hedra.

The instrument being hedged is the equal-weight portfolio of 10 Nifty 50
constituent stocks (RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, SBIN, LT,
BHARTIARTL, ITC, AXISBANK).

The hedging instrument is the Nifty 50 front-month futures contract.

The Nifty 50 spot index is used separately as:
  1. A feature input for the LSTM/GARCH risk model (market regime signal)
  2. A benchmark for portfolio performance comparison

The spot index is NOT the instrument being hedged.

Hedge formula:
    R_h(t) = R_portfolio(t) - h * R_futures(t)

where:
    R_portfolio(t)  = equal-weight average of the 10 stock log returns
    R_futures(t)    = log return of the Nifty continuous futures series
    h               = hedge ratio (scalar, determined by hedging strategy)
    R_h(t)          = hedged portfolio return

Run from the project root:
    python src/evaluation/portfolio.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Allow running directly from the project root: python src/evaluation/portfolio.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ── Constants ────────────────────────────────────────────────────────────────

RAW_DIR   = Path("data/raw")
PORTFOLIO = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]

SPLIT_DATES = {
    "train": ("2021-01-01", "2024-09-30"),
    "val":   ("2024-10-01", "2024-12-31"),
    "test":  ("2025-01-01", "2025-03-31"),
    "full":  ("2021-01-01", "2025-03-31"),
}

# ── Data loading ─────────────────────────────────────────────────────────────

def load_stock_prices() -> pd.DataFrame:
    """
    Load adjusted close prices for all 10 portfolio stocks.

    Returns a DataFrame indexed by date with one column per stock.
    Rows where any stock has a missing price are dropped so all series
    are perfectly aligned before computing returns.
    """
    frames = {}
    for sym in PORTFOLIO:
        path = RAW_DIR / f"{sym}.csv"
        df   = pd.read_csv(path, parse_dates=["date"])
        df   = df.sort_values("date").set_index("date")
        frames[sym] = df["close"]

    prices = pd.DataFrame(frames).dropna()
    return prices


def load_futures_prices() -> pd.Series:
    """
    Load the continuous Nifty futures close prices.
    This is the hedging instrument.

    Returns a Series indexed by date, named 'futures'.
    """
    path = RAW_DIR / "NIFTY_futures_continuous.csv"
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.sort_values("date").set_index("date")
    return df["close"].rename("futures")


def load_nifty_spot_prices() -> pd.Series:
    """
    Load Nifty 50 spot index close prices.

    Dual purpose:
      - Feature input: Nifty spot movement feeds into the LSTM/GARCH model
        as a market regime signal, informing hedge ratio decisions.
      - Benchmark: after backtesting, portfolio performance is compared
        against the Nifty 50 to measure alpha.

    The spot index is NOT the instrument being hedged.

    Returns a Series indexed by date, named 'nifty_spot'.
    """
    path = RAW_DIR / "NIFTY50_spot.csv"
    df   = pd.read_csv(path, parse_dates=["date"])
    df   = df.sort_values("date").set_index("date")
    return df["close"].rename("nifty_spot")


# ── Return computation ────────────────────────────────────────────────────────

def compute_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for each of the 10 portfolio stocks.
    First row (NaN) is dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_portfolio_returns(stock_returns: pd.DataFrame) -> pd.Series:
    """
    Compute daily equal-weight portfolio log returns.

    Each of the 10 stocks receives weight 1/10. The portfolio return
    on day t is the simple average of the 10 individual log returns.

    Returns a Series named 'portfolio'.
    """
    return stock_returns.mean(axis=1).rename("portfolio")


def compute_futures_returns(futures_prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns of the Nifty futures series.
    Returns a Series named 'futures'.
    """
    return np.log(futures_prices / futures_prices.shift(1)).dropna().rename("futures")


def compute_nifty_spot_returns(spot_prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns of the Nifty 50 spot index.
    Used as feature input and benchmark — not part of the hedge formula.
    Returns a Series named 'nifty_spot'.
    """
    return np.log(spot_prices / spot_prices.shift(1)).dropna().rename("nifty_spot")


# ── Hedged return series ──────────────────────────────────────────────────────

def apply_hedge(
    portfolio_returns: pd.Series,
    futures_returns: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """
    Apply a constant hedge ratio to produce hedged portfolio returns.

    Formula:
        R_h(t) = R_portfolio(t) - hedge_ratio * R_futures(t)

    The short futures position offsets systematic market exposure in the
    stock portfolio. When the market falls, futures gains partially
    compensate for portfolio losses.

    hedge_ratio = 0.0  →  no hedge (unhedged portfolio)
    hedge_ratio = 1.0  →  fully hedged

    Parameters
    ----------
    portfolio_returns : daily equal-weight portfolio log returns
    futures_returns   : daily Nifty futures log returns
    hedge_ratio       : scalar h in the hedge formula

    Returns
    -------
    Series of hedged portfolio returns, named 'hedged'.
    """
    aligned = pd.concat(
        [portfolio_returns, futures_returns], axis=1, join="inner"
    )
    hedged = aligned["portfolio"] - hedge_ratio * aligned["futures"]
    return hedged.rename("hedged")


def build_return_series(
    hedge_ratio: float,
    split: str = "full",
) -> pd.DataFrame:
    """
    Build a DataFrame with all return series aligned on common trading days.

    Columns:
        portfolio   - unhedged equal-weight portfolio returns (what is hedged)
        futures     - Nifty futures returns (hedging instrument)
        nifty_spot  - Nifty 50 spot returns (feature input + benchmark)
        hedged      - hedged portfolio returns at the given hedge_ratio

    Parameters
    ----------
    hedge_ratio : hedge ratio h to apply
    split       : 'full' | 'train' | 'val' | 'test'

    Returns
    -------
    DataFrame indexed by date with the four columns above.
    """
    # Load and compute returns
    stock_returns     = compute_stock_returns(load_stock_prices())
    portfolio_returns = compute_portfolio_returns(stock_returns)
    futures_returns   = compute_futures_returns(load_futures_prices())
    spot_returns      = compute_nifty_spot_returns(load_nifty_spot_prices())

    # Align all series on common trading days
    combined = pd.concat(
        [portfolio_returns, futures_returns, spot_returns],
        axis=1, join="inner"
    ).dropna()

    # Apply time-period split
    if split in SPLIT_DATES:
        start, end = SPLIT_DATES[split]
        combined = combined.loc[start:end]

    # Compute hedged portfolio returns
    hedged = (
        combined["portfolio"] - hedge_ratio * combined["futures"]
    ).rename("hedged")

    return combined.join(hedged)


def compute_portfolio_value(
    returns: pd.Series,
    initial_value: float = 1_000_000.0,
) -> pd.Series:
    """
    Convert a daily log return series into a cumulative portfolio value.

    Starts at initial_value (default ₹10,00,000) and compounds daily.

    Returns a Series of portfolio values indexed by date.
    """
    value = initial_value * np.exp(returns.cumsum())
    return value.rename("portfolio_value")


# ── Convenience wrapper ───────────────────────────────────────────────────────

def build_hedged_series(
    hedge_ratio: float,
    split: str = "test",
    initial_value: float = 1_000_000.0,
) -> dict:
    """
    Build all series needed for evaluation in one call.

    Returns a dict with keys:
        returns          - DataFrame (portfolio, futures, nifty_spot, hedged)
        portfolio_value  - cumulative value of the unhedged stock portfolio
        hedged_value     - cumulative value of the hedged stock portfolio
        nifty_value      - cumulative value of Nifty 50 benchmark (comparison)
        hedge_ratio      - the hedge ratio used
        split            - the data split used

    Note: portfolio_value and hedged_value reflect the 10-stock portfolio.
          nifty_value is purely a benchmark — it is not a tradeable position.
    """
    returns = build_return_series(hedge_ratio=hedge_ratio, split=split)

    return {
        "returns":         returns,
        "portfolio_value": compute_portfolio_value(
            returns["portfolio"], initial_value
        ),
        "hedged_value":    compute_portfolio_value(
            returns["hedged"], initial_value
        ),
        "nifty_value":     compute_portfolio_value(
            returns["nifty_spot"], initial_value
        ),
        "hedge_ratio":     hedge_ratio,
        "split":           split,
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building return series (full period, hedge_ratio=0.5) ...")
    result = build_hedged_series(hedge_ratio=0.5, split="full")
    r = result["returns"]

    print(f"\n  Date range      : {r.index[0].date()} to {r.index[-1].date()}")
    print(f"  Rows            : {len(r)}")
    print(f"  Columns         : {list(r.columns)}")
    print(f"\n  Portfolio return (mean daily) : {r['portfolio'].mean():.6f}")
    print(f"  Futures   return (mean daily) : {r['futures'].mean():.6f}")
    print(f"  Nifty spot return(mean daily) : {r['nifty_spot'].mean():.6f}")
    print(f"  Hedged   return (mean daily)  : {r['hedged'].mean():.6f}")
    print(f"\n  Portfolio cumulative value    : ₹{result['portfolio_value'].iloc[-1]:>12,.0f}")
    print(f"  Hedged    cumulative value    : ₹{result['hedged_value'].iloc[-1]:>12,.0f}")
    print(f"  Nifty benchmark value         : ₹{result['nifty_value'].iloc[-1]:>12,.0f}")
    print(f"\n  (Nifty benchmark is for comparison only — not a position in the portfolio)")