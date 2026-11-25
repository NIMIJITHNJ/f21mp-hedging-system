# Updates macro_combined.csv to cover 2018-01-01 to 2025-03-28.

import os
import numpy as np
import pandas as pd
import yfinance as yf

EXTERNAL_DIR = "data/external"
RAW_DIR      = "data/raw"

FROM_DATE = "2012-01-01"
TO_DATE   = "2026-03-31"

os.makedirs(EXTERNAL_DIR, exist_ok=True)


def fetch_yfinance(ticker, col_name, from_date, to_date):
    """Fetch daily close from yfinance, return a named Series."""
    print(f"  Fetching {ticker} ...")
    try:
        df = yf.download(ticker, start=from_date, end=to_date,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"    WARNING: No data returned for {ticker}")
            return None
        series = df["Close"].squeeze()
        series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
        series.name = col_name
        print(f"    OK — {len(series)} rows "
              f"({series.index[0].date()} to {series.index[-1].date()})")
        return series
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def load_bond_yield():
    """
    Load India 10Y bond yield from the existing CSV.
    Extends backward if needed by forward-filling from the earliest value.
    """
    bond_path = os.path.join(EXTERNAL_DIR, "raw_india_10y.csv")
    if not os.path.exists(bond_path):
        print("  WARNING: India 10Y bond CSV not found. Using constant 6.5%.")
        idx = pd.date_range(FROM_DATE, TO_DATE, freq="B")
        return pd.Series(6.5 / 100, index=idx, name="india_10y")

    print(f"  Loading India 10Y bond from CSV ...")
    df = pd.read_csv(bond_path)

    # Handle different column name formats
    date_col  = [c for c in df.columns if "date" in c.lower() or "Date" in c][0]
    price_col = [c for c in df.columns if "price" in c.lower() or "Price" in c or "%" in c][0]

    df[date_col]  = pd.to_datetime(df[date_col], dayfirst=True)
    df[price_col] = pd.to_numeric(
        df[price_col].astype(str).str.replace("%", "").str.replace(",", ""),
        errors="coerce"
    ) / 100   # convert % to decimal

    series = df.set_index(date_col)[price_col].sort_index()
    series.index = series.index.tz_localize(None).normalize()
    series.name = "india_10y"

    earliest_date = series.index.min()
    print(f"    Bond data range: {earliest_date.date()} to {series.index.max().date()}")

    # If bond data starts after 2018, backfill with earliest available value
    target_start = pd.Timestamp(FROM_DATE)
    if earliest_date > target_start:
        print(f"    Bond data starts at {earliest_date.date()} — "
              f"backfilling to {FROM_DATE} with value {series.iloc[0]:.4f}")
        backfill_idx = pd.date_range(FROM_DATE, earliest_date - pd.Timedelta(days=1), freq="B")
        backfill     = pd.Series(series.iloc[0], index=backfill_idx, name="india_10y")
        series       = pd.concat([backfill, series]).sort_index()

    return series


if __name__ == "__main__":
    print(f"Fetching macro data ({FROM_DATE} to {TO_DATE}) ...\n")

    # yfinance sources
    gold   = fetch_yfinance("GC=F",  "gold_usd",  FROM_DATE, TO_DATE)
    brent  = fetch_yfinance("BZ=F",  "brent_usd", FROM_DATE, TO_DATE)
    usdinr = fetch_yfinance("INR=X", "usdinr",    FROM_DATE, TO_DATE)

    # India 10Y bond
    print("\n  Loading India 10Y bond yield ...")
    bond = load_bond_yield()

    # Load NSE calendar from spot data
    spot_path = os.path.join(RAW_DIR, "NIFTY50_spot.csv")
    spot = pd.read_csv(spot_path, parse_dates=["date"])
    spot["date"] = pd.to_datetime(spot["date"]).dt.tz_localize(None).dt.normalize()
    nse_dates = spot.set_index("date").index

    print(f"\nAligning to NSE calendar ({len(nse_dates)} trading days) ...")

    # Combine all macro series
    macro = pd.DataFrame(index=nse_dates)
    macro.index.name = "date"

    for series, name in [(gold, "gold_usd"), (brent, "brent_usd"),
                         (usdinr, "usdinr"), (bond, "india_10y")]:
        if series is None:
            macro[name] = np.nan
        else:
            macro[name] = series.reindex(nse_dates, method="ffill")


    macro = macro.ffill().bfill()

    nan_counts = macro.isna().sum()
    if nan_counts.any():
        print(f"  WARNING: NaNs remaining after fill:\n{nan_counts[nan_counts > 0]}")
    else:
        print("  All NaNs filled successfully.")

    print(f"\n  Macro matrix shape: {macro.shape}")
    print(f"  Date range: {macro.index[0].date()} to {macro.index[-1].date()}")
    print(f"  Columns: {list(macro.columns)}")
    print(macro.tail(3))

    # Save
    out_path = os.path.join(EXTERNAL_DIR, "macro_combined.csv")
    macro.to_csv(out_path)
    print(f"\n  Saved: {out_path}  ({len(macro)} rows)")
    print("\nMacro data fetch complete.")
    print("Next: rerun src/features/build_features.py with new split dates.")
