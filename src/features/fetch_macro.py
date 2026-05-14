"""
fetch_macro.py

Fetches macroeconomic time series and saves them to data/external/.
These series are used as additional input features in the LSTM-GARCH model.

Sources:
    - Gold price in USD        : yfinance (GC=F)
    - Brent crude in USD       : yfinance (BZ=F)
    - USD/INR exchange rate    : yfinance (INR=X)
    - India 10Y bond yield (%) : investing.com CSV download
                                 (place at data/external/raw_india_10y.csv)

All series are aligned to the 557 NSE trading days (2023-01-02 to 2025-03-28)
using forward-fill for any calendar gaps.

Run from the project root:
    python src/features/fetch_macro.py
"""

import os
import pandas as pd
import yfinance as yf

EXTERNAL_DIR  = "data/external"
FROM_DATE     = "2023-01-01"
TO_DATE       = "2025-03-31"
REFERENCE_CSV = "data/raw/NIFTY50_spot.csv"

# Path where the investing.com bond yield CSV should be placed.
# Download from: https://in.investing.com/rates-bonds/india-10-year-bond-yield-historical-data
BOND_YIELD_CSV = "data/external/raw_india_10y.csv"

YFINANCE_SERIES = [
    ("GC=F",  "gold_usd",  "Gold price (USD/oz)"),
    ("BZ=F",  "brent_usd", "Brent crude (USD/barrel)"),
    ("INR=X", "usdinr",    "USD/INR exchange rate"),
]


def fetch_yfinance_series(ticker, name, description):
    """Download a single yfinance series and return a named Series by date."""
    print(f"  Fetching {description} ({ticker}) ...", end=" ")
    try:
        raw = yf.download(
            ticker,
            start=FROM_DATE,
            end=TO_DATE,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            print("WARNING: no data returned.")
            return None

        close = raw["Close"][ticker] if isinstance(raw.columns, pd.MultiIndex) \
                else raw["Close"]

        close.index = pd.to_datetime(close.index).normalize()
        close.name  = name
        print(f"{len(close)} rows")
        return close

    except Exception as e:
        print(f"ERROR: {e}")
        return None


def load_bond_yield(path):
    """
    Parse the investing.com India 10Y bond yield CSV.
    The file uses DD-MM-YYYY date format and a 'Price' column for the yield.
    """
    print(f"  Loading India 10Y bond yield from {path} ...", end=" ")
    if not os.path.exists(path):
        print(f"WARNING: file not found at {path}")
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"]  = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df = df.sort_values("Date").set_index("Date")

    series       = df["Price"].rename("india_10y")
    series.index = pd.DatetimeIndex(series.index).normalize()
    print(f"{len(series)} rows  ({series.index[0].date()} to {series.index[-1].date()})")
    return series


def align_to_reference(series_dict, reference_dates):
    """
    Reindex all macro series to the NSE trading calendar.
    Forward-fill gaps then back-fill any leading NaNs.
    """
    ref_idx = pd.DatetimeIndex(reference_dates)
    aligned = {}
    for name, series in series_dict.items():
        reindexed = (
            series
            .reindex(ref_idx.union(series.index))
            .sort_index()
            .ffill()
            .bfill()
            .reindex(ref_idx)
        )
        aligned[name] = reindexed
    return aligned


def save_csv(df, name):
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    path = os.path.join(EXTERNAL_DIR, f"{name}.csv")
    df.to_csv(path)
    print(f"  Saved {path} ({len(df)} rows)")


if __name__ == "__main__":
    # Load reference trading dates
    ref_df    = pd.read_csv(REFERENCE_CSV, parse_dates=["date"])
    ref_dates = pd.DatetimeIndex(ref_df["date"])
    print(f"Reference trading days: {len(ref_dates)} "
          f"({ref_dates[0].date()} to {ref_dates[-1].date()})\n")

    series_dict = {}

    # Fetch yfinance series
    print("Fetching yfinance series ...")
    for ticker, name, description in YFINANCE_SERIES:
        s = fetch_yfinance_series(ticker, name, description)
        if s is not None:
            series_dict[name] = s

    # Load bond yield from local CSV
    print("\nLoading bond yield ...")
    bond = load_bond_yield(BOND_YIELD_CSV)
    if bond is not None:
        series_dict["india_10y"] = bond

    # Align all series to NSE calendar
    print("\nAligning to NSE trading calendar ...")
    aligned = align_to_reference(series_dict, ref_dates)

    print("\nAlignment check:")
    for name, series in aligned.items():
        missing = series.isna().sum()
        status  = "OK" if missing == 0 else f"WARNING: {missing} NaN(s)"
        print(f"  {name:<15} {len(series)} rows  {status}")

    # Save individual CSVs and combined file
    print("\nSaving ...")
    for name, series in aligned.items():
        df = series.reset_index()
        df.columns = ["date", name]
        save_csv(df.set_index("date"), name)

    combined            = pd.DataFrame(aligned)
    combined.index.name = "date"
    save_csv(combined, "macro_combined")

    print(f"\nAll macro data saved to {EXTERNAL_DIR}/")
    print("\nPreview — first 3 rows:")
    print(combined.head(3).to_string())
    print("\nPreview — last 3 rows:")
    print(combined.tail(3).to_string())