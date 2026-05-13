"""
fetch_data.py

Fetches historical OHLCV and open interest data from Kite Connect for the
Nifty 50 spot index, Nifty front-month futures (rolled monthly), and ten
Nifty 50 constituent stocks. All data is saved as CSV files in data/raw/.

Training period  : January 2023 to September 2024
Validation period: October 2024 to December 2024
Test period      : January 2025 to March 2025

Run from the project root:
    python src/data/fetch_data.py
"""

import os
import time
from datetime import date

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

if not API_KEY or not ACCESS_TOKEN:
    raise ValueError("KITE_API_KEY or KITE_ACCESS_TOKEN not found in .env file.")

FROM_DATE    = date(2023, 1, 1)
TO_DATE      = date(2025, 3, 31)
RAW_DATA_DIR = "data/raw"

# Portfolio chosen to represent a diversified cross-section of Nifty 50 sectors:
# financials (HDFC, ICICI, SBI, AXIS), technology (INFY, TCS), energy (RELIANCE),
# industrials (LT), telecom (BHARTIARTL), and FMCG (ITC).
STOCK_INSTRUMENTS = [
    ("NSE", "NIFTY 50",   "NIFTY50_spot"),
    ("NSE", "RELIANCE",   "RELIANCE"),
    ("NSE", "HDFCBANK",   "HDFCBANK"),
    ("NSE", "ICICIBANK",  "ICICIBANK"),
    ("NSE", "INFY",       "INFY"),
    ("NSE", "TCS",        "TCS"),
    ("NSE", "SBIN",       "SBIN"),
    ("NSE", "LT",         "LT"),
    ("NSE", "BHARTIARTL", "BHARTIARTL"),
    ("NSE", "ITC",        "ITC"),
    ("NSE", "AXISBANK",   "AXISBANK"),
]

# Kite Connect has no single continuous futures symbol, so we roll manually.
# Each entry covers the window where that contract is the active front-month.
# Roll dates are set to the last trading day of the expiring contract so we
# avoid fetching thin near-expiry volume into the next contract's window.
NIFTY_FUTURES_EXPIRIES = [
    ("NFO", "NIFTY23JANFUT", date(2023, 1,  1),  date(2023, 1,  26)),
    ("NFO", "NIFTY23FEBFUT", date(2023, 1,  27), date(2023, 2,  23)),
    ("NFO", "NIFTY23MARFUT", date(2023, 2,  24), date(2023, 3,  30)),
    ("NFO", "NIFTY23APRFUT", date(2023, 3,  31), date(2023, 4,  27)),
    ("NFO", "NIFTY23MAYFUT", date(2023, 4,  28), date(2023, 5,  25)),
    ("NFO", "NIFTY23JUNFUT", date(2023, 5,  26), date(2023, 6,  29)),
    ("NFO", "NIFTY23JULFUT", date(2023, 6,  30), date(2023, 7,  27)),
    ("NFO", "NIFTY23AUGFUT", date(2023, 7,  28), date(2023, 8,  31)),
    ("NFO", "NIFTY23SEPFUT", date(2023, 9,  1),  date(2023, 9,  28)),
    ("NFO", "NIFTY23OCTFUT", date(2023, 9,  29), date(2023, 10, 26)),
    ("NFO", "NIFTY23NOVFUT", date(2023, 10, 27), date(2023, 11, 30)),
    ("NFO", "NIFTY23DECFUT", date(2023, 12, 1),  date(2023, 12, 28)),
    ("NFO", "NIFTY24JANFUT", date(2023, 12, 29), date(2024, 1,  25)),
    ("NFO", "NIFTY24FEBFUT", date(2024, 1,  26), date(2024, 2,  29)),
    ("NFO", "NIFTY24MARFUT", date(2024, 3,  1),  date(2024, 3,  28)),
    ("NFO", "NIFTY24APRFUT", date(2024, 3,  29), date(2024, 4,  25)),
    ("NFO", "NIFTY24MAYFUT", date(2024, 4,  26), date(2024, 5,  30)),
    ("NFO", "NIFTY24JUNFUT", date(2024, 5,  31), date(2024, 6,  27)),
    ("NFO", "NIFTY24JULFUT", date(2024, 6,  28), date(2024, 7,  25)),
    ("NFO", "NIFTY24AUGFUT", date(2024, 7,  26), date(2024, 8,  29)),
    ("NFO", "NIFTY24SEPFUT", date(2024, 8,  30), date(2024, 9,  26)),
    ("NFO", "NIFTY24OCTFUT", date(2024, 9,  27), date(2024, 10, 31)),
    ("NFO", "NIFTY24NOVFUT", date(2024, 11, 1),  date(2024, 11, 28)),
    ("NFO", "NIFTY24DECFUT", date(2024, 11, 29), date(2024, 12, 26)),
    ("NFO", "NIFTY25JANFUT", date(2024, 12, 27), date(2025, 1,  30)),
    ("NFO", "NIFTY25FEBFUT", date(2025, 1,  31), date(2025, 2,  27)),
    ("NFO", "NIFTY25MARFUT", date(2025, 2,  28), date(2025, 3,  27)),
]


def get_instrument_token(kite, exchange, tradingsymbol):
    """Return the numeric instrument token for a given exchange and symbol."""
    instruments = kite.instruments(exchange)
    for inst in instruments:
        if inst["tradingsymbol"] == tradingsymbol:
            return inst["instrument_token"]
    return None


def fetch_historical(kite, instrument_token, from_date, to_date, oi=False):
    """
    Fetch daily OHLCV data from Kite Connect.
    oi=True is only meaningful for futures — equities and the spot index
    do not carry open interest in Kite's response.
    """
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval="day",
        continuous=False,
        oi=oi,
    )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_csv(df, name):
    """Save a DataFrame to data/raw/<name>.csv."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)")


def fetch_spot_and_stocks(kite):
    """Fetch Nifty 50 spot index and ten constituent stocks."""
    print("\nFetching spot index and stocks ...")
    for exchange, symbol, name in STOCK_INSTRUMENTS:
        print(f"  {symbol}", end=" ... ")
        token = get_instrument_token(kite, exchange, symbol)
        if token is None:
            print("WARNING: symbol not found, skipping.")
            continue
        df = fetch_historical(kite, token, FROM_DATE, TO_DATE, oi=False)
        if df.empty:
            print("WARNING: no data returned, skipping.")
            continue
        save_csv(df, name)
        time.sleep(0.4)  # Kite rate limit is ~3 requests/sec


def fetch_nifty_futures_rolling(kite):
    """
    Fetch Nifty futures contract by contract and stitch into a single
    rolling series. Each contract covers its active front-month window.
    Duplicate dates at roll boundaries are dropped keeping the earlier
    contract, which has better liquidity further from expiry.
    """
    print("\nFetching Nifty futures (rolling) ...")
    all_dfs = []

    for exchange, symbol, start, end in NIFTY_FUTURES_EXPIRIES:
        print(f"  {symbol} ({start} to {end})", end=" ... ")
        token = get_instrument_token(kite, exchange, symbol)
        if token is None:
            print("WARNING: symbol not found, skipping.")
            continue
        df = fetch_historical(kite, token, start, end, oi=True)
        if df.empty:
            print("WARNING: no data returned, skipping.")
            continue
        # Keep track of which contract each row came from — useful for
        # identifying roll dates during preprocessing
        df["contract"] = symbol
        all_dfs.append(df)
        print(f"{len(df)} rows")
        time.sleep(0.4)

    if not all_dfs:
        print("ERROR: no futures data fetched.")
        return

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset="date", keep="first")
        .reset_index(drop=True)
    )

    save_csv(combined, "NIFTY_futures_rolling")
    print(f"\n  Combined series: {len(combined)} rows")
    print(f"  Date range    : {combined['date'].min()} to {combined['date'].max()}")
    print(f"  Columns       : {list(combined.columns)}")


def run_sanity_check():
    """Print shape and boundary rows for every CSV in data/raw/."""
    print("\nSanity check ...")
    for f in sorted(os.listdir(RAW_DATA_DIR)):
        if not f.endswith(".csv"):
            continue
        path = os.path.join(RAW_DATA_DIR, f)
        df = pd.read_csv(path)
        print(f"\n  {f}")
        print(f"    Shape : {df.shape}")
        print(f"    Cols  : {list(df.columns)}")
        print(f"    First : {df.iloc[0].to_dict()}")
        print(f"    Last  : {df.iloc[-1].to_dict()}")


if __name__ == "__main__":
    print("Connecting to Kite Connect ...")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)
    profile = kite.profile()
    print(f"Connected as: {profile['user_name']} ({profile['email']})")

    fetch_spot_and_stocks(kite)
    fetch_nifty_futures_rolling(kite)
    run_sanity_check()

    print("\nAll raw data saved to data/raw/")