"""
fetch_data.py

Fetches historical OHLCV and open interest data from Kite Connect for the
Nifty 50 spot index, Nifty continuous futures, and ten Nifty 50 constituent
stocks. All data is saved as CSV files in data/raw/.

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

# Instrument token for the current Nifty front-month futures contract.
# Confirmed from check_symbols.py — NIFTY26MAYFUT token=16914178.
# Kite's continuous=True flag stitches all expired contracts automatically
# when fetching historical data, so no manual rolling is needed.
NIFTY_CONTINUOUS_TOKEN = 16914178


def get_instrument_token(kite, exchange, tradingsymbol):
    """Return the numeric instrument token for a given exchange and symbol."""
    instruments = kite.instruments(exchange)
    for inst in instruments:
        if inst["tradingsymbol"] == tradingsymbol:
            return inst["instrument_token"]
    return None


def fetch_historical(kite, instrument_token, from_date, to_date,
                     oi=False, continuous=False):
    """
    Fetch daily OHLCV data from Kite Connect.
    oi=True includes open interest — only meaningful for futures.
    continuous=True stitches expired contracts into one unbroken series,
    which is how we get full historical futures data without needing
    individual expiry tokens.
    """
    data = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval="day",
        continuous=continuous,
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


def fetch_nifty_futures_continuous(kite):
    """
    Fetch a full continuous Nifty futures series using Kite's continuous
    data feature. Passing continuous=True with the current front-month token
    causes Kite to automatically stitch all expired contracts into one
    unbroken daily series going back several years, including OI data.
    This avoids the need to fetch and stitch individual monthly expiries.
    """
    print("\nFetching Nifty futures (continuous) ...")

    df = fetch_historical(
        kite,
        instrument_token=NIFTY_CONTINUOUS_TOKEN,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        oi=True,
        continuous=True,
    )

    if df.empty:
        print("ERROR: no data returned.")
        return

    save_csv(df, "NIFTY_futures_continuous")
    print(f"  Date range : {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns    : {list(df.columns)}")


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
    fetch_nifty_futures_continuous(kite)
    run_sanity_check()

    print("\nAll raw data saved to data/raw/")