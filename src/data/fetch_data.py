"""
2012-01-01 to 2026-03-31

Fetches daily OHLCV + OI data for:
  - NIFTY 50 spot index (token 256265, continuous=False)
  - NIFTY 50 continuous futures (token 15956226, continuous=True, oi=True)
  - 10 constituent stocks (continuous=False)

Futures note:
    Token 15956226 (NIFTY26JUNFUT) with continuous=True returns the full
    stitched continuous series back to 2012. OI is available from 2012 Q2
    onwards (2012 Q1 returns 0 rows for futures — data starts mid-2012).
    No spot proxy needed for this date range.

"""

import os
import time
import datetime
import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect

load_dotenv()

API_KEY      = os.getenv("KITE_API_KEY")
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")
RAW_DIR      = "data/raw"

FROM_DATE              = datetime.date(2012, 1, 1)
TO_DATE                = datetime.date(2026, 3, 31)
NIFTY_CONTINUOUS_TOKEN = 15956226   # NIFTY26JUNFUT — works with continuous=True
NIFTY_SPOT_TOKEN       = 256265

STOCK_TOKENS = {
    "RELIANCE":   738561,
    "HDFCBANK":   341249,
    "ICICIBANK":  1270529,
    "INFY":       408065,
    "TCS":        2953217,
    "SBIN":       779521,
    "LT":         2939649,
    "BHARTIARTL": 2714625,
    "ITC":        424961,
    "AXISBANK":   1510401,
}

os.makedirs(RAW_DIR, exist_ok=True)


def fetch_ohlcv(kite, token, from_date, to_date, label,
                continuous=False, oi=False):
    """
    Fetch daily OHLCV in 365-day chunks.
    Returns a clean tz-naive DataFrame with columns:
        date, open, high, low, close, volume, oi
    """
    all_rows    = []
    chunk_start = from_date

    while chunk_start <= to_date:
        chunk_end = min(
            chunk_start + datetime.timedelta(days=364),
            to_date,
        )
        try:
            data = kite.historical_data(
                token, chunk_start, chunk_end, "day",
                continuous=continuous,
                oi=oi,
            )
            all_rows.extend(data)
            print(f"    {chunk_start} to {chunk_end}: {len(data)} rows")
        except Exception as e:
            print(f"    {chunk_start} to {chunk_end}: ERROR — {e}")

        chunk_start = chunk_end + datetime.timedelta(days=1)
        time.sleep(0.35)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if "oi" not in df.columns:
        df["oi"] = 0

    return df[["date", "open", "high", "low", "close", "volume", "oi"]]


if __name__ == "__main__":
    print("Connecting to Kite ...")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    # ── Nifty spot ────────────────────────────────────────────────────────────
    print(f"\nFetching NIFTY50_spot ({FROM_DATE} to {TO_DATE}) ...")
    spot_df = fetch_ohlcv(
        kite, NIFTY_SPOT_TOKEN,
        FROM_DATE, TO_DATE,
        label="NIFTY50_spot",
        continuous=False, oi=False,
    )
    if spot_df is not None:
        path = os.path.join(RAW_DIR, "NIFTY50_spot.csv")
        spot_df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(spot_df)} rows)")

    # ── Continuous futures ────────────────────────────────────────────────────
    print(f"\nFetching NIFTY_futures_continuous ({FROM_DATE} to {TO_DATE}) ...")
    fut_df = fetch_ohlcv(
        kite, NIFTY_CONTINUOUS_TOKEN,
        FROM_DATE, TO_DATE,
        label="NIFTY_futures_continuous",
        continuous=True, oi=True,
    )
    if fut_df is not None:
        path = os.path.join(RAW_DIR, "NIFTY_futures_continuous.csv")
        fut_df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(fut_df)} rows)")
        oi_nonzero = (fut_df["oi"] > 0).sum()
        print(f"  OI non-zero: {oi_nonzero}/{len(fut_df)} rows")

    # ── Stocks ────────────────────────────────────────────────────────────────
    for symbol, token in STOCK_TOKENS.items():
        print(f"\nFetching {symbol} ...")
        df = fetch_ohlcv(
            kite, token,
            FROM_DATE, TO_DATE,
            label=symbol,
            continuous=False, oi=False,
        )
        if df is not None:
            path = os.path.join(RAW_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved: {path}  ({len(df)} rows)")
        time.sleep(0.5)

    print("\nKite data fetch complete.")
    print(f"All files saved to {RAW_DIR}/")
    print(f"\nExpected: ~3,500 rows per instrument (2012-2026)")
    print("Next: update macro data then run build_features.py")