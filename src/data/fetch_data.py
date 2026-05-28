# extended version — 2018-01-01 to 2025-03-28

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

FROM_DATE = datetime.date(2018, 1, 1)
TO_DATE   = datetime.date(2025, 3, 28)

FUTURES_CONTINUOUS_FROM = datetime.date(2021, 1, 1)
NIFTY_CONTINUOUS_TOKEN  = 16914178
NIFTY_SPOT_TOKEN        = 256265

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


def fetch_ohlcv(kite, token, from_date, to_date, label, oi=False):
    """Fetch daily OHLCV in 365-day chunks. Returns a clean DataFrame."""
    all_rows = []
    chunk_start = from_date

    while chunk_start <= to_date:
        chunk_end = min(
            chunk_start + datetime.timedelta(days=364),
            to_date,
        )
        try:
            data = kite.historical_data(token, chunk_start, chunk_end, "day", oi=oi)
            all_rows.extend(data)
            print(f"    {chunk_start} to {chunk_end}: {len(data)} rows")
        except Exception as e:
            print(f"    {chunk_start} to {chunk_end}: ERROR — {e}")

        chunk_start = chunk_end + datetime.timedelta(days=1)
        time.sleep(0.35)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    # Strip timezone so all dates are tz-naive for consistent comparison
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    if "oi" not in df.columns:
        df["oi"] = 0

    return df[["date", "open", "high", "low", "close", "volume", "oi"]]


def build_futures_series(kite, spot_df):
    """
    Build a continuous futures series for the full date range.
    2018-2020: spot proxy (OI=0)
    2021-2025: actual continuous futures token
    """
    print("\n  Building futures series ...")

    cutoff = pd.Timestamp(FUTURES_CONTINUOUS_FROM)  # tz-naive Timestamp

    # Proxy segment — spot data before 2021
    proxy_df = spot_df[spot_df["date"] < cutoff].copy()
    proxy_df["oi"] = 0
    print(f"    Proxy segment (spot): {len(proxy_df)} rows "
          f"({proxy_df['date'].min().date()} to {proxy_df['date'].max().date()})")

    # Actual futures segment — 2021 onwards
    print(f"    Fetching continuous futures from {FUTURES_CONTINUOUS_FROM} ...")
    fut_df = fetch_ohlcv(
        kite, NIFTY_CONTINUOUS_TOKEN,
        FUTURES_CONTINUOUS_FROM, TO_DATE,
        label="NIFTY_futures_continuous",
        oi=True,
    )

    if fut_df is None:
        print("    WARNING: Could not fetch futures. Using full spot proxy.")
        full_proxy = spot_df.copy()
        full_proxy["oi"] = 0
        return full_proxy

    print(f"    Futures segment: {len(fut_df)} rows "
          f"({fut_df['date'].min().date()} to {fut_df['date'].max().date()})")

    combined = pd.concat([proxy_df, fut_df], ignore_index=True)
    combined = combined.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    print(f"    Combined futures series: {len(combined)} rows")
    return combined


if __name__ == "__main__":
    print("Connecting to Kite ...")
    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(ACCESS_TOKEN)

    # ── Nifty spot ────────────────────────────────────────────────────────────
    print(f"\nFetching NIFTY50_spot ({FROM_DATE} to {TO_DATE}) ...")
    spot_df = fetch_ohlcv(kite, NIFTY_SPOT_TOKEN, FROM_DATE, TO_DATE,
                          label="NIFTY50_spot", oi=False)
    if spot_df is not None:
        path = os.path.join(RAW_DIR, "NIFTY50_spot.csv")
        spot_df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(spot_df)} rows)")

    # ── Futures (proxy + continuous) ──────────────────────────────────────────
    print(f"\nBuilding NIFTY_futures_continuous ...")
    fut_df = build_futures_series(kite, spot_df)
    if fut_df is not None:
        path = os.path.join(RAW_DIR, "NIFTY_futures_continuous.csv")
        fut_df.to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(fut_df)} rows)")

    # ── Stocks ────────────────────────────────────────────────────────────────
    for symbol, token in STOCK_TOKENS.items():
        print(f"\nFetching {symbol} ...")
        df = fetch_ohlcv(kite, token, FROM_DATE, TO_DATE, label=symbol, oi=False)
        if df is not None:
            path = os.path.join(RAW_DIR, f"{symbol}.csv")
            df.to_csv(path, index=False)
            print(f"  Saved: {path}  ({len(df)} rows)")
        time.sleep(0.5)

    print("\nKite data fetch complete.")
    print(f"All files saved to {RAW_DIR}/")
    print("\nNext: run fetch_macro_extended.py to update macro data")