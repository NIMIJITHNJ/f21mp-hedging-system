from pathlib import Path
from datetime import datetime, time
import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from kiteconnect import KiteConnect


ROOT_DIR = Path(__file__).resolve().parents[1]

SPOT_CSV = ROOT_DIR / "data" / "raw" / "NIFTY50_spot.csv"
FUTURES_CSV = ROOT_DIR / "data" / "raw" / "NIFTY_futures_continuous.csv"


def make_kite():
    load_dotenv(ROOT_DIR / ".env")

    api_key = os.getenv("KITE_API_KEY")
    access_token = os.getenv("KITE_ACCESS_TOKEN")

    if not api_key:
        raise RuntimeError("KITE_API_KEY not found in .env")

    if not access_token:
        raise RuntimeError("KITE_ACCESS_TOKEN not found in .env")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def find_instrument_token(kite, exchange, tradingsymbol):
    instruments = kite.instruments(exchange)

    for item in instruments:
        if str(item.get("tradingsymbol", "")).upper() == tradingsymbol.upper():
            return int(item["instrument_token"])

    available = [
        item.get("tradingsymbol")
        for item in instruments
        if tradingsymbol.upper() in str(item.get("tradingsymbol", "")).upper()
    ][:10]

    raise RuntimeError(
        f"Could not find instrument token for {exchange}:{tradingsymbol}. "
        f"Similar symbols found: {available}"
    )


def candle_to_row(candle):
    return {
        "date": pd.to_datetime(candle["date"]).date().isoformat(),
        "open": float(candle.get("open", 0.0)),
        "high": float(candle.get("high", 0.0)),
        "low": float(candle.get("low", 0.0)),
        "close": float(candle.get("close", 0.0)),
        "volume": int(candle.get("volume", 0) or 0),
        "oi": int(candle.get("oi", 0) or 0),
    }


def fetch_daily_candles(kite, instrument_token, from_date, to_date, include_oi=False):
    day_start = datetime.combine(from_date, time(9, 0))
    day_end = datetime.combine(to_date, time(16, 0))

    candles = kite.historical_data(
        instrument_token=instrument_token,
        from_date=day_start,
        to_date=day_end,
        interval="day",
        continuous=False,
        oi=include_oi,
    )

    if not candles:
        raise RuntimeError(f"No daily candles returned for {from_date} to {to_date}")

    return [candle_to_row(candle) for candle in candles]


def upsert_csv_rows(csv_path, rows):
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "oi"])

    required_columns = ["date", "open", "high", "low", "close", "volume", "oi"]

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0 if col != "date" else ""

    df = df[required_columns].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)

    update_df = pd.DataFrame(rows)
    update_df = update_df[required_columns].copy()
    update_df["date"] = pd.to_datetime(update_df["date"], errors="coerce").dt.date.astype(str)

    update_dates = set(update_df["date"].dropna().astype(str))
    df = df[~df["date"].isin(update_dates)]

    df = pd.concat([df, update_df], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(2)

    for col in ["volume", "oi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df.to_csv(csv_path, index=False)

    return df.tail(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Final EOD date in YYYY-MM-DD format")
    parser.add_argument("--from-date", default=None, help="Optional start date in YYYY-MM-DD format")
    parser.add_argument("--futures-symbol", required=True, help="Example: NIFTY26JUNFUT")
    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    if args.from_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    else:
        current = pd.read_csv(SPOT_CSV)
        current["date"] = pd.to_datetime(current["date"], errors="coerce")
        last_date = current["date"].dropna().max().date()
        from_date = (pd.Timestamp(last_date) + pd.Timedelta(days=1)).date()

    kite = make_kite()

    print(f"Fetching EOD market data from {from_date} to {target_date}...")

    nifty_token = find_instrument_token(kite, "NSE", "NIFTY 50")
    futures_token = find_instrument_token(kite, "NFO", args.futures_symbol)

    spot_rows = fetch_daily_candles(
        kite=kite,
        instrument_token=nifty_token,
        from_date=from_date,
        to_date=target_date,
        include_oi=False,
    )

    futures_rows = fetch_daily_candles(
        kite=kite,
        instrument_token=futures_token,
        from_date=from_date,
        to_date=target_date,
        include_oi=True,
    )

    print(f"\nFetched {len(spot_rows)} NIFTY 50 spot candles.")
    print(f"Fetched {len(futures_rows)} NIFTY futures candles.")

    print("\nLatest NIFTY 50 spot EOD candle:")
    print(spot_rows[-1])

    print("\nLatest NIFTY futures EOD candle:")
    print(futures_rows[-1])

    print("\nUpdating CSV files...")

    spot_tail = upsert_csv_rows(SPOT_CSV, spot_rows)
    futures_tail = upsert_csv_rows(FUTURES_CSV, futures_rows)

    print("\nUpdated NIFTY50_spot.csv tail:")
    print(spot_tail.to_string(index=False))

    print("\nUpdated NIFTY_futures_continuous.csv tail:")
    print(futures_tail.to_string(index=False))

    print("\nDone. Restart Streamlit so the dashboard reloads the updated CSV data.")


if __name__ == "__main__":
    main()
