"""
verify_data.py

Cross-verification of all raw CSV files in data/raw/.
Checks for completeness, missing values, date alignment,
price sanity, and futures OI validity.

Run from the project root:
    python src/data/verify_data.py
"""

import os
import pandas as pd
import numpy as np
from datetime import date

RAW_DATA_DIR = "data/raw"

# Expected date range
EXPECTED_START = "2023-01-02"
EXPECTED_END   = "2025-03-28"
EXPECTED_ROWS  = 557

# All files we expect to exist
EXPECTED_FILES = [
    "NIFTY50_spot.csv",
    "NIFTY_futures_continuous.csv",
    "RELIANCE.csv",
    "HDFCBANK.csv",
    "ICICIBANK.csv",
    "INFY.csv",
    "TCS.csv",
    "SBIN.csv",
    "LT.csv",
    "BHARTIARTL.csv",
    "ITC.csv",
    "AXISBANK.csv",
]

STOCK_FILES = [f for f in EXPECTED_FILES if f not in
               ("NIFTY50_spot.csv", "NIFTY_futures_continuous.csv")]

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def log(check, status, detail=""):
    tag = f"[{status}]"
    print(f"  {tag:<8} {check}")
    if detail:
        print(f"           {detail}")
    results.append((check, status))


def section(title):
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. File presence ──────────────────────────────────────────────────────────

section("1. File presence")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if os.path.exists(path):
        size_kb = os.path.getsize(path) // 1024
        log(fname, PASS, f"{size_kb} KB")
    else:
        log(fname, FAIL, "File not found")


# ── 2. Row count ──────────────────────────────────────────────────────────────

section("2. Row count (expected 557 per file)")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    rows = len(df)
    if rows == EXPECTED_ROWS:
        log(fname, PASS, f"{rows} rows")
    elif abs(rows - EXPECTED_ROWS) <= 3:
        log(fname, WARN, f"{rows} rows (expected {EXPECTED_ROWS} — minor holiday variation)")
    else:
        log(fname, FAIL, f"{rows} rows (expected {EXPECTED_ROWS})")


# ── 3. Date range ─────────────────────────────────────────────────────────────

section("3. Date range")

reference_dates = None

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path, parse_dates=["date"])
    first = df["date"].min().strftime("%Y-%m-%d")
    last  = df["date"].max().strftime("%Y-%m-%d")

    if first == EXPECTED_START and last == EXPECTED_END:
        log(fname, PASS, f"{first} to {last}")
    else:
        log(fname, WARN, f"{first} to {last} (expected {EXPECTED_START} to {EXPECTED_END})")

    if reference_dates is None:
        reference_dates = set(df["date"].dt.strftime("%Y-%m-%d"))


# ── 4. Date alignment across all files ───────────────────────────────────────

section("4. Date alignment (all files share the same trading days)")

if reference_dates:
    ref_file = EXPECTED_FILES[0]
    for fname in EXPECTED_FILES[1:]:
        path = os.path.join(RAW_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df    = pd.read_csv(path, parse_dates=["date"])
        dates = set(df["date"].dt.strftime("%Y-%m-%d"))
        diff  = reference_dates.symmetric_difference(dates)
        if not diff:
            log(f"{fname} vs {ref_file}", PASS, "Identical trading days")
        else:
            log(f"{fname} vs {ref_file}", WARN,
                f"{len(diff)} date(s) differ: {sorted(diff)[:5]}")


# ── 5. Missing values ─────────────────────────────────────────────────────────

section("5. Missing values")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df      = pd.read_csv(path)
    missing = df.isnull().sum().sum()
    zeros   = (df[["open", "high", "low", "close"]] == 0).sum().sum()

    if missing == 0 and zeros == 0:
        log(fname, PASS, "No missing values, no zero prices")
    elif missing > 0:
        log(fname, FAIL, f"{missing} missing value(s): {df.isnull().sum().to_dict()}")
    elif zeros > 0:
        log(fname, WARN, f"{zeros} zero price(s) in OHLC columns")


# ── 6. OHLC sanity ───────────────────────────────────────────────────────────

section("6. OHLC sanity (high >= low, close within high/low)")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)

    bad_hl    = (df["high"] < df["low"]).sum()
    bad_close = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()

    if bad_hl == 0 and bad_close == 0:
        log(fname, PASS, "All OHLC relationships valid")
    else:
        issues = []
        if bad_hl:    issues.append(f"{bad_hl} rows where high < low")
        if bad_close: issues.append(f"{bad_close} rows where close outside high/low")
        log(fname, FAIL, ", ".join(issues))


# ── 7. Duplicate dates ───────────────────────────────────────────────────────

section("7. Duplicate dates")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df   = pd.read_csv(path, parse_dates=["date"])
    dups = df["date"].duplicated().sum()
    if dups == 0:
        log(fname, PASS, "No duplicate dates")
    else:
        log(fname, FAIL, f"{dups} duplicate date(s)")


# ── 8. Futures-specific checks ───────────────────────────────────────────────

section("8. Futures: OI and volume checks")

fut_path = os.path.join(RAW_DATA_DIR, "NIFTY_futures_continuous.csv")
if os.path.exists(fut_path):
    df = pd.read_csv(fut_path)

    # OI column presence
    if "oi" in df.columns:
        log("OI column present", PASS)
    else:
        log("OI column present", FAIL, "Column 'oi' missing")

    # OI should be non-zero for a futures series
    if "oi" in df.columns:
        zero_oi = (df["oi"] == 0).sum()
        if zero_oi == 0:
            log("OI non-zero throughout", PASS, f"Min OI: {df['oi'].min():,}")
        else:
            log("OI non-zero throughout", WARN, f"{zero_oi} rows with zero OI")

    # Volume should be non-zero
    zero_vol = (df["volume"] == 0).sum()
    if zero_vol == 0:
        log("Volume non-zero throughout", PASS, f"Min volume: {df['volume'].min():,}")
    else:
        log("Volume non-zero throughout", WARN, f"{zero_vol} rows with zero volume")

    # OI range sanity — typical Nifty OI is in the millions
    if "oi" in df.columns:
        oi_min = df["oi"].min()
        oi_max = df["oi"].max()
        log("OI range", PASS if oi_min > 100_000 else WARN,
            f"Min: {oi_min:,}  Max: {oi_max:,}")


# ── 9. Price continuity ───────────────────────────────────────────────────────

section("9. Price continuity (no extreme daily moves > 15%)")

for fname in EXPECTED_FILES:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df       = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df["ret"] = df["close"].pct_change().abs()
    extreme  = df[df["ret"] > 0.15]

    if extreme.empty:
        log(fname, PASS, "No daily moves > 15%")
    else:
        dates = extreme["date"].dt.strftime("%Y-%m-%d").tolist()
        log(fname, WARN,
            f"{len(extreme)} move(s) > 15% on: {dates[:3]}"
            + (" ..." if len(dates) > 3 else ""))


# ── 10. Split period check ───────────────────────────────────────────────────

section("10. Train / Val / Test split row counts")

TRAIN_END = "2024-09-30"
VAL_END   = "2024-12-31"

for fname in ["NIFTY50_spot.csv", "NIFTY_futures_continuous.csv"]:
    path = os.path.join(RAW_DATA_DIR, fname)
    if not os.path.exists(path):
        continue
    df    = pd.read_csv(path, parse_dates=["date"])
    train = df[df["date"] <= TRAIN_END]
    val   = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test  = df[df["date"] > VAL_END]
    log(fname, PASS,
        f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")


# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")

passed = sum(1 for _, s in results if s == PASS)
warned = sum(1 for _, s in results if s == WARN)
failed = sum(1 for _, s in results if s == FAIL)
total  = len(results)

print(f"\n  Total checks : {total}")
print(f"  Passed       : {passed}")
print(f"  Warnings     : {warned}")
print(f"  Failed       : {failed}")

if failed == 0 and warned == 0:
    print("\n  All checks passed. Data is ready for feature engineering.")
elif failed == 0:
    print("\n  No failures. Review warnings before proceeding.")
else:
    print("\n  Failures detected. Investigate before proceeding.")