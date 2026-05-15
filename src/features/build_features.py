"""
build_features.py

Assembles the full 25-dimensional feature vector from raw price data,
technical indicators, OI features, GARCH volatility, and macroeconomic inputs.
Generates risk labels from realised volatility tertiles fitted on training data.
Applies StandardScaler fitted only on the training partition to prevent leakage.

Outputs saved to data/processed/:
    X_train.csv, X_val.csv, X_test.csv
    y_train.csv, y_val.csv, y_test.csv
    scaler.pkl
    feature_names.txt
    risk_label_thresholds.txt

Split boundaries:
    Training   : 2021-01-01 to 2024-09-30  (~930 rows)
    Validation : 2024-10-01 to 2024-12-31  ( 62 rows)
    Test       : 2025-01-01 to 2025-03-28  ( 62 rows)

Run from the project root:
    python src/features/build_features.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, "src")
from features.technical_indicators import (
    rsi, macd, vwap, atr, sma_crossover,
    oi_pct_change, oi_volume_ratio, oi_moving_average,
    log_return, rolling_volatility,
)

RAW_DIR        = "data/raw"
EXTERNAL_DIR   = "data/external"
PROCESSED_DIR  = "data/processed"

TRAIN_END = "2024-09-30"
VAL_END   = "2024-12-31"

STOCK_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]


def load_raw(symbol):
    """Load a raw CSV from data/raw/ indexed by date."""
    path = os.path.join(RAW_DIR, f"{symbol}.csv")
    df   = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def fit_garch(log_returns_train, log_returns_full):
    """
    Fit GARCH(1,1) on training data only and generate conditional volatility
    (sigma) for the full series by updating the recursion row by row.

    Returns a Series of daily sigma values aligned to log_returns_full.index.
    The first value is NaN since log returns require one prior close.
    """
    print("  Fitting GARCH(1,1) on training data ...")

    # arch_model expects returns scaled to percentage points for numerical stability
    train_returns = log_returns_train.dropna() * 100

    model  = arch_model(train_returns, vol="Garch", p=1, q=1,
                        mean="Zero", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]

    print(f"  GARCH params — omega={omega:.6f}  alpha={alpha:.4f}  beta={beta:.4f}")
    print(f"  Persistence (alpha+beta) = {alpha+beta:.4f}")

    # Generate sigma for the full series using the fitted parameters
    full_returns = (log_returns_full * 100).fillna(0).values
    n            = len(full_returns)
    sigma2       = np.zeros(n)

    # Initialise with unconditional variance from training fit
    sigma2[0] = result.conditional_volatility.iloc[-1] ** 2

    for t in range(1, n):
        sigma2[t] = omega + alpha * full_returns[t-1]**2 + beta * sigma2[t-1]

    # Convert back from percentage-scaled to decimal, then to daily sigma
    sigma = np.sqrt(sigma2) / 100
    sigma_series       = pd.Series(sigma, index=log_returns_full.index,
                                   name="garch_sigma")
    sigma_series.iloc[0] = np.nan  # first row has no prior return
    return sigma_series, result


def make_risk_labels(realised_vol_train, realised_vol_full):
    """
    Generate three-class risk labels using tertiles fitted on training data only.

    low    (0) : bottom tertile of training volatility
    medium (1) : middle tertile
    high   (2) : top tertile

    Thresholds are fitted on training data and applied to val/test to prevent
    any information leakage from future volatility regimes.
    """
    train_clean = realised_vol_train.dropna()
    q33 = train_clean.quantile(1/3)
    q67 = train_clean.quantile(2/3)

    print(f"  Risk label thresholds (training tertiles):")
    print(f"    Low    : vol < {q33:.4f}")
    print(f"    Medium : {q33:.4f} <= vol < {q67:.4f}")
    print(f"    High   : vol >= {q67:.4f}")

    labels = pd.cut(
        realised_vol_full,
        bins=[-np.inf, q33, q67, np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")
    labels.name = "risk_label"
    return labels, q33, q67


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading raw data ...")
    spot = load_raw("NIFTY50_spot")
    fut  = load_raw("NIFTY_futures_continuous")
    macro = pd.read_csv(
        os.path.join(EXTERNAL_DIR, "macro_combined.csv"),
        parse_dates=["date"],
    ).set_index("date")
    macro.index = pd.DatetimeIndex(macro.index).normalize()

    # Load individual stock log returns
    print("Computing stock log returns ...")
    stock_returns = {}
    for sym in STOCK_SYMBOLS:
        df = load_raw(sym)
        stock_returns[sym] = log_return(df["close"]).rename(f"ret_{sym}")

    # ── Technical indicators on spot ──────────────────────────────────────────
    print("\nComputing technical indicators ...")

    rsi14              = rsi(spot["close"], period=14)
    macd_line, macd_sig, _ = macd(spot["close"], fast=12, slow=26, signal=9)
    vwap20             = vwap(spot["high"], spot["low"],
                              spot["close"], fut["volume"])
    atr14              = atr(spot["high"], spot["low"], spot["close"], period=14)
    sma_cross          = sma_crossover(spot["close"], fast=10, slow=20)

    # ── OI features from futures ──────────────────────────────────────────────
    oi_pct             = oi_pct_change(fut["oi"], period=1)
    oi_ratio           = oi_volume_ratio(fut["oi"], fut["volume"])
    oi_ma20            = oi_moving_average(fut["oi"], window=20)

    # ── Volatility ────────────────────────────────────────────────────────────
    realised_vol       = rolling_volatility(spot["close"], window=20)
    spot_log_ret       = log_return(spot["close"])

    # ── GARCH(1,1) ────────────────────────────────────────────────────────────
    print("\nFitting GARCH(1,1) ...")
    train_mask         = spot_log_ret.index <= TRAIN_END
    garch_sigma, garch_result = fit_garch(
        spot_log_ret[train_mask],
        spot_log_ret,
    )

    # ── Assemble feature matrix ───────────────────────────────────────────────
    print("\nAssembling feature matrix ...")

    feature_df = pd.DataFrame(index=spot.index)

    # Group 1 — Technical
    feature_df["rsi_14"]          = rsi14
    feature_df["macd_12_26"]      = macd_line
    feature_df["macd_signal_9"]   = macd_sig
    feature_df["vwap_20"]         = vwap20
    feature_df["atr_14"]          = atr14
    feature_df["sma_cross_10_20"] = sma_cross

    # Group 2 — OI
    feature_df["oi_pct_change"]   = oi_pct
    feature_df["oi_vol_ratio"]    = oi_ratio
    feature_df["oi_ma_20"]        = oi_ma20

    # Group 3 — Volatility
    feature_df["realised_vol_20"] = realised_vol
    feature_df["garch_sigma"]     = garch_sigma

    # Group 4 — Macro (aligned already)
    feature_df["gold_usd"]        = macro["gold_usd"]
    feature_df["brent_usd"]       = macro["brent_usd"]
    feature_df["usdinr"]          = macro["usdinr"]
    feature_df["india_10y"]       = macro["india_10y"]

    # Group 5 — Stock log returns
    for sym in STOCK_SYMBOLS:
        feature_df[f"ret_{sym}"] = stock_returns[sym]

    print(f"  Feature matrix shape before dropna: {feature_df.shape}")
    print(f"  NaNs per column:\n{feature_df.isna().sum()[feature_df.isna().sum() > 0]}")

    # Drop warmup rows — all NaNs introduced by the longest rolling window (26 for MACD slow EMA)
    # We use dropna to remove any row with any NaN across all 25 features
    feature_df = feature_df.dropna()
    print(f"  Feature matrix shape after dropna : {feature_df.shape}")

    # ── Risk labels ───────────────────────────────────────────────────────────
    print("\nGenerating risk labels ...")
    train_vol_mask = (feature_df.index <= TRAIN_END)
    labels, q33, q67 = make_risk_labels(
        feature_df.loc[train_vol_mask, "realised_vol_20"],
        feature_df["realised_vol_20"],
    )
    feature_df["risk_label"] = labels

    # Drop any remaining NaN labels (shouldn't occur after dropna but safety check)
    feature_df = feature_df.dropna(subset=["risk_label"])
    print(f"  Label distribution:\n{feature_df['risk_label'].value_counts().sort_index()}")

    # ── Train / val / test split ──────────────────────────────────────────────
    print("\nSplitting into train / val / test ...")
    train_df = feature_df[feature_df.index <= TRAIN_END]
    val_df   = feature_df[(feature_df.index > TRAIN_END) & (feature_df.index <= VAL_END)]
    test_df  = feature_df[feature_df.index > VAL_END]

    print(f"  Train : {len(train_df)} rows  ({train_df.index[0].date()} to {train_df.index[-1].date()})")
    print(f"  Val   : {len(val_df)} rows  ({val_df.index[0].date()} to {val_df.index[-1].date()})")
    print(f"  Test  : {len(test_df)} rows  ({test_df.index[0].date()} to {test_df.index[-1].date()})")

    feature_cols = [c for c in feature_df.columns if c != "risk_label"]

    X_train = train_df[feature_cols]
    X_val   = val_df[feature_cols]
    X_test  = test_df[feature_cols]
    y_train = train_df["risk_label"]
    y_val   = val_df["risk_label"]
    y_test  = test_df["risk_label"]

    # ── Scale features ────────────────────────────────────────────────────────
    # Scaler is fit only on training data — applying it to val/test
    # prevents any leakage of future distribution statistics.
    print("\nFitting StandardScaler on training data only ...")
    scaler  = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index, columns=feature_cols,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        index=X_val.index, columns=feature_cols,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index, columns=feature_cols,
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving to data/processed/ ...")

    X_train_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_val_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_val.csv"))
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    y_val.to_csv(os.path.join(PROCESSED_DIR, "y_val.csv"))
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(PROCESSED_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    with open(os.path.join(PROCESSED_DIR, "risk_label_thresholds.txt"), "w") as f:
        f.write(f"low_threshold  = {q33:.6f}\n")
        f.write(f"high_threshold = {q67:.6f}\n")
        f.write("low=0  medium=1  high=2\n")

    print(f"\n  X_train : {X_train_scaled.shape}")
    print(f"  X_val   : {X_val_scaled.shape}")
    print(f"  X_test  : {X_test_scaled.shape}")
    print(f"  y_train : {y_train.shape}  labels: {sorted(y_train.unique())}")
    print(f"  y_val   : {y_val.shape}  labels: {sorted(y_val.unique())}")
    print(f"  y_test  : {y_test.shape}  labels: {sorted(y_test.unique())}")
    print(f"\n  Features ({len(feature_cols)}):")
    for i, name in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {name}")

    print("\nFeature engineering complete.")
    print(f"All outputs saved to {PROCESSED_DIR}/")