"""
build_features.py

Assembles the full 25-dimensional feature vector from raw price data,
technical indicators, OI features, GARCH volatility, and macroeconomic inputs.
Generates risk labels from FUTURE 5-day realised volatility tertiles fitted
on training data only.

Label design:
    The label on day t = volatility regime over the NEXT 5 trading days
    (t+1 to t+5), computed from Nifty 50 spot returns. This ensures the
    LSTM predicts future risk, not same-day risk (which would cause leakage
    since realised_vol_20 is also a feature). Thresholds are fitted on
    training data only.

Data note:
    Nifty continuous futures fetched using token 15956226 with continuous=True.
    OI is available from 2019 onwards; 2018 Q1 OI is zero (documented limitation).
    Infinity values in OI-ratio features (from zero-volume days) are replaced
    with 0 before scaling.

Split boundaries:
    Training   : 2018-01-01 to 2023-12-31  (~1,224 rows)
    Validation : 2024-01-01 to 2024-09-30  (~187 rows)
    Test       : 2024-10-01 to 2025-03-28  (~119 rows)

Outputs saved to data/processed/:
    X_train.csv, X_val.csv, X_test.csv
    y_train.csv, y_val.csv, y_test.csv
    scaler.pkl
    feature_names.txt
    risk_label_thresholds.txt

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

RAW_DIR       = "data/raw"
EXTERNAL_DIR  = "data/external"
PROCESSED_DIR = "data/processed"

TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
TRADING_DAYS  = 252
LABEL_HORIZON = 5

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
    for the full series by updating the recursion row by row.
    """
    print("  Fitting GARCH(1,1) on training data ...")

    train_returns = log_returns_train.dropna() * 100
    model  = arch_model(train_returns, vol="Garch", p=1, q=1,
                        mean="Zero", rescale=False)
    result = model.fit(disp="off", show_warning=False)

    omega = result.params["omega"]
    alpha = result.params["alpha[1]"]
    beta  = result.params["beta[1]"]

    print(f"  GARCH params — omega={omega:.6f}  alpha={alpha:.4f}  beta={beta:.4f}")
    print(f"  Persistence (alpha+beta) = {alpha+beta:.4f}")

    full_returns = (log_returns_full * 100).fillna(0).values
    n            = len(full_returns)
    sigma2       = np.zeros(n)
    sigma2[0]    = result.conditional_volatility.iloc[-1] ** 2

    for t in range(1, n):
        sigma2[t] = omega + alpha * full_returns[t-1]**2 + beta * sigma2[t-1]

    sigma                = np.sqrt(sigma2) / 100
    sigma_series         = pd.Series(sigma, index=log_returns_full.index,
                                     name="garch_sigma")
    sigma_series.iloc[0] = np.nan
    return sigma_series, result


def make_risk_labels(spot_close, feature_index, train_end, horizon=5):
    """
    Generate three-class forward-looking risk labels.

    The label on day t = volatility regime over the NEXT horizon trading days.
    Thresholds are tertiles fitted on training data only.
    Last horizon rows receive NaN (no future window available).
    """
    spot_ret    = np.log(spot_close / spot_close.shift(1))
    rolling_vol = spot_ret.rolling(horizon).std() * np.sqrt(TRADING_DAYS)
    future_vol  = rolling_vol.shift(-horizon)
    future_vol  = future_vol.reindex(feature_index)

    train_mask   = feature_index <= train_end
    train_future = future_vol[train_mask].dropna()

    q33 = train_future.quantile(1/3)
    q67 = train_future.quantile(2/3)

    print(f"  Forward-looking risk label thresholds (training tertiles, {horizon}-day horizon):")
    print(f"    Low    : future_vol < {q33:.4f}  ({q33*100:.2f}% annualised)")
    print(f"    Medium : {q33:.4f} <= future_vol < {q67:.4f}")
    print(f"    High   : future_vol >= {q67:.4f}  ({q67*100:.2f}% annualised)")

    labels = pd.cut(
        future_vol,
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

    print("Computing stock log returns ...")
    stock_returns = {}
    for sym in STOCK_SYMBOLS:
        df = load_raw(sym)
        stock_returns[sym] = log_return(df["close"]).rename(f"ret_{sym}")

    # ── Technical indicators ──────────────────────────────────────────────────
    print("\nComputing technical indicators ...")

    rsi14                  = rsi(spot["close"], period=14)
    macd_line, macd_sig, _ = macd(spot["close"], fast=12, slow=26, signal=9)
    vwap20                 = vwap(spot["high"], spot["low"],
                                  spot["close"], fut["volume"])
    atr14                  = atr(spot["high"], spot["low"], spot["close"], period=14)
    sma_cross              = sma_crossover(spot["close"], fast=10, slow=20)

    # ── OI features ───────────────────────────────────────────────────────────
    # OI is available from 2019 onwards; 2018 Q1 has zero OI.
    # pct_change and ratio on zero OI produce NaN/inf — handled below.
    oi_pct   = oi_pct_change(fut["oi"], period=1)
    oi_ratio = oi_volume_ratio(fut["oi"], fut["volume"])
    oi_ma20  = oi_moving_average(fut["oi"], window=20)

    # ── Volatility ────────────────────────────────────────────────────────────
    realised_vol = rolling_volatility(spot["close"], window=20)
    spot_log_ret = log_return(spot["close"])

    # ── GARCH(1,1) ────────────────────────────────────────────────────────────
    print("\nFitting GARCH(1,1) ...")
    train_mask = spot_log_ret.index <= TRAIN_END
    garch_sigma, garch_result = fit_garch(
        spot_log_ret[train_mask],
        spot_log_ret,
    )

    # ── Assemble feature matrix ───────────────────────────────────────────────
    print("\nAssembling feature matrix ...")

    feature_df = pd.DataFrame(index=spot.index)

    feature_df["rsi_14"]          = rsi14
    feature_df["macd_12_26"]      = macd_line
    feature_df["macd_signal_9"]   = macd_sig
    feature_df["vwap_20"]         = vwap20
    feature_df["atr_14"]          = atr14
    feature_df["sma_cross_10_20"] = sma_cross
    feature_df["oi_pct_change"]   = oi_pct
    feature_df["oi_vol_ratio"]    = oi_ratio
    feature_df["oi_ma_20"]        = oi_ma20
    feature_df["realised_vol_20"] = realised_vol
    feature_df["garch_sigma"]     = garch_sigma
    feature_df["gold_usd"]        = macro["gold_usd"]
    feature_df["brent_usd"]       = macro["brent_usd"]
    feature_df["usdinr"]          = macro["usdinr"]
    feature_df["india_10y"]       = macro["india_10y"]

    for sym in STOCK_SYMBOLS:
        feature_df[f"ret_{sym}"] = stock_returns[sym]

    print(f"  Feature matrix shape before dropna: {feature_df.shape}")

    # Replace inf/-inf before dropna (oi_vol_ratio can produce inf from
    # zero-volume days; oi_pct_change can produce inf from 0->nonzero transitions)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.dropna()
    print(f"  Feature matrix shape after dropna : {feature_df.shape}")

    # ── Forward-looking risk labels ───────────────────────────────────────────
    print("\nGenerating forward-looking risk labels ...")
    labels, q33, q67 = make_risk_labels(
        spot_close    = spot["close"],
        feature_index = feature_df.index,
        train_end     = TRAIN_END,
        horizon       = LABEL_HORIZON,
    )
    feature_df["risk_label"] = labels

    # Drop last LABEL_HORIZON rows where future label is NaN
    feature_df = feature_df.dropna(subset=["risk_label"])
    print(f"  Rows after dropping NaN-label tail : {len(feature_df)}")
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

    # ── Clean infinities before scaling ──────────────────────────────────────
    # Belt-and-braces: replace any remaining inf that survived dropna
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_val   = X_val.replace([np.inf, -np.inf], 0)
    X_test  = X_test.replace([np.inf, -np.inf], 0)

    # ── Scale features ────────────────────────────────────────────────────────
    # Scaler is fit ONLY on training data — applied to val/test without refitting
    print("\nFitting StandardScaler on training data only ...")
    scaler = StandardScaler()
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
    X_val_scaled.to_csv(os.path.join(PROCESSED_DIR,   "X_val.csv"))
    X_test_scaled.to_csv(os.path.join(PROCESSED_DIR,  "X_test.csv"))
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    y_val.to_csv(os.path.join(PROCESSED_DIR,   "y_val.csv"))
    y_test.to_csv(os.path.join(PROCESSED_DIR,  "y_test.csv"))

    with open(os.path.join(PROCESSED_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(PROCESSED_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    with open(os.path.join(PROCESSED_DIR, "risk_label_thresholds.txt"), "w") as f:
        f.write(f"label_horizon  = {LABEL_HORIZON} trading days (forward-looking)\n")
        f.write(f"low_threshold  = {q33:.6f}  (annualised vol)\n")
        f.write(f"high_threshold = {q67:.6f}  (annualised vol)\n")
        f.write("low=0  medium=1  high=2\n")
        f.write("\nNote: labels represent FUTURE 5-day volatility regime.\n")
        f.write("realised_vol_20 and garch_sigma are input features only.\n")
        f.write("OI features have zero values for 2018 Q1 (limited data).\n")

    print(f"\n  X_train : {X_train_scaled.shape}")
    print(f"  X_val   : {X_val_scaled.shape}")
    print(f"  X_test  : {X_test_scaled.shape}")
    print(f"  y_train : {y_train.shape}  labels: {sorted(y_train.dropna().unique())}")
    print(f"  y_val   : {y_val.shape}  labels: {sorted(y_val.dropna().unique())}")
    print(f"  y_test  : {y_test.shape}  labels: {sorted(y_test.dropna().unique())}")
    print(f"\n  Features ({len(feature_cols)}):")
    for i, name in enumerate(feature_cols, 1):
        print(f"    {i:2d}. {name}")

    print("\nFeature engineering complete.")
    print(f"All outputs saved to {PROCESSED_DIR}/")