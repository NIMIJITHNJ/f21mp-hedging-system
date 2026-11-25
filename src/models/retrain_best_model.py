import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Determinism flags — must be set BEFORE importing tensorflow
os.environ["PYTHONHASHSEED"]         = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, "src")

from features.build_features import make_risk_labels, fit_garch, load_raw
from features.technical_indicators import (
    rsi, macd, vwap, atr, sma_crossover,
    oi_pct_change, oi_volume_ratio, oi_moving_average,
    log_return, rolling_volatility,
)

# Paths
EXTERNAL_DIR = "data/external"
MODELS_DIR   = "outputs/models"
REPORTS_DIR  = "outputs/reports"
FIGURES_DIR  = "outputs/figures"

for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# Locked config: LSTM_C4_h10
HORIZON     = 10
LSTM_UNITS  = 64
DENSE_UNITS = 32
DROPOUT     = 0.2
BATCH_SIZE  = 32
LR          = 5e-4
SEQ_LEN     = 20
N_FEATURES  = 25
N_CLASSES   = 3
MAX_EPOCHS  = 100
ES_PATIENCE = 15
TRAIN_END   = "2022-12-31"
VAL_END     = "2023-12-31"

STOCK_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]


def build_feature_matrix():
    spot  = load_raw("NIFTY50_spot")
    fut   = load_raw("NIFTY_futures_continuous")
    macro = pd.read_csv(
        os.path.join(EXTERNAL_DIR, "macro_combined.csv"),
        parse_dates=["date"],
    ).set_index("date")
    macro.index = pd.DatetimeIndex(macro.index).normalize()

    stock_returns = {}
    for sym in STOCK_SYMBOLS:
        df = load_raw(sym)
        stock_returns[sym] = log_return(df["close"]).rename(f"ret_{sym}")

    rsi14                  = rsi(spot["close"], period=14)
    macd_line, macd_sig, _ = macd(spot["close"], fast=12, slow=26, signal=9)
    vwap20 = vwap(spot["high"], spot["low"], spot["close"], fut["volume"])
    atr14  = atr(spot["high"], spot["low"], spot["close"], period=14)
    sma_cross = sma_crossover(spot["close"], fast=10, slow=20)
    oi_pct   = oi_pct_change(fut["oi"], period=1).fillna(0)
    oi_ratio = oi_volume_ratio(fut["oi"], fut["volume"]).fillna(0)
    oi_ma20  = oi_moving_average(fut["oi"], window=20).fillna(0)
    realised_vol = rolling_volatility(spot["close"], window=20)
    spot_log_ret = log_return(spot["close"])

    train_mask = spot_log_ret.index <= TRAIN_END
    garch_sigma, _ = fit_garch(spot_log_ret[train_mask], spot_log_ret)

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

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).dropna()

    labels, q33, q67 = make_risk_labels(
        spot_close=spot["close"],
        feature_index=feature_df.index,
        train_end=TRAIN_END,
        horizon=HORIZON,
    )
    feature_df["risk_label"] = labels
    feature_df = feature_df.dropna(subset=["risk_label"])

    train_df = feature_df[feature_df.index <= TRAIN_END]
    val_df   = feature_df[(feature_df.index > TRAIN_END) & (feature_df.index <= VAL_END)]
    test_df  = feature_df[feature_df.index > VAL_END]

    feature_cols = [c for c in feature_df.columns if c != "risk_label"]

    X_tr = train_df[feature_cols].replace([np.inf, -np.inf], 0)
    X_v  = val_df[feature_cols].replace([np.inf, -np.inf], 0)
    X_te = test_df[feature_cols].replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_v_s  = scaler.transform(X_v)
    X_te_s = scaler.transform(X_te)

    y_tr = train_df["risk_label"].values.astype(np.int32)
    y_v  = val_df["risk_label"].values.astype(np.int32)
    y_te = test_df["risk_label"].values.astype(np.int32)

    print(f"  Train: {len(y_tr)}  Val: {len(y_v)}  Test: {len(y_te)}")
    return X_tr_s, X_v_s, X_te_s, y_tr, y_v, y_te


def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X) + 1):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def build_sequences_with_context(X_context, X_split, y_split, seq_len):
    assert len(X_context) == seq_len - 1
    X_full = np.vstack([X_context, X_split])
    Xs, ys = [], []
    for i in range(seq_len, len(X_full) + 1):
        Xs.append(X_full[i - seq_len: i])
        ys.append(y_split[i - seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


if __name__ == "__main__":
    print("=" * 60)
    print("RETRAINING LSTM_C4_h10 (locked best model)")
    print(f"  LSTM({LSTM_UNITS}), Dense({DENSE_UNITS}), dropout={DROPOUT}")
    print(f"  batch={BATCH_SIZE}, lr={LR}, horizon={HORIZON}d")
    print(f"  TF_DETERMINISTIC_OPS=1")
    print("=" * 60)

    # Build data
    print("\nBuilding feature matrix ...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = build_feature_matrix()

    print(f"\nBuilding {SEQ_LEN}-day sequences ...")
    X_train_seq, y_train_seq = build_sequences(X_tr, y_tr, SEQ_LEN)
    X_val_seq,   y_val_seq   = build_sequences_with_context(
        X_tr[-(SEQ_LEN-1):], X_v, y_v, SEQ_LEN)
    X_test_seq,  y_test_seq  = build_sequences_with_context(
        X_v[-(SEQ_LEN-1):], X_te, y_te, SEQ_LEN)

    print(f"  train={X_train_seq.shape}  val={X_val_seq.shape}  test={X_test_seq.shape}")

    y_train_cat = to_categorical(y_train_seq, N_CLASSES)
    y_val_cat   = to_categorical(y_val_seq,   N_CLASSES)

    cw = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_train_seq)
    class_weight = {0: cw[0], 1: cw[1], 2: cw[2]}
    print(f"  Class weights: {class_weight}")

    # Build model
    print("\nBuilding model ...")
    model = Sequential([
        Input(shape=(SEQ_LEN, N_FEATURES)),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(DROPOUT),
        LSTM(LSTM_UNITS, return_sequences=False),
        Dropout(DROPOUT),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(N_CLASSES,   activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Train
    print(f"\nTraining (max {MAX_EPOCHS} epochs, patience={ES_PATIENCE}) ...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", patience=7,
                          factor=0.5, min_lr=1e-6, verbose=1),
    ]
    hist = model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # Evaluate
    print("\nEvaluating ...")
    y_val_pred  = np.argmax(model.predict(X_val_seq,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)

    val_acc  = accuracy_score(y_val_seq,  y_val_pred)
    test_acc = accuracy_score(y_test_seq, y_test_pred)
    val_f1   = f1_score(y_val_seq,  y_val_pred,  average="macro",
                        labels=[0,1,2], zero_division=0)
    test_f1  = f1_score(y_test_seq, y_test_pred, average="macro",
                        labels=[0,1,2], zero_division=0)

    print(f"\n  Val  accuracy : {val_acc:.4f}   macro-F1 : {val_f1:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}   macro-F1 : {test_f1:.4f}")

    val_report  = classification_report(y_val_seq,  y_val_pred,
                      target_names=["Low","Medium","High"],
                      labels=[0,1,2], zero_division=0)
    test_report = classification_report(y_test_seq, y_test_pred,
                      target_names=["Low","Medium","High"],
                      labels=[0,1,2], zero_division=0)
    print("\n  Validation:\n", val_report)
    print("  Test:\n", test_report)

    # Save model
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    model.save(model_path)
    print(f"  Saved model: {model_path}")

    # Save report
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("LSTM_C4_h10 — LOCKED FINAL MODEL\n")
        f.write("=" * 60 + "\n")
        f.write(f"Architecture : 2x LSTM({LSTM_UNITS}) + Dense({DENSE_UNITS}) + Softmax(3)\n")
        f.write(f"Horizon      : {HORIZON}-day forward volatility\n")
        f.write(f"Sequence len : {SEQ_LEN} trading days\n")
        f.write(f"LR={LR}  batch={BATCH_SIZE}  dropout={DROPOUT}\n")
        f.write(f"Epochs run   : {len(hist.history['loss'])}\n\n")
        f.write(f"Val  accuracy : {val_acc:.4f}   macro-F1 : {val_f1:.4f}\n")
        f.write(f"Test accuracy : {test_acc:.4f}   macro-F1 : {test_f1:.4f}\n\n")
        f.write("VALIDATION SET\n" + "-"*40 + "\n")
        f.write(val_report + "\n")
        f.write("TEST SET\n" + "-"*40 + "\n")
        f.write(test_report + "\n")
    print(f"  Saved report: {report_path}")

    # Save training history
    history_df = pd.DataFrame(hist.history)
    history_df.to_csv(os.path.join(REPORTS_DIR, "training_history.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test_seq, y_test_pred, labels=[0,1,2])
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0d0f")
    ax.set_facecolor("#16161a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low","Medium","High"],
                yticklabels=["Low","Medium","High"],
                linewidths=0.5, linecolor="#2a2a30",
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted", color="#9090a0", fontsize=11)
    ax.set_ylabel("Actual",    color="#9090a0", fontsize=11)
    ax.set_title("Confusion Matrix — Test Set (LSTM_C4_h10)",
                 color="#f0f0f5", fontsize=12, pad=10)
    ax.tick_params(colors="#9090a0")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved confusion matrix: {cm_path}")

    print("\n" + "=" * 60)
    print("DONE — LSTM_C4_h10 locked and saved.")
    print(f"  Val  macro-F1 : {val_f1:.4f}")
    print(f"  Test macro-F1 : {test_f1:.4f}")
    print("=" * 60)
