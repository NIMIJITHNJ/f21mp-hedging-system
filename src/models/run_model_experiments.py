"""

Experiments:
    1. Horizon sweep: 3-day, 5-day, 10-day forward volatility labels
       (using current best LSTM config on each horizon)
    2. Hyperparameter sweep: 6 LSTM configs on best horizon
    3. Classical baselines: Random Forest and XGBoost on best horizon

Evaluation criteria (in priority order):
    1. Macro-F1
    2. High-risk recall
    3. Overall accuracy

All results saved to:
    outputs/reports/experiment_results.csv
    outputs/reports/experiment_summary.txt
    outputs/figures/experiment_confusion_matrices/

Fixed test period: 2024-01-01 to 2026-03-20 (chronological, no leakage)

"""

import os
import sys
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, "src")

from features.build_features import make_risk_labels, fit_garch, load_raw, TRADING_DAYS
from features.technical_indicators import (
    rsi, macd, vwap, atr, sma_crossover,
    oi_pct_change, oi_volume_ratio, oi_moving_average,
    log_return, rolling_volatility,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "outputs/reports"
FIGURES_DIR   = "outputs/figures/experiment_confusion_matrices"
MODELS_DIR    = "outputs/models"
RAW_DIR       = "data/raw"
EXTERNAL_DIR  = "data/external"

for d in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Fixed split boundaries ────────────────────────────────────────────────────
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"

# ── Experiment configurations ─────────────────────────────────────────────────
HORIZONS = [3, 5, 10]

LSTM_CONFIGS = {
    "C1": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2, "batch": 32, "lr": 1e-3},
    "C2": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.3, "batch": 32, "lr": 1e-3},
    "C3": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.3, "batch": 16, "lr": 1e-3},
    "C4": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2, "batch": 32, "lr": 5e-4},
    "C5": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.3, "batch": 16, "lr": 5e-4},
    "C6": {"lstm_units": 128, "dense_units": 64, "dropout": 0.3, "batch": 32, "lr": 5e-4},
}

SEQ_LEN    = 20
N_FEATURES = 25
N_CLASSES  = 3
MAX_EPOCHS = 100
ES_PATIENCE = 15


# ── Feature matrix builder ────────────────────────────────────────────────────

def build_feature_matrix(horizon):
    """
    Build the full feature matrix and labels for a given horizon.
    Returns X_train, X_val, X_test, y_train, y_val, y_test (numpy arrays).
    """
    spot  = load_raw("NIFTY50_spot")
    fut   = load_raw("NIFTY_futures_continuous")
    macro = pd.read_csv(
        os.path.join(EXTERNAL_DIR, "macro_combined.csv"),
        parse_dates=["date"],
    ).set_index("date")
    macro.index = pd.DatetimeIndex(macro.index).normalize()

    from sklearn.preprocessing import StandardScaler

    STOCK_SYMBOLS = [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
        "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
    ]

    stock_returns = {}
    for sym in STOCK_SYMBOLS:
        df = load_raw(sym)
        stock_returns[sym] = log_return(df["close"]).rename(f"ret_{sym}")

    rsi14                  = rsi(spot["close"], period=14)
    macd_line, macd_sig, _ = macd(spot["close"], fast=12, slow=26, signal=9)
    vwap20                 = vwap(spot["high"], spot["low"], spot["close"], fut["volume"])
    atr14                  = atr(spot["high"], spot["low"], spot["close"], period=14)
    sma_cross              = sma_crossover(spot["close"], fast=10, slow=20)
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
        horizon=horizon,
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

    print(f"    horizon={horizon}d: train={len(y_tr)} val={len(y_v)} test={len(y_te)}")
    return X_tr_s, X_v_s, X_te_s, y_tr, y_v, y_te


# ── Sequence builder ──────────────────────────────────────────────────────────

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


# ── LSTM builder ──────────────────────────────────────────────────────────────

def build_lstm(lstm_units, dense_units, dropout, lr):
    model = Sequential([
        Input(shape=(SEQ_LEN, N_FEATURES)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Metrics extractor ─────────────────────────────────────────────────────────

def get_metrics(y_true, y_pred, split="test"):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=[0, 1, 2], zero_division=0)
    report   = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2], zero_division=0, output_dict=True,
    )
    high_recall = report["High"]["recall"]
    low_recall  = report["Low"]["recall"]
    med_recall  = report["Medium"]["recall"]
    return {
        f"{split}_accuracy":    round(acc, 4),
        f"{split}_macro_f1":    round(macro_f1, 4),
        f"{split}_high_recall": round(high_recall, 4),
        f"{split}_low_recall":  round(low_recall, 4),
        f"{split}_med_recall":  round(med_recall, 4),
    }


# ── Confusion matrix plotter ──────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0d0d0f")
    ax.set_facecolor("#16161a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Med", "High"],
                yticklabels=["Low", "Med", "High"],
                linewidths=0.5, linecolor="#2a2a30",
                ax=ax, cbar=False)
    ax.set_xlabel("Predicted", color="#9090a0")
    ax.set_ylabel("Actual",    color="#9090a0")
    ax.set_title(title, color="#f0f0f5", fontsize=10, pad=8)
    ax.tick_params(colors="#9090a0")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# ── Run one LSTM experiment ───────────────────────────────────────────────────

def run_lstm_experiment(config_name, config, horizon,
                        X_tr, X_v, X_te, y_tr, y_v, y_te):
    tf.random.set_seed(42)
    np.random.seed(42)

    X_train_seq, y_train_seq = build_sequences(X_tr, y_tr, SEQ_LEN)
    X_val_seq,   y_val_seq   = build_sequences_with_context(
        X_tr[-(SEQ_LEN - 1):], X_v, y_v, SEQ_LEN)
    X_test_seq,  y_test_seq  = build_sequences_with_context(
        X_v[-(SEQ_LEN - 1):], X_te, y_te, SEQ_LEN)

    y_train_cat = to_categorical(y_train_seq, N_CLASSES)
    y_val_cat   = to_categorical(y_val_seq,   N_CLASSES)

    cw_vals = compute_class_weight("balanced",
                                   classes=np.array([0, 1, 2]),
                                   y=y_train_seq)
    class_weight = {0: cw_vals[0], 1: cw_vals[1], 2: cw_vals[2]}

    model = build_lstm(
        config["lstm_units"], config["dense_units"],
        config["dropout"], config["lr"],
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", patience=7,
                          factor=0.5, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_train_seq, y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=config["batch"],
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )

    y_val_pred  = np.argmax(model.predict(X_val_seq,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)

    val_metrics  = get_metrics(y_val_seq,  y_val_pred,  "val")
    test_metrics = get_metrics(y_test_seq, y_test_pred, "test")

    exp_label = f"LSTM_{config_name}_h{horizon}"
    save_confusion_matrix(
        y_test_seq, y_test_pred,
        f"{exp_label} (test)",
        os.path.join(FIGURES_DIR, f"cm_{exp_label}.png"),
    )

    return {
        "experiment":   exp_label,
        "model":        "LSTM",
        "config":       config_name,
        "horizon":      horizon,
        "lstm_units":   config["lstm_units"],
        "dense_units":  config["dense_units"],
        "dropout":      config["dropout"],
        "batch":        config["batch"],
        "lr":           config["lr"],
        **val_metrics,
        **test_metrics,
    }


# ── Run RF/XGBoost experiment ─────────────────────────────────────────────────

def run_classical_experiment(model_name, horizon,
                             X_tr, X_v, X_te, y_tr, y_v, y_te):
    if model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
    elif model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            print(f"    XGBoost not installed — skipping")
            return None

    clf.fit(X_tr, y_tr)

    y_val_pred  = clf.predict(X_v)
    y_test_pred = clf.predict(X_te)

    val_metrics  = get_metrics(y_v,  y_val_pred,  "val")
    test_metrics = get_metrics(y_te, y_test_pred, "test")

    exp_label = f"{model_name}_h{horizon}"
    save_confusion_matrix(
        y_te, y_test_pred,
        f"{exp_label} (test)",
        os.path.join(FIGURES_DIR, f"cm_{exp_label}.png"),
    )

    return {
        "experiment":   exp_label,
        "model":        model_name,
        "config":       "default",
        "horizon":      horizon,
        "lstm_units":   "-",
        "dense_units":  "-",
        "dropout":      "-",
        "batch":        "-",
        "lr":           "-",
        **val_metrics,
        **test_metrics,
    }


# ── Main experiment loop ──────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []
    total   = len(HORIZONS) * len(LSTM_CONFIGS) + len(HORIZONS) * 2
    done    = 0

    print("=" * 65)
    print("HEDRA EXPERIMENT RUNNER")
    print(f"Horizons: {HORIZONS}  |  LSTM configs: {len(LSTM_CONFIGS)}")
    print(f"Total experiments: {total}")
    print("=" * 65)

    # ── Phase 1: LSTM horizon + config sweep ──────────────────────────────────
    for horizon in HORIZONS:
        print(f"\n{'─'*65}")
        print(f"HORIZON {horizon}-day  —  building feature matrix ...")
        X_tr, X_v, X_te, y_tr, y_v, y_te = build_feature_matrix(horizon)

        for cfg_name, cfg in LSTM_CONFIGS.items():
            done += 1
            print(f"  [{done}/{total}] LSTM {cfg_name} h{horizon}  "
                  f"units={cfg['lstm_units']} drop={cfg['dropout']} "
                  f"batch={cfg['batch']} lr={cfg['lr']}")
            try:
                row = run_lstm_experiment(
                    cfg_name, cfg, horizon,
                    X_tr, X_v, X_te, y_tr, y_v, y_te,
                )
                results.append(row)
                print(f"    val_f1={row['val_macro_f1']:.3f}  "
                      f"test_f1={row['test_macro_f1']:.3f}  "
                      f"test_high_recall={row['test_high_recall']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # ── Phase 2: Classical models on each horizon ─────────────────────────────
    for horizon in HORIZONS:
        print(f"\n{'─'*65}")
        print(f"Classical models — horizon {horizon}-day ...")
        X_tr, X_v, X_te, y_tr, y_v, y_te = build_feature_matrix(horizon)

        for model_name in ["RandomForest", "XGBoost"]:
            done += 1
            print(f"  [{done}/{total}] {model_name} h{horizon}")
            try:
                row = run_classical_experiment(
                    model_name, horizon,
                    X_tr, X_v, X_te, y_tr, y_v, y_te,
                )
                if row:
                    results.append(row)
                    print(f"    val_f1={row['val_macro_f1']:.3f}  "
                          f"test_f1={row['test_macro_f1']:.3f}  "
                          f"test_high_recall={row['test_high_recall']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    csv_path   = os.path.join(REPORTS_DIR, "experiment_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n\nSaved results: {csv_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY (sorted by test_macro_f1 desc)")
    print("=" * 65)

    display_cols = [
        "experiment", "test_accuracy", "test_macro_f1",
        "test_high_recall", "val_macro_f1",
    ]
    summary = results_df[display_cols].sort_values(
        "test_macro_f1", ascending=False
    ).head(15)
    print(summary.to_string(index=False))

    # ── Best model ────────────────────────────────────────────────────────────
    best = results_df.sort_values(
        ["test_macro_f1", "test_high_recall"], ascending=False
    ).iloc[0]

    print(f"\n{'='*65}")
    print("BEST MODEL")
    print(f"  Experiment    : {best['experiment']}")
    print(f"  Test accuracy : {best['test_accuracy']:.4f}")
    print(f"  Test macro-F1 : {best['test_macro_f1']:.4f}")
    print(f"  Test High rec : {best['test_high_recall']:.4f}")
    print(f"  Val  macro-F1 : {best['val_macro_f1']:.4f}")
    print("=" * 65)

    # ── Save summary text ─────────────────────────────────────────────────────
    summary_path = os.path.join(REPORTS_DIR, "experiment_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HEDRA EXPERIMENT SUMMARY\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Dataset    : 2012-01-01 to 2026-03-20\n")
        f.write(f"Train      : up to 2022-12-31\n")
        f.write(f"Val        : 2023-01-01 to 2023-12-31\n")
        f.write(f"Test       : 2024-01-01 to 2026-03-20\n")
        f.write(f"Horizons   : {HORIZONS}\n")
        f.write(f"LSTM configs: {len(LSTM_CONFIGS)}\n\n")
        f.write(results_df[display_cols].sort_values(
            "test_macro_f1", ascending=False
        ).to_string(index=False))
        f.write(f"\n\nBEST: {best['experiment']}\n")
        f.write(f"  test_macro_f1={best['test_macro_f1']:.4f}  "
                f"test_high_recall={best['test_high_recall']:.4f}\n")

    print(f"Saved summary : {summary_path}")
    print(f"Saved figures : {FIGURES_DIR}/")
    print("\nExperiment run complete.")