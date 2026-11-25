import os, sys, warnings, random
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

from features.build_features import make_risk_labels, fit_garch, load_raw
from features.technical_indicators import (
    rsi, macd, vwap, atr, sma_crossover,
    oi_pct_change, oi_volume_ratio, oi_moving_average,
    log_return, rolling_volatility,
)

REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures/experiment_confusion_matrices"
EXTERNAL_DIR = "data/external"
TRAIN_END = "2022-12-31"
VAL_END   = "2023-12-31"
SEQ_LEN   = 20
N_FEATURES = 25
N_CLASSES  = 3
MAX_EPOCHS = 100
ES_PATIENCE = 15
HORIZON = 7

LSTM_CONFIGS = {
    "C1": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2, "batch": 32, "lr": 1e-3},
    "C2": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.3, "batch": 32, "lr": 1e-3},
    "C3": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.3, "batch": 16, "lr": 1e-3},
    "C4": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2, "batch": 32, "lr": 5e-4},
    "C5": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.3, "batch": 16, "lr": 5e-4},
    "C6": {"lstm_units": 128, "dense_units": 64, "dropout": 0.3, "batch": 32, "lr": 5e-4},
}

STOCK_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]


def build_feature_matrix(horizon):
    from sklearn.preprocessing import StandardScaler
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


def get_metrics(y_true, y_pred, split="test"):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=[0, 1, 2], zero_division=0)
    report   = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2], zero_division=0, output_dict=True,
    )
    return {
        f"{split}_accuracy":    round(acc, 4),
        f"{split}_macro_f1":    round(macro_f1, 4),
        f"{split}_high_recall": round(report["High"]["recall"], 4),
        f"{split}_low_recall":  round(report["Low"]["recall"], 4),
        f"{split}_med_recall":  round(report["Medium"]["recall"], 4),
    }


def save_cm(y_true, y_pred, title, path):
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


if __name__ == "__main__":
    results = []

    print("=" * 65)
    print(f"HORIZON {HORIZON}-day — building feature matrix ...")
    X_tr, X_v, X_te, y_tr, y_v, y_te = build_feature_matrix(HORIZON)

    # LSTM configs
    for cfg_name, cfg in LSTM_CONFIGS.items():
        print(f"  LSTM {cfg_name} h{HORIZON}  "
              f"units={cfg['lstm_units']} drop={cfg['dropout']} "
              f"batch={cfg['batch']} lr={cfg['lr']}")
        tf.random.set_seed(42); np.random.seed(42)

        X_tr_seq, y_tr_seq = build_sequences(X_tr, y_tr, SEQ_LEN)
        X_v_seq,  y_v_seq  = build_sequences_with_context(
            X_tr[-(SEQ_LEN-1):], X_v, y_v, SEQ_LEN)
        X_te_seq, y_te_seq = build_sequences_with_context(
            X_v[-(SEQ_LEN-1):], X_te, y_te, SEQ_LEN)

        y_tr_cat = to_categorical(y_tr_seq, N_CLASSES)
        y_v_cat  = to_categorical(y_v_seq,  N_CLASSES)

        cw = compute_class_weight("balanced",
                                  classes=np.array([0,1,2]), y=y_tr_seq)
        class_weight = {0: cw[0], 1: cw[1], 2: cw[2]}

        model = build_lstm(cfg["lstm_units"], cfg["dense_units"],
                           cfg["dropout"], cfg["lr"])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", patience=7,
                              factor=0.5, min_lr=1e-6, verbose=0),
        ]
        model.fit(X_tr_seq, y_tr_cat,
                  validation_data=(X_v_seq, y_v_cat),
                  epochs=MAX_EPOCHS, batch_size=cfg["batch"],
                  callbacks=callbacks, class_weight=class_weight,
                  verbose=0)

        y_v_pred  = np.argmax(model.predict(X_v_seq,  verbose=0), axis=1)
        y_te_pred = np.argmax(model.predict(X_te_seq, verbose=0), axis=1)

        val_m  = get_metrics(y_v_seq,  y_v_pred,  "val")
        test_m = get_metrics(y_te_seq, y_te_pred, "test")

        exp_label = f"LSTM_{cfg_name}_h{HORIZON}"
        save_cm(y_te_seq, y_te_pred, f"{exp_label} (test)",
                os.path.join(FIGURES_DIR, f"cm_{exp_label}.png"))

        row = {"experiment": exp_label, "model": "LSTM",
               "config": cfg_name, "horizon": HORIZON,
               "lstm_units": cfg["lstm_units"], "dense_units": cfg["dense_units"],
               "dropout": cfg["dropout"], "batch": cfg["batch"], "lr": cfg["lr"],
               **val_m, **test_m}
        results.append(row)
        print(f"    val_f1={row['val_macro_f1']:.3f}  "
              f"test_f1={row['test_macro_f1']:.3f}  "
              f"test_high_recall={row['test_high_recall']:.3f}")

    # Classical models
    for model_name in ["RandomForest", "XGBoost"]:
        print(f"  {model_name} h{HORIZON}")
        try:
            if model_name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                             class_weight="balanced",
                                             random_state=42, n_jobs=-1)
            else:
                from xgboost import XGBClassifier
                clf = XGBClassifier(n_estimators=200, max_depth=6,
                                    learning_rate=0.1, eval_metric="mlogloss",
                                    random_state=42, n_jobs=-1, verbosity=0)

            clf.fit(X_tr, y_tr)
            y_v_pred  = clf.predict(X_v)
            y_te_pred = clf.predict(X_te)

            val_m  = get_metrics(y_v,  y_v_pred,  "val")
            test_m = get_metrics(y_te, y_te_pred, "test")

            exp_label = f"{model_name}_h{HORIZON}"
            save_cm(y_te, y_te_pred, f"{exp_label} (test)",
                    os.path.join(FIGURES_DIR, f"cm_{exp_label}.png"))

            row = {"experiment": exp_label, "model": model_name,
                   "config": "default", "horizon": HORIZON,
                   "lstm_units": "-", "dense_units": "-",
                   "dropout": "-", "batch": "-", "lr": "-",
                   **val_m, **test_m}
            results.append(row)
            print(f"    val_f1={row['val_macro_f1']:.3f}  "
                  f"test_f1={row['test_macro_f1']:.3f}  "
                  f"test_high_recall={row['test_high_recall']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Append to existing results
    new_df   = pd.DataFrame(results)
    csv_path = os.path.join(REPORTS_DIR, "experiment_results.csv")

    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(csv_path, index=False)

    # Print updated summary
    display_cols = ["experiment", "test_accuracy", "test_macro_f1",
                    "test_high_recall", "val_macro_f1"]

    print("\n" + "=" * 65)
    print("H7 RESULTS")
    print("=" * 65)
    print(new_df[display_cols].sort_values(
        "test_macro_f1", ascending=False).to_string(index=False))

    print("\n" + "=" * 65)
    print("FULL UPDATED RANKING (top 10)")
    print("=" * 65)
    print(combined[display_cols].sort_values(
        "test_macro_f1", ascending=False).head(10).to_string(index=False))

    best = combined.sort_values(
        ["test_macro_f1", "test_high_recall"], ascending=False).iloc[0]
    print(f"\nBEST OVERALL: {best['experiment']}")
    print(f"  test_macro_f1={best['test_macro_f1']:.4f}  "
          f"test_high_recall={best['test_high_recall']:.4f}  "
          f"val_macro_f1={best['val_macro_f1']:.4f}")

    print(f"\nSaved: {csv_path}")
    print("Done.")