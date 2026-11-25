import os
import sys
import json
import pickle
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# Determinism — ALL must be set before importing TensorFlow
os.environ["PYTHONHASHSEED"]         = "42"
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
sys.path.insert(0, "src")

import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

from features.build_features import make_risk_labels, fit_garch, load_raw
from features.technical_indicators import (
    rsi, macd, vwap, atr, sma_crossover,
    oi_pct_change, oi_volume_ratio, oi_moving_average,
    log_return, rolling_volatility,
)

# Output paths
REPORTS_DIR  = "outputs/reports/final_walkforward"
MODELS_DIR   = "outputs/models/final_walkforward"
FIGURES_DIR  = "outputs/figures/final_walkforward"
EXTERNAL_DIR = "data/external"

for d in [REPORTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

CHECKPOINT_PATH = os.path.join(REPORTS_DIR,
                                "walkforward_results_checkpoint.csv")

# Walk-forward folds
FOLDS = [
    {"fold": 1, "train_end": "2018-12-31",
     "val_end": "2019-12-31", "eval_end": "2020-12-31"},
    {"fold": 2, "train_end": "2019-12-31",
     "val_end": "2020-12-31", "eval_end": "2021-12-31"},
    {"fold": 3, "train_end": "2020-12-31",
     "val_end": "2021-12-31", "eval_end": "2022-12-31"},
    {"fold": 4, "train_end": "2021-12-31",
     "val_end": "2022-12-31", "eval_end": "2023-12-31"},
    {"fold": 5, "train_end": "2022-12-31",
     "val_end": "2023-12-31", "eval_end": "2024-12-31"},
]

HOLDOUT = {
    "fold":      "final",
    "train_end": "2023-12-31",
    "val_end":   "2024-12-31",
    "eval_end":  "2026-03-31",
}

# Experiment grid
HORIZONS = [3, 5, 7, 10, 14]

LSTM_CONFIGS = {
    "C1": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.2,
           "batch": 16, "lr": 1e-3},
    "C2": {"lstm_units": 32,  "dense_units": 16, "dropout": 0.3,
           "batch": 16, "lr": 5e-4},
    "C3": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2,
           "batch": 32, "lr": 1e-3},
    "C4": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.2,
           "batch": 32, "lr": 5e-4},
    "C5": {"lstm_units": 64,  "dense_units": 32, "dropout": 0.3,
           "batch": 32, "lr": 5e-4},
    "C6": {"lstm_units": 128, "dense_units": 64, "dropout": 0.3,
           "batch": 32, "lr": 5e-4},
}

SEEDS              = [7, 21, 42, 84, 123]
FINAL_SEED         = 42          # fixed — not chosen based on results
SEQ_LEN            = 20
N_CLASSES          = 3
MAX_EPOCHS         = 100
ES_PATIENCE        = 15
MIN_CLASS_RECALL   = 0.15
MAX_COLLAPSE_RATE  = 0.20        # reject if min_recall < 0.05 in >20% of runs

STOCK_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY",
    "TCS", "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]

# Pre-load raw data once
print("Pre-loading raw data ...")
SPOT = load_raw("NIFTY50_spot")
FUT  = load_raw("NIFTY_futures_continuous")
MACRO = pd.read_csv(
    os.path.join(EXTERNAL_DIR, "macro_combined.csv"),
    parse_dates=["date"],
).set_index("date")
MACRO.index = pd.DatetimeIndex(MACRO.index).normalize()

STOCK_RETURNS = {}
for sym in STOCK_SYMBOLS:
    df = load_raw(sym)
    STOCK_RETURNS[sym] = log_return(df["close"]).rename(f"ret_{sym}")

print("Pre-computing technical indicators ...")
SPOT_LOG_RET = log_return(SPOT["close"])

BASE_FEATURES = pd.DataFrame(index=SPOT.index)
BASE_FEATURES["rsi_14"]          = rsi(SPOT["close"], period=14)
_ml, _ms, _                      = macd(SPOT["close"], fast=12, slow=26, signal=9)
BASE_FEATURES["macd_12_26"]      = _ml
BASE_FEATURES["macd_signal_9"]   = _ms
BASE_FEATURES["vwap_20"]         = vwap(SPOT["high"], SPOT["low"],
                                         SPOT["close"], FUT["volume"])
BASE_FEATURES["atr_14"]          = atr(SPOT["high"], SPOT["low"],
                                        SPOT["close"], period=14)
BASE_FEATURES["sma_cross_10_20"] = sma_crossover(SPOT["close"], fast=10, slow=20)
BASE_FEATURES["oi_pct_change"]   = oi_pct_change(FUT["oi"], period=1).fillna(0)
BASE_FEATURES["oi_vol_ratio"]    = oi_volume_ratio(FUT["oi"], FUT["volume"]).fillna(0)
BASE_FEATURES["oi_ma_20"]        = oi_moving_average(FUT["oi"], window=20).fillna(0)
BASE_FEATURES["realised_vol_20"] = rolling_volatility(SPOT["close"], window=20)
BASE_FEATURES["gold_usd"]        = MACRO["gold_usd"]
BASE_FEATURES["brent_usd"]       = MACRO["brent_usd"]
BASE_FEATURES["usdinr"]          = MACRO["usdinr"]
BASE_FEATURES["india_10y"]       = MACRO["india_10y"]
for sym in STOCK_SYMBOLS:
    BASE_FEATURES[f"ret_{sym}"]  = STOCK_RETURNS[sym]
BASE_FEATURES = BASE_FEATURES.replace([np.inf, -np.inf], np.nan)

print(f"Base features: {BASE_FEATURES.shape[1]} columns "
      f"(garch_sigma added per fold)")

# Fold data cache
FOLD_DATA_CACHE = {}


def build_fold_data(fold_def, horizon):
    """
    Build train/val/eval arrays for one fold and horizon.
    All boundaries purged by `horizon` business days to prevent
    forward-label leakage across period boundaries.
    GARCH, scaler, and label thresholds fitted on training only.
    """
    train_end_ts = pd.Timestamp(fold_def["train_end"])
    val_end_ts   = pd.Timestamp(fold_def["val_end"])
    eval_end_ts  = pd.Timestamp(fold_def["eval_end"])

    # Purge actual NSE trading rows (not generic weekdays)
    # BDay does not account for NSE market holidays — using actual
    # trading dates from SPOT index ensures correct purging.
    def get_safe_purge_end(end_date, horizon_days):
        trading_dates = SPOT.index[SPOT.index <= end_date]
        if len(trading_dates) <= horizon_days:
            raise ValueError(
                f"Not enough trading rows to purge {horizon_days} days "
                f"before {end_date.date()}"
            )
        return trading_dates[-(horizon_days + 1)]

    train_purge_end = get_safe_purge_end(train_end_ts, horizon)
    val_purge_end   = get_safe_purge_end(val_end_ts,   horizon)
    eval_purge_end  = get_safe_purge_end(eval_end_ts,  horizon)

    # GARCH on full training returns (not purged — returns have no forward look)
    garch_sigma, _ = fit_garch(
        SPOT_LOG_RET[SPOT_LOG_RET.index <= train_end_ts],
        SPOT_LOG_RET,
    )

    feat = BASE_FEATURES.copy()
    feat["garch_sigma"] = garch_sigma
    feat = feat.dropna()

    feature_cols = list(feat.columns)

    # Thresholds from PURGED training period only (fix 1+2)
    labels, q33, q67 = make_risk_labels(
        spot_close    = SPOT["close"],
        feature_index = feat.index,
        train_end     = train_purge_end.strftime("%Y-%m-%d"),
        horizon       = horizon,
    )
    feat["risk_label"] = labels
    feat = feat.dropna(subset=["risk_label"])

    all_cols = [c for c in feat.columns if c != "risk_label"]

    # All three splits purged at their boundaries
    train_df = feat[feat.index <= train_purge_end]
    val_df   = feat[(feat.index > train_end_ts) & (feat.index <= val_purge_end)]
    eval_df  = feat[(feat.index > val_end_ts)   & (feat.index <= eval_purge_end)]

    if len(train_df) < SEQ_LEN + 10 or len(val_df) < 5 or len(eval_df) < 5:
        return None

    X_tr_raw = train_df[all_cols].replace([np.inf, -np.inf], 0)
    X_v_raw  = val_df[all_cols].replace([np.inf, -np.inf], 0)
    X_e_raw  = eval_df[all_cols].replace([np.inf, -np.inf], 0)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)
    X_v    = scaler.transform(X_v_raw)
    X_e    = scaler.transform(X_e_raw)

    y_tr = train_df["risk_label"].values.astype(np.int32)
    y_v  = val_df["risk_label"].values.astype(np.int32)
    y_e  = eval_df["risk_label"].values.astype(np.int32)

    # Context windows: most recent SEQ_LEN-1 feature rows from each period
    # (feature rows only — no labels used as context)
    def ctx(df_slice):
        raw = df_slice[all_cols].replace([np.inf, -np.inf], 0).tail(SEQ_LEN - 1)
        return scaler.transform(raw)

    X_tr_context  = ctx(feat[feat.index <= train_end_ts])
    X_val_context = ctx(feat[(feat.index > train_end_ts) &
                              (feat.index <= val_end_ts)])

    n_features = X_tr.shape[1]

    return (X_tr, X_v, X_e,
            y_tr, y_v, y_e,
            scaler, n_features, all_cols,
            X_tr_context, X_val_context,
            q33, q67)


def get_cached_fold_data(fold_def, horizon):
    key = (fold_def["fold"], horizon)
    if key not in FOLD_DATA_CACHE:
        FOLD_DATA_CACHE[key] = build_fold_data(fold_def, horizon)
    return FOLD_DATA_CACHE[key]


# Sequence builders

def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X) + 1):
        Xs.append(X[i - seq_len: i])
        ys.append(y[i - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def build_sequences_with_context(X_ctx, X_split, y_split, seq_len):
    assert len(X_ctx) == seq_len - 1
    X_full = np.vstack([X_ctx, X_split])
    Xs, ys = [], []
    for i in range(seq_len, len(X_full) + 1):
        Xs.append(X_full[i - seq_len: i])
        ys.append(y_split[i - seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


# Model builder

def build_lstm(cfg, seed, input_features):
    tf.keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    model = Sequential([
        Input(shape=(SEQ_LEN, input_features)),
        LSTM(cfg["lstm_units"], return_sequences=True),
        Dropout(cfg["dropout"]),
        LSTM(cfg["lstm_units"], return_sequences=False),
        Dropout(cfg["dropout"]),
        Dense(cfg["dense_units"], activation="relu"),
        Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Metrics

def get_metrics(y_true, y_pred, split="eval"):
    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=[0, 1, 2], zero_division=0)
    rep      = classification_report(
        y_true, y_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2], zero_division=0, output_dict=True,
    )
    low_r  = rep["Low"]["recall"]
    med_r  = rep["Medium"]["recall"]
    high_r = rep["High"]["recall"]
    return {
        f"{split}_accuracy":    round(acc,                       4),
        f"{split}_macro_f1":    round(macro_f1,                  4),
        f"{split}_high_recall": round(high_r,                    4),
        f"{split}_low_recall":  round(low_r,                     4),
        f"{split}_med_recall":  round(med_r,                     4),
        f"{split}_min_recall":  round(min(low_r, med_r, high_r), 4),
    }


# Single run

def run_one(fold_def, horizon, cfg_name, cfg, seed):
    data = get_cached_fold_data(fold_def, horizon)
    if data is None:
        return None

    (X_tr, X_v, X_e, y_tr, y_v, y_e,
     scaler, n_features, cols,
     X_tr_context, X_val_context, q33, q67) = data

    X_tr_seq, y_tr_seq = build_sequences(X_tr, y_tr, SEQ_LEN)
    X_v_seq,  y_v_seq  = build_sequences_with_context(
        X_tr_context, X_v, y_v, SEQ_LEN)
    X_e_seq,  y_e_seq  = build_sequences_with_context(
        X_val_context, X_e, y_e, SEQ_LEN)

    if len(X_tr_seq) < 10 or len(X_v_seq) < 3 or len(X_e_seq) < 3:
        return None

    y_tr_cat = to_categorical(y_tr_seq, N_CLASSES)
    y_v_cat  = to_categorical(y_v_seq,  N_CLASSES)
    cw = compute_class_weight("balanced",
                              classes=np.array([0, 1, 2]), y=y_tr_seq)
    class_weight = {0: cw[0], 1: cw[1], 2: cw[2]}

    model = build_lstm(cfg, seed, X_tr_seq.shape[2])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", patience=7,
                          factor=0.5, min_lr=1e-6, verbose=0),
    ]

    model.fit(
        X_tr_seq, y_tr_cat,
        validation_data=(X_v_seq, y_v_cat),
        epochs=MAX_EPOCHS,
        batch_size=cfg["batch"],
        callbacks=callbacks,
        class_weight=class_weight,
        shuffle=False,
        verbose=0,
    )

    y_v_pred = np.argmax(model.predict(X_v_seq, verbose=0), axis=1)
    y_e_pred = np.argmax(model.predict(X_e_seq, verbose=0), axis=1)

    val_m  = get_metrics(y_v_seq, y_v_pred, "val")
    eval_m = get_metrics(y_e_seq, y_e_pred, "eval")

    tf.keras.backend.clear_session()

    return {
        "fold":       fold_def["fold"],
        "train_end":  fold_def["train_end"],
        "eval_end":   fold_def["eval_end"],
        "horizon":    horizon,
        "config":     cfg_name,
        "seed":       seed,
        "n_features": n_features,
        "train_rows": len(y_tr_seq),
        "val_rows":   len(y_v_seq),
        "eval_rows":  len(y_e_seq),
        **val_m,
        **eval_m,
    }


# Confusion matrix

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
    ax.set_title(title, color="#f0f0f5", fontsize=9, pad=6)
    ax.tick_params(colors="#9090a0")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()


# Main

if __name__ == "__main__":
    total = len(FOLDS) * len(HORIZONS) * len(LSTM_CONFIGS) * len(SEEDS)

    print("\n" + "=" * 70)
    print("HEDRA — WALK-FORWARD CV WITH FINAL HOLDOUT")
    print(f"  Folds     : {len(FOLDS)}  (eval 2020-2024)")
    print(f"  Holdout   : Jan 2025 - Mar 2026  (used ONCE)")
    print(f"  Horizons  : {HORIZONS}")
    print(f"  Configs   : {list(LSTM_CONFIGS.keys())}")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Total     : {total} runs")
    print(f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Checkpoint resume
    results        = []
    completed_keys = set()

    if os.path.exists(CHECKPOINT_PATH):
        existing_df = pd.read_csv(CHECKPOINT_PATH)
        results     = existing_df.to_dict("records")
        completed_keys = set(zip(
            existing_df["horizon"].astype(int),
            existing_df["config"],
            existing_df["fold"],
            existing_df["seed"].astype(int),
        ))
        print(f"Resuming from checkpoint: {len(results)} completed runs")

    done                   = 0
    executed_this_session  = 0
    start_time             = datetime.now()

    for horizon in HORIZONS:
        for fold_def in FOLDS:
            # Cache this fold×horizon once
            cache_key = (fold_def["fold"], horizon)
            if cache_key not in FOLD_DATA_CACHE:
                print(f"\n  Caching h{horizon} fold{fold_def['fold']} ...",
                      end=" ", flush=True)
                cached = get_cached_fold_data(fold_def, horizon)
                if cached is None:
                    print("SKIPPED (insufficient data)")
                    continue
                print(f"train={cached[3].shape[0]} "
                      f"val={cached[4].shape[0]} "
                      f"eval={cached[5].shape[0]}")

            for cfg_name, cfg in LSTM_CONFIGS.items():
                for seed in SEEDS:
                    run_key = (horizon, cfg_name, fold_def["fold"], seed)

                    if run_key in completed_keys:
                        done += 1
                        continue  # already done in previous run

                    done += 1
                    executed_this_session += 1
                    elapsed = int((datetime.now() - start_time).total_seconds())
                    eta_s   = int(elapsed / max(executed_this_session, 1)
                                  * (total - done))
                    eta_m   = eta_s // 60

                    print(f"  [{done:>3}/{total}] "
                          f"h{horizon} {cfg_name} f{fold_def['fold']} "
                          f"s={seed}  ETA~{eta_m}min",
                          end="  ", flush=True)

                    row = run_one(fold_def, horizon, cfg_name, cfg, seed)

                    if row:
                        results.append(row)
                        print(f"f1={row['eval_macro_f1']:.3f} "
                              f"hi={row['eval_high_recall']:.3f} "
                              f"min={row['eval_min_recall']:.3f}")
                    else:
                        print("SKIPPED")

            # Checkpoint after each horizon×fold block
            if results:
                pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)

    # Save full results
    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(REPORTS_DIR, "walkforward_results.csv"), index=False)
    print(f"\nSaved walkforward_results.csv  ({len(results_df)} rows)")

    # Aggregate
    agg = results_df.groupby(["horizon", "config"]).agg(
        mean_eval_accuracy   =("eval_accuracy",    "mean"),
        mean_eval_macro_f1   =("eval_macro_f1",    "mean"),
        mean_eval_high_recall=("eval_high_recall", "mean"),
        mean_eval_low_recall =("eval_low_recall",  "mean"),
        mean_eval_med_recall =("eval_med_recall",  "mean"),
        mean_eval_min_recall =("eval_min_recall",  "mean"),
        std_eval_macro_f1    =("eval_macro_f1",    "std"),
        std_eval_high_recall =("eval_high_recall", "std"),
        collapse_rate        =("eval_min_recall",
                               lambda x: (x < 0.05).mean()),
        n_runs               =("fold",             "count"),
    ).reset_index()

    # Valid = passes mean min-recall AND collapse-rate filters
    agg["passes_min_recall"] = (
        (agg["mean_eval_min_recall"] >= MIN_CLASS_RECALL) &
        (agg["collapse_rate"]        <= MAX_COLLAPSE_RATE)
    )
    agg.to_csv(
        os.path.join(REPORTS_DIR, "walkforward_aggregated.csv"), index=False)

    valid   = agg[agg["passes_min_recall"]].copy()
    rank_df = valid if len(valid) > 0 else agg

    display_cols = [
        "horizon", "config",
        "mean_eval_accuracy", "mean_eval_macro_f1",
        "mean_eval_high_recall", "mean_eval_min_recall",
        "std_eval_macro_f1", "collapse_rate", "n_runs",
    ]

    print("\n" + "=" * 70)
    print(f"VALID MODELS  (mean_min_recall>={MIN_CLASS_RECALL} "
          f"AND collapse_rate<={MAX_COLLAPSE_RATE})")
    print("=" * 70)
    print(rank_df.sort_values("mean_eval_macro_f1",
          ascending=False)[display_cols].to_string(index=False))
    print(f"\nRejected: {len(agg) - len(valid)} configs")

    best_row = rank_df.sort_values(
        ["mean_eval_macro_f1", "mean_eval_high_recall"],
        ascending=False,
    ).iloc[0]

    best_horizon = int(best_row["horizon"])
    best_config  = best_row["config"]
    best_cfg     = LSTM_CONFIGS[best_config]

    print(f"\n{'='*70}")
    print("BEST MODEL (walk-forward selection 2020-2024)")
    print(f"  Horizon       : {best_horizon}-day")
    print(f"  Config        : {best_config}  {best_cfg}")
    print(f"  Mean macro-F1 : {best_row['mean_eval_macro_f1']:.4f} "
          f"± {best_row['std_eval_macro_f1']:.4f}")
    print(f"  Mean High rec : {best_row['mean_eval_high_recall']:.4f}")
    print(f"  Mean min rec  : {best_row['mean_eval_min_recall']:.4f}")
    print(f"  Collapse rate : {best_row['collapse_rate']:.2f}")
    print(f"{'='*70}")

    # Retrain on max data, evaluate on holdout
    print(f"\nRetraining with seed={FINAL_SEED} (fixed, not chosen from results)")
    print("Train: 2012-2023  |  Val (early stop): 2024  |  Eval: 2025-Mar2026")

    holdout_data = build_fold_data(HOLDOUT, best_horizon)

    if holdout_data is None:
        print("ERROR: Could not build holdout data.")
    else:
        (X_tr, X_v, X_e, y_tr, y_v, y_e,
         best_scaler, n_feat, feat_cols,
         X_tr_context, X_val_context,
         h_q33, h_q67) = holdout_data

        X_tr_seq, y_tr_seq = build_sequences(X_tr, y_tr, SEQ_LEN)
        X_v_seq,  y_v_seq  = build_sequences_with_context(
            X_tr_context, X_v, y_v, SEQ_LEN)
        X_e_seq,  y_e_seq  = build_sequences_with_context(
            X_val_context, X_e, y_e, SEQ_LEN)

        y_tr_cat = to_categorical(y_tr_seq, N_CLASSES)
        y_v_cat  = to_categorical(y_v_seq,  N_CLASSES)
        cw = compute_class_weight("balanced",
                                  classes=np.array([0, 1, 2]), y=y_tr_seq)
        class_weight = {0: cw[0], 1: cw[1], 2: cw[2]}

        best_model = build_lstm(best_cfg, FINAL_SEED, X_tr_seq.shape[2])
        callbacks  = [
            EarlyStopping(monitor="val_loss", patience=ES_PATIENCE,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=7,
                              factor=0.5, min_lr=1e-6, verbose=1),
        ]

        best_model.fit(
            X_tr_seq, y_tr_cat,
            validation_data=(X_v_seq, y_v_cat),
            epochs=MAX_EPOCHS,
            batch_size=best_cfg["batch"],
            callbacks=callbacks,
            class_weight=class_weight,
            shuffle=False,
            verbose=1,
        )

        y_final      = np.argmax(
            best_model.predict(X_e_seq, verbose=0), axis=1)
        final_m      = get_metrics(y_e_seq, y_final, "holdout")
        final_report = classification_report(
            y_e_seq, y_final,
            target_names=["Low", "Medium", "High"],
            labels=[0, 1, 2], zero_division=0,
        )

        print(f"\nFINAL HOLDOUT (Jan 2025 - Mar 2026) — confirmatory only:")
        print(f"  Accuracy  : {final_m['holdout_accuracy']:.4f}")
        print(f"  Macro-F1  : {final_m['holdout_macro_f1']:.4f}")
        print(f"  High rec  : {final_m['holdout_high_recall']:.4f}")
        print(f"  Min rec   : {final_m['holdout_min_recall']:.4f}")
        print("\n", final_report)

        # Save model
        best_model.save(os.path.join(MODELS_DIR, "best_model.keras"))

        # Save scaler
        with open(os.path.join(MODELS_DIR,
                               "best_model_scaler.pkl"), "wb") as f:
            pickle.dump(best_scaler, f)

        # Save metadata with thresholds
        metadata = {
            "horizon":          best_horizon,
            "config_name":      best_config,
            "config":           best_cfg,
            "seed":             FINAL_SEED,
            "sequence_length":  SEQ_LEN,
            "feature_columns":  feat_cols,
            "n_features":       n_feat,
            "risk_label_definition": (
                f"{best_horizon}-day forward realised volatility tertiles"
            ),
            "risk_label_thresholds": {
                "q33": float(h_q33),
                "q67": float(h_q67),
                "low_label":    0,
                "medium_label": 1,
                "high_label":   2,
            },
            "training_end":     HOLDOUT["train_end"],
            "validation_end":   HOLDOUT["val_end"],
            "holdout_period":   "2025-01-01 to 2026-03-31",
            "min_class_recall": MIN_CLASS_RECALL,
            "max_collapse_rate":MAX_COLLAPSE_RATE,
            "selection_folds":  [f["eval_end"] for f in FOLDS],
            "walk_forward_mean_macro_f1":
                float(best_row["mean_eval_macro_f1"]),
            "walk_forward_std_macro_f1":
                float(best_row["std_eval_macro_f1"]),
            "walk_forward_mean_high_recall":
                float(best_row["mean_eval_high_recall"]),
            "walk_forward_collapse_rate":
                float(best_row["collapse_rate"]),
            "holdout_accuracy":    float(final_m["holdout_accuracy"]),
            "holdout_macro_f1":    float(final_m["holdout_macro_f1"]),
            "holdout_high_recall": float(final_m["holdout_high_recall"]),
        }
        with open(os.path.join(MODELS_DIR, "best_model_metadata.json"),
                  "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        save_cm(
            y_e_seq, y_final,
            f"h{best_horizon} {best_config} — Holdout 2025-Mar2026",
            os.path.join(FIGURES_DIR,
                         "best_model_confusion_matrix.png"),
        )

        with open(os.path.join(REPORTS_DIR,
                               "best_model_report.txt"), "w") as f:
            f.write("HEDRA — FINAL MODEL REPORT\n" + "="*60 + "\n\n")
            f.write("Selection : Walk-forward CV "
                    "(5 folds, eval 2020-2024)\n")
            f.write("Holdout   : Jan 2025 - Mar 2026 "
                    "(confirmatory, used ONCE)\n\n")
            f.write(f"Horizon   : {best_horizon}-day\n")
            f.write(f"Config    : {best_config}  {best_cfg}\n")
            f.write(f"Seed      : {FINAL_SEED} (fixed)\n\n")
            f.write("WALK-FORWARD SELECTION (primary result)\n")
            f.write("-"*40 + "\n")
            f.write(f"Mean macro-F1   : "
                    f"{best_row['mean_eval_macro_f1']:.4f}"
                    f" ± {best_row['std_eval_macro_f1']:.4f}\n")
            f.write(f"Mean High rec   : "
                    f"{best_row['mean_eval_high_recall']:.4f}\n")
            f.write(f"Mean min rec    : "
                    f"{best_row['mean_eval_min_recall']:.4f}\n")
            f.write(f"Collapse rate   : "
                    f"{best_row['collapse_rate']:.2f}\n\n")
            f.write("HOLDOUT (confirmatory)\n" + "-"*40 + "\n")
            f.write(final_report)

        print(f"\nSaved: best_model.keras")
        print(f"Saved: best_model_scaler.pkl")
        print(f"Saved: best_model_metadata.json")
        print(f"Saved: best_model_report.txt")
        print(f"Saved: best_model_confusion_matrix.png")

    # Summary
    with open(os.path.join(REPORTS_DIR,
                           "walkforward_summary.txt"), "w") as f:
        f.write("HEDRA — WALK-FORWARD CV SUMMARY\n" + "="*70 + "\n\n")
        f.write(f"Total runs : {total}  Completed : {len(results_df)}\n")
        f.write(f"Folds      : {len(FOLDS)} (eval 2020-2024)\n")
        f.write(f"Horizons   : {HORIZONS}  Seeds : {SEEDS}\n\n")
        f.write("TOP VALID MODELS\n" + "-"*70 + "\n")
        f.write(rank_df.sort_values("mean_eval_macro_f1",
                ascending=False)[display_cols].to_string(index=False))
        f.write(f"\n\nBEST: h{best_horizon} {best_config}\n")

    elapsed_total = int(
        (datetime.now() - start_time).total_seconds() // 60)
    print(f"\nCompleted : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runtime   : ~{elapsed_total} minutes")
    print(f"Outputs   : {REPORTS_DIR}/")