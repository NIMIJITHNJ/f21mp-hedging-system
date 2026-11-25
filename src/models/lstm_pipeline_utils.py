"""
Shared utilities for leakage-safe, reproducible LSTM model selection.

"""

from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# TensorFlow reproducibility flags must be set before TensorFlow is imported.
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# Allow execution from the repository root.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from features.build_features import fit_garch, load_raw, TRADING_DAYS  # noqa: E402
from features.technical_indicators import (  # noqa: E402
    atr,
    log_return,
    macd,
    oi_moving_average,
    oi_pct_change,
    oi_volume_ratio,
    rolling_volatility,
    rsi,
    sma_crossover,
    vwap,
)

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
OUTPUT_MODELS_DIR = ROOT / "outputs" / "models"
OUTPUT_REPORTS_DIR = ROOT / "outputs" / "reports"
OUTPUT_FIGURES_DIR = ROOT / "outputs" / "figures"

for directory in [OUTPUT_MODELS_DIR, OUTPUT_REPORTS_DIR, OUTPUT_FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

STOCK_SYMBOLS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS",
    "SBIN", "LT", "BHARTIARTL", "ITC", "AXISBANK",
]

SEQ_LEN = 20
N_FEATURES = 25
N_CLASSES = 3
CLASS_NAMES = ["Low", "Medium", "High"]


@dataclass(frozen=True)
class LSTMConfig:
    name: str
    lstm_units: int
    dense_units: int
    dropout: float
    batch_size: int
    learning_rate: float


CONFIGS: Dict[str, LSTMConfig] = {
    "C1": LSTMConfig("C1", 32, 16, 0.2, 16, 1e-3),
    "C2": LSTMConfig("C2", 32, 16, 0.3, 16, 5e-4),
    "C3": LSTMConfig("C3", 64, 32, 0.2, 32, 1e-3),
    "C4": LSTMConfig("C4", 64, 32, 0.2, 32, 5e-4),
    "C5": LSTMConfig("C5", 64, 32, 0.3, 32, 5e-4),
    "C6": LSTMConfig("C6", 128, 64, 0.3, 32, 5e-4),
}


def set_determinism(seed: int) -> None:
    """Reset TensorFlow/Keras and all relevant random generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        # Older TensorFlow versions may not expose this helper.
        pass


def _load_macro() -> pd.DataFrame:
    path = EXTERNAL_DIR / "macro_combined.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing macro file: {path}")
    macro = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    macro.index = pd.DatetimeIndex(macro.index).normalize()
    return macro


def _future_volatility_and_end_dates(
    spot_close: pd.Series,
    horizon: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return forward realised volatility and its target-window end date.

    For row t, the target is the annualised standard deviation of returns from
    t+1 to t+horizon. The end-date series is used to purge samples whose target
    window crosses a train/validation/test boundary.
    """
    if horizon < 2:
        raise ValueError("Horizon must be at least 2 trading days for volatility labels.")

    returns = np.log(spot_close / spot_close.shift(1))
    forward_vol = returns.rolling(horizon).std() * np.sqrt(TRADING_DAYS)
    forward_vol = forward_vol.shift(-horizon)

    index_series = pd.Series(spot_close.index, index=spot_close.index)
    forward_end = index_series.shift(-horizon)
    return forward_vol, forward_end


def build_feature_frame(
    garch_fit_end: str,
    label_threshold_fit_end: str,
    horizon: int,
) -> Tuple[pd.DataFrame, List[str], float, float]:
    """Build the full feature frame and forward-looking labels.

    GARCH parameters and risk-label thresholds are fitted only using dates up
    to the supplied boundaries. This function does not split or scale data.
    """
    spot = load_raw("NIFTY50_spot")
    fut = load_raw("NIFTY_futures_continuous")
    macro = _load_macro()

    stock_returns: Dict[str, pd.Series] = {}
    for symbol in STOCK_SYMBOLS:
        stock_returns[symbol] = log_return(load_raw(symbol)["close"]).rename(f"ret_{symbol}")

    macd_line, macd_signal, _ = macd(spot["close"], fast=12, slow=26, signal=9)
    spot_log_returns = log_return(spot["close"])
    garch_mask = spot_log_returns.index <= pd.Timestamp(garch_fit_end)
    garch_sigma, _ = fit_garch(spot_log_returns[garch_mask], spot_log_returns)

    frame = pd.DataFrame(index=spot.index)
    frame["rsi_14"] = rsi(spot["close"], period=14)
    frame["macd_12_26"] = macd_line
    frame["macd_signal_9"] = macd_signal
    frame["vwap_20"] = vwap(spot["high"], spot["low"], spot["close"], fut["volume"])
    frame["atr_14"] = atr(spot["high"], spot["low"], spot["close"], period=14)
    frame["sma_cross_10_20"] = sma_crossover(spot["close"], fast=10, slow=20)
    frame["oi_pct_change"] = oi_pct_change(fut["oi"], period=1).fillna(0)
    frame["oi_vol_ratio"] = oi_volume_ratio(fut["oi"], fut["volume"]).fillna(0)
    frame["oi_ma_20"] = oi_moving_average(fut["oi"], window=20).fillna(0)
    frame["realised_vol_20"] = rolling_volatility(spot["close"], window=20)
    frame["garch_sigma"] = garch_sigma
    frame["gold_usd"] = macro["gold_usd"]
    frame["brent_usd"] = macro["brent_usd"]
    frame["usdinr"] = macro["usdinr"]
    frame["india_10y"] = macro["india_10y"]
    for symbol in STOCK_SYMBOLS:
        frame[f"ret_{symbol}"] = stock_returns[symbol]

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    feature_cols = list(frame.columns)
    if len(feature_cols) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, found {len(feature_cols)}")

    forward_vol, forward_end = _future_volatility_and_end_dates(spot["close"], horizon)
    frame["future_vol"] = forward_vol.reindex(frame.index)
    frame["target_end_date"] = forward_end.reindex(frame.index)

    threshold_mask = (
        (frame.index <= pd.Timestamp(label_threshold_fit_end))
        & (frame["target_end_date"] <= pd.Timestamp(label_threshold_fit_end))
    )
    threshold_values = frame.loc[threshold_mask, "future_vol"].dropna()
    if threshold_values.empty:
        raise ValueError("No rows available to fit label thresholds.")

    q33 = float(threshold_values.quantile(1 / 3))
    q67 = float(threshold_values.quantile(2 / 3))
    frame["risk_label"] = pd.cut(
        frame["future_vol"],
        bins=[-np.inf, q33, q67, np.inf],
        labels=[0, 1, 2],
    ).astype("Int64")
    frame = frame.dropna(subset=["future_vol", "target_end_date", "risk_label"])
    frame["risk_label"] = frame["risk_label"].astype(np.int32)
    return frame, feature_cols, q33, q67


def _split_and_scale(
    frame: pd.DataFrame,
    feature_cols: List[str],
    train_end: str,
    validation_start: str,
    validation_end: str,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Create a purged chronological train/validation split and scale safely."""
    train_end_ts = pd.Timestamp(train_end)
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)

    train = frame[
        (frame.index <= train_end_ts)
        & (frame["target_end_date"] <= train_end_ts)
    ]
    validation = frame[
        (frame.index >= validation_start_ts)
        & (frame.index <= validation_end_ts)
        & (frame["target_end_date"] <= validation_end_ts)
    ]

    if len(train) <= SEQ_LEN:
        raise ValueError(f"Not enough training rows for horizon={horizon}: {len(train)}")
    if validation.empty:
        raise ValueError(f"No validation rows for horizon={horizon}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols])
    x_validation = scaler.transform(validation[feature_cols])
    y_train = train["risk_label"].to_numpy(dtype=np.int32)
    y_validation = validation["risk_label"].to_numpy(dtype=np.int32)
    return x_train, y_train, x_validation, y_validation, scaler


def build_cv_fold(
    horizon: int,
    validation_year: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one expanding-window CV fold with a one-year validation period."""
    train_end = f"{validation_year - 1}-12-31"
    validation_start = f"{validation_year}-01-01"
    validation_end = f"{validation_year}-12-31"
    frame, feature_cols, _, _ = build_feature_frame(
        garch_fit_end=train_end,
        label_threshold_fit_end=train_end,
        horizon=horizon,
    )
    x_train, y_train, x_validation, y_validation, _ = _split_and_scale(
        frame=frame,
        feature_cols=feature_cols,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        horizon=horizon,
    )
    return x_train, y_train, x_validation, y_validation


def build_final_train_test(
    horizon: int,
    pretest_end: str = "2023-12-31",
    test_start: str = "2024-01-01",
) -> Dict[str, Any]:
    """Build the final purged pre-test training set and untouched test set."""
    frame, feature_cols, q33, q67 = build_feature_frame(
        garch_fit_end=pretest_end,
        label_threshold_fit_end=pretest_end,
        horizon=horizon,
    )
    pretest_end_ts = pd.Timestamp(pretest_end)
    test_start_ts = pd.Timestamp(test_start)

    train = frame[
        (frame.index <= pretest_end_ts)
        & (frame["target_end_date"] <= pretest_end_ts)
    ]
    test = frame[frame.index >= test_start_ts]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols])
    x_test = scaler.transform(test[feature_cols])
    y_train = train["risk_label"].to_numpy(dtype=np.int32)
    y_test = test["risk_label"].to_numpy(dtype=np.int32)
    test_dates = test.index.to_numpy(dtype="datetime64[ns]")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "test_dates": test_dates,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "q33": q33,
        "q67": q67,
        "train_rows": len(train),
        "test_rows": len(test),
    }


def build_sequences(x: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN) -> Tuple[np.ndarray, np.ndarray]:
    x_sequences, y_sequences = [], []
    for i in range(seq_len, len(x) + 1):
        x_sequences.append(x[i - seq_len:i])
        y_sequences.append(y[i - 1])
    return np.asarray(x_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.int32)


def build_sequences_with_context(
    x_context: np.ndarray,
    x_split: np.ndarray,
    y_split: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(x_context) != seq_len - 1:
        raise ValueError(f"Expected {seq_len - 1} context rows, found {len(x_context)}")
    x_full = np.vstack([x_context, x_split])
    x_sequences, y_sequences = [], []
    for i in range(seq_len, len(x_full) + 1):
        x_sequences.append(x_full[i - seq_len:i])
        y_sequences.append(y_split[i - seq_len])
    return np.asarray(x_sequences, dtype=np.float32), np.asarray(y_sequences, dtype=np.int32)


def build_lstm(config: LSTMConfig) -> Sequential:
    model = Sequential([
        Input(shape=(SEQ_LEN, N_FEATURES)),
        LSTM(config.lstm_units, return_sequences=True),
        Dropout(config.dropout),
        LSTM(config.lstm_units, return_sequences=False),
        Dropout(config.dropout),
        Dense(config.dense_units, activation="relu"),
        Dense(N_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_balanced_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.array([0, 1, 2], dtype=np.int32)
    present = np.unique(y)
    if not np.array_equal(present, classes):
        raise ValueError(f"Training fold does not contain all classes: {present.tolist()}")
    values = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(cls): float(weight) for cls, weight in zip(classes, values)}


def extract_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "val") -> Dict[str, float]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        f"{prefix}_low_recall": float(report["Low"]["recall"]),
        f"{prefix}_medium_recall": float(report["Medium"]["recall"]),
        f"{prefix}_high_recall": float(report["High"]["recall"]),
    }


def train_for_validation(
    config: LSTMConfig,
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    max_epochs: int = 100,
    early_stopping_patience: int = 15,
) -> Tuple[Sequential, Dict[str, float], Dict[str, List[float]], int]:
    """Train one fresh LSTM run and return validation metrics."""
    set_determinism(seed)
    x_train_seq, y_train_seq = build_sequences(x_train, y_train)
    x_validation_seq, y_validation_seq = build_sequences_with_context(
        x_train[-(SEQ_LEN - 1):], x_validation, y_validation
    )
    y_train_categorical = to_categorical(y_train_seq, N_CLASSES)
    y_validation_categorical = to_categorical(y_validation_seq, N_CLASSES)

    model = build_lstm(config)
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=7,
            factor=0.5,
            min_lr=1e-6,
            verbose=0,
        ),
    ]
    history = model.fit(
        x_train_seq,
        y_train_categorical,
        validation_data=(x_validation_seq, y_validation_categorical),
        epochs=max_epochs,
        batch_size=config.batch_size,
        callbacks=callbacks,
        class_weight=compute_balanced_class_weights(y_train_seq),
        shuffle=False,
        verbose=0,
    ).history

    predictions = np.argmax(model.predict(x_validation_seq, verbose=0), axis=1)
    metrics = extract_metrics(y_validation_seq, predictions, prefix="val")
    best_epoch = int(np.argmin(history["val_loss"]) + 1)
    return model, metrics, history, best_epoch


def train_final_model(
    config: LSTMConfig,
    seed: int,
    epochs: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Sequential, Dict[str, List[float]]]:
    """Train the locked configuration on all pre-test rows for fixed epochs."""
    set_determinism(seed)
    x_train_seq, y_train_seq = build_sequences(x_train, y_train)
    y_train_categorical = to_categorical(y_train_seq, N_CLASSES)
    model = build_lstm(config)
    history = model.fit(
        x_train_seq,
        y_train_categorical,
        epochs=epochs,
        batch_size=config.batch_size,
        class_weight=compute_balanced_class_weights(y_train_seq),
        shuffle=False,
        verbose=1,
    ).history
    return model, history


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def config_to_dict(config: LSTMConfig) -> Dict[str, Any]:
    return asdict(config)


def save_scaler(path: Path, scaler: StandardScaler) -> None:
    with path.open("wb") as file:
        pickle.dump(scaler, file)