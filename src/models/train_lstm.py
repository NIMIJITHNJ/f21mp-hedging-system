# Trains a two-layer LSTM risk classifier on the 25-feature dataset.

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
 
# Reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
 
import tensorflow as tf
tf.random.set_seed(SEED)
 
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
 
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
 
# Paths
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "outputs/models"
REPORTS_DIR   = "outputs/reports"
FIGURES_DIR   = "outputs/figures"
 
for d in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)
 
# Hyperparameters
SEQ_LEN    = 20       # look-back window in trading days
N_FEATURES = 25
N_CLASSES  = 3
LSTM_UNITS = 64
DROPOUT    = 0.2
DENSE_UNITS= 32
LR         = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
ES_PATIENCE= 15
LR_PATIENCE= 7
LR_FACTOR  = 0.5
 
 
# Data loading
 
def load_split(split):
    """Load X and y for a given split ('train', 'val', 'test')."""
    X = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"X_{split}.csv"),
        index_col=0, parse_dates=True,
    ).values.astype(np.float32)
 
    y = pd.read_csv(
        os.path.join(PROCESSED_DIR, f"y_{split}.csv"),
        index_col=0, parse_dates=True,
    ).squeeze().values.astype(np.int32)
 
    return X, y
 
 
# Sequence builder
 
def build_sequences(X, y, seq_len):
    """
    Build overlapping sequences for the training split.

    """
    Xs, ys = [], []
    for i in range(seq_len, len(X) + 1):
        Xs.append(X[i - seq_len : i])
        ys.append(y[i - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)
 
 
def build_sequences_with_context(X_context, X_split, y_split, seq_len):
    """
    Build sequences for val/test using prior-split rows as context window.

    """
    assert len(X_context) == seq_len - 1, (
        f"X_context must have {seq_len - 1} rows, got {len(X_context)}"
    )
    X_full = np.vstack([X_context, X_split])  # (seq_len-1 + n, n_features)
 
    Xs, ys = [], []
    for i in range(seq_len, len(X_full) + 1):
        Xs.append(X_full[i - seq_len : i])
        ys.append(y_split[i - seq_len])       # label aligned to X_split[0], [1], ...
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)
 
 
# Model definition
 
def build_model(seq_len, n_features, n_classes,
                lstm_units, dropout, dense_units, lr):
    """
    Two-layer stacked LSTM with a dense head.

    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ])
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
 
 
# Plotting helpers
 
def plot_confusion_matrix(y_true, y_pred, path):
    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    labels = ["Low", "Medium", "High"]
 
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d0d0f")
    ax.set_facecolor("#16161a")
 
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="#2a2a30",
        ax=ax, cbar=False,
    )
    ax.set_xlabel("Predicted", color="#9090a0", fontsize=11)
    ax.set_ylabel("Actual",    color="#9090a0", fontsize=11)
    ax.set_title("Confusion Matrix — Validation Set", color="#f0f0f5",
                 fontsize=13, pad=12)
    ax.tick_params(colors="#9090a0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a30")
 
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
 
 
def plot_training_curves(history, path):
    epochs = range(1, len(history["loss"]) + 1)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d0d0f")
 
    for ax in [ax1, ax2]:
        ax.set_facecolor("#16161a")
        ax.tick_params(colors="#9090a0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a30")
 
    # Loss
    ax1.plot(epochs, history["loss"],     color="#6c63ff", label="Train loss")
    ax1.plot(epochs, history["val_loss"], color="#f87171", label="Val loss",
             linestyle="--")
    ax1.set_title("Loss",     color="#f0f0f5", fontsize=12)
    ax1.set_xlabel("Epoch",   color="#9090a0")
    ax1.set_ylabel("Loss",    color="#9090a0")
    ax1.legend(facecolor="#1e1e28", edgecolor="#2a2a30",
               labelcolor="#9090a0")
 
    # Accuracy
    ax2.plot(epochs, history["accuracy"],     color="#6c63ff", label="Train acc")
    ax2.plot(epochs, history["val_accuracy"], color="#f87171", label="Val acc",
             linestyle="--")
    ax2.set_title("Accuracy",  color="#f0f0f5", fontsize=12)
    ax2.set_xlabel("Epoch",    color="#9090a0")
    ax2.set_ylabel("Accuracy", color="#9090a0")
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(facecolor="#1e1e28", edgecolor="#2a2a30",
               labelcolor="#9090a0")
 
    fig.suptitle("LSTM Training History", color="#f0f0f5",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {path}")
 
 
# Main
 
if __name__ == "__main__":
 
    # Load data
    print("Loading processed data ...")
    X_train_raw, y_train_raw = load_split("train")
    X_val_raw,   y_val_raw   = load_split("val")
    X_test_raw,  y_test_raw  = load_split("test")
 
    print(f"  Raw shapes — train: {X_train_raw.shape}  "
          f"val: {X_val_raw.shape}  test: {X_test_raw.shape}")
 
    # Build sequences
    print(f"\nBuilding {SEQ_LEN}-day sequences ...")

    X_train, y_train = build_sequences(X_train_raw, y_train_raw, SEQ_LEN)
    X_val,   y_val   = build_sequences_with_context(
        X_context = X_train_raw[-(SEQ_LEN - 1):],
        X_split   = X_val_raw,
        y_split   = y_val_raw,
        seq_len   = SEQ_LEN,
    )
    X_test,  y_test  = build_sequences_with_context(
        X_context = X_val_raw[-(SEQ_LEN - 1):],
        X_split   = X_test_raw,
        y_split   = y_test_raw,
        seq_len   = SEQ_LEN,
    )
 
    print(f"  Sequence shapes — train: {X_train.shape}  "
          f"val: {X_val.shape}  test: {X_test.shape}")
 
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, N_CLASSES)
    y_val_cat   = to_categorical(y_val,   N_CLASSES)
    y_test_cat  = to_categorical(y_test,  N_CLASSES)
 
    # Build model
    print("\nBuilding LSTM model ...")
    model = build_model(
        seq_len=SEQ_LEN, n_features=N_FEATURES, n_classes=N_CLASSES,
        lstm_units=LSTM_UNITS, dropout=DROPOUT,
        dense_units=DENSE_UNITS, lr=LR,
    )
    model.summary()
 
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=ES_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", patience=LR_PATIENCE,
            factor=LR_FACTOR, min_lr=1e-6, verbose=1,
        ),
    ]
 
    # Train
    print(f"\nTraining (max {MAX_EPOCHS} epochs, early stopping patience={ES_PATIENCE}) ...")
    # Class weights — prevents model collapsing to predicting only Medium.
    from sklearn.utils.class_weight import compute_class_weight
    cw_values = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=y_train,
    )
    class_weight = {0: cw_values[0], 1: cw_values[1], 2: cw_values[2]}
    print(f"  Class weights: {class_weight}")
 
    hist = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
 
    # Evaluate
    print("\nEvaluating ...")
    y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
 
    val_acc  = accuracy_score(y_val,  y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
 
    print(f"\n  Validation accuracy : {val_acc:.4f}  ({val_acc*100:.1f}%)")
    print(f"  Test accuracy       : {test_acc:.4f}  ({test_acc*100:.1f}%)")
 
    print("\n  Validation classification report:")
    val_report = classification_report(
        y_val, y_val_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2],
        zero_division=0,
    )
    print(val_report)
 
    print("  Test classification report:")
    test_report = classification_report(
        y_test, y_test_pred,
        target_names=["Low", "Medium", "High"],
        labels=[0, 1, 2],
        zero_division=0,
    )
    print(test_report)
 
    # Save model
    model_path = os.path.join(MODELS_DIR, "lstm_model.keras")
    model.save(model_path)
    print(f"\n  Saved model : {model_path}")
 
    # Save training history
    history_df = pd.DataFrame(hist.history)
    history_df.index.name = "epoch"
    history_path = os.path.join(REPORTS_DIR, "training_history.csv")
    history_df.to_csv(history_path)
    print(f"  Saved history : {history_path}")
 
    # Save classification report
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM RISK CLASSIFIER — EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Architecture : 2x LSTM({LSTM_UNITS}) + Dense({DENSE_UNITS}) + Softmax(3)\n")
        f.write(f"Sequence len : {SEQ_LEN} trading days\n")
        f.write(f"Features     : {N_FEATURES}\n")
        f.write(f"Epochs run   : {len(hist.history['loss'])}\n\n")
        f.write(f"Validation accuracy : {val_acc:.4f}\n")
        f.write(f"Test accuracy       : {test_acc:.4f}\n\n")
        f.write("VALIDATION SET\n" + "-" * 40 + "\n")
        f.write(val_report + "\n")
        f.write("TEST SET\n" + "-" * 40 + "\n")
        f.write(test_report + "\n")
    print(f"  Saved report  : {report_path}")
 
    # Save figures
    print("\nSaving figures ...")
    plot_confusion_matrix(
        y_val, y_val_pred,
        os.path.join(FIGURES_DIR, "confusion_matrix.png"),
    )
    plot_training_curves(
        hist.history,
        os.path.join(FIGURES_DIR, "training_curves.png"),
    )
 
    print("\nWeek 4 complete.")
    print(f"  Model    : {model_path}")
    print(f"  Report   : {report_path}")
    print(f"  Figures  : {FIGURES_DIR}/")
