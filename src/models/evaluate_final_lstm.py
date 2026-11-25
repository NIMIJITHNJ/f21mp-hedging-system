"""Evaluate the locked LSTM once on the untouched final test period."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from lstm_pipeline_utils import (
    CLASS_NAMES,
    OUTPUT_FIGURES_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_REPORTS_DIR,
    build_sequences_with_context,
    extract_metrics,
    load_json,
    save_json,
)

FINAL_MODEL_PATH = OUTPUT_MODELS_DIR / "lstm_model.keras"
FINAL_TEST_DATA_PATH = OUTPUT_MODELS_DIR / "lstm_final_test_data.npz"
FINAL_METADATA_PATH = OUTPUT_MODELS_DIR / "lstm_model_metadata.json"
FINAL_METRICS_PATH = OUTPUT_REPORTS_DIR / "lstm_final_test_metrics.json"
FINAL_REPORT_PATH = OUTPUT_REPORTS_DIR / "lstm_final_classification_report.txt"
FINAL_PREDICTIONS_PATH = OUTPUT_REPORTS_DIR / "lstm_final_test_predictions.csv"
FINAL_CM_PATH = OUTPUT_FIGURES_DIR / "lstm_final_test_confusion_matrix.png"


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    ax.set_xticks([0, 1, 2], labels=CLASS_NAMES)
    ax.set_yticks([0, 1, 2], labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Locked LSTM — Final Untouched Test Set")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FINAL_CM_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    for required in [FINAL_MODEL_PATH, FINAL_TEST_DATA_PATH, FINAL_METADATA_PATH]:
        if not required.exists():
            raise FileNotFoundError(
                f"Missing {required}. Run: python src/models/train_final_lstm.py"
            )

    metadata = load_json(FINAL_METADATA_PATH)
    test_payload = np.load(FINAL_TEST_DATA_PATH, allow_pickle=False)
    x_test_seq, y_test_seq = build_sequences_with_context(
        test_payload["x_train_tail"],
        test_payload["x_test"],
        test_payload["y_test"],
    )
    dates = pd.to_datetime(test_payload["test_dates"])

    print("=" * 72)
    print("HEDRA — FINAL UNTOUCHED TEST EVALUATION")
    print(f"Config: {metadata['config_name']}")
    print(f"Horizon: {metadata['horizon']} trading days")
    print(f"Test rows: {len(y_test_seq)}")
    print("=" * 72)

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    probabilities = model.predict(x_test_seq, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    metrics = extract_metrics(y_test_seq, predictions, prefix="test")
    metrics.update({
        "config_name": metadata["config_name"],
        "horizon": metadata["horizon"],
        "seed": metadata["seed"],
        "fixed_epochs": metadata["fixed_epochs"],
        "test_rows": len(y_test_seq),
    })
    save_json(FINAL_METRICS_PATH, metrics)

    report = classification_report(
        y_test_seq,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    with FINAL_REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write("LOCKED LSTM — FINAL UNTOUCHED TEST REPORT\n")
        file.write("=" * 72 + "\n\n")
        file.write(json.dumps(metrics, indent=2))
        file.write("\n\n")
        file.write(report)

    prediction_frame = pd.DataFrame({
        "date": dates,
        "actual_label": y_test_seq,
        "predicted_label": predictions,
        "prob_low": probabilities[:, 0],
        "prob_medium": probabilities[:, 1],
        "prob_high": probabilities[:, 2],
    })
    prediction_frame.to_csv(FINAL_PREDICTIONS_PATH, index=False)
    save_confusion_matrix(y_test_seq, predictions)

    metadata["test_evaluated"] = True
    metadata["final_test_metrics_path"] = str(FINAL_METRICS_PATH)
    save_json(FINAL_METADATA_PATH, metadata)

    print("\nFinal metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(f"  Metrics     : {FINAL_METRICS_PATH}")
    print(f"  Report      : {FINAL_REPORT_PATH}")
    print(f"  Predictions : {FINAL_PREDICTIONS_PATH}")
    print(f"  Confusion   : {FINAL_CM_PATH}")


if __name__ == "__main__":
    main()