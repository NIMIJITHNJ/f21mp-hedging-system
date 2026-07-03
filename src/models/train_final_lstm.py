"""Train and save the locked LSTM model using all pre-test data.

Run only after find_best_lstm.py has produced best_lstm_config.json.
The final test data is prepared and saved, but not evaluated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_pipeline_utils import (
    CONFIGS,
    OUTPUT_MODELS_DIR,
    OUTPUT_REPORTS_DIR,
    SEQ_LEN,
    build_final_train_test,
    build_sequences_with_context,
    load_json,
    save_json,
    save_scaler,
    train_final_model,
)

BEST_CONFIG_PATH = OUTPUT_REPORTS_DIR / "lstm_tuning" / "best_lstm_config.json"
FINAL_MODEL_PATH = OUTPUT_MODELS_DIR / "lstm_model.keras"
FINAL_SCALER_PATH = OUTPUT_MODELS_DIR / "lstm_scaler.pkl"
FINAL_TEST_DATA_PATH = OUTPUT_MODELS_DIR / "lstm_final_test_data.npz"
FINAL_METADATA_PATH = OUTPUT_MODELS_DIR / "lstm_model_metadata.json"
FINAL_HISTORY_PATH = OUTPUT_REPORTS_DIR / "lstm_final_training_history.csv"


def main() -> None:
    if not BEST_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BEST_CONFIG_PATH}. Run: python src/models/find_best_lstm.py --mode quick"
        )

    selection = load_json(BEST_CONFIG_PATH)
    horizon = int(selection["horizon"])
    config_name = str(selection["config_name"])
    epochs = int(selection["recommended_fixed_epochs"])
    config = CONFIGS[config_name]

    # Use the middle seed from the controlled seed list for a deterministic final fit.
    seeds = [int(seed) for seed in selection["seeds"]]
    final_seed = sorted(seeds)[len(seeds) // 2]

    print("=" * 72)
    print("HEDRA — FINAL LSTM TRAINING")
    print(f"Config: {config_name}")
    print(f"Horizon: {horizon} trading days")
    print(f"Epochs: {epochs}")
    print(f"Seed: {final_seed}")
    print("Training data: all rows up to 2023-12-31, with boundary purge")
    print("Final test data is prepared but NOT evaluated in this script.")
    print("=" * 72)

    data = build_final_train_test(horizon=horizon)
    model, history = train_final_model(
        config=config,
        seed=final_seed,
        epochs=epochs,
        x_train=data["x_train"],
        y_train=data["y_train"],
    )

    model.save(FINAL_MODEL_PATH)
    save_scaler(FINAL_SCALER_PATH, data["scaler"])
    pd.DataFrame(history).to_csv(FINAL_HISTORY_PATH, index=False)

    np.savez_compressed(
        FINAL_TEST_DATA_PATH,
        x_train_tail=data["x_train"][-(SEQ_LEN - 1):],
        x_test=data["x_test"],
        y_test=data["y_test"],
        test_dates=data["test_dates"],
    )

    metadata = {
        "model_path": str(FINAL_MODEL_PATH),
        "scaler_path": str(FINAL_SCALER_PATH),
        "selection_file": str(BEST_CONFIG_PATH),
        "selection_method": selection["selection_method"],
        "config_name": config_name,
        "config": selection["config"],
        "horizon": horizon,
        "fixed_epochs": epochs,
        "seed": final_seed,
        "pretest_end": "2023-12-31",
        "final_test_start": "2024-01-01",
        "q33": data["q33"],
        "q67": data["q67"],
        "train_rows": data["train_rows"],
        "test_rows": data["test_rows"],
        "feature_names": data["feature_cols"],
        "test_evaluated": False,
    }
    save_json(FINAL_METADATA_PATH, metadata)

    print("\nSaved locked model and supporting files:")
    print(f"  Model    : {FINAL_MODEL_PATH}")
    print(f"  Scaler   : {FINAL_SCALER_PATH}")
    print(f"  Metadata : {FINAL_METADATA_PATH}")
    print(f"  History  : {FINAL_HISTORY_PATH}")
    print(f"  Test data: {FINAL_TEST_DATA_PATH}")
    print("\nNext command (run once for final reporting):")
    print("  python src/models/evaluate_final_lstm.py")


if __name__ == "__main__":
    main()
