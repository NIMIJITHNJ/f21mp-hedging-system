"""
Find the best reproducible LSTM configuration without touching the final test set.

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from lstm_pipeline_utils import (
    CONFIGS,
    OUTPUT_REPORTS_DIR,
    build_cv_fold,
    config_to_dict,
    save_json,
    train_for_validation,
)

HORIZONS = [3, 5, 7, 10, 14]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe LSTM tuning runner")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick: fewer folds/seeds; full: broader confirmation run",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidates promoted from stage 1 to walk-forward confirmation",
    )
    return parser.parse_args()


def settings_for_mode(mode: str) -> Tuple[List[int], List[int], List[int]]:
    if mode == "full":
        # Stage 1 uses recent folds; stage 2 confirms finalists over a wider range.
        return [2021, 2022, 2023], [2019, 2020, 2021, 2022, 2023], [7, 21, 42, 84, 123]
    return [2022, 2023], [2020, 2021, 2022, 2023], [7, 42, 123]


def run_candidate(
    horizon: int,
    config_name: str,
    validation_years: Iterable[int],
    seeds: Iterable[int],
    phase: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    config = CONFIGS[config_name]
    for validation_year in validation_years:
        print(f"\n  Building fold: horizon={horizon}, validation_year={validation_year}")
        x_train, y_train, x_validation, y_validation = build_cv_fold(
            horizon=horizon,
            validation_year=validation_year,
        )
        for seed in seeds:
            print(f"    Training {config_name}, seed={seed} ...", end=" ", flush=True)
            _, metrics, _, best_epoch = train_for_validation(
                config=config,
                seed=seed,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
            )
            row = {
                "phase": phase,
                "horizon": horizon,
                "config": config_name,
                "validation_year": validation_year,
                "seed": seed,
                "best_epoch": best_epoch,
                **metrics,
            }
            rows.append(row)
            print(
                f"val_macro_f1={row['val_macro_f1']:.3f}, "
                f"high_recall={row['val_high_recall']:.3f}, "
                f"accuracy={row['val_accuracy']:.3f}, epoch={best_epoch}"
            )
    return rows


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["horizon", "config"], as_index=False).agg(
        runs=("val_macro_f1", "size"),
        mean_val_macro_f1=("val_macro_f1", "mean"),
        std_val_macro_f1=("val_macro_f1", "std"),
        mean_val_high_recall=("val_high_recall", "mean"),
        mean_val_accuracy=("val_accuracy", "mean"),
        median_best_epoch=("best_epoch", "median"),
    )
    grouped["std_val_macro_f1"] = grouped["std_val_macro_f1"].fillna(0.0)
    return grouped.sort_values(
        ["mean_val_macro_f1", "mean_val_high_recall", "std_val_macro_f1", "mean_val_accuracy"],
        ascending=[False, False, True, False],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    stage1_years, stage2_years, seeds = settings_for_mode(args.mode)
    report_dir = OUTPUT_REPORTS_DIR / "lstm_tuning"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("HEDRA — REPRODUCIBLE LSTM MODEL SELECTION")
    print("Final test period is NOT used in this script.")
    print(f"Mode: {args.mode}")
    print(f"Horizons: {HORIZONS}")
    print(f"Configs: {list(CONFIGS)}")
    print(f"Seeds: {seeds}")
    print(f"Stage 1 validation years: {stage1_years}")
    print(f"Stage 2 validation years: {stage2_years}")
    print("=" * 72)

    stage1_rows: List[Dict[str, Any]] = []
    total_candidates = len(HORIZONS) * len(CONFIGS)
    current = 0
    for horizon in HORIZONS:
        for config_name in CONFIGS:
            current += 1
            print(f"\n[{current}/{total_candidates}] Stage 1 candidate: {config_name}, horizon={horizon}")
            stage1_rows.extend(
                run_candidate(horizon, config_name, stage1_years, seeds, phase="stage1")
            )

    stage1_df = pd.DataFrame(stage1_rows)
    stage1_summary = summarise(stage1_df)
    stage1_df.to_csv(report_dir / "stage1_all_runs.csv", index=False)
    stage1_summary.to_csv(report_dir / "stage1_summary.csv", index=False)

    print("\nSTAGE 1 TOP CANDIDATES")
    print(stage1_summary.head(args.top_k).to_string(index=False))

    finalists = stage1_summary.head(args.top_k)[["horizon", "config"]].to_dict("records")
    stage2_rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(finalists, start=1):
        horizon = int(candidate["horizon"])
        config_name = str(candidate["config"])
        print(f"\n[{index}/{len(finalists)}] Stage 2 finalist: {config_name}, horizon={horizon}")
        stage2_rows.extend(
            run_candidate(horizon, config_name, stage2_years, seeds, phase="stage2")
        )

    stage2_df = pd.DataFrame(stage2_rows)
    stage2_summary = summarise(stage2_df)
    stage2_df.to_csv(report_dir / "stage2_all_runs.csv", index=False)
    stage2_summary.to_csv(report_dir / "stage2_summary.csv", index=False)

    best = stage2_summary.iloc[0]
    best_config_name = str(best["config"])
    best_horizon = int(best["horizon"])
    best_epoch = max(1, int(round(float(best["median_best_epoch"]))))
    best_config = CONFIGS[best_config_name]

    best_payload = {
        "selection_method": "walk-forward validation only; final test untouched",
        "mode": args.mode,
        "horizon": best_horizon,
        "config_name": best_config_name,
        "config": config_to_dict(best_config),
        "recommended_fixed_epochs": best_epoch,
        "stage2_mean_val_macro_f1": float(best["mean_val_macro_f1"]),
        "stage2_std_val_macro_f1": float(best["std_val_macro_f1"]),
        "stage2_mean_val_high_recall": float(best["mean_val_high_recall"]),
        "stage2_mean_val_accuracy": float(best["mean_val_accuracy"]),
        "stage2_validation_years": stage2_years,
        "seeds": seeds,
        "final_test_start": "2024-01-01",
    }
    save_json(report_dir / "best_lstm_config.json", best_payload)

    print("\n" + "=" * 72)
    print("LOCKED CONFIGURATION FROM VALIDATION ONLY")
    print(json.dumps(best_payload, indent=2))
    print("=" * 72)
    print("\nNext command:")
    print("  python src/models/train_final_lstm.py")


if __name__ == "__main__":
    main()