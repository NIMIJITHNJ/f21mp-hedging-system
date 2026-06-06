"""
Leakage-safe walk-forward benchmark for NIFTY 50 volatility regression.

Models compared:
1. GARCH(1,1)
2. EGARCH(1,1) with Student-t errors
3. GJR-GARCH(1,1) with Student-t errors
4. HAR-inspired Ridge regression
5. XGBoost regression
6. LightGBM regression
7. GARCH-XGBoost hybrid
8. EGARCH-XGBoost hybrid
9. GJR-GARCH-XGBoost hybrid

Target:
    Annualised future realised-volatility proxy over the next h trading days:
    sqrt(mean(r[t+1:t+h]^2) * 252)

Key design choices:
- Statistical forecasts are proper horizon-matched forecasts.
- GARCH uses analytical multi-step forecasts.
- EGARCH and GJR-GARCH use deterministic simulation forecasts for h > 1.
- Statistical model parameters are refitted annually using data available only
  up to the end of the previous year.
- Conditional volatility recursion is updated using observed returns as time
  moves forward. No future returns enter any feature or forecast.
- ML hyperparameters are selected using validation data only.
- 2025 to Mar-2026 is an additional confirmatory period, not a tuning period.

Run:
    python src/models/regression/run_volatility_regression_walkforward.py
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from arch import arch_model
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from features.technical_indicators import (  # noqa: E402
    atr,
    log_return,
    macd,
    oi_moving_average,
    oi_pct_change,
    oi_volume_ratio,
    rsi,
    sma_crossover,
    vwap,
)

RAW_DIR = ROOT / "data" / "raw"
EXTERNAL_DIR = ROOT / "data" / "external"
REPORT_DIR = ROOT / "outputs" / "reports" / "regression_walkforward"
MODEL_DIR = ROOT / "outputs" / "models" / "regression_walkforward"
FIGURE_DIR = ROOT / "outputs" / "figures" / "regression_walkforward"

for directory in (REPORT_DIR, MODEL_DIR, FIGURE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252
HORIZONS = [1, 5, 10]
EPS = 1e-10
FINAL_SEED = 42
SIMULATIONS = 1000
OOS_STAT_START_YEAR = 2015

STOCK_SYMBOLS = [
    "RELIANCE",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "TCS",
    "SBIN",
    "LT",
    "BHARTIARTL",
    "ITC",
    "AXISBANK",
]

FOLDS = [
    {"fold": 1, "train_end": "2018-12-31", "val_end": "2019-12-31", "eval_end": "2020-12-31"},
    {"fold": 2, "train_end": "2019-12-31", "val_end": "2020-12-31", "eval_end": "2021-12-31"},
    {"fold": 3, "train_end": "2020-12-31", "val_end": "2021-12-31", "eval_end": "2022-12-31"},
    {"fold": 4, "train_end": "2021-12-31", "val_end": "2022-12-31", "eval_end": "2023-12-31"},
    {"fold": 5, "train_end": "2022-12-31", "val_end": "2023-12-31", "eval_end": "2024-12-31"},
]

CONFIRMATORY_FOLD = {
    "fold": "confirmatory",
    "train_end": "2023-12-31",
    "val_end": "2024-12-31",
    "eval_end": "2026-03-31",
}

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]

XGB_CONFIGS = [
    {
        "name": "xgb_c1",
        "n_estimators": 250,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "xgb_c2",
        "n_estimators": 400,
        "max_depth": 3,
        "learning_rate": 0.02,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "xgb_c3",
        "n_estimators": 300,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
    },
]

LGBM_CONFIGS = [
    {
        "name": "lgbm_c1",
        "n_estimators": 250,
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": -1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    },
    {
        "name": "lgbm_c2",
        "n_estimators": 400,
        "learning_rate": 0.02,
        "num_leaves": 15,
        "max_depth": -1,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
    },
    {
        "name": "lgbm_c3",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": -1,
        "subsample": 0.85,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.1,
        "reg_lambda": 3.0,
    },
]


@dataclass
class FoldData:
    train: pd.DataFrame
    val: pd.DataFrame
    eval: pd.DataFrame


def load_raw(symbol: str) -> pd.DataFrame:
    path = RAW_DIR / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df.sort_index()


def load_macro() -> pd.DataFrame:
    path = EXTERNAL_DIR / "macro_combined.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing macro file: {path}")

    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df.sort_index()


def get_safe_purge_end(
    trading_index: pd.DatetimeIndex,
    end_date: str,
    horizon: int,
) -> pd.Timestamp:
    """
    Return the last origin date whose target window remains inside the period.
    Uses actual NIFTY trading rows rather than generic weekdays.
    """
    end_ts = min(pd.Timestamp(end_date), pd.Timestamp(trading_index.max()))
    available_dates = trading_index[trading_index <= end_ts]

    if len(available_dates) <= horizon:
        raise ValueError(
            f"Not enough trading rows to purge {horizon} days before {end_ts.date()}"
        )

    return pd.Timestamp(available_dates[-(horizon + 1)])


def future_realised_volatility(close: pd.Series, horizon: int) -> pd.Series:
    """
    At date t:
        target[t] = sqrt(mean(r[t+1:t+h]^2) * 252)
    """
    ret2 = log_return(close).pow(2)
    future_rv = (
        ret2.shift(-1)
        .rolling(window=horizon)
        .mean()
        .shift(-(horizon - 1))
    )
    future_rv = np.sqrt(future_rv * TRADING_DAYS)
    future_rv.name = f"future_rv_{horizon}d"
    return future_rv


def build_base_features() -> tuple[pd.DataFrame, pd.Series]:
    print("Loading raw data and assembling base features ...")

    spot = load_raw("NIFTY50_spot")
    fut = load_raw("NIFTY_futures_continuous")
    macro = load_macro()

    stock_returns: dict[str, pd.Series] = {}
    for symbol in STOCK_SYMBOLS:
        stock_returns[symbol] = log_return(
            load_raw(symbol)["close"]
        ).rename(f"ret_{symbol}")

    macd_line, macd_signal, _ = macd(
        spot["close"],
        fast=12,
        slow=26,
        signal=9,
    )

    # OI is unavailable for part of the historical dataset.
    # Retain the full history and explicitly mark availability.
    oi_available = (fut["oi"] > 0).astype(int).rename("oi_available")
    oi_clean = fut["oi"].where(oi_available.eq(1), np.nan)

    oi_pct = (
        oi_pct_change(oi_clean, period=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    oi_ratio = (
        oi_volume_ratio(oi_clean, fut["volume"])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    oi_ma20 = oi_moving_average(oi_clean, window=20).fillna(0)

    ret = log_return(spot["close"])

    base = pd.DataFrame(index=spot.index)
    base["rsi_14"] = rsi(spot["close"], period=14)
    base["macd_12_26"] = macd_line
    base["macd_signal_9"] = macd_signal
    base["vwap_20"] = vwap(
        spot["high"],
        spot["low"],
        spot["close"],
        fut["volume"],
    )
    base["atr_14"] = atr(
        spot["high"],
        spot["low"],
        spot["close"],
        period=14,
    )
    base["sma_cross_10_20"] = sma_crossover(
        spot["close"],
        fast=10,
        slow=20,
    )

    base["oi_available"] = oi_available
    base["oi_pct_change"] = oi_pct
    base["oi_vol_ratio"] = oi_ratio
    base["oi_ma_20"] = oi_ma20

    # HAR-inspired daily-data volatility features.
    base["rv_1"] = np.sqrt(ret.pow(2) * TRADING_DAYS)
    base["rv_5"] = np.sqrt(ret.pow(2).rolling(5).mean() * TRADING_DAYS)
    base["rv_10"] = np.sqrt(ret.pow(2).rolling(10).mean() * TRADING_DAYS)
    base["rv_22"] = np.sqrt(ret.pow(2).rolling(22).mean() * TRADING_DAYS)
    base["ret_1"] = ret
    base["abs_ret_1"] = ret.abs()

    base["gold_usd"] = macro["gold_usd"]
    base["brent_usd"] = macro["brent_usd"]
    base["usdinr"] = macro["usdinr"]
    base["india_10y"] = macro["india_10y"]

    for symbol in STOCK_SYMBOLS:
        base[f"ret_{symbol}"] = stock_returns[symbol]

    base = base.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"  Base feature rows   : {len(base)}")
    print(f"  Base feature columns: {len(base.columns)}")
    print(f"  Coverage            : {base.index.min().date()} to {base.index.max().date()}")

    return base, spot["close"]


def statistical_spec(model_name: str) -> dict[str, Any]:
    if model_name == "garch":
        return {
            "mean": "Zero",
            "vol": "GARCH",
            "p": 1,
            "o": 0,
            "q": 1,
            "dist": "normal",
            "rescale": False,
        }

    if model_name == "egarch":
        return {
            "mean": "Zero",
            "vol": "EGARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "dist": "t",
            "rescale": False,
        }

    if model_name == "gjr_garch":
        return {
            "mean": "Zero",
            "vol": "GARCH",
            "p": 1,
            "o": 1,
            "q": 1,
            "dist": "t",
            "rescale": False,
        }

    raise ValueError(f"Unsupported statistical model: {model_name}")


def make_simulation_rng(
    distribution_name: str,
    nu: float | None,
    seed: int,
) -> Callable[[tuple[int, ...]], np.ndarray]:
    """
    Return a deterministic standardized innovation generator.

    Student-t innovations are standardized to unit variance.
    """
    generator = np.random.default_rng(seed)

    if distribution_name == "t":
        if nu is None or nu <= 2:
            raise ValueError("Student-t simulation requires nu > 2")

        scale = math.sqrt((nu - 2.0) / nu)

        def student_t_rng(size: tuple[int, ...]) -> np.ndarray:
            return generator.standard_t(df=nu, size=size) * scale

        return student_t_rng

    def normal_rng(size: tuple[int, ...]) -> np.ndarray:
        return generator.standard_normal(size=size)

    return normal_rng


def average_horizon_forecast(
    returns_until_year_end: pd.Series,
    fitted_params: pd.Series,
    model_name: str,
    horizon: int,
    forecast_start: str,
    seed: int,
) -> pd.Series:
    """
    Produce horizon-matched annualised volatility forecasts.

    For date t, the result is:
        sqrt(mean(forecast_variance[t+1:t+h]) * 252)

    Returns are supplied in percent units to arch, so the final forecast is
    divided by 100 after annualisation.
    """
    spec = statistical_spec(model_name)

    fixed = arch_model(
        returns_until_year_end,
        **spec,
    ).fix(fitted_params)

    if horizon == 1 or model_name == "garch":
        forecast = fixed.forecast(
            horizon=horizon,
            start=forecast_start,
            align="origin",
            method="analytic",
            reindex=True,
        )
    else:
        nu = (
            float(fitted_params["nu"])
            if "nu" in fitted_params.index
            else None
        )
        rng = make_simulation_rng(
            distribution_name=str(spec["dist"]),
            nu=nu,
            seed=seed,
        )

        forecast = fixed.forecast(
            horizon=horizon,
            start=forecast_start,
            align="origin",
            method="simulation",
            simulations=SIMULATIONS,
            rng=rng,
            reindex=True,
        )

    mean_variance = forecast.variance.iloc[:, :horizon].mean(axis=1)
    annualised_vol = np.sqrt(mean_variance * TRADING_DAYS) / 100
    annualised_vol.name = f"{model_name}_forecast"
    return annualised_vol


def build_oos_statistical_forecasts(
    spot_close: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    """
    Build leakage-safe statistical forecasts with annual expanding refits.

    Forecasts for year Y use parameters estimated only through 31 Dec Y-1.
    Inside year Y, the recursion updates using observed returns available by
    each forecast origin date.
    """
    print(f"  Building OOS statistical forecasts for horizon={horizon}d ...")

    returns_pct = (log_return(spot_close).dropna() * 100).astype(float)
    final_year = int(returns_pct.index.max().year)

    forecast_parts: list[pd.DataFrame] = []

    for year in range(OOS_STAT_START_YEAR, final_year + 1):
        train_end = pd.Timestamp(year=year - 1, month=12, day=31)
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31)

        estimation_returns = returns_pct.loc[:train_end]
        returns_until_year_end = returns_pct.loc[:year_end]

        if len(estimation_returns) < 500:
            continue

        year_frame = pd.DataFrame(index=returns_until_year_end.index)

        for model_offset, model_name in enumerate(
            ("garch", "egarch", "gjr_garch"),
            start=1,
        ):
            fitted = arch_model(
                estimation_returns,
                **statistical_spec(model_name),
            ).fit(
                disp="off",
                show_warning=False,
            )

            year_frame[f"{model_name}_forecast"] = average_horizon_forecast(
                returns_until_year_end=returns_until_year_end,
                fitted_params=fitted.params,
                model_name=model_name,
                horizon=horizon,
                forecast_start=str(year_start.date()),
                seed=FINAL_SEED + year + (100 * horizon) + model_offset,
            )

        year_frame = year_frame[
            (year_frame.index >= year_start)
            & (year_frame.index <= year_end)
        ]

        forecast_parts.append(year_frame)

    if not forecast_parts:
        raise RuntimeError("No statistical forecasts were generated")

    forecasts = pd.concat(forecast_parts).sort_index()
    forecasts = forecasts[~forecasts.index.duplicated(keep="last")]

    print(
        f"    Statistical forecast coverage: "
        f"{forecasts.index.min().date()} to {forecasts.index.max().date()}"
    )

    return forecasts


def build_horizon_frame(
    base_features: pd.DataFrame,
    spot_close: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    statistical_forecasts = build_oos_statistical_forecasts(
        spot_close=spot_close,
        horizon=horizon,
    )

    target = future_realised_volatility(
        spot_close,
        horizon,
    ).rename("target")

    frame = (
        base_features
        .join(statistical_forecasts, how="inner")
        .join(target, how="inner")
    )

    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"    Horizon frame rows: {len(frame)}")
    return frame


def split_fold_data(
    frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    fold: dict[str, Any],
    horizon: int,
) -> FoldData:
    train_purge_end = get_safe_purge_end(
        trading_index,
        fold["train_end"],
        horizon,
    )
    val_purge_end = get_safe_purge_end(
        trading_index,
        fold["val_end"],
        horizon,
    )
    eval_purge_end = get_safe_purge_end(
        trading_index,
        fold["eval_end"],
        horizon,
    )

    train_df = frame.loc[:train_purge_end].copy()

    val_df = frame[
        (frame.index > pd.Timestamp(fold["train_end"]))
        & (frame.index <= val_purge_end)
    ].copy()

    eval_df = frame[
        (frame.index > pd.Timestamp(fold["val_end"]))
        & (frame.index <= eval_purge_end)
    ].copy()

    if min(len(train_df), len(val_df), len(eval_df)) == 0:
        raise ValueError(
            f"Empty split for fold={fold['fold']} horizon={horizon}"
        )

    return FoldData(
        train=train_df,
        val=val_df,
        eval=eval_df,
    )


def clip_predictions(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(
        np.asarray(values, dtype=float),
        EPS,
        None,
    )


def qlike_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """
    QLIKE loss on variance forecasts. Lower is better.
    """
    true_var = np.square(clip_predictions(y_true))
    pred_var = np.square(clip_predictions(y_pred))
    return float(
        np.mean(
            np.log(pred_var)
            + (true_var / pred_var)
        )
    )


def directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference: np.ndarray,
) -> float:
    actual_direction = np.sign(y_true - reference)
    predicted_direction = np.sign(y_pred - reference)
    return float(
        np.mean(
            actual_direction == predicted_direction
        )
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    reference: pd.Series,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    pred = clip_predictions(y_pred)
    ref = np.asarray(reference, dtype=float)

    stress_threshold = float(np.quantile(y, 0.80))
    stress_mask = y >= stress_threshold

    return {
        "rmse": float(
            math.sqrt(
                mean_squared_error(y, pred)
            )
        ),
        "mae": float(
            mean_absolute_error(y, pred)
        ),
        "qlike": qlike_loss(y, pred),
        "r2": float(r2_score(y, pred)),
        "directional_accuracy": directional_accuracy(
            y,
            pred,
            ref,
        ),
        "stress_mae": float(
            mean_absolute_error(
                y[stress_mask],
                pred[stress_mask],
            )
        ),
    }


def make_xgb(
    config: dict[str, Any],
) -> XGBRegressor:
    params = {
        key: value
        for key, value in config.items()
        if key != "name"
    }

    return XGBRegressor(
        objective="reg:squarederror",
        random_state=FINAL_SEED,
        n_jobs=-1,
        tree_method="hist",
        **params,
    )


def make_lgbm(
    config: dict[str, Any],
) -> LGBMRegressor:
    params = {
        key: value
        for key, value in config.items()
        if key != "name"
    }

    return LGBMRegressor(
        objective="regression",
        random_state=FINAL_SEED,
        n_jobs=-1,
        verbosity=-1,
        **params,
    )


def select_best_config(
    model_family: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Any, str]:
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]

    candidates: list[tuple[float, float, Any, str]] = []

    if model_family == "ridge":
        for alpha in RIDGE_ALPHAS:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ]
            )

            model.fit(X_train, y_train)
            pred = clip_predictions(
                model.predict(X_val)
            )

            candidates.append(
                (
                    qlike_loss(y_val, pred),
                    mean_absolute_error(y_val, pred),
                    model,
                    f"ridge_alpha_{alpha}",
                )
            )

    elif model_family in {
        "xgboost",
        "garch_xgboost",
        "egarch_xgboost",
        "gjr_garch_xgboost",
    }:
        for config in XGB_CONFIGS:
            model = make_xgb(config)
            model.fit(X_train, y_train)

            pred = clip_predictions(
                model.predict(X_val)
            )

            candidates.append(
                (
                    qlike_loss(y_val, pred),
                    mean_absolute_error(y_val, pred),
                    model,
                    config["name"],
                )
            )

    elif model_family == "lightgbm":
        for config in LGBM_CONFIGS:
            model = make_lgbm(config)
            model.fit(X_train, y_train)

            pred = clip_predictions(
                model.predict(X_val)
            )

            candidates.append(
                (
                    qlike_loss(y_val, pred),
                    mean_absolute_error(y_val, pred),
                    model,
                    config["name"],
                )
            )

    else:
        raise ValueError(
            f"Unsupported ML model family: {model_family}"
        )

    candidates.sort(
        key=lambda row: (
            row[0],
            row[1],
        )
    )

    _, _, best_model, best_config_name = candidates[0]
    return best_model, best_config_name


def evaluate_one_model(
    fold_label: Any,
    horizon: int,
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    reference: pd.Series,
    config_name: str = "fixed",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "fold": fold_label,
        "horizon": horizon,
        "model": model_name,
        "config": config_name,
        "n_eval": len(y_true),
    }

    row.update(
        calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            reference=reference,
        )
    )

    return row


def run_fold(
    horizon_frame: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    fold: dict[str, Any],
    horizon: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    print(
        f"\nFold={fold['fold']} | horizon={horizon}d | "
        f"train<={fold['train_end']} "
        f"val<={fold['val_end']} "
        f"eval<={fold['eval_end']}"
    )

    data = split_fold_data(
        frame=horizon_frame,
        trading_index=trading_index,
        fold=fold,
        horizon=horizon,
    )

    train_df = data.train
    val_df = data.val
    eval_df = data.eval

    print(
        f"  rows: train={len(train_df)} "
        f"val={len(val_df)} "
        f"eval={len(eval_df)}"
    )

    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    reference_col = {
        1: "rv_1",
        5: "rv_5",
        10: "rv_10",
    }[horizon]

    reference = eval_df[reference_col]

    def record(
        model_name: str,
        config_name: str,
        predictions: np.ndarray | pd.Series,
    ) -> None:
        pred = clip_predictions(predictions)

        rows.append(
            evaluate_one_model(
                fold_label=fold["fold"],
                horizon=horizon,
                model_name=model_name,
                y_true=eval_df["target"],
                y_pred=pred,
                reference=reference,
                config_name=config_name,
            )
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": eval_df.index,
                    "fold": fold["fold"],
                    "horizon": horizon,
                    "model": model_name,
                    "config": config_name,
                    "actual": eval_df["target"].to_numpy(),
                    "prediction": pred,
                }
            )
        )

    # Horizon-matched statistical baselines.
    for stat_model in (
        "garch",
        "egarch",
        "gjr_garch",
    ):
        record(
            model_name=stat_model,
            config_name="annual_refit",
            predictions=eval_df[f"{stat_model}_forecast"],
        )

    har_cols = [
        "rv_1",
        "rv_5",
        "rv_22",
    ]

    statistical_cols = [
        "garch_forecast",
        "egarch_forecast",
        "gjr_garch_forecast",
    ]

    tree_cols = [
        column
        for column in train_df.columns
        if (
            column != "target"
            and column not in statistical_cols
        )
    ]

    # HAR-inspired Ridge baseline.
    ridge, ridge_config = select_best_config(
        model_family="ridge",
        train_df=train_df,
        val_df=val_df,
        feature_cols=har_cols,
    )

    record(
        model_name="har_ridge",
        config_name=ridge_config,
        predictions=ridge.predict(
            eval_df[har_cols]
        ),
    )

    # Pure XGBoost.
    xgb, xgb_config = select_best_config(
        model_family="xgboost",
        train_df=train_df,
        val_df=val_df,
        feature_cols=tree_cols,
    )

    record(
        model_name="xgboost",
        config_name=xgb_config,
        predictions=xgb.predict(
            eval_df[tree_cols]
        ),
    )

    # Pure LightGBM.
    lgbm, lgbm_config = select_best_config(
        model_family="lightgbm",
        train_df=train_df,
        val_df=val_df,
        feature_cols=tree_cols,
    )

    record(
        model_name="lightgbm",
        config_name=lgbm_config,
        predictions=lgbm.predict(
            eval_df[tree_cols]
        ),
    )

    # Compare three statistical-ML hybrids fairly.
    for hybrid_name, statistical_feature in (
        ("garch_xgboost", "garch_forecast"),
        ("egarch_xgboost", "egarch_forecast"),
        ("gjr_garch_xgboost", "gjr_garch_forecast"),
    ):
        hybrid_cols = tree_cols + [
            statistical_feature,
        ]

        hybrid, hybrid_config = select_best_config(
            model_family=hybrid_name,
            train_df=train_df,
            val_df=val_df,
            feature_cols=hybrid_cols,
        )

        record(
            model_name=hybrid_name,
            config_name=hybrid_config,
            predictions=hybrid.predict(
                eval_df[hybrid_cols]
            ),
        )

    return rows, prediction_frames


def summarise_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        results_df
        .groupby(
            [
                "horizon",
                "model",
            ]
        )
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_qlike=("qlike", "mean"),
            std_qlike=("qlike", "std"),
            mean_r2=("r2", "mean"),
            mean_directional_accuracy=(
                "directional_accuracy",
                "mean",
            ),
            mean_stress_mae=("stress_mae", "mean"),
            n_folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(
            [
                "mean_qlike",
                "mean_stress_mae",
                "mean_mae",
                "mean_rmse",
            ]
        )
    )

    return summary


def main() -> None:
    base_features, spot_close = build_base_features()
    trading_index = pd.DatetimeIndex(spot_close.index)

    all_rows: list[dict[str, Any]] = []
    all_prediction_frames: list[pd.DataFrame] = []

    horizon_frames: dict[int, pd.DataFrame] = {}

    print("\n" + "=" * 80)
    print("BUILDING HORIZON-MATCHED STATISTICAL FORECASTS")
    print("=" * 80)

    for horizon in HORIZONS:
        horizon_frames[horizon] = build_horizon_frame(
            base_features=base_features,
            spot_close=spot_close,
            horizon=horizon,
        )

    total_blocks = len(FOLDS) * len(HORIZONS)
    completed_blocks = 0

    print("\n" + "=" * 80)
    print("WALK-FORWARD VOLATILITY REGRESSION BENCHMARK")
    print(f"Planned fold-horizon blocks: {total_blocks}")
    print("=" * 80)

    for horizon in HORIZONS:
        for fold in FOLDS:
            completed_blocks += 1

            print(
                f"\n[{completed_blocks}/{total_blocks}] "
                f"Starting fold-horizon block ..."
            )

            rows, prediction_frames = run_fold(
                horizon_frame=horizon_frames[horizon],
                trading_index=trading_index,
                fold=fold,
                horizon=horizon,
            )

            all_rows.extend(rows)
            all_prediction_frames.extend(
                prediction_frames
            )

            pd.DataFrame(all_rows).to_csv(
                REPORT_DIR
                / "regression_walkforward_results_checkpoint.csv",
                index=False,
            )

            pd.concat(
                all_prediction_frames,
                ignore_index=True,
            ).to_csv(
                REPORT_DIR
                / "regression_walkforward_predictions_checkpoint.csv",
                index=False,
            )

    results_df = pd.DataFrame(all_rows)

    predictions_df = pd.concat(
        all_prediction_frames,
        ignore_index=True,
    )

    summary_df = summarise_results(
        results_df
    )

    results_path = (
        REPORT_DIR
        / "regression_walkforward_results.csv"
    )

    predictions_path = (
        REPORT_DIR
        / "regression_walkforward_predictions.csv"
    )

    summary_path = (
        REPORT_DIR
        / "regression_walkforward_summary.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    best_row = summary_df.iloc[0]

    selected_horizon = int(
        best_row["horizon"]
    )

    selected = {
        "selection_method": (
            "Walk-forward evaluation across 2020-2024. "
            "Lower mean QLIKE is preferred, followed by lower "
            "stress-period MAE, MAE and RMSE."
        ),
        "selected_horizon": selected_horizon,
        "selected_model": str(
            best_row["model"]
        ),
        "mean_qlike": float(
            best_row["mean_qlike"]
        ),
        "mean_stress_mae": float(
            best_row["mean_stress_mae"]
        ),
        "mean_mae": float(
            best_row["mean_mae"]
        ),
        "mean_rmse": float(
            best_row["mean_rmse"]
        ),
        "mean_r2": float(
            best_row["mean_r2"]
        ),
        "mean_directional_accuracy": float(
            best_row[
                "mean_directional_accuracy"
            ]
        ),
        "confirmatory_period": (
            "2025-01-01 to 2026-03-31"
        ),
        "notes": [
            (
                "The 2025-2026 period is an additional confirmatory "
                "evaluation and is not used for model selection."
            ),
            (
                "HAR-inspired features use daily returns because "
                "intraday realised-volatility data is unavailable."
            ),
            (
                "OI availability is explicitly represented by "
                "the oi_available feature."
            ),
            (
                "Statistical forecasts are horizon-matched and use "
                "annual expanding refits with prior-year data only."
            ),
        ],
    }

    selected_path = (
        REPORT_DIR
        / "selected_regression_model.json"
    )

    with open(
        selected_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            selected,
            file,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("TOP WALK-FORWARD RESULTS")
    print("=" * 80)
    print(
        summary_df
        .head(20)
        .to_string(index=False)
    )

    print("\n" + "=" * 80)
    print("SELECTED WALK-FORWARD CANDIDATE")
    print("=" * 80)
    print(
        json.dumps(
            selected,
            indent=2,
        )
    )

    print("\nRunning confirmatory period ...")

    holdout_rows, holdout_frames = run_fold(
        horizon_frame=horizon_frames[
            selected_horizon
        ],
        trading_index=trading_index,
        fold=CONFIRMATORY_FOLD,
        horizon=selected_horizon,
    )

    holdout_df = pd.DataFrame(
        holdout_rows
    )

    holdout_predictions_df = pd.concat(
        holdout_frames,
        ignore_index=True,
    )

    holdout_results_path = (
        REPORT_DIR
        / "regression_confirmatory_results.csv"
    )

    holdout_predictions_path = (
        REPORT_DIR
        / "regression_confirmatory_predictions.csv"
    )

    holdout_df.to_csv(
        holdout_results_path,
        index=False,
    )

    holdout_predictions_df.to_csv(
        holdout_predictions_path,
        index=False,
    )

    print("\n" + "=" * 80)
    print("CONFIRMATORY RESULTS")
    print("=" * 80)
    print(
        holdout_df
        .sort_values(
            [
                "qlike",
                "stress_mae",
                "mae",
                "rmse",
            ]
        )
        .to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"  {results_path}")
    print(f"  {predictions_path}")
    print(f"  {summary_path}")
    print(f"  {selected_path}")
    print(f"  {holdout_results_path}")
    print(f"  {holdout_predictions_path}")


if __name__ == "__main__":
    main()
