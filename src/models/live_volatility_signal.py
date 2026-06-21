from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from arch import arch_model


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SPOT_PATH = ROOT_DIR / "data" / "raw" / "NIFTY50_spot.csv"
TRADING_DAYS = 252


@dataclass
class LiveVolatilitySignal:
    source: str
    model: str
    data_date: str
    predicted_volatility: float
    risk_regime: str
    core_target_ratio: float
    low_threshold: float
    high_threshold: float
    latest_close: float


def _classify_regime(predicted_volatility: float, low_threshold: float, high_threshold: float) -> str:
    if predicted_volatility <= low_threshold:
        return "LOW"
    if predicted_volatility <= high_threshold:
        return "MEDIUM"
    return "HIGH"


def _core_ratio_for_regime(regime: str) -> float:
    regime = str(regime).upper()
    if regime == "LOW":
        return 0.00
    if regime == "MEDIUM":
        return 0.40
    if regime == "HIGH":
        return 0.70
    return 0.40


def get_live_volatility_signal(
    spot_path: Path = DEFAULT_SPOT_PATH,
    lookback_days: int = 1250,
    threshold_window_days: int = 1250,
) -> LiveVolatilitySignal:
    """Create a latest-available GJR-GARCH volatility signal for the dashboard.

    This is a dashboard-facing ML signal adapter. It uses the latest available
    NIFTY spot close data, fits a GJR-GARCH(1,1) model, forecasts next-day
    volatility, and maps that forecast into a Low/Medium/High risk regime.
    """

    if not spot_path.exists():
        raise FileNotFoundError(f"NIFTY spot file not found: {spot_path}")

    df = pd.read_csv(spot_path)
    required_cols = {"date", "close"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {spot_path}: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")

    if len(df) < 300:
        raise ValueError("Not enough NIFTY spot observations for live volatility estimation.")

    close = df.set_index("date")["close"]
    returns = np.log(close).diff().dropna()

    if len(returns) < 300:
        raise ValueError("Not enough return observations for live volatility estimation.")

    model_returns = returns.tail(lookback_days) * 100.0

    model = arch_model(
        model_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="t",
        rescale=False,
    )

    fitted = model.fit(disp="off")
    forecast = fitted.forecast(horizon=1, reindex=False)

    daily_variance_percent_squared = float(forecast.variance.iloc[-1, 0])
    daily_vol_decimal = math.sqrt(max(daily_variance_percent_squared, 0.0)) / 100.0
    annualised_volatility = daily_vol_decimal * math.sqrt(TRADING_DAYS)

    realised_volatility = returns.rolling(20).std() * math.sqrt(TRADING_DAYS)
    threshold_sample = realised_volatility.dropna().tail(threshold_window_days)

    if len(threshold_sample) < 100:
        low_threshold = 0.12
        high_threshold = 0.22
    else:
        low_threshold = float(threshold_sample.quantile(0.33))
        high_threshold = float(threshold_sample.quantile(0.66))

    risk_regime = _classify_regime(
        annualised_volatility,
        low_threshold,
        high_threshold,
    )

    return LiveVolatilitySignal(
        source="latest_available_nifty_spot_gjr_garch",
        model="GJR-GARCH(1,1), Student-t errors",
        data_date=close.index[-1].strftime("%Y-%m-%d"),
        predicted_volatility=float(annualised_volatility),
        risk_regime=risk_regime,
        core_target_ratio=_core_ratio_for_regime(risk_regime),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        latest_close=float(close.iloc[-1]),
    )


if __name__ == "__main__":
    signal = get_live_volatility_signal()
    print(json.dumps(asdict(signal), indent=2))
