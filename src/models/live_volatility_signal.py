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
    signal_mode: str = "confirmed_history"
    live_quote_used: bool = False
    live_quote_close: float | None = None
    live_quote_timestamp: str = ""
    data_freshness_note: str = "Latest confirmed historical close used"


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


def _parse_live_quote_timestamp(value):
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = (
            value.replace(" IST", "")
            .replace(" UTC", "")
            .replace(" GMT", "")
            .strip()
        )
        parsed = pd.to_datetime(cleaned, errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")

    if pd.isna(parsed):
        return None

    parsed = pd.Timestamp(parsed)

    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)

    return parsed.normalize()


def _append_live_quote_if_available(close, live_close, live_timestamp, use_live_price):
    if not use_live_price or live_close is None:
        return close, False, "", None, "Market feed not fresh/open; latest confirmed historical close used"

    try:
        live_close = float(live_close)
    except Exception:
        return close, False, "", None, "Live quote could not be converted to a valid price"

    if not np.isfinite(live_close) or live_close <= 0:
        return close, False, "", None, "Live quote was invalid; latest confirmed historical close used"

    live_date = _parse_live_quote_timestamp(live_timestamp)
    if live_date is None:
        live_date = pd.Timestamp.utcnow().normalize()

    close = close.copy().sort_index()
    latest_historical_date = pd.Timestamp(close.index[-1]).normalize()

    if live_date < latest_historical_date:
        return (
            close,
            False,
            str(live_timestamp or ""),
            live_close,
            "Live quote date is older than the local historical dataset; latest confirmed historical close used",
        )

    close.loc[live_date] = live_close
    close = close.sort_index()

    if live_date == latest_historical_date:
        note = "Live Kite quote replaced the latest historical close as a provisional intraday observation"
    else:
        note = "Live Kite quote appended as a provisional latest observation for intraday volatility estimation"

    return close, True, str(live_timestamp or live_date.date()), live_close, note


def get_live_volatility_signal(
    spot_path: Path = DEFAULT_SPOT_PATH,
    lookback_days: int = 1250,
    threshold_window_days: int = 1250,
    live_close: float | None = None,
    live_timestamp=None,
    use_live_price: bool = False,
) -> LiveVolatilitySignal:
    """
     Create a dashboard-facing GJR-GARCH volatility signal.

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

    close = df.set_index("date")["close"].sort_index()

    close, live_quote_used, live_quote_timestamp, live_quote_close, freshness_note = (
        _append_live_quote_if_available(
            close=close,
            live_close=live_close,
            live_timestamp=live_timestamp,
            use_live_price=use_live_price,
        )
    )

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

    source = (
        "live_intraday_nifty_spot_gjr_garch"
        if live_quote_used
        else "latest_available_nifty_spot_gjr_garch"
    )

    signal_mode = "intraday_provisional" if live_quote_used else "confirmed_history"

    return LiveVolatilitySignal(
        source=source,
        model="GJR-GARCH(1,1), Student-t errors",
        data_date=close.index[-1].strftime("%Y-%m-%d"),
        predicted_volatility=float(annualised_volatility),
        risk_regime=risk_regime,
        core_target_ratio=_core_ratio_for_regime(risk_regime),
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        latest_close=float(close.iloc[-1]),
        signal_mode=signal_mode,
        live_quote_used=live_quote_used,
        live_quote_close=live_quote_close,
        live_quote_timestamp=live_quote_timestamp,
        data_freshness_note=freshness_note,
    )


if __name__ == "__main__":
    signal = get_live_volatility_signal()
    print(json.dumps(asdict(signal), indent=2))