"""
technical_indicators.py

Pure pandas/numpy implementations of all technical and volatility indicators
used in the feature engineering pipeline. Each function takes a DataFrame
and returns a named Series aligned to the same index.

No file I/O here — this module is imported by build_features.py.

Indicators implemented:
    Technical  : RSI(14), MACD(12,26,9), VWAP, ATR(14), SMA20 crossover
    OI         : OI % change, OI/volume ratio, OI 20-day MA
    Volatility : 20-day rolling std of log returns, log return
"""

import numpy as np
import pandas as pd


# ── Technical indicators ──────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EMA with alpha=1/period).
    Values range 0–100. Overbought above 70, oversold below 30.

    Formula:
        RSI = 100 - 100 / (1 + RS)
        RS  = avg_gain / avg_loss  (Wilder smoothed)
    """
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing — equivalent to EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi.name = f"rsi_{period}"
    return rsi


def macd(close: pd.Series,
         fast: int = 12, slow: int = 26, signal: int = 9
         ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence.

    Returns:
        macd_line   : EMA(fast) - EMA(slow)
        signal_line : EMA(macd_line, signal)
        histogram   : macd_line - signal_line
    """
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line

    macd_line.name   = f"macd_{fast}_{slow}"
    signal_line.name = f"macd_signal_{signal}"
    histogram.name   = f"macd_hist"
    return macd_line, signal_line, histogram


def vwap(high: pd.Series, low: pd.Series,
         close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Volume Weighted Average Price using daily OHLCV.
    Since we have daily bars (not intraday), VWAP is approximated using
    a 20-day rolling cumulative calculation to preserve the volume-weighting
    signal over time.

    Formula: VWAP_t = sum(TP * V, window) / sum(V, window)
    where TP = (high + low + close) / 3

    Note: pass futures volume, not spot index volume — the Nifty 50 spot
    index carries zero volume since it is not a tradeable instrument.
    If volume is zero throughout, falls back to the 20-day SMA of TP.
    """
    tp      = (high + low + close) / 3
    cum_tpv = (tp * volume).rolling(window=20).sum()
    cum_vol = volume.rolling(window=20).sum()

    # Fallback to simple 20-day MA of typical price when volume is unavailable
    vwap_val = cum_tpv / cum_vol.replace(0, np.nan)
    fallback = tp.rolling(window=20).mean()
    vwap_val = vwap_val.fillna(fallback)

    vwap_val.name = "vwap_20"
    return vwap_val


def atr(high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder's smoothing.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = Wilder EMA of True Range over `period` days.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr_val      = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    atr_val.name = f"atr_{period}"
    return atr_val


def sma_crossover(close: pd.Series,
                  fast: int = 10, slow: int = 20) -> pd.Series:
    """
    SMA crossover signal.
    Returns +1 when fast SMA > slow SMA (uptrend), -1 otherwise.
    Used as a binary trend direction feature.
    """
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    signal   = (sma_fast > sma_slow).astype(int) * 2 - 1
    signal.name = f"sma_cross_{fast}_{slow}"
    return signal


# ── Open interest features ────────────────────────────────────────────────────

def oi_pct_change(oi: pd.Series, period: int = 1) -> pd.Series:
    """
    Percentage change in open interest over `period` days.
    Rising OI with rising price confirms trend; falling OI suggests reversal.
    """
    pct        = oi.pct_change(periods=period) * 100
    pct.name   = f"oi_pct_change_{period}d"
    return pct


def oi_volume_ratio(oi: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Ratio of open interest to trading volume.
    High ratio indicates stronger participant commitment to existing positions.
    """
    ratio      = oi / volume.replace(0, np.nan)
    ratio.name = "oi_vol_ratio"
    return ratio


def oi_moving_average(oi: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling mean of open interest over `window` days.
    Smooths the daily OI signal to identify broader positioning trends.
    """
    ma      = oi.rolling(window=window).mean()
    ma.name = f"oi_ma_{window}"
    return ma


# ── Volatility features ───────────────────────────────────────────────────────

def log_return(close: pd.Series) -> pd.Series:
    """
    Daily log return: ln(close_t / close_{t-1}).
    Used directly as a feature and as input to the volatility calculations.
    """
    ret      = np.log(close / close.shift(1))
    ret.name = "log_return"
    return ret


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualised rolling standard deviation of log returns.
    Multiplied by sqrt(252) to convert daily std to annualised volatility.
    This is the realised volatility estimate used for risk label generation.
    """
    log_ret  = log_return(close)
    vol      = log_ret.rolling(window=window).std() * np.sqrt(252)
    vol.name = f"realised_vol_{window}d"
    return vol