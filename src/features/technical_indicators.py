import numpy as np
import pandas as pd


# Technical indicators

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (EMA with alpha=1/period).
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

    """
    sma_fast = close.rolling(window=fast).mean()
    sma_slow = close.rolling(window=slow).mean()
    signal   = (sma_fast > sma_slow).astype(int) * 2 - 1
    signal.name = f"sma_cross_{fast}_{slow}"
    return signal


# Open interest features

def oi_pct_change(oi: pd.Series, period: int = 1) -> pd.Series:
    """
    Percentage change in open interest over `period` days.

    """
    pct        = oi.pct_change(periods=period) * 100
    pct.name   = f"oi_pct_change_{period}d"
    return pct


def oi_volume_ratio(oi: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Ratio of open interest to trading volume.

    """
    ratio      = oi / volume.replace(0, np.nan)
    ratio.name = "oi_vol_ratio"
    return ratio


def oi_moving_average(oi: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling mean of open interest over `window` days.

    """
    ma      = oi.rolling(window=window).mean()
    ma.name = f"oi_ma_{window}"
    return ma


# Volatility features

def log_return(close: pd.Series) -> pd.Series:
    """
    Daily log return: ln(close_t / close_{t-1}).

    """
    ret      = np.log(close / close.shift(1))
    ret.name = "log_return"
    return ret


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """
    Annualised rolling standard deviation of log returns.

    """
    log_ret  = log_return(close)
    vol      = log_ret.rolling(window=window).std() * np.sqrt(252)
    vol.name = f"realised_vol_{window}d"
    return vol
