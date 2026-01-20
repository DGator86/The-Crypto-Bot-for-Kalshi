import pandas as pd
import numpy as np

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean() # Simple rolling for simplicity or ewm
    # Standard RSI uses smoothed moving average usually. 
    # Let's match pandas_ta default which often uses wilder's smoothing (RMA).
    # But simple EMA is often close enough for baselines. 
    # Let's use Wilder's Smoothing: alpha = 1/length
    
    # Using ewm for Wilder's
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # pandas_ta rsi uses RMA (Wilder's)
    ma_up = up.ewm(com=length - 1, adjust=False).mean()
    ma_down = down.ewm(com=length - 1, adjust=False).mean()
    
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd(series: pd.Series, fast: int, slow: int, signal: int) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
        f"MACDh_{fast}_{slow}_{signal}": hist
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR is usually RMA of TR
    return tr.ewm(com=length - 1, adjust=False).mean()

def bbands(series: pd.Series, length: int, std: float) -> pd.DataFrame:
    mid = series.rolling(window=length).mean()
    s = series.rolling(window=length).std()
    upper = mid + std * s
    lower = mid - std * s
    # Calculate Bandwidth and Percent B
    # Bandwidth = (Upper - Lower) / Middle
    # Percent B = (Price - Lower) / (Upper - Lower)
    
    # pandas_ta returns columns like BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0
    return pd.DataFrame({
        f"BBL_{length}_{std}": lower,
        f"BBM_{length}_{std}": mid,
        f"BBU_{length}_{std}": upper,
        f"BBB_{length}_{std}": (upper - lower) / mid,
        f"BBP_{length}_{std}": (series - lower) / (upper - lower)
    })
