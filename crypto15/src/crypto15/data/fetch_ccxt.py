"""
CCXT data fetching module.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, List

import ccxt
import pandas as pd

import logging

logger = logging.getLogger(__name__)

_TIMEFRAME_MULTIPLIERS = {
    'm': 60,
    'h': 3600,
    'd': 86400,
    'w': 604800,
}


def timeframe_to_timedelta(timeframe: str) -> timedelta:
    """Convert a CCXT timeframe string into a timedelta."""
    timeframe = timeframe.strip()
    value = int(timeframe[:-1])
    unit = timeframe[-1].lower()
    if unit not in _TIMEFRAME_MULTIPLIERS:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")
    seconds = value * _TIMEFRAME_MULTIPLIERS[unit]
    return timedelta(seconds=seconds)


def to_pandas_freq(timeframe: str) -> str:
    """Convert a CCXT timeframe string to a pandas frequency string."""
    timeframe = timeframe.strip()
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    if unit == 'm':
        return f"{value}T"
    if unit == 'h':
        return f"{value}H"
    if unit == 'd':
        return f"{value}D"
    if unit == 'w':
        return f"{value}W"
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to a different timeframe using standard OHLC aggregation."""
    if df.empty:
        return df

    freq = to_pandas_freq(target_timeframe)
    agg_map = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    resampled = df.resample(freq, label='right', closed='right').agg(agg_map)
    resampled = resampled.dropna(how='any')
    return resampled


def get_exchange(exchange_name: str = "binance") -> ccxt.Exchange:
    """
    Initialize and return a CCXT exchange instance.
    
    Args:
        exchange_name: Name of the exchange (default: binance)
    
    Returns:
        CCXT exchange instance
    """
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        'enableRateLimit': True,
    })
    return exchange


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    since: Optional[datetime] = None,
    limit: int = 1000,
    exchange_name: str = "binance",
    exchange: Optional[ccxt.Exchange] = None
) -> pd.DataFrame:
    """
    Fetch OHLCV data from exchange using CCXT.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '1h', '1d')
        since: Start datetime for historical data
        limit: Number of candles to fetch
        exchange_name: Name of the exchange
    
    Returns:
        DataFrame with OHLCV data
    """
    logger.debug(f"Fetching {symbol} {timeframe} data from {exchange_name} (limit={limit})")
    
    exchange = exchange or get_exchange(exchange_name)
    
    # Convert datetime to timestamp if provided
    since_ts = None
    if since:
        since_ts = int(since.timestamp() * 1000)
    
    # Fetch OHLCV data
    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        since=since_ts,
        limit=limit
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    if df.empty:
        return df

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    logger.debug(f"Fetched {len(df)} candles for {symbol}")
    
    return df


def fetch_historical_data(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
    exchange_name: str = "binance",
    limit: int = 1000,
    max_batches: int = 5000,
    sleep_seconds: float = 0.0,
    exchange: Optional[ccxt.Exchange] = None
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a specified period.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        days: Number of days of historical data
        exchange_name: Name of the exchange
        limit: Number of candles per request
        max_batches: Maximum number of fetch iterations to avoid infinite loops
        sleep_seconds: Optional sleep between API calls to respect rate limits
    
    Returns:
        DataFrame with historical OHLCV data
    """
    timeframe_delta = timeframe_to_timedelta(timeframe)
    now = datetime.utcnow()
    since = now - timedelta(days=days)
    
    all_data: List[pd.DataFrame] = []
    current_since = since
    iterations = 0

    exchange_instance = exchange or get_exchange(exchange_name)

    while current_since < now and iterations < max_batches:
        df = fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=current_since,
            limit=limit,
            exchange_name=exchange_name,
            exchange=exchange_instance
        )
        
        if df.empty:
            logger.debug("No more data returned; stopping fetch loop")
            break
        
        # Filter out any data we already have to prevent duplicates
        if all_data:
            last_timestamp = all_data[-1].index[-1]
            df = df[df.index > last_timestamp]
        
        if df.empty:
            logger.debug("Received duplicate data; advancing cursor")
            current_since = current_since + timeframe_delta
            iterations += 1
            if sleep_seconds:
                time.sleep(sleep_seconds)
            continue

        all_data.append(df)
        current_since = df.index[-1] + timeframe_delta
        iterations += 1

        if sleep_seconds:
            time.sleep(sleep_seconds)
    
    if not all_data:
        logger.warning(f"No historical data fetched for {symbol}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Concatenate all data
    result = pd.concat(all_data)
    result = result[~result.index.duplicated(keep='first')]
    result.sort_index(inplace=True)
    
    return result
