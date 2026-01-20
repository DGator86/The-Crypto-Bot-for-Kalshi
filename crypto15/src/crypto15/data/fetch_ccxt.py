"""
CCXT data fetching module.
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


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
    exchange_name: str = "binance"
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
    logger.info(f"Fetching {symbol} {timeframe} data from {exchange_name}")
    
    exchange = get_exchange(exchange_name)
    
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
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Fetched {len(df)} candles for {symbol}")
    
    return df


def fetch_historical_data(
    symbol: str,
    timeframe: str = "1h",
    days: int = 365,
    exchange_name: str = "binance"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a specified period.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        days: Number of days of historical data
        exchange_name: Name of the exchange
    
    Returns:
        DataFrame with historical OHLCV data
    """
    since = datetime.now() - timedelta(days=days)
    
    all_data = []
    current_since = since
    
    while current_since < datetime.now():
        df = fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=current_since,
            limit=1000,
            exchange_name=exchange_name
        )
        
        if df.empty:
            break
        
        all_data.append(df)
        current_since = df.index[-1]
    
    if not all_data:
        return pd.DataFrame()
    
    # Concatenate all data
    result = pd.concat(all_data)
    result = result[~result.index.duplicated(keep='first')]
    result.sort_index(inplace=True)
    
    return result
