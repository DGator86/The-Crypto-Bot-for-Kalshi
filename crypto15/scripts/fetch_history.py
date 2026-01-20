#!/usr/bin/env python3
"""
Script to fetch and save historical data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config
from crypto15.data import fetch_historical_data, save_data
from crypto15.util import setup_logging

import logging

logger = setup_logging()


def main():
    """Fetch historical data and save to disk."""
    logger.info("Starting historical data fetch")
    
    # Load configuration
    config = get_data_config()
    exchange = config.get('exchange', 'binance')
    symbols = config.get('symbols', ['BTC/USDT'])
    timeframe = config.get('timeframe', '1h')
    history_days = config.get('history_days', 365)
    
    # Fetch data for each symbol
    for symbol in symbols:
        logger.info(f"Fetching {symbol} data...")
        
        try:
            df = fetch_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                days=history_days,
                exchange_name=exchange
            )
            
            # Save data
            filename = f"{symbol.replace('/', '_')}_{timeframe}"
            save_data(df, filename, format='parquet')
            
            logger.info(f"Successfully fetched and saved {len(df)} records for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
    
    logger.info("Historical data fetch completed")


if __name__ == "__main__":
    main()
