#!/usr/bin/env python3
"""
Script to make live predictions using saved model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config
from crypto15.data import fetch_ohlcv
from crypto15.features import create_features
from crypto15.model import XGBModel
from crypto15.util import setup_logging

import logging
from datetime import datetime, timedelta

logger = setup_logging()


def main():
    """Make live predictions using saved model."""
    logger.info("Starting live prediction")
    
    # Load configuration
    data_config = get_data_config()
    
    symbol = data_config.get('symbols', ['BTC/USDT'])[0]
    timeframe = data_config.get('timeframe', '1h')
    exchange = data_config.get('exchange', 'binance')
    
    # Load model
    model_path = Path(__file__).parent.parent / "models"
    model_file = model_path / f"{symbol.replace('/', '_')}_{timeframe}_model.pkl"
    
    if not model_file.exists():
        logger.error(f"Model file not found: {model_file}")
        logger.error("Please run train_full_and_save.py first")
        return
    
    logger.info(f"Loading model from {model_file}")
    model = XGBModel.load(str(model_file))
    
    # Fetch recent data
    logger.info(f"Fetching recent data for {symbol}")
    since = datetime.now() - timedelta(days=60)  # Get 60 days of data for features
    
    df = fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        since=since,
        limit=1440,  # ~60 days of hourly data
        exchange_name=exchange
    )
    
    # Create features
    logger.info("Creating features")
    df = create_features(df)
    
    # Get latest data point
    latest = df.iloc[-1:]
    latest_features = latest[model.feature_names]
    
    # Make prediction
    prediction = model.predict(latest_features)[0]
    
    # Display results
    current_price = df['close'].iloc[-1]
    predicted_return = prediction
    predicted_price = current_price * (1 + predicted_return)
    
    logger.info("=" * 60)
    logger.info(f"Live Prediction for {symbol}")
    logger.info("=" * 60)
    logger.info(f"Current Time: {df.index[-1]}")
    logger.info(f"Current Price: ${current_price:.2f}")
    logger.info(f"Predicted Return: {predicted_return * 100:.2f}%")
    logger.info(f"Predicted Price: ${predicted_price:.2f}")
    
    if predicted_return > 0:
        logger.info("Signal: BUY / LONG")
    elif predicted_return < 0:
        logger.info("Signal: SELL / SHORT")
    else:
        logger.info("Signal: HOLD / NEUTRAL")
    
    logger.info("=" * 60)
    logger.info("Live prediction completed")


if __name__ == "__main__":
    main()
