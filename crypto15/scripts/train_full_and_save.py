#!/usr/bin/env python3
"""
Script to train model on full dataset and save.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, get_model_config
from crypto15.data import load_data
from crypto15.features import create_features
from crypto15.model import XGBModel
from crypto15.util import setup_logging

import logging

logger = setup_logging()


def main():
    """Train model on full dataset and save."""
    logger.info("Starting full model training")
    
    # Load configuration
    data_config = get_data_config()
    model_config = get_model_config()
    
    # Load data
    symbol = data_config.get('symbols', ['BTC/USDT'])[0]
    timeframe = data_config.get('timeframe', '1h')
    filename = f"{symbol.replace('/', '_')}_{timeframe}"
    
    logger.info(f"Loading data for {symbol}")
    df = load_data(filename)
    
    # Create features
    logger.info("Creating features")
    df = create_features(df)
    
    # Add target (next hour return)
    df['target'] = df['close'].shift(-1) / df['close'] - 1
    df = df.dropna()
    
    # Train model
    logger.info("Training model on full dataset")
    model = XGBModel(params=model_config)
    X, y = model.prepare_data(df, target_col='target')
    model.train(X, y)
    
    # Show feature importance
    importance = model.get_feature_importance()
    logger.info("Top 10 important features:")
    for _, row in importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "models"
    model_path.mkdir(exist_ok=True)
    model_file = model_path / f"{symbol.replace('/', '_')}_{timeframe}_model.pkl"
    
    model.save(str(model_file))
    logger.info(f"Model saved to {model_file}")
    
    logger.info("Full model training completed")


if __name__ == "__main__":
    main()
