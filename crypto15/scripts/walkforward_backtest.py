#!/usr/bin/env python3
"""
Script to run walk-forward backtest.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, get_backtest_config, get_model_config
from crypto15.data import load_data
from crypto15.features import create_features
from crypto15.model import XGBModel
from crypto15.backtest import WalkForwardBacktest
from crypto15.util import setup_logging

import pandas as pd
import numpy as np
import logging

logger = setup_logging()


def main():
    """Run walk-forward backtest."""
    logger.info("Starting walk-forward backtest")
    
    # Load configuration
    data_config = get_data_config()
    backtest_config = get_backtest_config()
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
    
    # Setup backtest
    n_splits = backtest_config.get('walkforward_windows', 12)
    train_ratio = backtest_config.get('train_test_split', 0.8)
    
    backtest = WalkForwardBacktest(n_splits=n_splits, train_ratio=train_ratio)
    
    # Define train, predict, and evaluate functions
    def train_func(df_train):
        """Train model on training data."""
        model = XGBModel(params=model_config)
        X, y = model.prepare_data(df_train, target_col='target')
        model.train(X, y)
        return model
    
    def predict_func(model, df_test):
        """Make predictions on test data."""
        X = df_test[model.feature_names]
        return model.predict(X)
    
    def evaluate_func(df_test, predictions):
        """Evaluate predictions."""
        actual = df_test['target'].values
        
        # Calculate metrics
        mse = np.mean((predictions - actual) ** 2)
        mae = np.mean(np.abs(predictions - actual))
        
        # Directional accuracy
        pred_direction = (predictions > 0).astype(int)
        actual_direction = (actual > 0).astype(int)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
    
    # Run backtest
    logger.info("Running walk-forward backtest")
    results = backtest.run_backtest(
        df=df,
        train_func=train_func,
        predict_func=predict_func,
        evaluate_func=evaluate_func
    )
    
    # Summarize results
    summary = backtest.summarize_results(results)
    
    logger.info("Backtest Results:")
    for key, value in summary.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Walk-forward backtest completed")


if __name__ == "__main__":
    main()
