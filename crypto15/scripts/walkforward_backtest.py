#!/usr/bin/env python3
"""Script to run walk-forward backtest with the enhanced look-ahead model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn import metrics

from crypto15.backtest import TradingSimulator, WalkForwardBacktest
from crypto15.config import (
    get_backtest_config,
    get_data_config,
    get_feature_config,
    get_model_config,
    get_trading_config,
)
from crypto15.data import load_data
from crypto15.features import create_features
from crypto15.model import LookaheadModel
from crypto15.util import setup_logging

import numpy as np

logger = setup_logging()


def main():
    """Run walk-forward backtest."""
    logger.info("Starting walk-forward backtest with look-ahead model")

    # Load configuration
    data_config = get_data_config()
    feature_config = get_feature_config()
    backtest_config = get_backtest_config()
    model_config = get_model_config()
    trading_config = get_trading_config()

    primary_id = data_config['primary_symbol_id']
    target_tf = data_config.get('target_timeframe', data_config.get('timeframe', '15m'))
    dataset_name = f"{primary_id}_{target_tf}_dataset"

    logger.info("Loading dataset %s", dataset_name)
    df = load_data(dataset_name)

    logger.info("Creating features (version %s)", feature_config.get('version', 'v1'))
    df_features = create_features(df, config=feature_config)

    logger.info("Dataset after feature engineering: %d rows, %d columns", len(df_features), len(df_features.columns))

    n_splits = backtest_config.get('walkforward_windows', 12)
    train_ratio = backtest_config.get('train_test_split', 0.8)
    purge_ratio = backtest_config.get('purge_ratio', 0.0)
    embargo_ratio = backtest_config.get('embargo_ratio', 0.0)

    backtest = WalkForwardBacktest(
        n_splits=n_splits,
        train_ratio=train_ratio,
        purge_ratio=purge_ratio,
        embargo_ratio=embargo_ratio,
    )

    simulator = TradingSimulator(
        initial_capital=backtest_config.get('initial_capital', 10000.0),
        max_position_size=trading_config.get('max_position_size', 0.1),
        stop_loss=trading_config.get('stop_loss', 0.02),
        take_profit=trading_config.get('take_profit', 0.05),
        commission=backtest_config.get('commission', trading_config.get('commission', 0.0)),
        annualization_factor=backtest_config.get('annualization_factor', 252),
    )

    def train_func(df_train):
        model = LookaheadModel(config=model_config)
        model.train(df_train)
        return model

    def predict_func(model, df_test):
        return model.predict(df_test)

    def evaluate_func(df_test, prediction_df):
        actual_returns = df_test['target_return']
        actual_up = df_test['target_up'] if 'target_up' in df_test.columns else (actual_returns > 0).astype(int)

        expected_return = prediction_df['expected_return']
        prob_up = prediction_df['probability_up']
        signals = prediction_df['signal']

        mse = float(np.mean((expected_return - actual_returns) ** 2))
        mae = float(np.mean(np.abs(expected_return - actual_returns)))
        directional_accuracy = float(np.mean((expected_return > 0) == (actual_returns > 0)))

        try:
            roc_auc = float(metrics.roc_auc_score(actual_up, prob_up))
        except Exception:  # pylint: disable=broad-except
            roc_auc = float('nan')
        pred_labels = (prob_up >= 0.5).astype(int)
        precision = metrics.precision_score(actual_up, pred_labels, zero_division=0)
        recall = metrics.recall_score(actual_up, pred_labels, zero_division=0)
        f1 = metrics.f1_score(actual_up, pred_labels, zero_division=0)

        coverage = float(np.mean(signals != 0))
        signal_direction_match = float(np.mean(np.sign(signals) == np.sign(actual_returns)))

        sim_results = simulator.simulate(df_test, signals, price_col='primary_close')
        sim_metrics = {f"sim_{k}": v for k, v in sim_results.metrics.items()}

        return {
            'mse': mse,
            'mae': mae,
            'directional_accuracy': directional_accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'signal_coverage': coverage,
            'signal_direction_match': signal_direction_match,
            **sim_metrics,
        }

    results = backtest.run_backtest(
        df=df_features,
        train_func=train_func,
        predict_func=predict_func,
        evaluate_func=evaluate_func,
    )

    summary = backtest.summarize_results(results)

    logger.info("Backtest Summary:")
    for key, value in summary.items():
        logger.info("  %s: %.4f", key, value)

    logger.info("Walk-forward backtest completed")


if __name__ == "__main__":
    main()
