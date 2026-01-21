#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna.

Usage:
    # Quick tuning (30 trials)
    python tune_hyperparameters.py --quick

    # Full tuning (50 trials)
    python tune_hyperparameters.py

    # Extended tuning (100 trials)
    python tune_hyperparameters.py --trials 100

    # Optimize for Sharpe ratio
    python tune_hyperparameters.py --metric sharpe

    # Save study for later resumption
    python tune_hyperparameters.py --storage sqlite:///tuning.db
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, get_feature_config, load_config
from crypto15.data import build_lookahead_dataset
from crypto15.data.store import load_data, get_data_dir
from crypto15.features import create_features
from crypto15.util import setup_logging

logger = setup_logging()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune model hyperparameters using Optuna"
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick tuning with 30 trials and 3-fold CV"
    )

    parser.add_argument(
        "--metric",
        choices=["roc_auc", "sharpe", "f1", "accuracy", "expectancy"],
        default="roc_auc",
        help="Metric to optimize (default: roc_auc)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds (default: 3600 = 1 hour)"
    )

    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation splits (default: 5)"
    )

    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///tuning.db)"
    )

    parser.add_argument(
        "--study-name",
        type=str,
        default="crypto_tuning",
        help="Name for the Optuna study"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Path to existing data file (skip fetching)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Check if optuna is available
    try:
        from crypto15.model.tuning import HyperparameterTuner, TuningConfig, quick_tune
    except ImportError:
        logger.error("Optuna is required for hyperparameter tuning.")
        logger.error("Install it with: pip install optuna")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Hyperparameter Tuning")
    logger.info("=" * 60)

    # Load configuration
    data_config = get_data_config()
    feature_config = get_feature_config()

    symbol = data_config.get('primary_symbol', 'BTC/USDT')
    symbol_id = data_config.get('primary_symbol_id', 'BTCUSDT')
    timeframe = data_config.get('target_timeframe', '15m')

    logger.info("Symbol: %s", symbol)
    logger.info("Timeframe: %s", timeframe)
    logger.info("Metric: %s", args.metric)
    logger.info("Trials: %d", 30 if args.quick else args.trials)
    logger.info("CV Splits: %d", 3 if args.quick else args.cv_splits)
    logger.info("=" * 60)

    # Load or fetch data
    if args.data_file:
        logger.info("Loading data from %s", args.data_file)
        dataset = load_data(args.data_file)
    else:
        # Try to load existing data first
        data_dir = get_data_dir()
        data_file = data_dir / f"{symbol_id}_{timeframe}_dataset.parquet"

        if data_file.exists():
            logger.info("Loading existing data from %s", data_file)
            dataset = load_data(str(data_file))
        else:
            logger.info("Fetching fresh data...")
            dataset, _ = build_lookahead_dataset(data_config)

    logger.info("Dataset: %d rows", len(dataset))

    # Generate features
    logger.info("Generating features...")
    features_df = create_features(dataset, config=feature_config)
    logger.info("Features: %d columns", len(features_df.columns))

    # Drop rows with NaN targets
    if 'target_return' in features_df.columns:
        initial_len = len(features_df)
        features_df = features_df.dropna(subset=['target_return'])
        logger.info("Dropped %d rows with NaN targets", initial_len - len(features_df))

    # Run tuning
    if args.quick:
        logger.info("Running quick tuning...")
        result = quick_tune(
            features_df,
            n_trials=30,
            optimize_metric=args.metric,
        )
    else:
        config = TuningConfig(
            n_trials=args.trials,
            timeout=args.timeout,
            n_cv_splits=args.cv_splits,
            optimize_metric=args.metric,
        )

        tuner = HyperparameterTuner(config)

        result = tuner.tune(
            features_df,
            study_name=args.study_name,
            storage=args.storage,
        )

    # Print results
    logger.info("=" * 60)
    logger.info("Tuning Results")
    logger.info("=" * 60)
    logger.info("Best %s: %.4f", args.metric, result.best_score)
    logger.info("Study Stats: %s", result.study_stats)
    logger.info("")
    logger.info("Best Parameters:")

    model_config = result.to_model_config()

    logger.info("Classification params:")
    for k, v in model_config['classification_params'].items():
        logger.info("  %s: %s", k, v)

    logger.info("Regression params:")
    for k, v in model_config['regression_params'].items():
        logger.info("  %s: %s", k, v)

    logger.info("Thresholds:")
    for k, v in model_config['thresholds'].items():
        logger.info("  %s: %.4f", k, v)

    logger.info("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "models"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{symbol_id}_{timeframe}_tuning_results.json"

    results_dict = {
        "symbol": symbol,
        "timeframe": timeframe,
        "metric": args.metric,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "model_config": model_config,
        "study_stats": result.study_stats,
    }

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info("Results saved to %s", output_path)

    # Print config snippet for easy copy-paste
    logger.info("")
    logger.info("Add to config.yaml:")
    logger.info("")
    logger.info("model:")
    logger.info("  classification_params:")
    for k, v in model_config['classification_params'].items():
        if isinstance(v, float):
            logger.info("    %s: %.4f", k, v)
        else:
            logger.info("    %s: %s", k, v)
    logger.info("  regression_params:")
    for k, v in model_config['regression_params'].items():
        if isinstance(v, float):
            logger.info("    %s: %.4f", k, v)
        else:
            logger.info("    %s: %s", k, v)
    logger.info("  thresholds:")
    for k, v in model_config['thresholds'].items():
        logger.info("    %s: %.4f", k, v)


if __name__ == "__main__":
    main()
