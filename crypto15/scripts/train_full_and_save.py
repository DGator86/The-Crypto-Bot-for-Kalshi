#!/usr/bin/env python3
"""Train the look-ahead model on the full dataset and persist artifacts."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, get_feature_config, get_model_config
from crypto15.data import load_data
from crypto15.features import create_features
from crypto15.model import LookaheadModel
from crypto15.util import setup_logging

logger = setup_logging()


def main():
    """Train look-ahead model on full dataset and save."""
    logger.info("Starting full model training for look-ahead predictions")

    data_config = get_data_config()
    feature_config = get_feature_config()
    model_config = get_model_config()

    primary_symbol = data_config['primary_symbol']
    primary_id = data_config['primary_symbol_id']
    target_tf = data_config.get('target_timeframe', data_config.get('timeframe', '15m'))
    dataset_name = f"{primary_id}_{target_tf}_dataset"

    logger.info("Loading dataset %s", dataset_name)
    df = load_data(dataset_name)

    logger.info("Engineering features (version %s)", feature_config.get('version', 'v1'))
    df_features = create_features(df, config=feature_config)
    logger.info("Feature matrix size: %d rows, %d columns", len(df_features), len(df_features.columns))

    model = LookaheadModel(config=model_config)
    model.train(df_features)

    # Evaluate on holdout portion if configured
    test_size = model_config.get('training', {}).get('test_size', 0.2)
    if 0 < test_size < 1:
        split_idx = int(len(df_features) * (1 - test_size))
        holdout_df = df_features.iloc[split_idx:]
        scores = model.evaluate_holdout(holdout_df)
        if scores:
            logger.info("Holdout metrics:")
            for key, value in scores.items():
                logger.info("  %s: %.4f", key, value)

    importance = model.get_feature_importance()
    logger.info("Top 10 combined feature importances:")
    for _, row in importance.head(10).iterrows():
        logger.info(
            "  %s: reg=%.4f cls=%.4f", row['feature'], row['importance_regression'], row['importance_classification']
        )

    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_file = models_dir / f"{primary_id}_{target_tf}_lookahead.pkl"
    model.save(str(model_file))

    importance_file = models_dir / f"{primary_id}_{target_tf}_feature_importance.csv"
    importance.to_csv(importance_file, index=False)
    logger.info("Persisted feature importance to %s", importance_file)

    logger.info("Model saved to %s", model_file)
    logger.info("Full model training completed")


if __name__ == "__main__":
    main()
