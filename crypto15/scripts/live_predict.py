#!/usr/bin/env python3
"""Generate live predictions using the trained look-ahead model."""

import copy
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, get_feature_config, get_model_config
from crypto15.data import build_lookahead_dataset
from crypto15.features import create_features
from crypto15.model import LookaheadModel
from crypto15.util import setup_logging

logger = setup_logging()


def main():
    """Fetch latest data, score with look-ahead model, and emit trading signal."""
    logger.info("Starting live look-ahead prediction")

    data_config = get_data_config()
    feature_config = get_feature_config()
    model_config = get_model_config()

    primary_symbol = data_config['primary_symbol']
    primary_id = data_config['primary_symbol_id']
    target_tf = data_config.get('target_timeframe', '15m')

    model_dir = Path(__file__).parent.parent / "models"
    model_file = model_dir / f"{primary_id}_{target_tf}_lookahead.pkl"

    if not model_file.exists():
        logger.error("Model file not found: %s", model_file)
        logger.error("Please train the model with train_full_and_save.py before running live predictions")
        return

    logger.info("Loading look-ahead model from %s", model_file)
    model = LookaheadModel.load(str(model_file))

    # Build a fresh dataset using a shorter live lookback window
    live_history_hours = data_config.get('live', {}).get('history_hours', 96)
    live_history_days = max(2, math.ceil(live_history_hours / 24))
    live_data_config = copy.deepcopy(data_config)
    live_data_config['history_days'] = live_history_days

    dataset, _ = build_lookahead_dataset(live_data_config)
    logger.info("Fetched %d rows of recent data for %s", len(dataset), primary_symbol)

    features_df = create_features(dataset, config=feature_config)
    latest_row = features_df.iloc[[-1]]

    predictions = model.predict(latest_row)
    latest_pred = predictions.iloc[0]

    current_price = latest_row['primary_close'].iloc[0]
    predicted_return = latest_pred['expected_return']
    probability_up = latest_pred['probability_up']
    signal = int(latest_pred['signal'])

    predicted_price = current_price * (1 + predicted_return)

    logger.info("=" * 80)
    logger.info("Live Look-Ahead Prediction for %s (%s)", primary_symbol, target_tf)
    logger.info("Current Bar: %s", latest_row.index[-1])
    logger.info("Current Price: %.2f", current_price)
    logger.info("Expected Return (next %s): %.4f (%.2f%%)", target_tf, predicted_return, predicted_return * 100)
    logger.info("Probability Up: %.2f%%", probability_up * 100)
    logger.info("Expected Price: %.2f", predicted_price)
    logger.info("Signal: %s", {1: 'LONG', -1: 'SHORT', 0: 'NEUTRAL'}.get(signal, 'NEUTRAL'))
    logger.info("Expected Value: %.6f", latest_pred['expected_value'])
    logger.info("=" * 80)

    logger.info("Live prediction completed")


if __name__ == "__main__":
    main()
