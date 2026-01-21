#!/usr/bin/env python3
"""
Script to fetch, merge, and persist historical datasets for the look-ahead model.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config
from crypto15.data import build_lookahead_dataset, save_data
from crypto15.util import setup_logging
from crypto15.data.store import get_data_dir

logger = setup_logging()


def main():
    """Fetch historical data, build the modeling dataset, and persist it."""
    logger.info("Starting look-ahead dataset build")

    data_config = get_data_config()

    try:
        dataset, metadata = build_lookahead_dataset(data_config)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to build dataset: %s", exc)
        raise

    primary_id = metadata['primary_symbol_id']
    target_tf = metadata['target_timeframe']
    filename = f"{primary_id}_{target_tf}_dataset"

    save_data(dataset, filename, format='parquet')
    logger.info("Saved dataset '%s' with %d rows and %d columns", filename, len(dataset), len(dataset.columns))

    # Persist metadata alongside dataset for downstream scripts
    data_dir = get_data_dir()
    metadata_path = data_dir / f"{filename}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote dataset metadata to %s", metadata_path)

    logger.info("Historical dataset build completed")


if __name__ == "__main__":
    main()
