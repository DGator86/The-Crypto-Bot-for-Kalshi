"""
Feature engineering module.
"""

from typing import Any, Dict, Optional

from .feature_set_v1 import FeatureSetV1
from .feature_set_v2 import FeatureSetV2

__all__ = ["FeatureSetV1", "FeatureSetV2", "create_features"]


def create_features(df, config: Optional[Dict[str, Any]] = None):
    """Factory helper that instantiates the configured feature set."""
    config = config or {}
    version = config.get('version') or config.get('feature_version') or 'v1'

    if version.lower() == 'v2':
        feature_set = FeatureSetV2(config=config)
    else:
        lookback_periods = config.get('lookback_periods')
        feature_set = FeatureSetV1(lookback_periods=lookback_periods)

    return feature_set.create_features(df)
