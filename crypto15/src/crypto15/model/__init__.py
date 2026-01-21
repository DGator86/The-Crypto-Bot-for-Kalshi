"""
Machine learning model module.
"""

from .lookahead import LookaheadModel
from .xgb import XGBModel, train_model, predict

# Conditional import for tuning (requires optuna)
try:
    from .tuning import HyperparameterTuner, TuningConfig, TuningResult, quick_tune
    _TUNING_AVAILABLE = True
except ImportError:
    _TUNING_AVAILABLE = False

__all__ = ["XGBModel", "LookaheadModel", "train_model", "predict"]

if _TUNING_AVAILABLE:
    __all__.extend(["HyperparameterTuner", "TuningConfig", "TuningResult", "quick_tune"])
