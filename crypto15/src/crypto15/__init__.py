"""
The Crypto Bot for Kalshi - Main Package

A comprehensive crypto trading bot with:
- ML-based price prediction (XGBoost dual-head model)
- Walk-forward backtesting with purging
- Live trading with paper mode support
- Kalshi API integration
- Hyperparameter tuning with Optuna
"""

__version__ = "0.2.0"
__author__ = "Crypto Bot Team"

from .config import load_config
from .util import setup_logging

__all__ = ["load_config", "setup_logging"]

# Lazy imports for submodules
def __getattr__(name):
    if name == "exchange":
        from . import exchange
        return exchange
    elif name == "trading":
        from . import trading
        return trading
    elif name == "model":
        from . import model
        return model
    elif name == "data":
        from . import data
        return data
    elif name == "features":
        from . import features
        return features
    elif name == "backtest":
        from . import backtest
        return backtest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
