"""
Machine learning model module.
"""

from .xgb import XGBModel, train_model, predict

__all__ = ["XGBModel", "train_model", "predict"]
