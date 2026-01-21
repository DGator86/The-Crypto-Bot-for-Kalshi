"""Dual-head XGBoost model for probabilistic 15-minute look-ahead predictions."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import metrics

logger = logging.getLogger(__name__)


@dataclass
class LookaheadPredictions:
    expected_return: np.ndarray
    probability_up: np.ndarray
    signal: np.ndarray
    expected_value: np.ndarray


class LookaheadModel:
    """Wrapper around twin XGBoost models (classification + regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.classification_params = dict(self.config.get('classification_params', {}))
        self.regression_params = dict(self.config.get('regression_params', {}))
        self.training_config = dict(self.config.get('training', {}))
        self.thresholds = dict(self.config.get('thresholds', {}))

        # Sensible defaults
        self.classification_params.setdefault('max_depth', 4)
        self.classification_params.setdefault('learning_rate', 0.07)
        self.classification_params.setdefault('n_estimators', 400)
        self.classification_params.setdefault('subsample', 0.9)
        self.classification_params.setdefault('colsample_bytree', 0.8)
        self.classification_params.setdefault('min_child_weight', 3)
        self.classification_params.setdefault('gamma', 0.0)
        self.classification_params.setdefault('use_label_encoder', False)
        self.classification_params.setdefault(
            'eval_metric',
            self.training_config.get('eval_metric_classification', 'auc')
        )

        self.regression_params.setdefault('max_depth', 5)
        self.regression_params.setdefault('learning_rate', 0.05)
        self.regression_params.setdefault('n_estimators', 600)
        self.regression_params.setdefault('subsample', 0.9)
        self.regression_params.setdefault('colsample_bytree', 0.75)
        self.regression_params.setdefault('min_child_weight', 2)
        self.regression_params.setdefault('gamma', 0.0)
        self.regression_params.setdefault('objective', 'reg:squarederror')

        self.training_config.setdefault('test_size', 0.2)
        self.training_config.setdefault('early_stopping_rounds', 50)
        self.training_config.setdefault('random_state', 42)
        self.training_config.setdefault('eval_metric_classification', 'auc')
        self.training_config.setdefault('eval_metric_regression', 'rmse')

        self.thresholds.setdefault('probability_long', 0.55)
        self.thresholds.setdefault('probability_short', 0.45)
        self.thresholds.setdefault('min_expected_return', 0.0)
        self.thresholds.setdefault('neutral_absolute_return', 0.0)

        # Internal state
        self.feature_names: List[str] = []
        self.regressor: Optional[xgb.XGBRegressor] = None
        self.classifier: Optional[xgb.XGBClassifier] = None

    @staticmethod
    def _split_train_validation(
        X: pd.DataFrame,
        y_reg: pd.Series,
        y_cls: pd.Series,
        test_size: float
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series, pd.Series], Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]]]:
        if test_size <= 0 or test_size >= 1:
            return (X, y_reg, y_cls), None

        split_idx = int(len(X) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(X):
            return (X, y_reg, y_cls), None

        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_reg_train = y_reg.iloc[:split_idx]
        y_reg_val = y_reg.iloc[split_idx:]
        y_cls_train = y_cls.iloc[:split_idx]
        y_cls_val = y_cls.iloc[split_idx:]

        return (X_train, y_reg_train, y_cls_train), (X_val, y_reg_val, y_cls_val)

    def _prepare_targets(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        if 'target_return' not in df.columns:
            raise ValueError("target_return column is required for LookaheadModel")

        target_return = df['target_return'].astype(np.float32)

        if 'target_up' in df.columns:
            target_up = df['target_up'].astype(int)
        elif 'target_label' in df.columns:
            target_up = (df['target_label'] > 0).astype(int)
        elif 'target_direction' in df.columns:
            target_up = df['target_direction'].astype(int)
        else:
            target_up = (target_return > 0).astype(int)

        return target_return, target_up

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        target_columns = [col for col in df.columns if col.startswith('target_')]
        feature_frame = df.drop(columns=target_columns, errors='ignore')
        feature_frame = feature_frame.select_dtypes(include=[np.number])
        feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        self.feature_names = list(feature_frame.columns)
        return feature_frame.astype(np.float32)

    def train(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Cannot train LookaheadModel on empty DataFrame")

        y_reg, y_cls = self._prepare_targets(df)
        X = self._select_features(df)

        (X_train, y_reg_train, y_cls_train), val_split = self._split_train_validation(
            X,
            y_reg,
            y_cls,
            self.training_config.get('test_size', 0.2)
        )

        evals_result = {}

        self.regressor = xgb.XGBRegressor(**self.regression_params)
        if val_split is not None:
            X_val, y_reg_val, _ = val_split
            self.regressor.fit(
                X_train,
                y_reg_train,
                eval_set=[(X_val, y_reg_val)],
                early_stopping_rounds=self.training_config.get('early_stopping_rounds'),
                verbose=False,
                evals_result=evals_result,
            )
        else:
            self.regressor.fit(X_train, y_reg_train, verbose=False)

        self.classifier = xgb.XGBClassifier(**self.classification_params)
        if val_split is not None:
            X_val, _, y_cls_val = val_split
            self.classifier.fit(
                X_train,
                y_cls_train,
                eval_set=[(X_val, y_cls_val)],
                early_stopping_rounds=self.training_config.get('early_stopping_rounds'),
                verbose=False,
            )
        else:
            self.classifier.fit(X_train, y_cls_train, verbose=False)

        logger.info(
            "LookaheadModel training complete: %s features, %s samples",
            len(self.feature_names),
            len(X),
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.regressor is None or self.classifier is None:
            raise ValueError("Model has not been trained")
        X = df[self.feature_names].astype(np.float32)
        expected_return = self.regressor.predict(X)
        prob_up = self.classifier.predict_proba(X)[:, 1]
        expected_value = prob_up * expected_return
        signal = self._generate_signal(prob_up, expected_return)
        predictions = pd.DataFrame(
            {
                'expected_return': expected_return,
                'probability_up': prob_up,
                'expected_value': expected_value,
                'signal': signal,
            },
            index=df.index,
        )
        return predictions

    def _generate_signal(self, probability_up: np.ndarray, expected_return: np.ndarray) -> np.ndarray:
        prob_long = self.thresholds.get('probability_long', 0.55)
        prob_short = self.thresholds.get('probability_short', 0.45)
        min_ret = self.thresholds.get('min_expected_return', 0.0)
        neutral_band = self.thresholds.get('neutral_absolute_return', 0.0)

        signal = np.zeros_like(expected_return, dtype=int)

        long_mask = (probability_up >= prob_long) & (expected_return >= max(min_ret, neutral_band))
        short_mask = (probability_up <= prob_short) & (expected_return <= -max(min_ret, neutral_band))
        neutral_mask = np.abs(expected_return) < neutral_band

        signal[long_mask] = 1
        signal[short_mask] = -1
        signal[neutral_mask] = 0

        return signal

    def evaluate_holdout(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {}
        y_reg, y_cls = self._prepare_targets(df)
        preds = self.predict(df)
        scores: Dict[str, float] = {}
        scores['mse'] = float(np.mean((preds['expected_return'] - y_reg) ** 2))
        scores['mae'] = float(np.mean(np.abs(preds['expected_return'] - y_reg)))
        scores['directional_accuracy'] = float(np.mean((preds['expected_return'] > 0) == (y_reg > 0)))
        try:
            scores['roc_auc'] = float(metrics.roc_auc_score(y_cls, preds['probability_up']))
        except Exception:  # pylint: disable=broad-except
            scores['roc_auc'] = float('nan')
        return scores

    def get_feature_importance(self) -> pd.DataFrame:
        if self.regressor is None or self.classifier is None:
            raise ValueError("Model not trained")
        reg_imp = self.regressor.feature_importances_
        cls_imp = self.classifier.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_regression': reg_imp,
            'importance_classification': cls_imp,
        })
        df['importance_mean'] = (df['importance_regression'] + df['importance_classification']) / 2
        df = df.sort_values('importance_mean', ascending=False)
        return df

    def save(self, filepath: str) -> None:
        if self.regressor is None or self.classifier is None:
            raise ValueError("Cannot save an untrained model")
        payload = {
            'config': self.config,
            'feature_names': self.feature_names,
            'regressor': self.regressor,
            'classifier': self.classifier,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(payload, f)
        logger.info("Lookahead model saved to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> 'LookaheadModel':
        with open(filepath, 'rb') as f:
            payload = pickle.load(f)
        model = cls(config=payload.get('config'))
        model.feature_names = payload.get('feature_names', [])
        model.regressor = payload.get('regressor')
        model.classifier = payload.get('classifier')
        logger.info("Loaded lookahead model from %s", filepath)
        return model
