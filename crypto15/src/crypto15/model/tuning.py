"""
Hyperparameter tuning module using Optuna.

Provides automated hyperparameter optimization for the LookaheadModel
using Bayesian optimization with pruning for efficiency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

import xgboost as xgb
from sklearn import metrics

logger = logging.getLogger(__name__)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    # Tuning parameters
    n_trials: int = 50
    timeout: Optional[int] = 3600  # 1 hour timeout
    n_cv_splits: int = 5
    purge_bars: int = 2
    random_state: int = 42

    # Optimization target
    optimize_metric: str = "roc_auc"  # roc_auc, sharpe, f1, accuracy
    direction: str = "maximize"

    # Search space bounds
    classification_space: Dict[str, Tuple[Any, ...]] = field(default_factory=lambda: {
        "max_depth": (2, 8),
        "learning_rate": (0.01, 0.3),
        "n_estimators": (100, 1000),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 1.0),
        "reg_alpha": (0.0, 1.0),
        "reg_lambda": (0.0, 2.0),
    })

    regression_space: Dict[str, Tuple[Any, ...]] = field(default_factory=lambda: {
        "max_depth": (2, 10),
        "learning_rate": (0.01, 0.2),
        "n_estimators": (200, 1200),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 1.0),
        "reg_alpha": (0.0, 1.0),
        "reg_lambda": (0.0, 2.0),
    })

    threshold_space: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "probability_long": (0.52, 0.70),
        "probability_short": (0.30, 0.48),
        "min_expected_return": (0.0, 0.002),
    })


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_params: Dict[str, Any]
    best_score: float
    study_stats: Dict[str, Any]
    all_trials: List[Dict[str, Any]]
    feature_names: List[str]

    def to_model_config(self) -> Dict[str, Any]:
        """Convert tuning results to model config format."""
        classification_params = {}
        regression_params = {}
        thresholds = {}

        for key, value in self.best_params.items():
            if key.startswith("cls_"):
                classification_params[key[4:]] = value
            elif key.startswith("reg_"):
                regression_params[key[4:]] = value
            elif key.startswith("thr_"):
                thresholds[key[4:]] = value
            elif key in ("probability_long", "probability_short", "min_expected_return"):
                thresholds[key] = value

        return {
            "classification_params": classification_params,
            "regression_params": regression_params,
            "thresholds": thresholds,
        }


class PurgedTimeSeriesSplit:
    """
    Time series cross-validator with purging to prevent data leakage.

    Removes samples between train and test sets to prevent lookahead bias.
    """

    def __init__(self, n_splits: int = 5, purge_bars: int = 2):
        self.n_splits = n_splits
        self.purge_bars = purge_bars

    def split(self, X: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purging."""
        n_samples = len(X)
        fold_size = n_samples // (self.n_splits + 1)

        splits = []
        for i in range(self.n_splits):
            # Training set: everything before this fold
            train_end = fold_size * (i + 1)
            train_end = max(0, train_end - self.purge_bars)  # Purge

            # Test set: this fold
            test_start = fold_size * (i + 1)
            test_end = fold_size * (i + 2)

            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


class HyperparameterTuner:
    """
    Hyperparameter tuner for the crypto trading model.

    Uses Optuna for Bayesian optimization with:
    - Time-series cross-validation with purging
    - Early stopping via pruning
    - Multiple optimization targets (ROC-AUC, Sharpe, F1)

    Example:
        tuner = HyperparameterTuner(TuningConfig(n_trials=100))
        result = tuner.tune(df_features)

        # Apply results to model
        model_config = result.to_model_config()
        model = LookaheadModel(config=model_config)
    """

    def __init__(self, config: Optional[TuningConfig] = None):
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install it with: pip install optuna"
            )

        self.config = config or TuningConfig()
        self._feature_names: List[str] = []
        self._target_col_reg = "target_return"
        self._target_col_cls = "target_up"

    def _prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features and targets from dataframe."""
        # Identify target columns
        if self._target_col_reg not in df.columns:
            raise ValueError(f"Missing target column: {self._target_col_reg}")

        # Extract targets
        y_reg = df[self._target_col_reg].astype(np.float32)

        if self._target_col_cls in df.columns:
            y_cls = df[self._target_col_cls].astype(int)
        else:
            y_cls = (y_reg > 0).astype(int)

        # Extract features
        target_columns = [col for col in df.columns if col.startswith("target_")]
        X = df.drop(columns=target_columns, errors="ignore")
        X = X.select_dtypes(include=[np.number])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        self._feature_names = list(X.columns)

        return X.astype(np.float32), y_reg, y_cls

    def _create_objective(
        self,
        X: pd.DataFrame,
        y_reg: pd.Series,
        y_cls: pd.Series,
    ) -> Callable[[optuna.Trial], float]:
        """Create Optuna objective function."""

        cv = PurgedTimeSeriesSplit(
            n_splits=self.config.n_cv_splits,
            purge_bars=self.config.purge_bars,
        )
        splits = cv.split(X)

        def objective(trial: optuna.Trial) -> float:
            # Sample classification parameters
            cls_params = {
                "max_depth": trial.suggest_int(
                    "cls_max_depth",
                    *self.config.classification_space["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "cls_learning_rate",
                    *self.config.classification_space["learning_rate"],
                    log=True
                ),
                "n_estimators": trial.suggest_int(
                    "cls_n_estimators",
                    *self.config.classification_space["n_estimators"]
                ),
                "subsample": trial.suggest_float(
                    "cls_subsample",
                    *self.config.classification_space["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "cls_colsample_bytree",
                    *self.config.classification_space["colsample_bytree"]
                ),
                "min_child_weight": trial.suggest_int(
                    "cls_min_child_weight",
                    *self.config.classification_space["min_child_weight"]
                ),
                "gamma": trial.suggest_float(
                    "cls_gamma",
                    *self.config.classification_space["gamma"]
                ),
                "reg_alpha": trial.suggest_float(
                    "cls_reg_alpha",
                    *self.config.classification_space["reg_alpha"]
                ),
                "reg_lambda": trial.suggest_float(
                    "cls_reg_lambda",
                    *self.config.classification_space["reg_lambda"]
                ),
                "use_label_encoder": False,
                "eval_metric": "auc",
                "random_state": self.config.random_state,
            }

            # Sample regression parameters
            reg_params = {
                "max_depth": trial.suggest_int(
                    "reg_max_depth",
                    *self.config.regression_space["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "reg_learning_rate",
                    *self.config.regression_space["learning_rate"],
                    log=True
                ),
                "n_estimators": trial.suggest_int(
                    "reg_n_estimators",
                    *self.config.regression_space["n_estimators"]
                ),
                "subsample": trial.suggest_float(
                    "reg_subsample",
                    *self.config.regression_space["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "reg_colsample_bytree",
                    *self.config.regression_space["colsample_bytree"]
                ),
                "min_child_weight": trial.suggest_int(
                    "reg_min_child_weight",
                    *self.config.regression_space["min_child_weight"]
                ),
                "gamma": trial.suggest_float(
                    "reg_gamma",
                    *self.config.regression_space["gamma"]
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_reg_alpha",
                    *self.config.regression_space["reg_alpha"]
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_reg_lambda",
                    *self.config.regression_space["reg_lambda"]
                ),
                "objective": "reg:squarederror",
                "random_state": self.config.random_state,
            }

            # Sample threshold parameters
            prob_long = trial.suggest_float(
                "probability_long",
                *self.config.threshold_space["probability_long"]
            )
            prob_short = trial.suggest_float(
                "probability_short",
                *self.config.threshold_space["probability_short"]
            )
            min_ret = trial.suggest_float(
                "min_expected_return",
                *self.config.threshold_space["min_expected_return"]
            )

            # Cross-validation
            fold_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_cls_train = y_cls.iloc[train_idx]
                y_cls_test = y_cls.iloc[test_idx]
                y_reg_train = y_reg.iloc[train_idx]
                y_reg_test = y_reg.iloc[test_idx]

                # Train classifier
                classifier = xgb.XGBClassifier(**cls_params)
                classifier.fit(X_train, y_cls_train, verbose=False)

                # Train regressor
                regressor = xgb.XGBRegressor(**reg_params)
                regressor.fit(X_train, y_reg_train, verbose=False)

                # Predict
                prob_up = classifier.predict_proba(X_test)[:, 1]
                expected_return = regressor.predict(X_test)

                # Generate signals
                signals = np.zeros(len(prob_up))
                signals[(prob_up >= prob_long) & (expected_return >= min_ret)] = 1
                signals[(prob_up <= prob_short) & (expected_return <= -min_ret)] = -1

                # Calculate score based on metric
                if self.config.optimize_metric == "roc_auc":
                    try:
                        score = metrics.roc_auc_score(y_cls_test, prob_up)
                    except ValueError:
                        score = 0.5
                elif self.config.optimize_metric == "f1":
                    pred_cls = (prob_up >= 0.5).astype(int)
                    score = metrics.f1_score(y_cls_test, pred_cls, zero_division=0)
                elif self.config.optimize_metric == "accuracy":
                    pred_cls = (prob_up >= 0.5).astype(int)
                    score = metrics.accuracy_score(y_cls_test, pred_cls)
                elif self.config.optimize_metric == "sharpe":
                    # Simple Sharpe approximation
                    strategy_returns = signals[:-1] * y_reg_test.values[1:]
                    if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                        score = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 * 96)
                    else:
                        score = 0.0
                elif self.config.optimize_metric == "expectancy":
                    # Expected profit per trade
                    strategy_returns = signals[:-1] * y_reg_test.values[1:]
                    n_trades = np.sum(np.abs(signals) > 0)
                    if n_trades > 0:
                        score = np.sum(strategy_returns) / n_trades
                    else:
                        score = 0.0
                else:
                    score = metrics.roc_auc_score(y_cls_test, prob_up)

                fold_scores.append(score)

                # Report intermediate value for pruning
                trial.report(np.mean(fold_scores), fold_idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(fold_scores)

        return objective

    def tune(
        self,
        df: pd.DataFrame,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> TuningResult:
        """
        Run hyperparameter tuning.

        Args:
            df: DataFrame with features and targets.
            study_name: Optional name for the study (for persistence).
            storage: Optional storage URL (e.g., "sqlite:///tuning.db").

        Returns:
            TuningResult with best parameters and scores.
        """
        logger.info("Starting hyperparameter tuning with %d trials", self.config.n_trials)

        X, y_reg, y_cls = self._prepare_data(df)
        logger.info("Prepared %d samples with %d features", len(X), len(self._feature_names))

        # Create study
        sampler = TPESampler(seed=self.config.random_state)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        if study_name and storage:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True,
            )
        else:
            study = optuna.create_study(
                direction=self.config.direction,
                sampler=sampler,
                pruner=pruner,
            )

        # Create objective
        objective = self._create_objective(X, y_reg, y_cls)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True,
        )

        # Collect results
        all_trials = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                all_trials.append({
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                })

        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            study_stats={
                "n_trials": len(study.trials),
                "n_complete": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                "optimization_metric": self.config.optimize_metric,
            },
            all_trials=all_trials,
            feature_names=self._feature_names,
        )

        logger.info(
            "Tuning complete. Best %s: %.4f",
            self.config.optimize_metric,
            study.best_value
        )

        return result

    def tune_thresholds_only(
        self,
        df: pd.DataFrame,
        classifier: xgb.XGBClassifier,
        regressor: xgb.XGBRegressor,
        n_trials: int = 100,
    ) -> Dict[str, float]:
        """
        Tune only threshold parameters with fixed models.

        Faster than full tuning when you just want to optimize thresholds.

        Args:
            df: DataFrame with features and targets.
            classifier: Trained classifier.
            regressor: Trained regressor.
            n_trials: Number of trials.

        Returns:
            Dict with optimal thresholds.
        """
        X, y_reg, y_cls = self._prepare_data(df)

        # Get predictions
        prob_up = classifier.predict_proba(X)[:, 1]
        expected_return = regressor.predict(X)

        def objective(trial: optuna.Trial) -> float:
            prob_long = trial.suggest_float("probability_long", 0.52, 0.70)
            prob_short = trial.suggest_float("probability_short", 0.30, 0.48)
            min_ret = trial.suggest_float("min_expected_return", 0.0, 0.002)

            # Generate signals
            signals = np.zeros(len(prob_up))
            signals[(prob_up >= prob_long) & (expected_return >= min_ret)] = 1
            signals[(prob_up <= prob_short) & (expected_return <= -min_ret)] = -1

            # Calculate expectancy (profit per trade)
            strategy_returns = signals[:-1] * y_reg.values[1:]
            n_trades = np.sum(np.abs(signals) > 0)

            if n_trades < 10:  # Require minimum trades
                return -1.0

            # Return expectancy adjusted for trade frequency
            expectancy = np.sum(strategy_returns) / n_trades
            trade_freq = n_trades / len(signals)

            # Balance expectancy with reasonable trade frequency
            score = expectancy * min(trade_freq * 10, 1.0)

            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        return {
            "probability_long": study.best_params["probability_long"],
            "probability_short": study.best_params["probability_short"],
            "min_expected_return": study.best_params["min_expected_return"],
        }


def quick_tune(
    df: pd.DataFrame,
    n_trials: int = 30,
    optimize_metric: str = "roc_auc",
) -> TuningResult:
    """
    Quick hyperparameter tuning with sensible defaults.

    Args:
        df: DataFrame with features and targets.
        n_trials: Number of trials (default: 30 for quick results).
        optimize_metric: Metric to optimize.

    Returns:
        TuningResult with best parameters.
    """
    config = TuningConfig(
        n_trials=n_trials,
        n_cv_splits=3,  # Fewer splits for speed
        optimize_metric=optimize_metric,
    )
    tuner = HyperparameterTuner(config)
    return tuner.tune(df)
