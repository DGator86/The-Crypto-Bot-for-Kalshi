"""
XGBoost model implementation.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class XGBModel:
    """
    XGBoost-based trading model.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            params: XGBoost parameters
        """
        if params is None:
            params = {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'objective': 'reg:squarederror',
                'random_state': 42,
            }
        
        self.params = params
        self.model = None
        self.feature_names = None
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target',
        feature_cols: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names (if None, use all except target)
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        if feature_cols is None:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        self.feature_names = feature_cols
        
        return X, y
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: Optional[int] = 10
    ):
        """
        Train the XGBoost model.
        
        Args:
            X: Features DataFrame
            y: Target Series
            eval_set: Optional evaluation set (X_val, y_val)
            early_stopping_rounds: Early stopping rounds
        """
        logger.info(f"Training XGBoost model with {len(X)} samples and {len(X.columns)} features")
        
        self.model = xgb.XGBRegressor(**self.params)
        
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_set_formatted = [(X_val, y_val)]
            self.model.fit(
                X, y,
                eval_set=eval_set_formatted,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        else:
            self.model.fit(X, y)
        
        logger.info("Model training completed")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
        
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance = self.model.feature_importances_
        
        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df_importance
    
    def save(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_data = {
            'model': self.model,
            'params': self.params,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'XGBModel':
        """
        Load model from file.
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Loaded XGBModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(params=model_data['params'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


def train_model(
    df: pd.DataFrame,
    target_col: str = 'target',
    test_size: float = 0.2,
    params: Optional[Dict[str, Any]] = None
) -> XGBModel:
    """
    Train XGBoost model with train/test split.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        test_size: Proportion of test set
        params: Model parameters
    
    Returns:
        Trained XGBModel instance
    """
    model = XGBModel(params=params)
    X, y = model.prepare_data(df, target_col=target_col)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Train with validation
    model.train(X_train, y_train, eval_set=(X_test, y_test))
    
    # Evaluate
    train_score = model.model.score(X_train, y_train)
    test_score = model.model.score(X_test, y_test)
    
    logger.info(f"Train R²: {train_score:.4f}, Test R²: {test_score:.4f}")
    
    return model


def predict(model: XGBModel, df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using trained model.
    
    Args:
        model: Trained XGBModel instance
        df: DataFrame with features
    
    Returns:
        Array of predictions
    """
    X = df[model.feature_names]
    return model.predict(X)
