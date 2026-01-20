"""
Feature Set V1 - Initial feature engineering implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FeatureSetV1:
    """
    Feature set version 1 - Basic technical indicators and features.
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize feature set.
        
        Args:
            lookback_periods: List of lookback periods for rolling features
        """
        if lookback_periods is None:
            lookback_periods = [24, 168, 720]  # 1 day, 1 week, 1 month (hourly)
        
        self.lookback_periods = lookback_periods
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price return features."""
        df = df.copy()
        
        # Simple returns
        df['return_1h'] = df['close'].pct_change(1)
        df['return_4h'] = df['close'].pct_change(4)
        df['return_24h'] = df['close'].pct_change(24)
        
        return df
    
    def add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistical features."""
        df = df.copy()
        
        for period in self.lookback_periods:
            # Rolling mean
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            
            # Rolling std
            df[f'std_{period}'] = df['close'].rolling(window=period).std()
            
            # Price relative to MA
            df[f'close_ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df = df.copy()
        
        # Volume change
        df['volume_change'] = df['volume'].pct_change(1)
        
        # Volume rolling average
        df['volume_ma_24'] = df['volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_24']
        
        return df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        df = df.copy()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        df = df.copy()
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added features
        """
        logger.info("Creating features...")
        
        df = df.copy()
        
        # Add all feature groups
        df = self.add_returns(df)
        df = self.add_rolling_features(df)
        df = self.add_volume_features(df)
        df = self.add_momentum_features(df)
        df = self.add_volatility_features(df)
        
        # Drop NaN values created by rolling windows
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Created features. Dropped {initial_rows - len(df)} rows with NaN values")
        
        return df


def create_features(
    df: pd.DataFrame,
    lookback_periods: List[int] = None
) -> pd.DataFrame:
    """
    Convenience function to create features.
    
    Args:
        df: DataFrame with OHLCV data
        lookback_periods: List of lookback periods
    
    Returns:
        DataFrame with features
    """
    feature_set = FeatureSetV1(lookback_periods=lookback_periods)
    return feature_set.create_features(df)
