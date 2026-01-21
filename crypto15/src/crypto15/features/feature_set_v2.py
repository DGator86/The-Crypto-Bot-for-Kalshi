"""Advanced feature engineering for 15-minute look-ahead modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, List, Set

import logging

logger = logging.getLogger(__name__)


class FeatureSetV2:
    """Richer technical, cross-asset, and temporal features for the primary symbol."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.lookback_bars: List[int] = sorted(set(self.config.get('lookback_bars', [1, 2, 4, 8, 16, 32, 64])))
        self.volatility_windows: List[int] = sorted(set(self.config.get('volatility_windows', [8, 16, 32, 96])))
        self.volume_windows: List[int] = sorted(set(self.config.get('volume_windows', [4, 16, 64])))
        self.context_corr_windows: List[int] = sorted(set(self.config.get('context_correlation_windows', [8, 32, 96])))
        self.include_intraday: bool = bool(self.config.get('include_intraday', True))
        self.include_regime: bool = bool(self.config.get('include_regime', True))
        self.include_interactions: bool = bool(self.config.get('include_interactions', True))

        target_cfg = self.config.get('target', {})
        self.target_horizon: int = int(target_cfg.get('horizon_bars', 1))
        self.target_threshold: float = float(target_cfg.get('classification_threshold', 0.0))
        self.target_neutral: float = float(target_cfg.get('neutral_band', 0.0))

    @staticmethod
    def _context_prefixes(columns: Iterable[str]) -> Set[str]:
        prefixes: Set[str] = set()
        for col in columns:
            if col.startswith('ctx_') and '_' in col:
                prefix = col.rsplit('_', 1)[0]
                prefixes.add(prefix)
        return prefixes

    @staticmethod
    def _safe_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / (std + 1e-9)

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def _add_primary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'primary_close' not in df.columns:
            raise ValueError("primary_close column is required for FeatureSetV2")

        close = df['primary_close']
        high = df.get('primary_high', close)
        low = df.get('primary_low', close)
        volume = df.get('primary_volume')
        log_price = np.log(close)

        df['primary_return_1'] = close.pct_change().replace([np.inf, -np.inf], np.nan)
        df['primary_log_return_1'] = log_price.diff()

        for window in self.lookback_bars:
            df[f'primary_return_{window}'] = close.pct_change(window)
            df[f'primary_log_return_{window}'] = log_price.diff(window)
            df[f'primary_ma_{window}'] = close.rolling(window).mean()
            df[f'primary_ema_{window}'] = close.ewm(span=window, adjust=False).mean()
            df[f'primary_std_{window}'] = close.rolling(window).std()
            df[f'primary_zscore_{window}'] = self._safe_zscore(close, window)
            df[f'primary_max_{window}'] = close.rolling(window).max()
            df[f'primary_min_{window}'] = close.rolling(window).min()
            df[f'primary_range_{window}'] = df[f'primary_max_{window}'] - df[f'primary_min_{window}']
            df[f'primary_pct_range_{window}'] = df[f'primary_range_{window}'] / (df[f'primary_min_{window}'] + 1e-9)

        # Price momentum / oscillators
        df['primary_rsi_14'] = self._compute_rsi(close, 14)
        ema_fast = close.ewm(span=12, adjust=False).mean()
        ema_slow = close.ewm(span=26, adjust=False).mean()
        df['primary_macd'] = ema_fast - ema_slow
        df['primary_macd_signal'] = df['primary_macd'].ewm(span=9, adjust=False).mean()
        df['primary_macd_hist'] = df['primary_macd'] - df['primary_macd_signal']

        # Volatility / ATR
        true_range = pd.concat([
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        for window in self.volatility_windows:
            df[f'primary_realized_vol_{window}'] = df['primary_return_1'].rolling(window).std()
            df[f'primary_atr_{window}'] = true_range.rolling(window).mean()

        if volume is not None:
            df['primary_volume_change'] = volume.pct_change()
            for window in self.volume_windows:
                df[f'primary_volume_ma_{window}'] = volume.rolling(window).mean()
                df[f'primary_volume_z_{window}'] = self._safe_zscore(volume, window)
                df[f'primary_volume_ratio_{window}'] = volume / (df[f'primary_volume_ma_{window}'] + 1e-9)

        return df

    def _add_intraday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.include_intraday:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex; skipping intraday features")
            return df

        minutes_per_day = 24 * 60
        minute_of_day = df.index.hour * 60 + df.index.minute
        df['intra_sin_daily'] = np.sin(2 * np.pi * minute_of_day / minutes_per_day)
        df['intra_cos_daily'] = np.cos(2 * np.pi * minute_of_day / minutes_per_day)

        day_of_week = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        df['is_weekend'] = (day_of_week >= 5).astype(int)

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.include_regime:
            return df
        vol_window = max(self.volatility_windows) if self.volatility_windows else 32
        long_window = vol_window * 4
        realized = df['primary_return_1'].rolling(vol_window).std()
        long_realized = df['primary_return_1'].rolling(long_window).std()
        df['primary_vol_regime'] = (realized > long_realized).astype(int)
        df['primary_vol_ratio'] = realized / (long_realized + 1e-9)
        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        context_prefixes = self._context_prefixes(df.columns)
        if not context_prefixes:
            return df

        primary_close = df['primary_close']
        primary_returns = df['primary_return_1']

        for prefix in context_prefixes:
            ctx_close_col = f'{prefix}_close'
            if ctx_close_col not in df.columns:
                continue
            ctx_close = df[ctx_close_col]
            ctx_returns = ctx_close.pct_change()
            df[f'{prefix}_return_1'] = ctx_returns

            for window in self.context_corr_windows:
                df[f'corr_{prefix}_primary_{window}'] = primary_returns.rolling(window).corr(ctx_returns)
                df[f'spread_{prefix}_{window}'] = primary_close - ctx_close
                ctx_ratio = (primary_close + 1e-9) / (ctx_close + 1e-9)
                df[f'ratio_{prefix}_{window}'] = ctx_ratio.rolling(window).mean()

            if self.include_interactions:
                df[f'interaction_{prefix}'] = primary_returns * ctx_returns

        return df

    def _add_derivatives_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'primary_funding_rate' in df.columns:
            df['primary_funding_rate_change'] = df['primary_funding_rate'].diff()
        if 'primary_open_interest' in df.columns:
            df['primary_open_interest_change'] = df['primary_open_interest'].diff()
            df['primary_open_interest_z_64'] = self._safe_zscore(df['primary_open_interest'], 64)
        return df

    def _add_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        horizon = max(self.target_horizon, 1)
        future_price = df['primary_close'].shift(-horizon)
        df['target_return'] = future_price / df['primary_close'] - 1

        upper = self.target_threshold
        lower = -self.target_threshold if self.target_threshold else -upper
        neutral_band = self.target_neutral

        df['target_up'] = (df['target_return'] > upper + neutral_band).astype(int)
        df['target_down'] = (df['target_return'] < lower - neutral_band).astype(int)
        df['target_label'] = 0
        df.loc[df['target_return'] > upper + neutral_band, 'target_label'] = 1
        df.loc[df['target_return'] < lower - neutral_band, 'target_label'] = -1
        df['target_direction'] = (df['target_return'] > 0).astype(int)

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create feature-rich dataset for modeling."""
        df = df.copy()

        df = self._add_primary_features(df)
        df = self._add_intraday_features(df)
        df = self._add_regime_features(df)
        df = self._add_context_features(df)
        df = self._add_derivatives_features(df)
        df = self._add_targets(df)

        # Drop rows with missing values introduced by rolling calculations
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df


def create_features(df: pd.DataFrame, config: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Convenience wrapper mirroring FeatureSetV1 signature."""
    feature_set = FeatureSetV2(config=config)
    return feature_set.create_features(df)
