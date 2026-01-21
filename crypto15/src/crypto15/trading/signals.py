"""
Signal generation module.

Converts model predictions into actionable trading signals with risk management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SignalAction(Enum):
    """Trading signal actions."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"
    KILL = "KILL"  # Emergency stop


@dataclass
class TradingSignal:
    """Complete trading signal with all metadata."""

    timestamp: datetime
    symbol: str
    action: SignalAction
    probability_up: float
    expected_return: float
    confidence: float
    position_size: float  # Fraction of portfolio (0.0 to 1.0)
    current_price: float
    predicted_price: float
    volatility_z: float
    regime: str
    kill_switch_active: bool
    metadata: Dict[str, Any]

    @property
    def is_trade(self) -> bool:
        """Returns True if this signal suggests entering a trade."""
        return self.action in (SignalAction.LONG, SignalAction.SHORT) and not self.kill_switch_active

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "action": self.action.value,
            "probability_up": round(self.probability_up, 4),
            "expected_return": round(self.expected_return, 6),
            "confidence": round(self.confidence, 4),
            "position_size": round(self.position_size, 4),
            "current_price": round(self.current_price, 2),
            "predicted_price": round(self.predicted_price, 2),
            "volatility_z": round(self.volatility_z, 2),
            "regime": self.regime,
            "kill_switch_active": self.kill_switch_active,
            "metadata": self.metadata,
        }


@dataclass
class SignalConfig:
    """Configuration for signal generation."""

    # Thresholds
    probability_long: float = 0.55
    probability_short: float = 0.45
    min_expected_return: float = 0.0003
    neutral_band: float = 0.0002

    # Kill switch parameters
    max_volatility_z: float = 3.5
    min_probability_gap: float = 0.025  # Minimum |prob - 0.5|

    # Position sizing
    max_position_size: float = 0.20
    confidence_scaling: float = 5.0  # position = min(max_pos, confidence * scaling)

    # Regime-specific thresholds
    regime_thresholds: Dict[str, Dict[str, float]] = None

    def __post_init__(self):
        if self.regime_thresholds is None:
            self.regime_thresholds = {
                "normal": {
                    "probability_long": self.probability_long,
                    "probability_short": self.probability_short,
                },
                "trend": {
                    "probability_long": self.probability_long + 0.05,
                    "probability_short": self.probability_short - 0.05,
                },
                "high_vol": {
                    "probability_long": self.probability_long + 0.08,
                    "probability_short": self.probability_short - 0.08,
                },
                "chop": {
                    "probability_long": self.probability_long - 0.02,
                    "probability_short": self.probability_short + 0.02,
                },
            }


class SignalGenerator:
    """
    Generates trading signals from model predictions.

    Includes:
    - Threshold-based signal generation
    - Regime-aware thresholds
    - Kill switch for extreme conditions
    - Confidence-based position sizing
    - Volatility normalization

    Example:
        generator = SignalGenerator(SignalConfig(
            probability_long=0.55,
            max_position_size=0.15
        ))

        signal = generator.generate(
            predictions_df=model.predict(features),
            features_df=features,
            symbol="BTC/USDT"
        )

        if signal.is_trade:
            execute_trade(signal)
    """

    def __init__(self, config: Optional[SignalConfig] = None):
        self.config = config or SignalConfig()
        self._volatility_history: list = []
        self._volatility_window = 96  # 24 hours of 15m bars

    def _detect_regime(self, features_df: pd.DataFrame) -> str:
        """Detect current market regime from features."""
        if features_df.empty:
            return "normal"

        latest = features_df.iloc[-1]

        # Check for volatility regime indicators
        vol_cols = [c for c in features_df.columns if "vol" in c.lower() or "std" in c.lower()]
        trend_cols = [c for c in features_df.columns if "trend" in c.lower() or "momentum" in c.lower()]

        # Simple regime detection
        if vol_cols:
            vol_z = latest.get(vol_cols[0], 0)
            if isinstance(vol_z, (int, float)) and vol_z > 2.0:
                return "high_vol"

        if trend_cols:
            trend = latest.get(trend_cols[0], 0)
            if isinstance(trend, (int, float)) and abs(trend) > 1.5:
                return "trend"

        # Check for choppy market (low trend, normal vol)
        if "regime" in latest:
            return str(latest["regime"])

        return "normal"

    def _calculate_volatility_z(self, features_df: pd.DataFrame) -> float:
        """Calculate current volatility z-score."""
        if features_df.empty:
            return 0.0

        # Look for volatility columns
        vol_cols = [c for c in features_df.columns if "realized_vol" in c.lower() or "volatility" in c.lower()]

        if not vol_cols:
            # Fallback: calculate from returns if available
            ret_cols = [c for c in features_df.columns if "return" in c.lower() and "target" not in c.lower()]
            if ret_cols:
                recent_vol = features_df[ret_cols[0]].tail(16).std()
                long_vol = features_df[ret_cols[0]].std()
                if long_vol > 0:
                    return (recent_vol - long_vol) / long_vol
            return 0.0

        # Use existing volatility column
        vol_series = features_df[vol_cols[0]].dropna()
        if len(vol_series) < 2:
            return 0.0

        current_vol = vol_series.iloc[-1]
        mean_vol = vol_series.mean()
        std_vol = vol_series.std()

        if std_vol > 0:
            return (current_vol - mean_vol) / std_vol

        return 0.0

    def _calculate_confidence(self, probability_up: float, expected_return: float) -> float:
        """Calculate confidence score from predictions."""
        # Confidence based on distance from 0.5
        prob_confidence = abs(probability_up - 0.5) * 2  # 0 to 1 scale

        # Adjust for expected return magnitude
        return_confidence = min(abs(expected_return) / 0.005, 1.0)  # Cap at 0.5% return

        # Combine
        return (prob_confidence + return_confidence) / 2

    def _calculate_position_size(self, confidence: float, volatility_z: float) -> float:
        """Calculate position size based on confidence and volatility."""
        base_size = min(
            self.config.max_position_size,
            confidence * self.config.confidence_scaling
        )

        # Reduce size in high volatility
        if volatility_z > 1.5:
            vol_factor = max(0.3, 1 - (volatility_z - 1.5) * 0.2)
            base_size *= vol_factor

        return max(0.0, min(base_size, self.config.max_position_size))

    def _check_kill_switch(
        self,
        probability_up: float,
        volatility_z: float,
    ) -> bool:
        """Check if kill switch should be activated."""
        # Extreme volatility
        if volatility_z > self.config.max_volatility_z:
            logger.warning("Kill switch: extreme volatility (z=%.2f)", volatility_z)
            return True

        # Probability too close to 0.5
        if abs(probability_up - 0.5) < self.config.min_probability_gap:
            logger.debug("Kill switch: low confidence (prob=%.3f)", probability_up)
            return True

        return False

    def generate(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> TradingSignal:
        """
        Generate trading signal from predictions.

        Args:
            predictions_df: DataFrame with model predictions (expected_return, probability_up).
            features_df: DataFrame with features used for regime detection.
            symbol: Trading symbol.
            current_price: Current price (auto-detected if not provided).

        Returns:
            TradingSignal with action and metadata.
        """
        if predictions_df.empty or features_df.empty:
            raise ValueError("Empty predictions or features DataFrame")

        # Get latest prediction
        latest_pred = predictions_df.iloc[-1]
        latest_features = features_df.iloc[-1]

        probability_up = float(latest_pred.get("probability_up", 0.5))
        expected_return = float(latest_pred.get("expected_return", 0.0))

        # Detect regime
        regime = self._detect_regime(features_df)

        # Get regime-specific thresholds
        regime_thresholds = self.config.regime_thresholds.get(regime, {})
        prob_long = regime_thresholds.get("probability_long", self.config.probability_long)
        prob_short = regime_thresholds.get("probability_short", self.config.probability_short)

        # Calculate volatility z-score
        volatility_z = self._calculate_volatility_z(features_df)

        # Check kill switch
        kill_switch = self._check_kill_switch(probability_up, volatility_z)

        # Determine action
        if kill_switch:
            action = SignalAction.KILL
        elif probability_up >= prob_long and expected_return >= self.config.min_expected_return:
            action = SignalAction.LONG
        elif probability_up <= prob_short and expected_return <= -self.config.min_expected_return:
            action = SignalAction.SHORT
        else:
            action = SignalAction.FLAT

        # Calculate confidence and position size
        confidence = self._calculate_confidence(probability_up, expected_return)
        position_size = 0.0 if kill_switch else self._calculate_position_size(confidence, volatility_z)

        # Get current price
        if current_price is None:
            price_cols = [c for c in features_df.columns if "close" in c.lower() or "price" in c.lower()]
            if price_cols:
                current_price = float(latest_features[price_cols[0]])
            else:
                current_price = 0.0

        predicted_price = current_price * (1 + expected_return)

        signal = TradingSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            action=action,
            probability_up=probability_up,
            expected_return=expected_return,
            confidence=confidence,
            position_size=position_size,
            current_price=current_price,
            predicted_price=predicted_price,
            volatility_z=volatility_z,
            regime=regime,
            kill_switch_active=kill_switch,
            metadata={
                "threshold_long": prob_long,
                "threshold_short": prob_short,
                "model_signal": int(latest_pred.get("signal", 0)),
            }
        )

        logger.info(
            "Signal generated: %s @ %.2f (prob=%.2f%%, exp_ret=%.4f%%, conf=%.2f, size=%.1f%%)",
            action.value,
            current_price,
            probability_up * 100,
            expected_return * 100,
            confidence,
            position_size * 100,
        )

        return signal

    def generate_batch(
        self,
        predictions_df: pd.DataFrame,
        features_df: pd.DataFrame,
        symbol: str,
    ) -> pd.DataFrame:
        """
        Generate signals for all rows in predictions DataFrame.

        Useful for backtesting.

        Returns:
            DataFrame with signal for each row.
        """
        signals = []

        for idx in predictions_df.index:
            try:
                pred_row = predictions_df.loc[[idx]]
                feat_row = features_df.loc[:idx]  # All features up to this point

                signal = self.generate(pred_row, feat_row, symbol)
                signals.append({
                    "timestamp": idx,
                    "action": signal.action.value,
                    "probability_up": signal.probability_up,
                    "expected_return": signal.expected_return,
                    "confidence": signal.confidence,
                    "position_size": signal.position_size,
                    "kill_switch": signal.kill_switch_active,
                    "regime": signal.regime,
                })
            except Exception as e:
                logger.warning("Failed to generate signal for %s: %s", idx, e)

        return pd.DataFrame(signals)
