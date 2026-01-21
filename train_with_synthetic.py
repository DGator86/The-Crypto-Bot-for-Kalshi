#!/usr/bin/env python3
"""
Train the 30-minute lookahead model with synthetic BTC-like data.

This script generates realistic crypto price data and trains the model,
allowing us to validate the ML pipeline and optimize for Kalshi wagers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
import json

# Import from our extended model
from crypto_ml_15m_extended import (
    MLConfig,
    AssetPredictor,
    MultiAssetML,
    DataProvider,
    build_features,
    build_labels,
    eval_cv_metrics,
    XGBOOST_AVAILABLE
)

print(f"XGBoost available: {XGBOOST_AVAILABLE}")


class SyntheticDataProvider(DataProvider):
    """
    Generates realistic BTC-like price data for testing.

    Features:
    - Geometric Brownian Motion base
    - Mean reversion
    - Volatility clustering (GARCH-like)
    - Trend regimes
    - Intraday patterns
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)

        # Parse timeframe
        if timeframe == "15m":
            interval_minutes = 15
        elif timeframe == "1h":
            interval_minutes = 60
        else:
            interval_minutes = 15

        # Generate timestamps
        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        end_time = end_time.replace(minute=(end_time.minute // interval_minutes) * interval_minutes)

        timestamps = [end_time - timedelta(minutes=interval_minutes * i) for i in range(limit)]
        timestamps = timestamps[::-1]  # Oldest first

        # Parameters based on symbol
        if symbol.upper() in ["BTC", "BTCUSDT"]:
            initial_price = 95000
            annual_vol = 0.60  # 60% annual volatility
            mean_reversion = 0.02
            trend_prob = 0.01
        elif symbol.upper() in ["ETH", "ETHUSDT"]:
            initial_price = 3200
            annual_vol = 0.70
            mean_reversion = 0.015
            trend_prob = 0.012
        elif symbol.upper() in ["SOL", "SOLUSDT"]:
            initial_price = 180
            annual_vol = 0.90
            mean_reversion = 0.01
            trend_prob = 0.015
        else:
            initial_price = 100
            annual_vol = 0.50
            mean_reversion = 0.02
            trend_prob = 0.01

        # Convert to per-bar volatility (15-min bars)
        bars_per_year = 365 * 24 * 4  # 4 bars per hour
        bar_vol = annual_vol / np.sqrt(bars_per_year)

        # Generate price path with regime switching
        prices = np.zeros(limit)
        prices[0] = initial_price

        # Volatility state (for GARCH-like clustering)
        vol_state = np.ones(limit)
        current_vol_regime = 1.0

        # Trend state
        trend = 0.0
        trend_strength = 0.0

        for i in range(1, limit):
            # Update volatility regime (GARCH-like)
            if np.random.random() < 0.02:  # 2% chance to change vol regime
                current_vol_regime = np.random.choice([0.5, 1.0, 1.5, 2.0, 2.5])
            vol_state[i] = current_vol_regime

            # Update trend regime
            if np.random.random() < trend_prob:
                trend = np.random.choice([-1, 0, 1]) * np.random.uniform(0.001, 0.003)
                trend_strength = np.random.uniform(0.5, 1.0)
            else:
                trend_strength *= 0.99  # Decay trend

            # Intraday pattern (higher vol during US/EU sessions)
            hour = timestamps[i].hour
            if 13 <= hour <= 21:  # US session
                intraday_mult = 1.3
            elif 7 <= hour <= 16:  # EU session
                intraday_mult = 1.1
            else:
                intraday_mult = 0.8

            # Calculate return
            drift = trend * trend_strength
            shock = np.random.normal(0, bar_vol * vol_state[i] * intraday_mult)

            # Mean reversion
            log_price = np.log(prices[i-1])
            log_fair = np.log(initial_price)
            reversion = -mean_reversion * (log_price - log_fair) * 0.01

            ret = drift + shock + reversion
            prices[i] = prices[i-1] * np.exp(ret)

        # Generate OHLCV from closes
        df_data = []
        for i in range(limit):
            c = prices[i]

            # Generate intrabar volatility
            intrabar_vol = bar_vol * vol_state[i] * 0.3

            h = c * (1 + abs(np.random.normal(0, intrabar_vol)))
            l = c * (1 - abs(np.random.normal(0, intrabar_vol)))

            # Open is previous close with small gap
            if i > 0:
                o = prices[i-1] * (1 + np.random.normal(0, intrabar_vol * 0.1))
            else:
                o = c * (1 + np.random.normal(0, intrabar_vol * 0.1))

            # Ensure OHLC consistency
            h = max(h, o, c)
            l = min(l, o, c)

            # Volume (correlated with volatility and price movement)
            base_vol = 1000 + np.random.exponential(500)
            vol_mult = 1 + abs(c / prices[max(0, i-1)] - 1) * 50  # Higher vol on big moves
            vol_mult *= vol_state[i]  # Higher vol in high-vol regimes
            v = base_vol * vol_mult

            df_data.append([timestamps[i], o, h, l, c, v])

        df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp").sort_index()

        print(f"Generated {len(df)} bars for {symbol}")
        print(f"  Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"  Mean: ${df['close'].mean():.2f}")

        return df


def train_and_evaluate():
    """Train model and evaluate predictions."""
    print("="*80)
    print("TRAINING 30-MINUTE LOOKAHEAD MODEL FOR KALSHI WAGERS")
    print("="*80)

    # Configuration optimized for 30-minute Kalshi wagers
    cfg = MLConfig(
        timeframe="15m",
        horizon_bars=2,  # 30 minutes
        n_splits=8,
        purge_bars=4,
        embargo_bars=4,
        optimize_thresholds=True,
        calibrate_probabilities=True,
        use_xgboost=XGBOOST_AVAILABLE,
        min_edge_for_trade=0.03,  # 3% edge minimum for Kalshi
    )

    print(f"\nConfiguration:")
    print(f"  Horizon: {cfg.horizon_bars * 15} minutes")
    print(f"  CV Splits: {cfg.n_splits}")
    print(f"  Purge/Embargo: {cfg.purge_bars}/{cfg.embargo_bars} bars")
    print(f"  XGBoost: {cfg.use_xgboost}")

    # Generate synthetic data
    provider = SyntheticDataProvider(seed=42)

    # Train for BTC
    print("\n" + "="*60)
    print("Training model for BTC")
    print("="*60)

    df = provider.fetch_ohlcv("BTC", cfg.timeframe, limit=5000)

    # Build features and labels
    X = build_features(df, cfg)
    y = build_labels(df, cfg)
    X = X.loc[y.index]
    df = df.loc[X.index]

    print(f"\nDataset prepared:")
    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Class balance: UP={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    # Train predictor
    predictor = AssetPredictor(cfg)
    report = predictor.fit(df)

    # Print results
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)

    print("\nCross-Validation Metrics:")
    for metric in ["accuracy", "precision", "recall", "f1", "auc", "brier"]:
        if metric in report:
            print(f"  {metric:15s}: {report[metric]:.4f}")

    print(f"\nOptimized Thresholds:")
    print(f"  p_long: {report['p_long']:.2f}")
    print(f"  p_short: {report['p_short']:.2f}")

    print(f"\nRegime Distribution:")
    for regime, count in report.get("regime_counts", {}).items():
        print(f"  {regime}: {count}")

    print(f"\nRegime-Specific Thresholds:")
    for regime, thr in report.get("regime_thresholds", {}).items():
        print(f"  {regime}: p_long={thr['p_long']:.2f}, p_short={thr['p_short']:.2f}")

    # Lookahead validation
    if "lookahead_validation" in report:
        val = report["lookahead_validation"]
        print(f"\nLookahead Validation:")
        print(f"  Validation Accuracy: {val.get('validation_accuracy', 0):.1f}%")
        if val.get("validation_auc"):
            print(f"  Validation AUC: {val.get('validation_auc', 0):.4f}")
        if val.get("high_conf_accuracy"):
            print(f"  High-Conf Accuracy: {val.get('high_conf_accuracy', 0):.1f}% ({val.get('high_conf_samples', 0)} samples)")

    # Test live prediction
    print("\n" + "="*60)
    print("LIVE PREDICTION TEST")
    print("="*60)

    # Get the most recent data for prediction
    recent_df = provider.fetch_ohlcv("BTC", cfg.timeframe, limit=500)

    trade = predictor.decide_trade(recent_df)

    print(f"\nCurrent Signal:")
    print(f"  Timestamp: {trade['timestamp']}")
    print(f"  Price: ${trade['current_price']:.2f}")
    print(f"  Probability UP: {trade['probability_up']*100:.1f}%")
    print(f"  Action: {trade['action']}")
    print(f"  Regime: {trade['regime']}")
    print(f"  Edge: {trade['edge']*100:.1f}%")
    print(f"  Confidence: {trade['confidence']*100:.1f}%")
    print(f"  Position Size: {trade['position_size']:.2f}")
    print(f"  Kill Switch: {'ACTIVE' if trade['kill_switch'] else 'OFF'}")

    kalshi = trade.get("kalshi_wager", {})
    print(f"\nKalshi Wager Suggestion:")
    print(f"  Side: {kalshi.get('side', 'N/A')}")
    print(f"  Fair Price: ${kalshi.get('fair_price', 0)*100:.0f}")
    print(f"  Recommended: {'YES' if kalshi.get('recommended') else 'NO'}")

    # Save model
    predictor.save("models_30m_kalshi/BTC")
    print(f"\nModel saved to: models_30m_kalshi/BTC")

    # Important disclaimer about accuracy
    print("\n" + "="*80)
    print("IMPORTANT NOTES ON ACCURACY")
    print("="*80)
    print("""
Achieving >90% accuracy in crypto direction prediction is not realistic.
Here's why and what the model actually achieves:

1. MARKET EFFICIENCY: Crypto markets, while less efficient than traditional
   markets, still incorporate significant information. Achieving sustained
   90%+ accuracy would imply a massive market inefficiency.

2. REALISTIC EXPECTATIONS:
   - 50% = Random guessing
   - 52-54% = Weak but potentially profitable with proper sizing
   - 55-58% = Good edge for a quantitative model
   - 60%+ = Exceptional (rare and usually temporary)

3. WHAT THIS MODEL PROVIDES:
   - Probabilistic predictions (not binary)
   - Calibrated probabilities for Kalshi pricing
   - Regime-aware thresholds
   - Risk management (kill switch, position sizing)
   - Edge detection (only trade when edge > 3%)

4. FOR KALSHI WAGERS:
   - Use probability_up to price YES contracts
   - Use (1 - probability_up) to price NO contracts
   - Only trade when edge >= min_edge_for_trade (3%)
   - The model's value is in probability calibration, not accuracy
""")

    return report


if __name__ == "__main__":
    report = train_and_evaluate()
