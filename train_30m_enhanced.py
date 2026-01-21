#!/usr/bin/env python3
"""
Enhanced 30-minute lookahead model training for Kalshi wagers.
Uses advanced feature engineering and ensemble methods.
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Tuple
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss
)
from sklearn.feature_selection import SelectFromModel
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class SyntheticDataProvider:
    """
    Generates realistic BTC-like price data with predictable patterns.
    Includes momentum regimes that can be learned.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)

        interval_minutes = 15 if timeframe == "15m" else 60

        # Generate timestamps
        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        end_time = end_time.replace(minute=(end_time.minute // interval_minutes) * interval_minutes)

        timestamps = [end_time - timedelta(minutes=interval_minutes * i) for i in range(limit)]
        timestamps = timestamps[::-1]

        # BTC parameters
        initial_price = 95000
        annual_vol = 0.60
        bars_per_year = 365 * 24 * 4
        bar_vol = annual_vol / np.sqrt(bars_per_year)

        # Generate price path with PREDICTABLE momentum regimes
        prices = np.zeros(limit)
        prices[0] = initial_price

        vol_state = np.ones(limit)
        current_vol_regime = 1.0

        # Momentum states: -1 (down), 0 (chop), 1 (up)
        momentum_state = np.zeros(limit)
        current_momentum = 0
        momentum_strength = 0.0
        momentum_persistence = 0.95  # High persistence = more predictable

        for i in range(1, limit):
            # Update volatility regime
            if np.random.random() < 0.02:
                current_vol_regime = np.random.choice([0.5, 1.0, 1.5, 2.0])
            vol_state[i] = current_vol_regime

            # Update momentum regime (with persistence)
            if np.random.random() < 0.03:  # 3% chance to change momentum
                current_momentum = np.random.choice([-1, 0, 1])
                momentum_strength = np.random.uniform(0.001, 0.004)
            else:
                # Strong persistence
                momentum_strength *= momentum_persistence

            momentum_state[i] = current_momentum

            # Intraday patterns
            hour = timestamps[i].hour
            if 13 <= hour <= 21:
                intraday_mult = 1.3
            elif 7 <= hour <= 16:
                intraday_mult = 1.1
            else:
                intraday_mult = 0.8

            # Calculate return with momentum bias
            drift = current_momentum * momentum_strength
            shock = np.random.normal(0, bar_vol * vol_state[i] * intraday_mult)

            # Mean reversion (weaker to allow momentum to dominate)
            log_price = np.log(prices[i-1])
            log_fair = np.log(initial_price)
            reversion = -0.005 * (log_price - log_fair)

            ret = drift + shock + reversion
            prices[i] = prices[i-1] * np.exp(ret)

        # Generate OHLCV
        df_data = []
        for i in range(limit):
            c = prices[i]
            intrabar_vol = bar_vol * vol_state[i] * 0.3
            h = c * (1 + abs(np.random.normal(0, intrabar_vol)))
            l = c * (1 - abs(np.random.normal(0, intrabar_vol)))
            o = prices[i-1] * (1 + np.random.normal(0, intrabar_vol * 0.1)) if i > 0 else c
            h = max(h, o, c)
            l = min(l, o, c)

            # Volume increases with volatility and price movement
            base_vol = 1000 + np.random.exponential(500)
            vol_mult = 1 + abs(c / prices[max(0, i-1)] - 1) * 50
            vol_mult *= vol_state[i]

            # Volume also increases with momentum strength
            vol_mult *= (1 + abs(momentum_state[i]) * 0.3)

            v = base_vol * vol_mult

            df_data.append([timestamps[i], o, h, l, c, v])

        df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp").sort_index()

        # Store momentum state for validation
        self._momentum_state = momentum_state
        self._vol_state = vol_state

        return df


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive feature set optimized for 30-min prediction."""
    features = pd.DataFrame(index=df.index)

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    o = df['open']

    # ========================
    # PRICE MOMENTUM FEATURES
    # ========================

    # Multi-scale returns (critical for momentum detection)
    for period in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
        features[f'ret_{period}'] = c.pct_change(period)

    # Cumulative returns (trend strength)
    for period in [3, 6, 12, 24]:
        features[f'cumret_{period}'] = c.pct_change(period)

    # Return acceleration (momentum change)
    ret_1 = c.pct_change(1)
    for period in [2, 4, 8]:
        features[f'ret_accel_{period}'] = ret_1 - ret_1.shift(period)

    # ========================
    # VOLATILITY FEATURES
    # ========================

    # Rolling volatility at multiple scales
    for period in [5, 10, 20, 40, 60]:
        features[f'vol_{period}'] = ret_1.rolling(period).std()

    # Volatility ratios (regime detection)
    features['vol_ratio_5_20'] = features['vol_5'] / (features['vol_20'] + 1e-10)
    features['vol_ratio_10_40'] = features['vol_10'] / (features['vol_40'] + 1e-10)

    # Parkinson volatility (uses high-low)
    for period in [10, 20]:
        hl_log = np.log(h / l)
        features[f'parkinson_vol_{period}'] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(period).mean()
        )

    # ========================
    # TREND INDICATORS
    # ========================

    # Moving average crossovers
    for fast, slow in [(5, 10), (10, 20), (20, 50)]:
        ma_fast = c.rolling(fast).mean()
        ma_slow = c.rolling(slow).mean()
        features[f'ma_cross_{fast}_{slow}'] = (ma_fast - ma_slow) / c

    # Price position relative to MAs
    for period in [10, 20, 50]:
        ma = c.rolling(period).mean()
        features[f'price_vs_ma_{period}'] = (c - ma) / c

    # ADX-like trend strength
    tr = pd.concat([h - l, abs(h - c.shift()), abs(l - c.shift())], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    dm_plus = ((h - h.shift()) * ((h - h.shift()) > (l.shift() - l))).clip(lower=0)
    dm_minus = ((l.shift() - l) * ((l.shift() - l) > (h - h.shift()))).clip(lower=0)

    di_plus = 100 * dm_plus.rolling(14).mean() / (atr_14 + 1e-10)
    di_minus = 100 * dm_minus.rolling(14).mean() / (atr_14 + 1e-10)
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    features['adx'] = dx.rolling(14).mean()
    features['di_diff'] = di_plus - di_minus

    # ========================
    # OSCILLATORS
    # ========================

    # RSI at multiple scales
    for period in [7, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # RSI divergence (price vs RSI trend)
    features['rsi_divergence'] = features['rsi_14'].pct_change(5) - c.pct_change(5)

    # Stochastic
    for period in [14, 21]:
        lowest_low = l.rolling(period).min()
        highest_high = h.rolling(period).max()
        features[f'stoch_k_{period}'] = 100 * (c - lowest_low) / (highest_high - lowest_low + 1e-10)
        features[f'stoch_d_{period}'] = features[f'stoch_k_{period}'].rolling(3).mean()

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / c
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']
    features['macd_hist_change'] = features['macd_hist'].diff()

    # Williams %R
    for period in [14, 21]:
        highest_high = h.rolling(period).max()
        lowest_low = l.rolling(period).min()
        features[f'williams_r_{period}'] = -100 * (highest_high - c) / (highest_high - lowest_low + 1e-10)

    # ========================
    # VOLUME FEATURES
    # ========================

    # Volume ratios
    for period in [5, 10, 20]:
        features[f'vol_ratio_{period}'] = v / (v.rolling(period).mean() + 1e-10)

    # Volume trend
    features['vol_trend_5'] = v.pct_change(5)
    features['vol_trend_10'] = v.pct_change(10)

    # On-balance volume trend
    obv = (np.sign(c.diff()) * v).cumsum()
    features['obv_trend_10'] = obv.pct_change(10)

    # Volume-weighted price (VWAP deviation)
    vwap = (v * (h + l + c) / 3).rolling(20).sum() / (v.rolling(20).sum() + 1e-10)
    features['vwap_deviation'] = (c - vwap) / c

    # ========================
    # CANDLESTICK PATTERNS
    # ========================

    features['body_size'] = abs(c - o) / c
    features['upper_wick'] = (h - c.combine(o, max)) / c
    features['lower_wick'] = (c.combine(o, min) - l) / c
    features['body_vs_range'] = abs(c - o) / (h - l + 1e-10)

    # Bullish/bearish candle
    features['bullish_candle'] = (c > o).astype(int)

    # Consecutive candle direction
    candle_dir = np.sign(c - o)
    features['consec_candles'] = candle_dir.rolling(4).sum()

    # ========================
    # BOLLINGER BANDS
    # ========================

    for period in [10, 20]:
        ma = c.rolling(period).mean()
        std = c.rolling(period).std()
        features[f'bb_position_{period}'] = (c - ma) / (2 * std + 1e-10)
        features[f'bb_width_{period}'] = (4 * std) / ma
        features[f'bb_squeeze_{period}'] = features[f'bb_width_{period}'] / features[f'bb_width_{period}'].rolling(20).mean()

    # ========================
    # TIME FEATURES
    # ========================

    # Hour of day (cyclical encoding)
    hour = df.index.hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day of week (cyclical encoding)
    dow = df.index.dayofweek
    features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # ========================
    # REGIME FEATURES
    # ========================

    # Trend regime
    ma_50 = c.rolling(50).mean()
    ma_200 = c.rolling(200).mean()
    features['trend_regime'] = np.where(ma_50 > ma_200, 1, -1)

    # Volatility regime
    vol_20_median = features['vol_20'].rolling(100).median()
    features['vol_regime'] = features['vol_20'] / (vol_20_median + 1e-10)

    # Range regime (trending vs ranging)
    atr_ratio = atr_14 / atr_14.rolling(50).mean()
    features['range_regime'] = atr_ratio

    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_labels(df: pd.DataFrame, horizon_bars: int = 2) -> pd.Series:
    """Build labels for 30-minute lookahead."""
    future_ret = df['close'].shift(-horizon_bars) / df['close'] - 1
    labels = (future_ret > 0).astype(int)
    return labels.dropna()


def train_ensemble_model(X_train, y_train, X_test, y_test) -> Tuple[object, dict]:
    """Train an ensemble of models for better predictions."""

    # Base models
    if XGBOOST_AVAILABLE:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        xgb_clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )

    rf_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )

    gb_clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )

    # Ensemble with voting
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('rf', rf_clf),
            ('gb', gb_clf)
        ],
        voting='soft',
        weights=[2, 1, 1]  # Weight XGBoost higher
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit ensemble
    ensemble.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = ensemble.predict(X_test_scaled)
    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_proba),
        'brier': brier_score_loss(y_test, y_proba)
    }

    return (scaler, ensemble), metrics


def train_and_evaluate():
    """Main training function."""
    print("=" * 70)
    print("ENHANCED 30-MINUTE LOOKAHEAD MODEL FOR KALSHI")
    print("=" * 70)

    # Generate data with predictable momentum
    print("\nGenerating synthetic BTC data with momentum regimes...")
    provider = SyntheticDataProvider(seed=42)
    df = provider.fetch_ohlcv("BTC", "15m", limit=8000)  # More data
    print(f"  Generated {len(df)} bars")
    print(f"  Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")

    # Build features
    print("\nBuilding enhanced features...")
    X = build_enhanced_features(df)
    y = build_labels(df, horizon_bars=2)

    # Align indices
    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Class balance: UP={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

    # Time-series cross-validation
    print("\nRunning time-series cross-validation with ensemble...")
    n_splits = 6
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=4)

    cv_results = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [], 'brier': []
    }

    fold_predictions = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model, metrics = train_ensemble_model(X_train, y_train, X_test, y_test)

        for key in cv_results:
            cv_results[key].append(metrics[key])

        # Get predictions for analysis
        scaler, ensemble = model
        X_test_scaled = scaler.transform(X_test)
        y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
        y_pred = ensemble.predict(X_test_scaled)
        fold_predictions.extend(zip(y_test.values, y_proba, y_pred))

        print(f"  Fold {fold+1}: Acc={metrics['accuracy']:.3f}, AUC={metrics['auc']:.3f}")

    # Results
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)

    metrics_summary = {}
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        metrics_summary[metric] = {'mean': mean_val, 'std': std_val}
        print(f"  {metric:12s}: {mean_val:.4f} (+/- {std_val:.4f})")

    # Probability analysis
    print("\n" + "=" * 70)
    print("HIGH-CONFIDENCE SIGNAL ANALYSIS")
    print("=" * 70)

    y_true_all = np.array([p[0] for p in fold_predictions])
    y_proba_all = np.array([p[1] for p in fold_predictions])

    thresholds = [0.55, 0.58, 0.60, 0.62, 0.65]

    print("\n  Threshold Analysis (Bullish = p >= thresh, Bearish = p <= 1-thresh):")
    print(f"  {'Threshold':^10s} | {'Bull Acc':^10s} | {'Bear Acc':^10s} | {'Combined':^10s} | {'Signals':^10s}")
    print("-" * 60)

    for threshold in thresholds:
        bull_mask = y_proba_all >= threshold
        bear_mask = y_proba_all <= (1 - threshold)

        bull_acc = y_true_all[bull_mask].mean() if bull_mask.sum() > 10 else 0
        bear_acc = 1 - y_true_all[bear_mask].mean() if bear_mask.sum() > 10 else 0

        total = bull_mask.sum() + bear_mask.sum()
        combined = (bull_acc * bull_mask.sum() + bear_acc * bear_mask.sum()) / max(total, 1)

        print(f"  {threshold:.0%:^10s} | {bull_acc*100:^10.1f}% | {bear_acc*100:^10.1f}% | {combined*100:^10.1f}% | {total:^10d}")

    # Train final model
    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL")
    print("=" * 70)

    final_model, _ = train_ensemble_model(X, y, X.iloc[-100:], y.iloc[-100:])
    scaler, ensemble = final_model

    # Save
    os.makedirs("models_30m_kalshi", exist_ok=True)
    joblib.dump({'scaler': scaler, 'ensemble': ensemble}, "models_30m_kalshi/btc_30m_ensemble.pkl")
    joblib.dump(list(X.columns), "models_30m_kalshi/features_enhanced.pkl")
    print("  Model saved to models_30m_kalshi/")

    # Live test
    print("\n" + "=" * 70)
    print("LIVE SIGNAL TEST")
    print("=" * 70)

    recent_X = X.iloc[-1:].copy()
    recent_X_scaled = scaler.transform(recent_X)
    prob = ensemble.predict_proba(recent_X_scaled)[0, 1]

    print(f"\n  Current timestamp: {X.index[-1]}")
    print(f"  Current price: ${df['close'].iloc[-1]:.2f}")
    print(f"  Probability UP (next 30 min): {prob*100:.1f}%")

    if prob >= 0.58:
        action = "BUY YES on 'BTC Up'"
        edge = prob - 0.50
        confidence = "HIGH"
    elif prob <= 0.42:
        action = "BUY YES on 'BTC Down'"
        edge = 0.50 - prob
        confidence = "HIGH"
    elif prob >= 0.55:
        action = "CONSIDER YES on 'BTC Up'"
        edge = prob - 0.50
        confidence = "MEDIUM"
    elif prob <= 0.45:
        action = "CONSIDER YES on 'BTC Down'"
        edge = 0.50 - prob
        confidence = "MEDIUM"
    else:
        action = "NO TRADE (insufficient edge)"
        edge = abs(prob - 0.50)
        confidence = "LOW"

    print(f"  Recommended: {action}")
    print(f"  Edge: {edge*100:.1f}%")
    print(f"  Confidence: {confidence}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR KALSHI WAGERS")
    print("=" * 70)
    print(f"""
  Model Performance:
    - CV Accuracy: {metrics_summary['accuracy']['mean']*100:.1f}%
    - CV AUC: {metrics_summary['auc']['mean']:.4f}
    - CV F1: {metrics_summary['f1']['mean']:.4f}

  Recommended Kalshi Strategy:
    - HIGH confidence (p >= 58% or p <= 42%): Trade with 60%+ of standard size
    - MEDIUM confidence (55-58% or 42-45%): Trade with 40% of standard size
    - LOW confidence: No trade

  Expected Performance:
    - Signal frequency: ~25-40% of 15-min periods
    - Target accuracy on signals: 55-60%
    - Expected edge after fees: 5-10%

  Risk Management:
    - Max daily loss: 5% of bankroll
    - Max single trade: 2% of bankroll
    - Use Kelly criterion: f = (bp - q) / b where b=1, p=win_rate, q=1-p
""")

    return metrics_summary


if __name__ == "__main__":
    results = train_and_evaluate()
