#!/usr/bin/env python3
"""
Run 3 simulations comparing real vs synthetic data performance.
1. Real market data (if available)
2. Synthetic data with seed 42
3. Synthetic data with seed 123
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Optional
import json

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ============================================================================
# DATA PROVIDERS
# ============================================================================

class RealDataProvider:
    """Fetches real BTC data from public APIs."""

    def __init__(self):
        self.sources = [
            self._fetch_coinbase,
            self._fetch_coingecko,
            self._fetch_binance,
        ]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Try multiple sources for real data."""
        import requests

        for source_fn in self.sources:
            try:
                df = source_fn(symbol, timeframe, limit)
                if df is not None and len(df) > 100:
                    print(f"    [Real data source: {source_fn.__name__}]")
                    return df
            except Exception as e:
                continue

        return None

    def _fetch_coinbase(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch from Coinbase API."""
        import requests

        granularity = 900 if timeframe == "15m" else 3600  # seconds
        product_id = "BTC-USD"

        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        params = {"granularity": granularity}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data:
            return None

        # Coinbase returns [time, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        df = df.set_index('timestamp').sort_index()

        return df.tail(limit)

    def _fetch_coingecko(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch from CoinGecko API (limited to daily/hourly)."""
        import requests

        # CoinGecko doesn't have 15m data, so we'll use hourly and resample
        days = min(90, limit // 4)
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {"vs_currency": "usd", "days": days}

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data:
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['volume'] = 1000  # CoinGecko OHLC doesn't include volume
        df = df.set_index('timestamp').sort_index()

        return df.tail(limit)

    def _fetch_binance(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch from Binance API."""
        import requests

        interval = "15m" if timeframe == "15m" else "1h"
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": interval,
            "limit": min(limit, 1000)
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return None

        data = response.json()
        if not data:
            return None

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        df = df.set_index('timestamp').sort_index()

        return df


class SyntheticDataProvider:
    """Generates realistic BTC-like price data with momentum regimes."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)

        interval_minutes = 15 if timeframe == "15m" else 60

        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        end_time = end_time.replace(minute=(end_time.minute // interval_minutes) * interval_minutes)

        timestamps = [end_time - timedelta(minutes=interval_minutes * i) for i in range(limit)]
        timestamps = timestamps[::-1]

        # BTC parameters
        initial_price = 95000
        annual_vol = 0.60
        bars_per_year = 365 * 24 * 4
        bar_vol = annual_vol / np.sqrt(bars_per_year)

        prices = np.zeros(limit)
        prices[0] = initial_price

        vol_state = np.ones(limit)
        current_vol_regime = 1.0

        # Momentum regimes (makes data somewhat predictable)
        current_momentum = 0
        momentum_strength = 0.0
        momentum_persistence = 0.95

        for i in range(1, limit):
            if np.random.random() < 0.02:
                current_vol_regime = np.random.choice([0.5, 1.0, 1.5, 2.0])
            vol_state[i] = current_vol_regime

            if np.random.random() < 0.03:
                current_momentum = np.random.choice([-1, 0, 1])
                momentum_strength = np.random.uniform(0.001, 0.004)
            else:
                momentum_strength *= momentum_persistence

            hour = timestamps[i].hour
            if 13 <= hour <= 21:
                intraday_mult = 1.3
            elif 7 <= hour <= 16:
                intraday_mult = 1.1
            else:
                intraday_mult = 0.8

            drift = current_momentum * momentum_strength
            shock = np.random.normal(0, bar_vol * vol_state[i] * intraday_mult)

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

            base_vol = 1000 + np.random.exponential(500)
            vol_mult = 1 + abs(c / prices[max(0, i-1)] - 1) * 50
            v = base_vol * vol_mult * vol_state[i]

            df_data.append([timestamps[i], o, h, l, c, v])

        df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp").sort_index()
        return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build comprehensive feature set."""
    features = pd.DataFrame(index=df.index)

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    o = df['open']

    # Returns
    for period in [1, 2, 4, 8, 16, 32]:
        features[f'ret_{period}'] = c.pct_change(period)

    # Volatility
    ret_1 = c.pct_change(1)
    for period in [5, 10, 20, 40]:
        features[f'vol_{period}'] = ret_1.rolling(period).std()

    # Momentum
    for period in [5, 10, 20]:
        features[f'mom_{period}'] = (c - c.shift(period)) / c.shift(period)

    # RSI
    for period in [7, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    features['macd'] = (ema12 - ema26) / c
    features['macd_signal'] = features['macd'].ewm(span=9).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    for period in [10, 20]:
        ma = c.rolling(period).mean()
        std = c.rolling(period).std()
        features[f'bb_position_{period}'] = (c - ma) / (2 * std + 1e-10)
        features[f'bb_width_{period}'] = (4 * std) / ma

    # Volume
    for period in [5, 10, 20]:
        features[f'vol_ratio_{period}'] = v / (v.rolling(period).mean() + 1e-10)

    # Candlestick
    features['body_size'] = abs(c - o) / c
    features['hl_range'] = (h - l) / c

    # MA crossovers
    for fast, slow in [(5, 10), (10, 20), (20, 50)]:
        ma_fast = c.rolling(fast).mean()
        ma_slow = c.rolling(slow).mean()
        features[f'ma_cross_{fast}_{slow}'] = (ma_fast - ma_slow) / c

    # ADX
    tr = pd.concat([h - l, abs(h - c.shift()), abs(l - c.shift())], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    features['atr_ratio'] = atr / c

    # Time features
    hour = df.index.hour
    features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_labels(df: pd.DataFrame, horizon_bars: int = 2) -> pd.Series:
    """Build labels for 30-minute lookahead."""
    future_ret = df['close'].shift(-horizon_bars) / df['close'] - 1
    labels = (future_ret > 0).astype(int)
    return labels.dropna()


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, name: str) -> Dict:
    """Train model and return metrics."""

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=4)

    cv_results = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'auc': [], 'brier': []
    }

    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Build ensemble
        if XGBOOST_AVAILABLE:
            xgb_clf = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0,
                use_label_encoder=False, eval_metric='logloss'
            )
        else:
            xgb_clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
            )

        rf_clf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )

        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
            voting='soft', weights=[2, 1]
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ensemble.fit(X_train_scaled, y_train)

        y_pred = ensemble.predict(X_test_scaled)
        y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

        cv_results['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_results['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_results['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_results['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        cv_results['auc'].append(roc_auc_score(y_test, y_proba))
        cv_results['brier'].append(brier_score_loss(y_test, y_proba))

        all_predictions.extend(zip(y_test.values, y_proba))

    # Aggregate results
    results = {
        'name': name,
        'samples': len(y),
        'features': len(X.columns),
        'class_balance': y.mean(),
    }

    for metric, values in cv_results.items():
        results[f'{metric}_mean'] = np.mean(values)
        results[f'{metric}_std'] = np.std(values)

    # High-confidence analysis
    y_true_all = np.array([p[0] for p in all_predictions])
    y_proba_all = np.array([p[1] for p in all_predictions])

    high_conf_results = {}
    for threshold in [0.55, 0.60, 0.65]:
        bull_mask = y_proba_all >= threshold
        bear_mask = y_proba_all <= (1 - threshold)

        bull_acc = y_true_all[bull_mask].mean() if bull_mask.sum() > 10 else 0
        bear_acc = 1 - y_true_all[bear_mask].mean() if bear_mask.sum() > 10 else 0

        total = bull_mask.sum() + bear_mask.sum()
        combined = (bull_acc * bull_mask.sum() + bear_acc * bear_mask.sum()) / max(total, 1)

        high_conf_results[threshold] = {
            'accuracy': combined,
            'signals': total,
            'bull_signals': int(bull_mask.sum()),
            'bear_signals': int(bear_mask.sum()),
        }

    results['high_confidence'] = high_conf_results

    return results


# ============================================================================
# MAIN
# ============================================================================

def run_simulation(provider, name: str, limit: int = 5000) -> Optional[Dict]:
    """Run a single simulation."""
    print(f"\n{'='*60}")
    print(f"SIMULATION: {name}")
    print(f"{'='*60}")

    # Fetch data
    print(f"\n  Fetching data...")
    df = provider.fetch_ohlcv("BTC", "15m", limit=limit)

    if df is None or len(df) < 500:
        print(f"  ERROR: Could not fetch sufficient data")
        return None

    print(f"  Data points: {len(df)}")
    print(f"  Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Build features
    print(f"\n  Building features...")
    X = build_features(df)
    y = build_labels(df, horizon_bars=2)

    common_idx = X.index.intersection(y.index)
    X = X.loc[common_idx]
    y = y.loc[common_idx]

    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Class balance: {y.mean()*100:.1f}% UP")

    # Train and evaluate
    print(f"\n  Training ensemble model...")
    results = train_and_evaluate(X, y, name)

    # Print results
    print(f"\n  RESULTS:")
    print(f"  {'Accuracy:':<15} {results['accuracy_mean']*100:.1f}% (+/- {results['accuracy_std']*100:.1f}%)")
    print(f"  {'AUC:':<15} {results['auc_mean']:.4f} (+/- {results['auc_std']:.4f})")
    print(f"  {'F1:':<15} {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
    print(f"  {'Brier:':<15} {results['brier_mean']:.4f}")

    print(f"\n  HIGH-CONFIDENCE SIGNALS:")
    for thresh, data in results['high_confidence'].items():
        print(f"    {thresh*100:.0f}% threshold: {data['accuracy']*100:.1f}% accuracy on {data['signals']} signals")

    return results


def compare_results(results_list: list):
    """Compare results from multiple simulations."""
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    # Filter out None results
    results_list = [r for r in results_list if r is not None]

    if not results_list:
        print("No valid results to compare.")
        return

    # Header
    print(f"\n{'Simulation':<25} | {'Samples':<8} | {'Accuracy':<12} | {'AUC':<10} | {'F1':<10} | {'Brier':<10}")
    print("-" * 85)

    for r in results_list:
        print(f"{r['name']:<25} | {r['samples']:<8} | {r['accuracy_mean']*100:>5.1f}% +/-{r['accuracy_std']*100:>4.1f} | {r['auc_mean']:>8.4f} | {r['f1_mean']:>8.4f} | {r['brier_mean']:>8.4f}")

    # High-confidence comparison
    print(f"\n{'HIGH-CONFIDENCE SIGNALS (65% threshold):'}")
    print(f"{'Simulation':<25} | {'Accuracy':<12} | {'Signals':<10} | {'Signal Rate':<12}")
    print("-" * 65)

    for r in results_list:
        hc = r['high_confidence'].get(0.65, {})
        acc = hc.get('accuracy', 0) * 100
        signals = hc.get('signals', 0)
        rate = signals / r['samples'] * 100 if r['samples'] > 0 else 0
        print(f"{r['name']:<25} | {acc:>10.1f}% | {signals:>10} | {rate:>10.1f}%")

    # Best performer
    print("\n" + "-" * 80)
    best_acc = max(results_list, key=lambda x: x['accuracy_mean'])
    best_auc = max(results_list, key=lambda x: x['auc_mean'])
    best_hc = max(results_list, key=lambda x: x['high_confidence'].get(0.65, {}).get('accuracy', 0))

    print(f"Best overall accuracy: {best_acc['name']} ({best_acc['accuracy_mean']*100:.1f}%)")
    print(f"Best AUC: {best_auc['name']} ({best_auc['auc_mean']:.4f})")
    print(f"Best high-confidence: {best_hc['name']} ({best_hc['high_confidence'].get(0.65, {}).get('accuracy', 0)*100:.1f}%)")

    # Kalshi recommendation
    print("\n" + "=" * 80)
    print("KALSHI WAGER RECOMMENDATION")
    print("=" * 80)

    avg_hc_acc = np.mean([r['high_confidence'].get(0.65, {}).get('accuracy', 0) for r in results_list])

    if avg_hc_acc >= 0.58:
        recommendation = "FAVORABLE - High-confidence signals show consistent edge"
        sizing = "Use 50-60% Kelly sizing on high-confidence signals"
    elif avg_hc_acc >= 0.55:
        recommendation = "MARGINAL - Small edge, proceed with caution"
        sizing = "Use 25-30% Kelly sizing, strict risk limits"
    else:
        recommendation = "NOT RECOMMENDED - Insufficient edge after fees"
        sizing = "Paper trade only until model improves"

    print(f"\n  Average high-confidence accuracy: {avg_hc_acc*100:.1f}%")
    print(f"  Recommendation: {recommendation}")
    print(f"  Position sizing: {sizing}")


def main():
    print("=" * 80)
    print("30-MINUTE KALSHI WAGER SIMULATION COMPARISON")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")

    results = []

    # Simulation 1: Real data
    print("\n" + "#" * 80)
    print("# SIMULATION 1: REAL MARKET DATA")
    print("#" * 80)

    real_provider = RealDataProvider()
    real_result = run_simulation(real_provider, "Real BTC Data", limit=2000)
    results.append(real_result)

    # Simulation 2: Synthetic (seed 42)
    print("\n" + "#" * 80)
    print("# SIMULATION 2: SYNTHETIC DATA (SEED 42)")
    print("#" * 80)

    synth_provider_1 = SyntheticDataProvider(seed=42)
    synth_result_1 = run_simulation(synth_provider_1, "Synthetic (seed=42)", limit=5000)
    results.append(synth_result_1)

    # Simulation 3: Synthetic (seed 123)
    print("\n" + "#" * 80)
    print("# SIMULATION 3: SYNTHETIC DATA (SEED 123)")
    print("#" * 80)

    synth_provider_2 = SyntheticDataProvider(seed=123)
    synth_result_2 = run_simulation(synth_provider_2, "Synthetic (seed=123)", limit=5000)
    results.append(synth_result_2)

    # Compare all results
    compare_results(results)

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
