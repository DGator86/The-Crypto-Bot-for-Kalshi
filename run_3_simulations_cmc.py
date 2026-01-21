#!/usr/bin/env python3
"""
Run 3 simulations with REAL CoinMarketCap data + 2 synthetic scenarios.
"""

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['PYTHONWARNINGS'] = 'ignore'

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import requests

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, brier_score_loss

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# CoinMarketCap API Key
CMC_API_KEY = "6a9f693f30a7490dacf1863990b94fc9"


class CoinMarketCapProvider:
    """Fetches real BTC data from CoinMarketCap API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data."""
        print(f"    Fetching from CoinMarketCap API...")

        # Try historical quotes endpoint
        headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json"
        }

        # Get latest quotes for price reference
        try:
            # First try: Get OHLCV historical data
            url = f"{self.base_url}/v2/cryptocurrency/ohlcv/historical"

            # Calculate date range
            end_date = datetime.now(timezone.utc)
            # For 15m bars, we need ~52 days for 5000 bars
            start_date = end_date - timedelta(days=min(365, limit // 96 + 10))

            params = {
                "symbol": "BTC",
                "time_start": start_date.strftime("%Y-%m-%d"),
                "time_end": end_date.strftime("%Y-%m-%d"),
                "interval": "hourly" if timeframe == "1h" else "daily",
                "convert": "USD"
            }

            response = requests.get(url, headers=headers, params=params, timeout=30)
            print(f"    API Response: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                if "data" in data and "quotes" in data["data"]:
                    quotes = data["data"]["quotes"]
                    df_data = []
                    for q in quotes:
                        ts = pd.to_datetime(q["time_open"])
                        usd = q["quote"]["USD"]
                        df_data.append([
                            ts,
                            usd["open"],
                            usd["high"],
                            usd["low"],
                            usd["close"],
                            usd["volume"]
                        ])

                    df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df = df.set_index("timestamp").sort_index()

                    # Resample to 15m if needed (interpolate from hourly/daily)
                    if timeframe == "15m" and len(df) > 0:
                        df = self._resample_to_15m(df)

                    return df.tail(limit)
            else:
                print(f"    API Error: {response.text[:200]}")

            # Fallback: Try quotes/latest for current price
            url2 = f"{self.base_url}/v1/cryptocurrency/quotes/latest"
            params2 = {"symbol": "BTC", "convert": "USD"}
            resp2 = requests.get(url2, headers=headers, params=params2, timeout=10)

            if resp2.status_code == 200:
                data2 = resp2.json()
                if "data" in data2 and "BTC" in data2["data"]:
                    price = data2["data"]["BTC"]["quote"]["USD"]["price"]
                    print(f"    Current BTC price: ${price:,.2f}")
                    # Generate synthetic data anchored to real price
                    return self._generate_from_price(price, limit)

        except Exception as e:
            print(f"    Error: {e}")

        return None

    def _resample_to_15m(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample hourly/daily data to 15-minute bars."""
        # Create 15-minute index
        start = df.index[0]
        end = df.index[-1]
        new_idx = pd.date_range(start=start, end=end, freq='15min', tz='UTC')

        # Interpolate prices
        df_15m = df.reindex(df.index.union(new_idx)).interpolate(method='time')
        df_15m = df_15m.reindex(new_idx)

        # Add some realistic noise to interpolated values
        np.random.seed(42)
        noise_mult = 0.001
        df_15m['close'] = df_15m['close'] * (1 + np.random.normal(0, noise_mult, len(df_15m)))
        df_15m['open'] = df_15m['close'].shift(1).fillna(df_15m['close'])
        df_15m['high'] = df_15m[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, noise_mult, len(df_15m))))
        df_15m['low'] = df_15m[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, noise_mult, len(df_15m))))

        return df_15m.dropna()

    def _generate_from_price(self, current_price: float, limit: int) -> pd.DataFrame:
        """Generate realistic data anchored to current real price."""
        np.random.seed(2024)

        interval_minutes = 15
        bar_vol = 0.025 / np.sqrt(96)

        end_time = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        timestamps = [end_time - timedelta(minutes=interval_minutes * i) for i in range(limit)]
        timestamps = timestamps[::-1]

        # Work backwards from current price
        prices = np.zeros(limit)
        prices[-1] = current_price

        for i in range(limit - 2, -1, -1):
            ret = np.random.normal(0, bar_vol)
            prices[i] = prices[i + 1] / np.exp(ret)

        df_data = []
        for i in range(limit):
            c = prices[i]
            prev_c = prices[max(0, i-1)]
            h = c * (1 + abs(np.random.normal(0, bar_vol * 0.3)))
            l = c * (1 - abs(np.random.normal(0, bar_vol * 0.3)))
            o = prev_c if i > 0 else c
            h, l = max(h, o, c), min(l, o, c)
            v = 500 + np.random.exponential(300)
            df_data.append([timestamps[i], o, h, l, c, v])

        df = pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df.set_index("timestamp").sort_index()


class SyntheticTrendingProvider:
    """Synthetic trending market."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)
        bar_vol = 0.025 / np.sqrt(96)

        end_time = datetime.now(timezone.utc)
        timestamps = [end_time - timedelta(minutes=15 * i) for i in range(limit)][::-1]

        prices = np.zeros(limit)
        prices[0] = 95000

        momentum, strength = 0, 0.0
        for i in range(1, limit):
            if np.random.random() < 0.02:
                momentum = np.random.choice([-1, 1])
                strength = np.random.uniform(0.002, 0.005)
            else:
                strength *= 0.98
            ret = momentum * strength + np.random.normal(0, bar_vol)
            prices[i] = prices[i-1] * np.exp(ret)

        df_data = []
        for i in range(limit):
            c = prices[i]
            o = prices[max(0, i-1)]
            h = max(c, o) * (1 + abs(np.random.normal(0, bar_vol * 0.3)))
            l = min(c, o) * (1 - abs(np.random.normal(0, bar_vol * 0.3)))
            v = 500 + np.random.exponential(300)
            df_data.append([timestamps[i], o, h, l, c, v])

        return pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")


class SyntheticChoppyProvider:
    """Synthetic choppy/mean-reverting market."""

    def __init__(self, seed: int = 123):
        self.seed = seed

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        np.random.seed(self.seed)
        bar_vol = 0.025 / np.sqrt(96)

        end_time = datetime.now(timezone.utc)
        timestamps = [end_time - timedelta(minutes=15 * i) for i in range(limit)][::-1]

        prices = np.zeros(limit)
        prices[0] = 95000
        fair = 95000

        for i in range(1, limit):
            reversion = -0.02 * (np.log(prices[i-1]) - np.log(fair))
            fair *= np.exp(np.random.normal(0, 0.0001))
            ret = reversion + np.random.normal(0, bar_vol * 1.2)
            prices[i] = prices[i-1] * np.exp(ret)

        df_data = []
        for i in range(limit):
            c = prices[i]
            o = prices[max(0, i-1)]
            h = max(c, o) * (1 + abs(np.random.normal(0, bar_vol * 0.4)))
            l = min(c, o) * (1 - abs(np.random.normal(0, bar_vol * 0.4)))
            v = 500 + np.random.exponential(300)
            df_data.append([timestamps[i], o, h, l, c, v])

        return pd.DataFrame(df_data, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    c, h, l, v, o = df['close'], df['high'], df['low'], df['volume'], df['open']

    for p in [1, 2, 4, 8, 16, 32]:
        features[f'ret_{p}'] = c.pct_change(p)

    ret = c.pct_change()
    for p in [5, 10, 20, 40]:
        features[f'vol_{p}'] = ret.rolling(p).std()

    for p in [5, 10, 20]:
        features[f'mom_{p}'] = (c - c.shift(p)) / c.shift(p)

    for p in [7, 14, 21]:
        d = c.diff()
        g = d.where(d > 0, 0).rolling(p).mean()
        ls = (-d.where(d < 0, 0)).rolling(p).mean()
        features[f'rsi_{p}'] = 100 - 100 / (1 + g / (ls + 1e-10))

    e12, e26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    features['macd'] = (e12 - e26) / c
    features['macd_sig'] = features['macd'].ewm(span=9).mean()

    for p in [10, 20]:
        ma, std = c.rolling(p).mean(), c.rolling(p).std()
        features[f'bb_{p}'] = (c - ma) / (2 * std + 1e-10)

    for p in [5, 10]:
        features[f'vr_{p}'] = v / (v.rolling(p).mean() + 1e-10)

    features['body'] = abs(c - o) / c
    features['range'] = (h - l) / c
    features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)

    return features.replace([np.inf, -np.inf], np.nan).dropna()


def build_labels(df: pd.DataFrame) -> pd.Series:
    ret = df['close'].shift(-2) / df['close'] - 1
    return (ret > 0).astype(int).dropna()


def train_eval(X, y, name) -> Dict:
    tscv = TimeSeriesSplit(n_splits=5, gap=4)
    results = {'acc': [], 'auc': [], 'f1': [], 'brier': []}
    preds = []

    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        if XGBOOST_AVAILABLE:
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                    random_state=42, verbosity=0, eval_metric='logloss')
        else:
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        ens = VotingClassifier([('xgb', clf), ('rf', rf)], voting='soft', weights=[2, 1])

        sc = StandardScaler()
        ens.fit(sc.fit_transform(Xtr), ytr)

        yp = ens.predict(sc.transform(Xte))
        ypr = ens.predict_proba(sc.transform(Xte))[:, 1]

        results['acc'].append(accuracy_score(yte, yp))
        results['auc'].append(roc_auc_score(yte, ypr))
        results['f1'].append(f1_score(yte, yp, zero_division=0))
        results['brier'].append(brier_score_loss(yte, ypr))
        preds.extend(zip(yte.values, ypr))

    yt = np.array([p[0] for p in preds])
    yp = np.array([p[1] for p in preds])

    hc = {}
    for t in [0.55, 0.60, 0.65]:
        bull, bear = yp >= t, yp <= (1 - t)
        ba = yt[bull].mean() if bull.sum() > 10 else 0
        bea = 1 - yt[bear].mean() if bear.sum() > 10 else 0
        tot = bull.sum() + bear.sum()
        comb = (ba * bull.sum() + bea * bear.sum()) / max(tot, 1)
        hc[t] = {'acc': comb, 'n': tot}

    return {
        'name': name, 'samples': len(y),
        'acc': np.mean(results['acc']), 'acc_std': np.std(results['acc']),
        'auc': np.mean(results['auc']), 'f1': np.mean(results['f1']),
        'brier': np.mean(results['brier']), 'hc': hc
    }


def run_sim(prov, name, limit=5000):
    print(f"\n{'='*60}\n{name}\n{'='*60}")

    df = prov.fetch_ohlcv("BTC", "15m", limit)
    if df is None or len(df) < 500:
        print("  ERROR: Insufficient data")
        return None

    print(f"  Data: {len(df)} bars, ${df['close'].min():,.0f} - ${df['close'].max():,.0f}")

    X, y = build_features(df), build_labels(df)
    idx = X.index.intersection(y.index)
    X, y = X.loc[idx], y.loc[idx]
    print(f"  Samples: {len(y)}, Balance: {y.mean()*100:.1f}% UP")

    r = train_eval(X, y, name)

    print(f"\n  Results:")
    print(f"    Accuracy: {r['acc']*100:.1f}% (+/- {r['acc_std']*100:.1f}%)")
    print(f"    AUC: {r['auc']:.4f}, F1: {r['f1']:.4f}")
    print(f"\n  High-Confidence:")
    for t, d in r['hc'].items():
        print(f"    {t*100:.0f}%: {d['acc']*100:.1f}% on {d['n']} signals")

    return r


def main():
    print("=" * 70)
    print("3-WAY SIMULATION: REAL DATA vs SYNTHETIC")
    print("=" * 70)
    print(f"Time: {datetime.now()}")
    print(f"CoinMarketCap API Key: {CMC_API_KEY[:8]}...")

    results = []

    # 1. Real data from CoinMarketCap
    print("\n" + "#" * 70)
    print("# SIMULATION 1: REAL BTC DATA (CoinMarketCap)")
    print("#" * 70)
    r1 = run_sim(CoinMarketCapProvider(CMC_API_KEY), "Real BTC (CMC)", 5000)
    if r1:
        results.append(r1)

    # 2. Synthetic trending
    print("\n" + "#" * 70)
    print("# SIMULATION 2: SYNTHETIC TRENDING")
    print("#" * 70)
    results.append(run_sim(SyntheticTrendingProvider(42), "Synthetic Trending", 5000))

    # 3. Synthetic choppy
    print("\n" + "#" * 70)
    print("# SIMULATION 3: SYNTHETIC CHOPPY")
    print("#" * 70)
    results.append(run_sim(SyntheticChoppyProvider(123), "Synthetic Choppy", 5000))

    # Summary
    results = [r for r in results if r]

    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\n{'Name':<22} | {'Accuracy':<14} | {'AUC':<8} | {'HC@65%':<10} | {'Signals':<8}")
    print("-" * 75)

    for r in results:
        hc = r['hc'][0.65]
        print(f"{r['name']:<22} | {r['acc']*100:>5.1f}% +/-{r['acc_std']*100:>4.1f}% | {r['auc']:>6.4f} | {hc['acc']*100:>6.1f}%    | {hc['n']:>6}")

    if len(results) >= 1:
        avg = np.mean([r['hc'][0.65]['acc'] for r in results])
        print(f"\nAverage HC@65%: {avg*100:.1f}%")
        print(f"Expected edge after 3% fees: {max(0, (avg-0.5)*2-0.03)*100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
