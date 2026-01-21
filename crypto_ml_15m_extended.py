"""
Crypto 15m/30m Direction ML (BTC/ETH/SOL) — Extended Bot-Ready Edition for Kalshi

Optimized for Kalshi 15-minute crypto up/down wagers with 30-minute lookahead.

Adds:
1) Data adapters: CSVProvider + BinanceRESTProvider (public REST klines)
2) Purged & Embargoed walk-forward CV (reduces leakage)
3) Threshold optimization (p_long/p_short) per asset using CV folds + costs
4) Regime gating with per-regime thresholds (trend / chop / high-vol)
5) Tree + logistic ensemble with calibrated meta layer
6) Live predict_and_signal() + decide_trade() for execution bots (position sizing + kill switch)
7) Model/threshold persistence
8) 30-minute lookahead optimization for Kalshi wagers
9) Integration with crypto15 package modules

Dependencies:
  pip install pandas numpy scikit-learn joblib requests xgboost

Important:
- This predicts the next 15m/30m direction probabilistically; it does not guarantee profits.
- For Kalshi wagers: use probability_up to price YES/NO positions.
- Realistic accuracy expectations: 55-65% directional accuracy is excellent for crypto.
"""

from __future__ import annotations

import os
import json
import math
import argparse
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Tuple, Optional, List, Iterator, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from joblib import dump, load
from sklearn.pipeline import Pipeline

from kalshi.fees import FeeSchedule
from kalshi.orderbook import TopOfBook
from strategy.ev import TradeEV
from strategy.gates import GateConfig
from strategy.policy import TradeDecision as PolicyTradeDecision, decide_trade as policy_decide_trade
from strategy.sizing import SizingConfig
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)

# Optional XGBoost for better performance
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ----------------------------
# Configuration
# ----------------------------

@dataclass
class MLConfig:
    timeframe: str = "15m"
    # horizon_bars=2 means 30 minutes lookahead for 15m candles
    horizon_bars: int = 2  # Changed from 1 to 2 for 30-minute window
    label_return_threshold: float = 0.0  # log-return threshold; raise to filter noise
    drop_last_n: int = 2  # Drop last N bars (matches horizon_bars)

    # Feature windows - expanded for better signal capture
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Additional lookback windows for multi-scale features
    lookback_bars: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    volatility_windows: Tuple[int, ...] = (4, 8, 16, 32, 64)

    # Leakage control for CV
    n_splits: int = 8  # Increased from 6
    purge_bars: int = 4  # Increased for 30-min horizon
    embargo_bars: int = 4

    # Modeling
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"
    use_xgboost: bool = True  # Use XGBoost if available

    # Default thresholds (will be optimized if enabled)
    p_long: float = 0.55
    p_short: float = 0.45
    allow_short: bool = True

    # Threshold optimization
    optimize_thresholds: bool = True
    p_long_grid: Tuple[float, float, float] = (0.52, 0.68, 0.02)
    p_short_grid: Tuple[float, float, float] = (0.32, 0.48, 0.02)
    threshold_objective: str = "net_expectancy"

    # Costs for objective/backtest (Kalshi has different fee structure)
    fee_bps: float = 0.0  # Kalshi doesn't charge per-contract fees
    slippage_bps: float = 1.0  # Bid-ask spread approximation

    # Regime gating parameters
    regime_trend_quantile: float = 0.65
    regime_vol_quantile: float = 0.7

    # Risk management & execution
    kill_switch_prob_gap: float = 0.03
    kill_switch_vol_z_limit: float = 3.0
    allow_kill_switch: bool = True
    position_size_max: float = 1.0
    position_size_min: float = 0.0
    position_size_scale: float = 5.0

    # Live inference
    min_history_bars: int = 150  # Increased for more features

    # Kalshi-specific settings
    kalshi_wager_type: str = "crypto_15m"  # For logging/tracking
    min_edge_for_trade: float = 0.03  # Minimum edge (prob - 0.5) to trade


@dataclass
class KalshiPolicyConfig:
    """Configuration bundle for the EV-based Kalshi execution policy."""

    fee_schedule: FeeSchedule = FeeSchedule()
    gate: GateConfig = GateConfig()
    sizing: SizingConfig = SizingConfig()
    is_maker: bool = False


# ----------------------------
# Data Providers
# ----------------------------

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


class DataProvider:
    """Abstract interface."""
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        raise NotImplementedError


class CSVProvider(DataProvider):
    """Loads OHLCV from CSV per symbol path mapping."""
    def __init__(self, symbol_to_path: Dict[str, str]):
        self.symbol_to_path = symbol_to_path

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        if symbol not in self.symbol_to_path:
            raise ValueError(f"CSVProvider: no path configured for symbol={symbol}")
        path = self.symbol_to_path[symbol]
        df = pd.read_csv(path)

        ts_col = None
        for c in ["timestamp", "time", "date", "datetime"]:
            if c in df.columns:
                ts_col = c
                break
        if ts_col is None:
            raise ValueError(f"{path}: missing timestamp column (timestamp/time/date/datetime).")

        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing required columns: {missing}")

        df = df[REQUIRED_COLS].astype(float)
        if limit is not None and limit > 0:
            df = df.iloc[-limit:]
        return df


class BinanceRESTProvider(DataProvider):
    """
    Public Binance REST klines with retry logic. No API key needed.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, quote: str = "USDT", session: Optional[requests.Session] = None,
                 timeout: int = 30, max_retries: int = 3):
        self.quote = quote
        self.sess = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries

    def _to_binance_symbol(self, symbol: str) -> str:
        s = symbol.upper()
        if s.endswith(self.quote):
            return s
        return f"{s}{self.quote}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        bsymbol = self._to_binance_symbol(symbol)
        params = {"symbol": bsymbol, "interval": timeframe, "limit": int(min(limit, 1500))}

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                r = self.sess.get(self.BASE_URL, params=params, timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                break
            except (requests.RequestException, Exception) as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        rows = []
        for k in data:
            open_time = pd.to_datetime(int(k[0]), unit="ms", utc=True)
            rows.append([open_time, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp").sort_index()

        # Fetch more data if needed
        if limit > 1500:
            all_dfs = [df]
            while len(pd.concat(all_dfs)) < limit:
                oldest = all_dfs[0].index[0]
                params["endTime"] = int(oldest.timestamp() * 1000) - 1
                try:
                    r = self.sess.get(self.BASE_URL, params=params, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                    if not data:
                        break
                    rows = []
                    for k in data:
                        open_time = pd.to_datetime(int(k[0]), unit="ms", utc=True)
                        rows.append([open_time, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])
                    new_df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    new_df = new_df.set_index("timestamp").sort_index()
                    all_dfs.insert(0, new_df)
                    time.sleep(0.2)  # Rate limiting
                except Exception:
                    break
            df = pd.concat(all_dfs).sort_index()
            df = df[~df.index.duplicated(keep='first')]

        return df.iloc[-limit:] if len(df) > limit else df


# ----------------------------
# Enhanced Feature Engineering
# ----------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    gain = up.ewm(span=period, adjust=False).mean()
    loss = down.ewm(span=period, adjust=False).mean()
    rs = gain / (loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def _stochastic_rsi(close: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
    rsi = _rsi(close, period)
    stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min() + 1e-10)
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.fillna(0.5), d.fillna(0.5)

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()

def _bollinger(close: pd.Series, period: int, k: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower

def _macd(close: pd.Series, fast: int, slow: int, signal: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    sig = _ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index - measures trend strength."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = _true_range(high, low, close)
    atr = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.fillna(25)

def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R momentum indicator."""
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr.fillna(-50)

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    return (np.sign(close.diff()) * volume).fillna(0).cumsum()

def _money_flow_index(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index - volume-weighted RSI."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    delta = typical_price.diff()
    positive_flow = money_flow.where(delta > 0, 0).rolling(period).sum()
    negative_flow = money_flow.where(delta < 0, 0).abs().rolling(period).sum()

    mfi = 100 - (100 / (1 + positive_flow / (negative_flow + 1e-10)))
    return mfi.fillna(50)

def build_features(df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    """Build comprehensive feature set for 30-minute prediction."""
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])

    features = {}

    # Basic returns at multiple scales
    logret = np.log(c).diff()
    features["logret"] = logret
    features["ret1"] = c.pct_change()

    for bars in cfg.lookback_bars:
        features[f"ret_{bars}"] = c.pct_change(bars)
        features[f"logret_{bars}"] = np.log(c).diff(bars)

    # Realized volatility at multiple windows
    for window in cfg.volatility_windows:
        features[f"rv_{window}"] = logret.rolling(window).std() * math.sqrt(window)
        # Volatility z-score
        rv = features[f"rv_{window}"]
        features[f"rv_z_{window}"] = (rv - rv.rolling(96).mean()) / (rv.rolling(96).std() + 1e-10)

    # Price momentum
    for span in [8, 21, 55, 89]:
        ema = _ema(c, span)
        features[f"ema_{span}_dist"] = (c - ema) / c

    # Trend features
    ema_8 = _ema(c, 8)
    ema_21 = _ema(c, 21)
    ema_55 = _ema(c, 55)
    features["trend_8_21"] = (ema_8 - ema_21) / c
    features["trend_21_55"] = (ema_21 - ema_55) / c
    features["trend_8_55"] = (ema_8 - ema_55) / c

    # RSI and Stochastic RSI
    features["rsi"] = _rsi(c, cfg.rsi_period)
    features["rsi_7"] = _rsi(c, 7)
    features["rsi_21"] = _rsi(c, 21)
    stoch_k, stoch_d = _stochastic_rsi(c)
    features["stoch_rsi_k"] = stoch_k
    features["stoch_rsi_d"] = stoch_d

    # ATR and volatility
    atr = _atr(h, l, c, cfg.atr_period)
    features["atrp"] = (atr / c).replace([np.inf, -np.inf], np.nan)
    features["atr_ratio"] = atr / atr.rolling(48).mean()

    # Bollinger Bands
    bb_ma, bb_up, bb_lo = _bollinger(c, cfg.bb_period, cfg.bb_k)
    features["bb_width"] = (bb_up - bb_lo) / (bb_ma.replace(0.0, np.nan))
    features["bb_pos"] = (c - bb_lo) / ((bb_up - bb_lo).replace(0.0, np.nan))

    # MACD
    macd, macd_sig, macd_hist = _macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    features["macd"] = macd / c * 100
    features["macd_sig"] = macd_sig / c * 100
    features["macd_hist"] = macd_hist / c * 100
    features["macd_cross"] = np.sign(macd - macd_sig)

    # ADX - trend strength
    features["adx"] = _adx(h, l, c, 14)
    features["adx_strong"] = (features["adx"] > 25).astype(float)

    # Williams %R
    features["williams_r"] = _williams_r(h, l, c, 14)

    # Money Flow Index
    features["mfi"] = _money_flow_index(h, l, c, v, 14)

    # Candlestick patterns
    body = c - o
    features["body"] = body / c
    features["body_abs"] = np.abs(body) / c
    features["range"] = (h - l) / c
    features["upper_wick"] = (h - np.maximum(o, c)) / c
    features["lower_wick"] = (np.minimum(o, c) - l) / c
    features["body_to_range"] = np.abs(body) / (h - l + 1e-10)

    # Doji pattern (small body relative to range)
    features["doji"] = (features["body_abs"] < features["range"] * 0.1).astype(float)

    # Volume features
    features["vol_z"] = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=0).replace(0.0, np.nan))
    features["vol_chg"] = v.pct_change()
    features["vol_ratio"] = v / v.rolling(24).mean()

    # OBV momentum
    obv = _obv(c, v)
    features["obv_slope"] = obv.diff(4) / (obv.rolling(48).std() + 1e-10)

    # VWAP distance
    typical = (h + l + c) / 3.0
    for window in [24, 48, 96]:
        vwap = (typical.mul(v).rolling(window).sum() / v.rolling(window).sum().replace(0.0, np.nan))
        features[f"vwap_{window}_dist"] = (c - vwap) / c

    # Price position in recent range
    for window in [24, 48, 96]:
        rolling_high = h.rolling(window).max()
        rolling_low = l.rolling(window).min()
        features[f"price_pos_{window}"] = (c - rolling_low) / (rolling_high - rolling_low + 1e-10)

    # Intraday seasonality
    idx = df.index
    hour = idx.hour
    minute = idx.minute
    dow = idx.dayofweek

    # Cyclical encoding
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["minute_sin"] = np.sin(2 * np.pi * (hour * 60 + minute) / 1440.0)
    features["minute_cos"] = np.cos(2 * np.pi * (hour * 60 + minute) / 1440.0)
    features["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Weekend effect
    features["is_weekend"] = (dow >= 5).astype(float)

    # Asian/European/US session indicators
    features["asian_session"] = ((hour >= 0) & (hour < 8)).astype(float)
    features["european_session"] = ((hour >= 7) & (hour < 16)).astype(float)
    features["us_session"] = ((hour >= 13) & (hour < 22)).astype(float)

    # Higher timeframe momentum (simulate 1H and 4H)
    c_1h = c.resample('1h').last().reindex(c.index, method='ffill')
    if len(c_1h.dropna()) > 20:
        features["ret_1h"] = c_1h.pct_change(4)  # 4 x 15m = 1H
        features["ret_4h"] = c_1h.pct_change(16)  # 16 x 15m = 4H

    # Autocorrelation features
    for lag in [1, 2, 4]:
        features[f"ret_autocorr_{lag}"] = logret.rolling(48).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
        )

    # Convert to DataFrame
    X = pd.DataFrame(features, index=df.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)

    return X


def build_labels(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """Build labels for 30-minute lookahead."""
    c = df["close"]
    # Forward return over horizon_bars (2 bars = 30 minutes for 15m candles)
    fwd = np.log(c.shift(-cfg.horizon_bars) / c)
    y = (fwd > cfg.label_return_threshold).astype(float)
    if cfg.drop_last_n > 0:
        y.iloc[-cfg.drop_last_n:] = np.nan
    return y.dropna().astype(int)


def build_forward_returns(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    """Get actual forward returns for validation."""
    c = df["close"]
    fwd = (c.shift(-cfg.horizon_bars) / c) - 1
    return fwd


# ----------------------------
# Purged / Embargoed Time Series Split
# ----------------------------

class PurgedTimeSeriesSplit:
    """
    Time-series split with purge and embargo for 30-minute horizon.
    """

    def __init__(self, n_splits: int, purge_bars: int = 0, embargo_bars: int = 0):
        self.n_splits = int(n_splits)
        self.purge_bars = int(purge_bars)
        self.embargo_bars = int(embargo_bars)

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")

        fold_size = n // (self.n_splits + 1)
        if fold_size < 10:
            raise ValueError("Not enough data for the requested number of splits.")

        for i in range(1, self.n_splits + 1):
            test_start = i * fold_size
            test_end = min(test_start + fold_size, n)

            train_end = max(0, test_start - self.purge_bars)
            train_idx = np.arange(0, train_end)

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < 50 or len(test_idx) < 20:
                continue

            yield train_idx, test_idx


# ----------------------------
# Modeling with XGBoost support
# ----------------------------

def make_tree_pipeline(use_xgboost: bool = True) -> Pipeline:
    if use_xgboost and XGBOOST_AVAILABLE:
        return Pipeline([
            ("scaler", RobustScaler()),
            ("clf", xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.05,
                n_estimators=500,
                subsample=0.85,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        return Pipeline([
            ("scaler", RobustScaler()),
            ("clf", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.05,
                max_iter=500,
                l2_regularization=1e-3,
                random_state=42
            ))
        ])


def make_logit_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", RobustScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced"
        ))
    ])


def _fit_with_time_series_calibration(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    cfg: MLConfig,
    pipeline_factory,
) -> Tuple[Pipeline, Optional[CalibratedClassifierCV]]:
    pipeline = pipeline_factory()

    if not cfg.calibrate_probabilities:
        pipeline.fit(X_tr, y_tr)
        return pipeline, None

    # Tail-split calibration to avoid leakage
    cut = int(len(X_tr) * 0.8)
    cut = max(1, min(len(X_tr) - 1, cut))
    X_fit, y_fit = X_tr.iloc[:cut], y_tr.iloc[:cut]
    X_cal, y_cal = X_tr.iloc[cut:], y_tr.iloc[cut:]

    if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
        pipeline.fit(X_tr, y_tr)
        return pipeline, None

    pipeline.fit(X_fit, y_fit)
    # For sklearn >= 1.3, use cv=None with pre-fitted estimator
    try:
        calibrator = CalibratedClassifierCV(pipeline, method=cfg.calibration_method, cv="prefit")
        calibrator.fit(X_cal, y_cal)
    except (TypeError, ValueError):
        # Fallback for newer sklearn versions
        from sklearn.calibration import _SigmoidCalibration, IsotonicRegression
        # Simple calibration without CalibratedClassifierCV
        proba = pipeline.predict_proba(X_cal)[:, 1]
        if cfg.calibration_method == "isotonic":
            calibrator_model = IsotonicRegression(out_of_bounds='clip')
        else:
            calibrator_model = _SigmoidCalibration()
        calibrator_model.fit(proba, y_cal)
        # Wrap in a simple object
        class SimpleCalibrator:
            def __init__(self, base_estimator, calibrator_model):
                self.base_estimator = base_estimator
                self.calibrator_model = calibrator_model
            def predict_proba(self, X):
                proba = self.base_estimator.predict_proba(X)[:, 1]
                calibrated = self.calibrator_model.predict(proba)
                calibrated = np.clip(calibrated, 0.01, 0.99)
                return np.column_stack([1 - calibrated, calibrated])
        calibrator = SimpleCalibrator(pipeline, calibrator_model)
    return pipeline, calibrator


def _predict_proba(model: Pipeline, calibrator: Optional[CalibratedClassifierCV], X: pd.DataFrame) -> np.ndarray:
    if calibrator is not None:
        return calibrator.predict_proba(X)[:, 1]
    return model.predict_proba(X)[:, 1]


REGIME_NAMES: Tuple[str, ...] = ("chop", "trend", "high_vol")


def _build_meta_features(
    probs_tree: np.ndarray,
    probs_logit: np.ndarray,
    X: pd.DataFrame,
) -> np.ndarray:
    trend_abs = np.nan_to_num(np.abs(X["trend_8_21"]).to_numpy(dtype=float), nan=0.0)
    rv = np.nan_to_num(X["rv_16"].to_numpy(dtype=float) if "rv_16" in X.columns else np.zeros(len(X)), nan=0.0)
    adx = np.nan_to_num(X["adx"].to_numpy(dtype=float) if "adx" in X.columns else np.zeros(len(X)), nan=0.0)
    rsi = np.nan_to_num(X["rsi"].to_numpy(dtype=float) if "rsi" in X.columns else np.zeros(len(X)), nan=0.0)

    return np.column_stack([
        probs_tree,
        probs_logit,
        trend_abs,
        rv,
        adx / 100,  # Normalize
        (rsi - 50) / 50,  # Center and normalize
    ])


def _compute_regime_thresholds(X: pd.DataFrame, cfg: MLConfig) -> Dict[str, float]:
    trend_metric = np.abs(X["trend_8_21"]).fillna(0.0)
    vol_col = "rv_16" if "rv_16" in X.columns else "rv_4"
    vol_metric = np.abs(X[vol_col]).fillna(0.0) if vol_col in X.columns else pd.Series(0, index=X.index)
    return {
        "trend": float(trend_metric.quantile(cfg.regime_trend_quantile)),
        "high_vol": float(vol_metric.quantile(cfg.regime_vol_quantile)),
    }


def _assign_regimes(X: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    regimes = pd.Series("chop", index=X.index, dtype=object)
    trend_metric = np.abs(X["trend_8_21"]).fillna(0.0)
    vol_col = "rv_16" if "rv_16" in X.columns else "rv_4"
    vol_metric = np.abs(X[vol_col]).fillna(0.0) if vol_col in X.columns else pd.Series(0, index=X.index)

    vol_thr = thresholds.get("high_vol", np.inf)
    trend_thr = thresholds.get("trend", np.inf)

    vol_mask = vol_metric >= vol_thr
    regimes.loc[vol_mask] = "high_vol"

    trend_mask = (trend_metric >= trend_thr) & (~vol_mask)
    regimes.loc[trend_mask] = "trend"
    return regimes


def make_meta_model() -> LogisticRegression:
    return LogisticRegression(
        C=0.5,
        penalty="l2",
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced"
    )


# ----------------------------
# Threshold objective
# ----------------------------

def _net_expectancy_from_probs(
    probs: np.ndarray,
    df_prices: pd.DataFrame,
    p_long: float,
    p_short: float,
    cfg: MLConfig,
) -> float:
    """Expectancy for 30-minute lookahead."""
    pos = np.zeros_like(probs, dtype=float)
    pos[probs >= p_long] = 1.0
    if cfg.allow_short:
        pos[probs <= p_short] = -1.0

    close = df_prices["close"].values
    # Return over horizon_bars
    ret_cc = np.zeros_like(close, dtype=float)
    horizon = cfg.horizon_bars
    if len(close) > horizon:
        ret_cc[:-horizon] = (close[horizon:] / close[:-horizon]) - 1.0
    ret_cc[-horizon:] = 0.0

    gross = pos * ret_cc[: len(pos)]

    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    turnover = np.zeros_like(pos)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    costs = turnover * cost_rate

    net = gross - costs
    return float(np.nanmean(net))


def _sharpe_from_probs(
    probs: np.ndarray,
    df_prices: pd.DataFrame,
    p_long: float,
    p_short: float,
    cfg: MLConfig,
) -> float:
    pos = np.zeros_like(probs, dtype=float)
    pos[probs >= p_long] = 1.0
    if cfg.allow_short:
        pos[probs <= p_short] = -1.0

    close = df_prices["close"].values
    horizon = cfg.horizon_bars
    ret_cc = np.zeros_like(close, dtype=float)
    if len(close) > horizon:
        ret_cc[:-horizon] = (close[horizon:] / close[:-horizon]) - 1.0
    ret_cc[-horizon:] = 0.0

    gross = pos * ret_cc[: len(pos)]
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    turnover = np.zeros_like(pos)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turnover * cost_rate

    mu = np.nanmean(net)
    sd = np.nanstd(net)
    if sd <= 1e-12:
        return -1e9
    # Annualize for 15m bars (96 bars/day, 365 days)
    return float(mu / sd * np.sqrt(96 * 365))


def optimize_thresholds_cv(
    X: pd.DataFrame,
    y: pd.Series,
    df_prices: pd.DataFrame,
    cfg: MLConfig,
) -> Tuple[float, float]:
    """Grid search p_long/p_short over CV test folds."""
    splitter = PurgedTimeSeriesSplit(cfg.n_splits, cfg.purge_bars, cfg.embargo_bars)

    pL_start, pL_stop, pL_step = cfg.p_long_grid
    pS_start, pS_stop, pS_step = cfg.p_short_grid
    pL_vals = np.round(np.arange(pL_start, pL_stop + 1e-9, pL_step), 4)
    pS_vals = np.round(np.arange(pS_start, pS_stop + 1e-9, pS_step), 4)

    objective_fn = _net_expectancy_from_probs if cfg.threshold_objective == "net_expectancy" else _sharpe_from_probs

    best_score = -1e18
    best = (cfg.p_long, cfg.p_short)

    for p_long in pL_vals:
        for p_short in pS_vals:
            if p_short >= p_long:
                continue

            scores = []
            for tr_idx, te_idx in splitter.split(X):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) == 0:
                    continue

                model, cal = _fit_with_time_series_calibration(
                    X_tr, y_tr, cfg, lambda: make_tree_pipeline(cfg.use_xgboost)
                )
                probs = _predict_proba(model, cal, X_te)

                df_te = df_prices.loc[X_te.index]
                sc = objective_fn(probs, df_te, p_long, p_short, cfg)
                scores.append(sc)

            if not scores:
                continue

            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best = (float(p_long), float(p_short))

    return best


# ----------------------------
# Validation and Reporting
# ----------------------------

def eval_cv_metrics(X: pd.DataFrame, y: pd.Series, cfg: MLConfig) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics."""
    splitter = PurgedTimeSeriesSplit(cfg.n_splits, cfg.purge_bars, cfg.embargo_bars)

    metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc", "brier", "logloss", "directional_acc"]}

    for tr_idx, te_idx in splitter.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 1:
            continue

        tree_model, tree_cal = _fit_with_time_series_calibration(
            X_tr, y_tr, cfg, lambda: make_tree_pipeline(cfg.use_xgboost)
        )
        logit_model, logit_cal = _fit_with_time_series_calibration(X_tr, y_tr, cfg, make_logit_pipeline)

        probs_tree_tr = _predict_proba(tree_model, tree_cal, X_tr)
        probs_logit_tr = _predict_proba(logit_model, logit_cal, X_tr)
        probs_tree_te = _predict_proba(tree_model, tree_cal, X_te)
        probs_logit_te = _predict_proba(logit_model, logit_cal, X_te)

        meta_tr = _build_meta_features(probs_tree_tr, probs_logit_tr, X_tr)
        meta_te = _build_meta_features(probs_tree_te, probs_logit_te, X_te)

        if len(np.unique(y_tr)) < 2:
            proba = probs_tree_te
        else:
            meta_clf = make_meta_model()
            meta_clf.fit(meta_tr, y_tr)
            proba = meta_clf.predict_proba(meta_te)[:, 1]

        pred = (proba >= 0.5).astype(int)

        metrics["accuracy"].append(accuracy_score(y_te, pred))
        metrics["precision"].append(precision_score(y_te, pred, zero_division=0))
        metrics["recall"].append(recall_score(y_te, pred, zero_division=0))
        metrics["f1"].append(f1_score(y_te, pred, zero_division=0))
        metrics["brier"].append(brier_score_loss(y_te, proba))

        # Directional accuracy (for trading)
        metrics["directional_acc"].append(accuracy_score(y_te, pred))

        if len(np.unique(y_te)) == 2:
            metrics["auc"].append(roc_auc_score(y_te, proba))
            metrics["logloss"].append(log_loss(y_te, np.vstack([1 - proba, proba]).T, labels=[0, 1]))
        else:
            metrics["auc"].append(np.nan)
            metrics["logloss"].append(np.nan)

    return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in metrics.items()}


def validate_lookahead_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    df_prices: pd.DataFrame,
    cfg: MLConfig,
    model: "AssetPredictor"
) -> Dict[str, Any]:
    """Validate that lookahead predictions are accurate."""
    # Get forward returns
    fwd_returns = build_forward_returns(df_prices, cfg)
    fwd_returns = fwd_returns.loc[y.index]

    # Get predictions
    features = X.loc[y.index]

    results = {
        "total_samples": len(y),
        "up_samples": int(y.sum()),
        "down_samples": int((1 - y).sum()),
        "actual_up_pct": float(y.mean() * 100),
    }

    # Validate predictions on last portion of data
    test_start = int(len(features) * 0.8)
    test_features = features.iloc[test_start:]
    test_y = y.iloc[test_start:]
    test_returns = fwd_returns.iloc[test_start:]

    predictions = []
    actuals = []

    for i in range(len(test_features)):
        try:
            # Build feature row
            row_idx = test_features.index[i]
            hist_end = df_prices.index.get_loc(row_idx) + 1
            hist_df = df_prices.iloc[:hist_end]

            if len(hist_df) < cfg.min_history_bars:
                continue

            pred = model.predict_proba_up(hist_df)
            predictions.append(pred)
            actuals.append(test_y.iloc[i])
        except Exception:
            continue

    if len(predictions) < 10:
        results["validation_error"] = "Not enough predictions"
        return results

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Calculate validation metrics
    pred_binary = (predictions >= 0.5).astype(int)

    results.update({
        "validation_samples": len(predictions),
        "validation_accuracy": float(accuracy_score(actuals, pred_binary) * 100),
        "validation_auc": float(roc_auc_score(actuals, predictions)) if len(np.unique(actuals)) == 2 else None,
        "validation_brier": float(brier_score_loss(actuals, predictions)),
        "mean_prediction": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    })

    # High confidence predictions
    high_conf_mask = np.abs(predictions - 0.5) > 0.1
    if high_conf_mask.sum() > 10:
        results["high_conf_accuracy"] = float(
            accuracy_score(actuals[high_conf_mask], pred_binary[high_conf_mask]) * 100
        )
        results["high_conf_samples"] = int(high_conf_mask.sum())

    return results


# ----------------------------
# Asset model wrapper
# ----------------------------

class AssetPredictor:
    def __init__(self, cfg: MLConfig):
        self.cfg = cfg
        self.tree_pipeline: Optional[Pipeline] = None
        self.tree_calibrator: Optional[CalibratedClassifierCV] = None
        self.logit_pipeline: Optional[Pipeline] = None
        self.logit_calibrator: Optional[CalibratedClassifierCV] = None
        self.meta_model: Optional[LogisticRegression] = None
        self.feature_columns: Optional[List[str]] = None

        self.global_thresholds: Tuple[float, float] = (cfg.p_long, cfg.p_short)
        self.regime_thresholds: Dict[str, Tuple[float, float]] = {reg: self.global_thresholds for reg in REGIME_NAMES}
        self.regime_params: Dict[str, float] = {}

    def _ensure_ready(self) -> None:
        if self.tree_pipeline is None or self.logit_pipeline is None:
            raise RuntimeError("Model not fitted/loaded.")

    def _classify_regime_from_features(self, feature_row: pd.Series) -> str:
        if not self.regime_params:
            return "chop"
        trend_val = float(abs(feature_row.get("trend_8_21", 0.0)))
        vol_col = "rv_16" if "rv_16" in feature_row.index else "rv_4"
        vol_val = float(abs(feature_row.get(vol_col, 0.0)))
        vol_thr = self.regime_params.get("high_vol", float("inf"))
        trend_thr = self.regime_params.get("trend", float("inf"))
        if vol_val >= vol_thr:
            return "high_vol"
        if trend_val >= trend_thr:
            return "trend"
        return "chop"

    def fit(self, df: pd.DataFrame) -> Dict[str, Any]:
        X = build_features(df, self.cfg)
        y = build_labels(df, self.cfg)
        X = X.loc[y.index]
        df = df.loc[X.index]

        if len(y) < max(200, self.cfg.n_splits * 50):
            raise ValueError("Not enough samples to train the ensemble model reliably.")

        self.feature_columns = list(X.columns)

        print(f"Training with {len(y)} samples, {len(self.feature_columns)} features")
        print(f"Class distribution: UP={y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")

        report = eval_cv_metrics(X, y, self.cfg)

        self.regime_params = _compute_regime_thresholds(X, self.cfg)
        regimes = _assign_regimes(X, self.regime_params)
        report["regime_counts"] = {reg: int((regimes == reg).sum()) for reg in REGIME_NAMES}

        if self.cfg.optimize_thresholds:
            print("Optimizing thresholds...")
            try:
                self.global_thresholds = optimize_thresholds_cv(X, y, df, self.cfg)
            except ValueError:
                self.global_thresholds = (self.cfg.p_long, self.cfg.p_short)
        else:
            self.global_thresholds = (self.cfg.p_long, self.cfg.p_short)

        report["p_long"] = float(self.global_thresholds[0])
        report["p_short"] = float(self.global_thresholds[1])

        # Regime-specific thresholds
        thresholds_by_regime: Dict[str, Tuple[float, float]] = {}
        for regime in REGIME_NAMES:
            mask = regimes == regime
            if self.cfg.optimize_thresholds and mask.sum() >= max(120, self.cfg.n_splits * 30):
                try:
                    thresholds_by_regime[regime] = optimize_thresholds_cv(
                        X.loc[mask],
                        y.loc[mask],
                        df.loc[mask],
                        self.cfg,
                    )
                except ValueError:
                    thresholds_by_regime[regime] = self.global_thresholds
            else:
                thresholds_by_regime[regime] = self.global_thresholds

        self.regime_thresholds = thresholds_by_regime
        report["regime_thresholds"] = {
            reg: {"p_long": float(vals[0]), "p_short": float(vals[1])}
            for reg, vals in thresholds_by_regime.items()
        }

        print("Training final models...")
        self.tree_pipeline, self.tree_calibrator = _fit_with_time_series_calibration(
            X, y, self.cfg, lambda: make_tree_pipeline(self.cfg.use_xgboost)
        )
        self.logit_pipeline, self.logit_calibrator = _fit_with_time_series_calibration(
            X, y, self.cfg, make_logit_pipeline
        )

        probs_tree_full = _predict_proba(self.tree_pipeline, self.tree_calibrator, X)
        probs_logit_full = _predict_proba(self.logit_pipeline, self.logit_calibrator, X)
        meta_feat_full = _build_meta_features(probs_tree_full, probs_logit_full, X)

        if len(np.unique(y)) >= 2:
            self.meta_model = make_meta_model()
            self.meta_model.fit(meta_feat_full, y)
        else:
            self.meta_model = None

        report["ensemble_ready"] = self.meta_model is not None

        # Validate lookahead predictions
        print("Validating lookahead predictions...")
        validation = validate_lookahead_predictions(X, y, df, self.cfg, self)
        report["lookahead_validation"] = validation

        return report

    def predict_components(self, df: pd.DataFrame) -> Dict[str, Any]:
        self._ensure_ready()
        if len(df) < self.cfg.min_history_bars:
            raise ValueError(f"Need at least {self.cfg.min_history_bars} bars for stable inference.")

        features = build_features(df, self.cfg)
        latest = features.iloc[[-1]]
        latest_row = latest.iloc[0]

        prob_tree = float(_predict_proba(self.tree_pipeline, self.tree_calibrator, latest)[0])
        prob_logit = float(_predict_proba(self.logit_pipeline, self.logit_calibrator, latest)[0])
        meta_input = _build_meta_features(
            np.array([prob_tree]),
            np.array([prob_logit]),
            latest,
        )
        if self.meta_model is not None:
            prob_final = float(self.meta_model.predict_proba(meta_input)[0, 1])
        else:
            prob_final = float(0.5 * (prob_tree + prob_logit))

        regime = self._classify_regime_from_features(latest_row)
        thresholds = self.regime_thresholds.get(regime, self.global_thresholds)
        vol_z = float(latest_row.get("vol_z", 0.0))

        return {
            "prob_tree": prob_tree,
            "prob_logit": prob_logit,
            "probability_up": prob_final,
            "regime": regime,
            "thresholds": {"p_long": float(thresholds[0]), "p_short": float(thresholds[1])},
            "vol_z": vol_z,
            "features_row": latest_row,
            "timestamp": df.index[-1],
            "current_price": float(df["close"].iloc[-1]),
        }

    def predict_proba_up(self, df: pd.DataFrame) -> float:
        return self.predict_components(df)["probability_up"]

    def signal(self, df: pd.DataFrame) -> Tuple[str, float]:
        data = self.predict_components(df)
        p_long = data["thresholds"]["p_long"]
        p_short = data["thresholds"]["p_short"]
        prob = data["probability_up"]

        if prob >= p_long:
            return "LONG", prob
        if self.cfg.allow_short and prob <= p_short:
            return "SHORT", prob
        return "FLAT", prob

    def decide_trade(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trade decision with Kalshi-specific outputs."""
        components = self.predict_components(df)
        prob = components["probability_up"]
        p_long = components["thresholds"]["p_long"]
        p_short = components["thresholds"]["p_short"]
        vol_z = components["vol_z"]

        action = "FLAT"
        if prob >= p_long:
            action = "LONG"
        elif self.cfg.allow_short and prob <= p_short:
            action = "SHORT"

        confidence = abs(prob - 0.5)
        edge = confidence  # For Kalshi, edge is distance from 50%

        kill_switch = False
        if self.cfg.allow_kill_switch:
            if not np.isfinite(prob) or confidence < self.cfg.kill_switch_prob_gap:
                kill_switch = True
            if abs(vol_z) >= self.cfg.kill_switch_vol_z_limit:
                kill_switch = True

        position_size = 0.0
        if not kill_switch and action != "FLAT":
            raw_size = confidence * self.cfg.position_size_scale
            position_size = min(self.cfg.position_size_max, max(self.cfg.position_size_min, raw_size))
            if action == "SHORT":
                position_size = -position_size
        elif kill_switch:
            action = "KILL"

        # Kalshi-specific: suggested wager
        kalshi_wager = {
            "side": "YES" if prob > 0.5 else "NO",
            "fair_price": prob if prob > 0.5 else (1 - prob),
            "edge": edge,
            "recommended": edge >= self.cfg.min_edge_for_trade and not kill_switch,
        }

        components.pop("features_row", None)
        components.update({
            "action": action,
            "probability_up": prob,
            "confidence": confidence,
            "edge": edge,
            "position_size": position_size,
            "kill_switch": kill_switch,
            "kalshi_wager": kalshi_wager,
            "horizon_minutes": self.cfg.horizon_bars * 15,  # 30 minutes for horizon_bars=2
        })
        return components

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        dump(self.tree_pipeline, os.path.join(path, "tree_pipeline.joblib"))
        dump(self.tree_calibrator, os.path.join(path, "tree_calibrator.joblib"))
        dump(self.logit_pipeline, os.path.join(path, "logit_pipeline.joblib"))
        dump(self.logit_calibrator, os.path.join(path, "logit_calibrator.joblib"))
        meta_model_path = os.path.join(path, "meta_model.joblib")
        if self.meta_model is not None:
            dump(self.meta_model, meta_model_path)
        elif os.path.exists(meta_model_path):
            os.remove(meta_model_path)

        meta = {
            "cfg": asdict(self.cfg),
            "feature_columns": self.feature_columns,
            "global_thresholds": list(self.global_thresholds),
            "regime_thresholds": {k: list(v) for k, v in self.regime_thresholds.items()},
            "regime_params": self.regime_params,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AssetPredictor":
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        # Handle tuple fields
        cfg_dict = meta["cfg"]
        for key in ["lookback_bars", "volatility_windows", "p_long_grid", "p_short_grid"]:
            if key in cfg_dict and isinstance(cfg_dict[key], list):
                cfg_dict[key] = tuple(cfg_dict[key])

        cfg = MLConfig(**cfg_dict)
        obj = cls(cfg)
        obj.tree_pipeline = load(os.path.join(path, "tree_pipeline.joblib"))
        try:
            obj.tree_calibrator = load(os.path.join(path, "tree_calibrator.joblib"))
        except FileNotFoundError:
            obj.tree_calibrator = None
        obj.logit_pipeline = load(os.path.join(path, "logit_pipeline.joblib"))
        try:
            obj.logit_calibrator = load(os.path.join(path, "logit_calibrator.joblib"))
        except FileNotFoundError:
            obj.logit_calibrator = None
        meta_model_path = os.path.join(path, "meta_model.joblib")
        obj.meta_model = load(meta_model_path) if os.path.exists(meta_model_path) else None

        obj.feature_columns = meta.get("feature_columns")
        obj.global_thresholds = tuple(meta.get("global_thresholds", [cfg.p_long, cfg.p_short]))
        obj.regime_thresholds = {
            k: tuple(v) for k, v in meta.get("regime_thresholds", {}).items()
        } or {reg: obj.global_thresholds for reg in REGIME_NAMES}
        obj.regime_params = {k: float(v) for k, v in meta.get("regime_params", {}).items()}
        return obj


# ----------------------------
# Multi-asset manager + Live Engine
# ----------------------------


def _ev_to_dict(ev: Optional[TradeEV]) -> Optional[Dict[str, float]]:
    if ev is None:
        return None
    return {
        "side": ev.side,
        "price": float(ev.price),
        "fee": float(ev.fee),
        "ev": float(ev.ev),
        "edge_vs_price": float(ev.edge_vs_price),
    }


class MultiAssetML:
    def __init__(self, cfg: MLConfig, provider: DataProvider):
        self.cfg = cfg
        self.provider = provider
        self.models: Dict[str, AssetPredictor] = {}

    def train(self, symbols: List[str], limit: int = 5000) -> Dict[str, Dict[str, Any]]:
        reports: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            print(f"\n{'='*60}")
            print(f"Training model for {sym}")
            print(f"{'='*60}")

            df = self.provider.fetch_ohlcv(sym, self.cfg.timeframe, limit=limit)
            df = df.dropna()
            if df.empty:
                raise ValueError(f"No data returned for symbol={sym} when training.")

            print(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")

            m = AssetPredictor(self.cfg)
            rep = m.fit(df)
            self.models[sym] = m
            reports[sym] = rep
        return reports

    def save(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        cfg_dict = asdict(self.cfg)
        # Convert tuples to lists for JSON
        for key in ["lookback_bars", "volatility_windows", "p_long_grid", "p_short_grid"]:
            if key in cfg_dict:
                cfg_dict[key] = list(cfg_dict[key])

        with open(os.path.join(root, "cfg.json"), "w") as f:
            json.dump(cfg_dict, f, indent=2)

        for sym, model in self.models.items():
            model.save(os.path.join(root, sym))

    @classmethod
    def load(cls, root: str, provider: DataProvider) -> "MultiAssetML":
        with open(os.path.join(root, "cfg.json"), "r") as f:
            cfg_dict = json.load(f)

        # Convert lists back to tuples
        for key in ["lookback_bars", "volatility_windows", "p_long_grid", "p_short_grid"]:
            if key in cfg_dict and isinstance(cfg_dict[key], list):
                cfg_dict[key] = tuple(cfg_dict[key])

        cfg = MLConfig(**cfg_dict)
        obj = cls(cfg, provider)
        for sym in os.listdir(root):
            sym_path = os.path.join(root, sym)
            if os.path.isdir(sym_path) and os.path.exists(os.path.join(sym_path, "meta.json")):
                obj.models[sym] = AssetPredictor.load(sym_path)
        return obj

    def decide_trade(self, symbol: str, limit: int = 500) -> Dict[str, Any]:
        if symbol not in self.models:
            raise ValueError(f"No model loaded/trained for symbol={symbol}")

        df = self.provider.fetch_ohlcv(symbol, self.cfg.timeframe, limit=limit).dropna()
        if df.empty:
            raise ValueError(f"No data returned for symbol={symbol} when generating trade signal.")
        trade = self.models[symbol].decide_trade(df)
        trade.update({
            "symbol": symbol,
            "timeframe": self.cfg.timeframe,
            "timestamp_utc": str(df.index[-1]),
        })
        return trade

    def decide_trade_with_policy(
        self,
        symbol: str,
        book: TopOfBook,
        seconds_to_expiry: int,
        bankroll: float,
        *,
        limit: int = 500,
        policy_cfg: Optional[KalshiPolicyConfig] = None,
    ) -> Dict[str, Any]:
        if symbol not in self.models:
            raise ValueError(f"No model loaded/trained for symbol={symbol}")

        df = self.provider.fetch_ohlcv(symbol, self.cfg.timeframe, limit=limit).dropna()
        if df.empty:
            raise ValueError(f"No data returned for symbol={symbol} when generating trade signal.")

        components = self.models[symbol].predict_components(df)
        cfg = policy_cfg or KalshiPolicyConfig()
        decision: PolicyTradeDecision = policy_decide_trade(
            p_yes=float(components["probability_up"]),
            book=book,
            t_left_seconds=int(seconds_to_expiry),
            bankroll=float(bankroll),
            is_maker=bool(cfg.is_maker),
            fee_schedule=cfg.fee_schedule,
            gate_cfg=cfg.gate,
            size_cfg=cfg.sizing,
        )

        return {
            "symbol": symbol,
            "timeframe": self.cfg.timeframe,
            "timestamp_utc": str(df.index[-1]),
            "probability_up": float(components["probability_up"]),
            "prob_tree": float(components["prob_tree"]),
            "prob_logit": float(components["prob_logit"]),
            "regime": components["regime"],
            "thresholds": components["thresholds"],
            "vol_z": float(components["vol_z"]),
            "policy_action": decision.action,
            "policy_reason": decision.reason,
            "policy_contracts": int(decision.contracts),
            "policy_price": float(decision.price) if decision.price is not None else None,
            "policy_ev": _ev_to_dict(decision.ev),
        }

    def predict_and_signal(self, symbol: str, limit: int = 500) -> Dict[str, Any]:
        """Returns condensed live signal information for Kalshi wagers."""
        trade = self.decide_trade(symbol, limit=limit)
        thresholds = trade.get("thresholds", {})
        kalshi = trade.get("kalshi_wager", {})

        return {
            "symbol": trade["symbol"],
            "timeframe": trade["timeframe"],
            "timestamp_utc": trade["timestamp_utc"],
            "horizon_minutes": trade.get("horizon_minutes", 30),
            "p_up": float(trade.get("probability_up", np.nan)),
            "signal": trade.get("action"),
            "p_long": float(thresholds.get("p_long", self.cfg.p_long)),
            "p_short": float(thresholds.get("p_short", self.cfg.p_short)),
            "regime": trade.get("regime"),
            "confidence": float(trade.get("confidence", 0.0)),
            "edge": float(trade.get("edge", 0.0)),
            "position_size": float(trade.get("position_size", 0.0)),
            "kill_switch": bool(trade.get("kill_switch", False)),
            "prob_tree": float(trade.get("prob_tree", np.nan)),
            "prob_logit": float(trade.get("prob_logit", np.nan)),
            "vol_z": float(trade.get("vol_z", 0.0)),
            "current_price": float(trade.get("current_price", 0.0)),
            # Kalshi-specific
            "kalshi_side": kalshi.get("side", "UNKNOWN"),
            "kalshi_fair_price": float(kalshi.get("fair_price", 0.5)),
            "kalshi_recommended": bool(kalshi.get("recommended", False)),
        }


# ----------------------------
# CLI
# ----------------------------

def _print_reports(reports: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "="*80)
    print("TRAINING RESULTS (30-MINUTE LOOKAHEAD)")
    print("="*80)

    for sym, rep in reports.items():
        print(f"\n{sym}")
        print("-" * 40)

        # Core metrics
        for k in ["accuracy", "precision", "recall", "f1", "auc", "brier", "logloss"]:
            if k in rep:
                v = rep[k]
                if isinstance(v, float):
                    print(f"  {k:15s}: {v:.4f}")

        # Thresholds
        print(f"\n  Thresholds:")
        print(f"    p_long: {rep.get('p_long', 0.55):.2f}")
        print(f"    p_short: {rep.get('p_short', 0.45):.2f}")

        # Regime info
        if "regime_counts" in rep:
            print(f"\n  Regime Distribution:")
            for regime, count in rep["regime_counts"].items():
                print(f"    {regime}: {count}")

        if "regime_thresholds" in rep:
            print(f"\n  Regime Thresholds:")
            for regime, thr in rep["regime_thresholds"].items():
                print(f"    {regime}: p_long={thr['p_long']:.2f}, p_short={thr['p_short']:.2f}")

        # Lookahead validation
        if "lookahead_validation" in rep:
            val = rep["lookahead_validation"]
            print(f"\n  Lookahead Validation:")
            print(f"    Validation Accuracy: {val.get('validation_accuracy', 0):.1f}%")
            if val.get("validation_auc"):
                print(f"    Validation AUC: {val.get('validation_auc', 0):.4f}")
            if val.get("high_conf_accuracy"):
                print(f"    High-Conf Accuracy: {val.get('high_conf_accuracy', 0):.1f}% ({val.get('high_conf_samples', 0)} samples)")


def main():
    ap = argparse.ArgumentParser(description="Crypto ML for Kalshi 15m/30m wagers")
    ap.add_argument("--mode", choices=["train_csv", "train_binance", "signal_csv", "signal_binance"], required=True)
    ap.add_argument("--model_dir", type=str, default="models_30m_kalshi")
    ap.add_argument("--symbols", type=str, default="BTC,ETH,SOL")

    # CSV mode
    ap.add_argument("--btc_csv", type=str, default="")
    ap.add_argument("--eth_csv", type=str, default="")
    ap.add_argument("--sol_csv", type=str, default="")

    # Binance mode
    ap.add_argument("--quote", type=str, default="USDT")
    ap.add_argument("--limit", type=int, default=5000)

    # Horizon
    ap.add_argument("--horizon", type=int, default=2, help="Lookahead bars (2=30min for 15m candles)")

    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    cfg = MLConfig(horizon_bars=args.horizon)
    print(f"Configuration: {args.horizon * 15}-minute lookahead")

    if args.mode == "train_csv":
        mapping = {}
        if args.btc_csv: mapping["BTC"] = args.btc_csv
        if args.eth_csv: mapping["ETH"] = args.eth_csv
        if args.sol_csv: mapping["SOL"] = args.sol_csv
        if not mapping:
            raise SystemExit("train_csv requires --btc_csv/--eth_csv/--sol_csv")
        provider = CSVProvider(mapping)

        mam = MultiAssetML(cfg, provider)
        reports = mam.train(symbols=[s for s in symbols if s in mapping], limit=args.limit)
        _print_reports(reports)
        mam.save(args.model_dir)
        print(f"\nModel saved to: {args.model_dir}")

    elif args.mode == "train_binance":
        provider = BinanceRESTProvider(quote=args.quote)
        mam = MultiAssetML(cfg, provider)
        reports = mam.train(symbols=symbols, limit=args.limit)
        _print_reports(reports)
        mam.save(args.model_dir)
        print(f"\nModel saved to: {args.model_dir}")

    elif args.mode in ("signal_csv", "signal_binance"):
        if args.mode == "signal_csv":
            mapping = {}
            if args.btc_csv: mapping["BTC"] = args.btc_csv
            if args.eth_csv: mapping["ETH"] = args.eth_csv
            if args.sol_csv: mapping["SOL"] = args.sol_csv
            if not mapping:
                raise SystemExit("signal_csv requires --btc_csv/--eth_csv/--sol_csv")
            provider = CSVProvider(mapping)
        else:
            provider = BinanceRESTProvider(quote=args.quote)

        mam = MultiAssetML.load(args.model_dir, provider)

        print("\n" + "="*80)
        print("LIVE SIGNALS")
        print("="*80)

        for sym in symbols:
            out = mam.predict_and_signal(sym, limit=min(args.limit, 1000))
            print(f"\n{sym} @ {out['timestamp_utc']}")
            print(f"  Horizon: {out['horizon_minutes']} minutes")
            print(f"  Probability UP: {out['p_up']*100:.1f}%")
            print(f"  Signal: {out['signal']}")
            print(f"  Regime: {out['regime']}")
            print(f"  Edge: {out['edge']*100:.1f}%")
            print(f"  Kalshi Side: {out['kalshi_side']}")
            print(f"  Kalshi Fair Price: {out['kalshi_fair_price']:.2f}")
            print(f"  Recommended Trade: {'YES' if out['kalshi_recommended'] else 'NO'}")
            print(f"  Kill Switch: {'ACTIVE' if out['kill_switch'] else 'OFF'}")

    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
