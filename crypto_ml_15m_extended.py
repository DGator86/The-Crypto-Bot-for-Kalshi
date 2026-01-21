"""
Crypto 15m Direction ML (BTC/ETH/SOL) — Extended Bot-Ready Edition

Adds:
1) Data adapters: CSVProvider + BinanceRESTProvider (public REST klines)
2) Purged & Embargoed walk-forward CV (reduces leakage)
3) Threshold optimization (p_long/p_short) per asset using CV folds + costs
4) Regime gating with per-regime thresholds (trend / chop / high-vol)
5) Tree + logistic ensemble with calibrated meta layer
6) Live predict_and_signal() + decide_trade() for execution bots (position sizing + kill switch)
7) Model/threshold persistence

Dependencies:
  pip install pandas numpy scikit-learn joblib requests

Important:
- This predicts the next 15m direction probabilistically; it does not guarantee profits.
- For “world-class” accuracy later: add L2 microstructure, funding/OI/liquidations, regime gating, ensembles.
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, List, Iterator, Any

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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss
)

# ----------------------------
# Configuration
# ----------------------------

@dataclass
class MLConfig:
    timeframe: str = "15m"
    horizon_bars: int = 1
    label_return_threshold: float = 0.0  # log-return threshold; raise to filter noise
    drop_last_n: int = 1

    # Feature windows
    rsi_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_k: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Leakage control for CV
    n_splits: int = 6
    purge_bars: int = 2      # remove a small window before test from training
    embargo_bars: int = 2    # skip bars after test (optional; conservative)

    # Modeling
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"  # isotonic tends to calibrate better but can overfit on tiny data

    # Default thresholds (will be optimized if enabled)
    p_long: float = 0.55
    p_short: float = 0.45
    allow_short: bool = True

    # Threshold optimization
    optimize_thresholds: bool = True
    p_long_grid: Tuple[float, float, float] = (0.52, 0.70, 0.02)  # start, stop, step
    p_short_grid: Tuple[float, float, float] = (0.30, 0.48, 0.02)
    threshold_objective: str = "net_expectancy"  # "net_expectancy" | "sharpe"

    # Costs for objective/backtest
    fee_bps: float = 6.0
    slippage_bps: float = 2.0

    # Regime gating parameters
    regime_trend_quantile: float = 0.65
    regime_vol_quantile: float = 0.7

    # Risk management & execution
    kill_switch_prob_gap: float = 0.025
    kill_switch_vol_z_limit: float = 3.5
    allow_kill_switch: bool = True
    position_size_max: float = 1.0
    position_size_min: float = 0.0
    position_size_scale: float = 5.0

    # Live inference
    min_history_bars: int = 120  # need enough history for stable indicators


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
    Public Binance REST klines. No API key needed.
    Notes:
      - Uses https://api.binance.com/api/v3/klines
      - symbol must be Binance format, e.g., BTCUSDT, ETHUSDT, SOLUSDT
      - interval: 15m, 1h, etc.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, quote: str = "USDT", session: Optional[requests.Session] = None, timeout: int = 20):
        self.quote = quote
        self.sess = session or requests.Session()
        self.timeout = timeout

    def _to_binance_symbol(self, symbol: str) -> str:
        # Accept "BTC" or "BTCUSDT"
        s = symbol.upper()
        if s.endswith(self.quote):
            return s
        return f"{s}{self.quote}"

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
        bsymbol = self._to_binance_symbol(symbol)
        params = {"symbol": bsymbol, "interval": timeframe, "limit": int(limit)}
        r = self.sess.get(self.BASE_URL, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()

        # Binance kline schema:
        # [ open_time, open, high, low, close, volume, close_time, quote_asset_volume, trades, ...]
        rows = []
        for k in data:
            open_time = pd.to_datetime(int(k[0]), unit="ms", utc=True)
            rows.append([open_time, float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.set_index("timestamp").sort_index()
        return df


# ----------------------------
# Feature Engineering
# ----------------------------

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    gain = up.rolling(period).mean()
    loss = down.rolling(period).mean()
    rs = gain / (loss.replace(0.0, np.nan))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.rolling(period).mean()

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

def build_features(df: pd.DataFrame, cfg: MLConfig) -> pd.DataFrame:
    o, h, l, c, v = (df["open"], df["high"], df["low"], df["close"], df["volume"])

    logret = np.log(c).diff()
    ret1 = c.pct_change()
    rv_4 = logret.rolling(4).std() * math.sqrt(4)
    rv_16 = logret.rolling(16).std() * math.sqrt(16)

    ema_8 = _ema(c, 8)
    ema_21 = _ema(c, 21)
    ema_55 = _ema(c, 55)
    trend_8_21 = (ema_8 - ema_21) / c
    trend_21_55 = (ema_21 - ema_55) / c

    rsi = _rsi(c, cfg.rsi_period)
    atr = _atr(h, l, c, cfg.atr_period)
    atrp = (atr / c).replace([np.inf, -np.inf], np.nan)

    bb_ma, bb_up, bb_lo = _bollinger(c, cfg.bb_period, cfg.bb_k)
    bb_width = (bb_up - bb_lo) / (bb_ma.replace(0.0, np.nan))
    bb_pos = (c - bb_lo) / ((bb_up - bb_lo).replace(0.0, np.nan))

    macd, macd_sig, macd_hist = _macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

    body = (c - o) / c
    range_ = (h - l) / c
    upper_wick = (h - np.maximum(o, c)) / c
    lower_wick = (np.minimum(o, c) - l) / c

    vol_z = (v - v.rolling(48).mean()) / (v.rolling(48).std(ddof=0).replace(0.0, np.nan))
    vol_chg = v.pct_change()

    typical = (h + l + c) / 3.0
    vwap_48 = (typical.mul(v).rolling(48).sum() / v.rolling(48).sum().replace(0.0, np.nan))
    vwap_dist = (c - vwap_48) / c

    idx = df.index
    hour = idx.hour
    dow = idx.dayofweek
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)

    X = pd.DataFrame({
        "logret": logret,
        "ret1": ret1,
        "rv_4": rv_4,
        "rv_16": rv_16,
        "trend_8_21": trend_8_21,
        "trend_21_55": trend_21_55,
        "rsi": rsi,
        "atrp": atrp,
        "bb_width": bb_width,
        "bb_pos": bb_pos,
        "macd": macd,
        "macd_sig": macd_sig,
        "macd_hist": macd_hist,
        "body": body,
        "range": range_,
        "upper_wick": upper_wick,
        "lower_wick": lower_wick,
        "vol_z": vol_z,
        "vol_chg": vol_chg,
        "vwap_dist": vwap_dist,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }, index=df.index)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.ffill().fillna(0.0)
    return X

def build_labels(df: pd.DataFrame, cfg: MLConfig) -> pd.Series:
    c = df["close"]
    fwd = np.log(c.shift(-cfg.horizon_bars) / c)
    y = (fwd > cfg.label_return_threshold).astype(float)
    if cfg.drop_last_n > 0:
        y.iloc[-cfg.drop_last_n:] = np.nan
    return y.dropna().astype(int)


# ----------------------------
# Purged / Embargoed Time Series Split
# ----------------------------

class PurgedTimeSeriesSplit:
    """
    Time-series split with purge and embargo.
    - Purge removes last `purge_bars` from training before the test window.
    - Embargo skips `embargo_bars` after the test window (conservative; prevents overlap effects).
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

            # embargo means we do not allow training samples after the test window (for these splits we already train only before)
            # but if you later expand to include "future" training blocks, embargo matters.
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) < 50 or len(test_idx) < 20:
                continue

            yield train_idx, test_idx


# ----------------------------
# Modeling
# ----------------------------

def make_tree_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.05,
            max_iter=700,
            l2_regularization=1e-3,
            random_state=42
        ))
    ])

def make_logit_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.5,
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

    pipeline.fit(X_fit, y_fit)
    calibrator = CalibratedClassifierCV(pipeline, method=cfg.calibration_method, cv="prefit")
    calibrator.fit(X_cal, y_cal)
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
    rv = np.nan_to_num(X["rv_16"].to_numpy(dtype=float), nan=0.0)
    return np.column_stack([
        probs_tree,
        probs_logit,
        trend_abs,
        rv,
    ])


def _compute_regime_thresholds(X: pd.DataFrame, cfg: MLConfig) -> Dict[str, float]:
    trend_metric = np.abs(X["trend_8_21"]).fillna(0.0)
    vol_metric = np.abs(X["rv_16"]).fillna(0.0)
    return {
        "trend": float(trend_metric.quantile(cfg.regime_trend_quantile)),
        "high_vol": float(vol_metric.quantile(cfg.regime_vol_quantile)),
    }


def _assign_regimes(X: pd.DataFrame, thresholds: Dict[str, float]) -> pd.Series:
    regimes = pd.Series("chop", index=X.index, dtype=object)
    trend_metric = np.abs(X["trend_8_21"]).fillna(0.0)
    vol_metric = np.abs(X["rv_16"]).fillna(0.0)

    vol_thr = thresholds.get("high_vol", np.inf)
    trend_thr = thresholds.get("trend", np.inf)

    vol_mask = vol_metric >= vol_thr
    regimes.loc[vol_mask] = "high_vol"

    trend_mask = (trend_metric >= trend_thr) & (~vol_mask)
    regimes.loc[trend_mask] = "trend"
    return regimes


def make_meta_model() -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
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
    """
    Expectancy proxy on the test window:
      position decided at time t using prob(t), applied to next-bar close-to-close return.
    Costs applied on turnover.
    """
    # positions
    pos = np.zeros_like(probs, dtype=float)
    pos[probs >= p_long] = 1.0
    if cfg.allow_short:
        pos[probs <= p_short] = -1.0

    # next-bar return from t close -> t+1 close
    close = df_prices["close"].values
    ret_cc = np.zeros_like(close, dtype=float)
    ret_cc[:-1] = (close[1:] / close[:-1]) - 1.0
    ret_cc[-1] = 0.0

    gross = pos * ret_cc[: len(pos)]

    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    turnover = np.zeros_like(pos)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    costs = turnover * cost_rate

    net = gross - costs
    return float(np.nanmean(net))  # per-bar average net return

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
    ret_cc = np.zeros_like(close, dtype=float)
    ret_cc[:-1] = (close[1:] / close[:-1]) - 1.0
    ret_cc[-1] = 0.0

    gross = pos * ret_cc[: len(pos)]
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    turnover = np.zeros_like(pos)
    turnover[1:] = np.abs(pos[1:] - pos[:-1])
    net = gross - turnover * cost_rate

    mu = np.nanmean(net)
    sd = np.nanstd(net)
    if sd <= 1e-12:
        return -1e9
    return float(mu / sd)


def optimize_thresholds_cv(
    X: pd.DataFrame,
    y: pd.Series,
    df_prices: pd.DataFrame,
    cfg: MLConfig,
) -> Tuple[float, float]:
    """
    Grid search p_long/p_short over CV test folds.
    Objective computed on each fold, then averaged.
    """
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
            if p_short >= p_long:  # invalid band
                continue

            scores = []
            for tr_idx, te_idx in splitter.split(X):
                X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) == 0:
                    continue

                model, cal = _fit_with_time_series_calibration(X_tr, y_tr, cfg, make_tree_pipeline)
                probs = _predict_proba(model, cal, X_te)

                # Align df_prices for test indices
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
# Reporting
# ----------------------------

def eval_cv_metrics(X: pd.DataFrame, y: pd.Series, cfg: MLConfig) -> Dict[str, float]:
    splitter = PurgedTimeSeriesSplit(cfg.n_splits, cfg.purge_bars, cfg.embargo_bars)

    metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc", "brier", "logloss"]}

    for tr_idx, te_idx in splitter.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 1:
            continue

        tree_model, tree_cal = _fit_with_time_series_calibration(X_tr, y_tr, cfg, make_tree_pipeline)
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

        if len(np.unique(y_te)) == 2:
            metrics["auc"].append(roc_auc_score(y_te, proba))
            metrics["logloss"].append(log_loss(y_te, np.vstack([1 - proba, proba]).T, labels=[0, 1]))
        else:
            metrics["auc"].append(np.nan)
            metrics["logloss"].append(np.nan)

    return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in metrics.items()}


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
        vol_val = float(abs(feature_row.get("rv_16", 0.0)))
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

        report = eval_cv_metrics(X, y, self.cfg)

        self.regime_params = _compute_regime_thresholds(X, self.cfg)
        regimes = _assign_regimes(X, self.regime_params)
        report["regime_counts"] = {reg: int((regimes == reg).sum()) for reg in REGIME_NAMES}

        if self.cfg.optimize_thresholds:
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

        self.tree_pipeline, self.tree_calibrator = _fit_with_time_series_calibration(
            X, y, self.cfg, make_tree_pipeline
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

        components.pop("features_row", None)
        components.update(
            {
                "action": action,
                "probability_up": prob,
                "confidence": confidence,
                "position_size": position_size,
                "kill_switch": kill_switch,
            }
        )
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
        cfg = MLConfig(**meta["cfg"])
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

    def train(self, symbols: List[str], limit: int = 3000) -> Dict[str, Dict[str, Any]]:
        reports: Dict[str, Dict[str, Any]] = {}
        for sym in symbols:
            df = self.provider.fetch_ohlcv(sym, self.cfg.timeframe, limit=limit)
            df = df.dropna()
            if df.empty:
                raise ValueError(f"No data returned for symbol={sym} when training.")
            m = AssetPredictor(self.cfg)
            rep = m.fit(df)
            self.models[sym] = m
            reports[sym] = rep
        return reports

    def save(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "cfg.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        for sym, model in self.models.items():
            model.save(os.path.join(root, sym))

    @classmethod
    def load(cls, root: str, provider: DataProvider) -> "MultiAssetML":
        with open(os.path.join(root, "cfg.json"), "r") as f:
            cfg = MLConfig(**json.load(f))
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
        trade.update(
            {
                "symbol": symbol,
                "timeframe": self.cfg.timeframe,
                "timestamp_utc": str(df.index[-1]),
            }
        )
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
        """
        Backwards-compatible alias that returns condensed live signal information.
        """
        trade = self.decide_trade(symbol, limit=limit)
        thresholds = trade.get("thresholds", {})
        return {
            "symbol": trade["symbol"],
            "timeframe": trade["timeframe"],
            "timestamp_utc": trade["timestamp_utc"],
            "p_up": float(trade.get("probability_up", np.nan)),
            "signal": trade.get("action"),
            "p_long": float(thresholds.get("p_long", self.cfg.p_long)),
            "p_short": float(thresholds.get("p_short", self.cfg.p_short)),
            "regime": trade.get("regime"),
            "confidence": float(trade.get("confidence", 0.0)),
            "position_size": float(trade.get("position_size", 0.0)),
            "kill_switch": bool(trade.get("kill_switch", False)),
            "prob_tree": float(trade.get("prob_tree", np.nan)),
            "prob_logit": float(trade.get("prob_logit", np.nan)),
            "vol_z": float(trade.get("vol_z", 0.0)),
        }


# ----------------------------
# CLI
# ----------------------------

def _print_reports(reports: Dict[str, Dict[str, Any]]) -> None:
    print("\nLEAKAGE-RESISTANT CV METRICS (purged walk-forward avg)")
    for sym, rep in reports.items():
        print(f"\n{sym}")
        for k in ["accuracy", "precision", "recall", "f1", "auc", "brier", "logloss", "p_long", "p_short"]:
            if k in rep:
                v = rep[k]
                if isinstance(v, float):
                    print(f"  {k:10s}: {v:.4f}")
                else:
                    print(f"  {k:10s}: {v}")
        if "ensemble_ready" in rep:
            print(f"  {'ensemble':10s}: {rep['ensemble_ready']}")
        if "regime_counts" in rep:
            print(f"  {'regime_counts':10s}: {rep['regime_counts']}")
        if "regime_thresholds" in rep:
            print(f"  {'regime_thr':10s}: {rep['regime_thresholds']}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train_csv", "train_binance", "signal_csv", "signal_binance"], required=True)

    ap.add_argument("--model_dir", type=str, default="models_15m_ext")
    ap.add_argument("--symbols", type=str, default="BTC,ETH,SOL")

    # CSV mode
    ap.add_argument("--btc_csv", type=str, default="")
    ap.add_argument("--eth_csv", type=str, default="")
    ap.add_argument("--sol_csv", type=str, default="")

    # Binance mode
    ap.add_argument("--quote", type=str, default="USDT")
    ap.add_argument("--limit", type=int, default=3000)

    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    cfg = MLConfig()

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
        print(f"\nSaved to: {args.model_dir}")

    elif args.mode == "train_binance":
        provider = BinanceRESTProvider(quote=args.quote)
        mam = MultiAssetML(cfg, provider)
        reports = mam.train(symbols=symbols, limit=args.limit)
        _print_reports(reports)
        mam.save(args.model_dir)
        print(f"\nSaved to: {args.model_dir}")

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
        for sym in symbols:
            out = mam.predict_and_signal(sym, limit=min(args.limit, 1000))
            print(json.dumps(out, indent=2))

    else:
        raise SystemExit("Unknown mode")


if __name__ == "__main__":
    main()
