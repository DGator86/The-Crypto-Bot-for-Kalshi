from __future__ import annotations
import numpy as np
import pandas as pd
import crypto15.ta_lib as ta
from crypto15.util import ensure_utc_index

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input df: columns open, high, low, close, volume; UTC index.
    Output: feature DF (no labels), same index, no scaling.
    All features must be causal (rolling/past-only).
    """
    df = ensure_utc_index(df.copy())
    x = df.copy()
    
    # Returns / volatility
    x["ret_1"] = x["close"].pct_change(1)
    x["ret_4"] = x["close"].pct_change(4)
    x["ret_16"] = x["close"].pct_change(16)
    
    x["vol_16"] = x["ret_1"].rolling(16).std()
    x["vol_64"] = x["ret_1"].rolling(64).std()
    
    # Trend
    x["ema_12"] = ta.ema(x["close"], length=12)
    x["ema_26"] = ta.ema(x["close"], length=26)
    x["ema_spread"] = (x["ema_12"] / x["ema_26"]) - 1.0
    
    # Momentum
    x["rsi_14"] = ta.rsi(x["close"], length=14)
    macd = ta.macd(x["close"], fast=12, slow=26, signal=9)
    x["macd"] = macd["MACD_12_26_9"]
    x["macd_signal"] = macd["MACDs_12_26_9"]
    x["macd_hist"] = macd["MACDh_12_26_9"]
    
    # Range / structure
    x["atr_14"] = ta.atr(x["high"], x["low"], x["close"], length=14)
    bb = ta.bbands(x["close"], length=20, std=2.0)
    x["bb_bw"] = bb["BBB_20_2.0"]        # bandwidth
    x["bb_pct"] = bb["BBP_20_2.0"]       # percent b
    
    # Volume
    x["vol_z_64"] = (x["volume"] - x["volume"].rolling(64).mean()) / (x["volume"].rolling(64).std() + 1e-12)
    
    # Time features (UTC)
    x["hour"] = x.index.hour
    x["dow"] = x.index.dayofweek
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    
    # Clean: remove raw columns that can dominate scaling unless you explicitly want them
    feats = x[
        [
            "ret_1","ret_4","ret_16",
            "vol_16","vol_64",
            "ema_spread",
            "rsi_14","macd","macd_signal","macd_hist",
            "atr_14","bb_bw","bb_pct",
            "vol_z_64",
            "hour","dow","is_weekend",
        ]
    ]
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats