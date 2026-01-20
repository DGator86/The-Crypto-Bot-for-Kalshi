from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List
import ccxt
import pandas as pd
from crypto15.util import ensure_utc_index

@dataclass
class OHLCV:
    df: pd.DataFrame  # columns: open, high, low, close, volume; index: UTC timestamp

def _make_exchange(name: str) -> ccxt.Exchange:
    cls = getattr(ccxt, name)
    ex = cls({"enableRateLimit": True})
    return ex

def fetch_ohlcv_history(
    exchange: str,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: Optional[int] = None,
    limit: int = 1000,
    sleep_s: float = 0.25,
) -> OHLCV:
    """
    Fetches candles in chunks using ccxt.fetch_ohlcv.
    - symbol must be like "BTC/USDT" for binance in ccxt.
    """
    ex = _make_exchange(exchange)
    now_ms = ex.milliseconds()
    end_ms = until_ms if until_ms is not None else now_ms
    
    rows: List[list] = []
    cur = since_ms
    
    while cur < end_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cur, limit=limit)
        if not batch:
            break
        
        rows.extend(batch)
        last_ts = batch[-1][0]
        cur = last_ts + 1
        
        if last_ts >= end_ms:
            break
        
        time.sleep(sleep_s)
        
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = ensure_utc_index(df)
    df = df[~df.index.duplicated(keep="last")]
    
    return OHLCV(df=df)

def fetch_latest_closed(
    exchange: str,
    symbol: str,
    timeframe: str,
    limit: int = 300,
) -> OHLCV:
    ex = _make_exchange(exchange)
    batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    
    df = pd.DataFrame(batch, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = ensure_utc_index(df)
    df = df[~df.index.duplicated(keep="last")]
    
    return OHLCV(df=df)