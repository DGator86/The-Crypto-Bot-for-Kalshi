from __future__ import annotations
import pandas as pd

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    else:
        df = df.tz_convert("UTC")
    return df.sort_index()

def bps_to_frac(bps: float) -> float:
    return float(bps) / 10_000.0