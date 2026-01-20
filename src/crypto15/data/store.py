from __future__ import annotations
from pathlib import Path
import pandas as pd

def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=True)

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)