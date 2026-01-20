from __future__ import annotations
from pathlib import Path
import pandas as pd
from crypto15.data.store import load_parquet

def load_panel(symbols: list[str], data_dir: str = "data", timeframe: str = "15m") -> dict[str, pd.DataFrame]:
    panel = {}
    for s in symbols:
        safe_sym = s.replace("/", "")
        path = Path(data_dir) / f"{safe_sym}_{timeframe}.parquet"
        if not path.exists():
            print(f"Warning: {path} not found.")
            continue
        panel[s] = load_parquet(path)
    
    # Align
    common_idx = None
    for df in panel.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)
            
    if common_idx is not None:
        aligned = {}
        for s, df in panel.items():
            aligned[s] = df.loc[common_idx].sort_index()
        return aligned
    return {}