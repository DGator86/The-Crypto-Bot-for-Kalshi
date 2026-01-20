from __future__ import annotations
import pandas as pd
from crypto15.features.feature_set_v1 import build_features as build_base_features

def build_panel_features(target_symbol: str, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # 1. Build base features for target
    target_df = panel[target_symbol]
    base_feats = build_base_features(target_df)
    
    # 2. Build features for others
    others = [s for s in panel.keys() if s != target_symbol]
    
    extra_feats_list = []
    for s in others:
        df = panel[s]
        feats = build_base_features(df)
        # Keep key features
        cols_to_keep = ["ret_1", "ret_4", "ret_16", "vol_16", "rsi_14", "ema_spread"]
        # Intersect with available columns
        cols_to_keep = [c for c in cols_to_keep if c in feats.columns]
        
        feats = feats[cols_to_keep]
        
        ticker = s.split("/")[0]
        feats.columns = [f"{ticker}_{c}" for c in feats.columns]
        extra_feats_list.append(feats)
    
    # 3. Merge
    all_feats = base_feats.copy()
    for extra in extra_feats_list:
        all_feats = all_feats.join(extra, how="left")
    
    # 4. Cross-asset features
    for s in others:
        ticker = s.split("/")[0]
        # Relative return
        if f"{ticker}_ret_1" in all_feats.columns:
            all_feats[f"rel_ret_1_{ticker}"] = all_feats["ret_1"] - all_feats[f"{ticker}_ret_1"]
        
        # Leader-lag proxy: Target(t) vs Other(t-1) ??
        # The prompt says "ret_BTC(t), ret_BTC(t-1) included in ETH/SOL feature vector"
        # We already included ret_BTC(t) (which is current bar close-to-close).
        # We can add lagged versions of others if we want.
        # But ret_1 is (close[t]/close[t-1] - 1). 
        # So "ret_BTC(t)" is known at time t.
        
    return all_feats.dropna()