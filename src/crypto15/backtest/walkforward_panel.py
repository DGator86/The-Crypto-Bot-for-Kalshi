from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from crypto15.backtest.sim import Costs, make_tradeable_labels, simulate_dual_model
from crypto15.features.feature_set_panel_v1 import build_panel_features
from crypto15.model.xgb import fit_xgb_classifier, predict_proba
from crypto15.model.xgb_reg import fit_xgb_regressor, predict_reg
from crypto15.backtest.walkforward import WalkforwardResult, _slice_by_days

def run_walkforward_panel(
    target_symbol: str,
    panel: dict[str, pd.DataFrame],
    train_days: int,
    test_days: int,
    step_days: int,
    horizon_bars: int,
    no_trade_band_bps: float,
    p_enter: float,
    risk_fraction: float,
    costs: Costs,
) -> WalkforwardResult:
    
    # Build features
    feats = build_panel_features(target_symbol, panel)
    
    ohlcv = panel[target_symbol]
    
    # Labels
    y_class = make_tradeable_labels(ohlcv, horizon_bars, no_trade_band_bps)
    
    # Regression label: forward return
    o = ohlcv["open"]
    entry = o.shift(-1)
    exit_ = o.shift(-(1 + horizon_bars))
    y_reg_all = (exit_ / entry) - 1.0
    
    # Align
    common_idx = feats.index.intersection(y_reg_all.index) 
    
    feats = feats.loc[common_idx]
    y_class = y_class.loc[common_idx]
    y_reg = y_reg_all.loc[common_idx]
    raw = ohlcv.loc[common_idx]
    
    start = feats.index.min().normalize()
    last = feats.index.max().normalize()
    
    fold_rows = []
    equity_parts = []
    
    cur = start
    fold = 0
    
    min_exp_ret = costs.round_trip_cost_frac * 1.5 # Buffer
    
    while cur + pd.Timedelta(days=train_days + test_days) <= last:
        train_df = _slice_by_days(feats, cur, train_days)
        test_df = _slice_by_days(feats, cur + pd.Timedelta(days=train_days), test_days)
        
        if len(train_df) < 500 or len(test_df) < 50:
            cur = cur + pd.Timedelta(days=step_days)
            continue
            
        # Class model training
        y_train_class = y_class.loc[train_df.index].dropna()
        X_train_class = train_df.loc[y_train_class.index].to_numpy()
        y_train_class_np = y_train_class.to_numpy().astype(int)
        
        artifact_class = fit_xgb_classifier(X_train_class, y_train_class_np, feature_names=list(train_df.columns))
        
        # Reg model training
        y_train_reg = y_reg.loc[train_df.index].dropna()
        # Align
        X_train_reg = train_df.loc[y_train_reg.index].to_numpy()
        y_train_reg_np = y_train_reg.to_numpy()
        
        artifact_reg = fit_xgb_regressor(X_train_reg, y_train_reg_np, feature_names=list(train_df.columns))
        
        # Predict
        X_test = test_df.to_numpy()
        p_up = predict_proba(artifact_class, X_test)
        e_ret = predict_reg(artifact_reg, X_test)
        
        # Sim
        raw_test = raw.loc[test_df.index]
        sim = simulate_dual_model(
            idx=test_df.index,
            ohlcv=raw_test,
            prob_up=p_up,
            exp_ret=e_ret,
            horizon_bars=horizon_bars,
            p_enter=p_enter,
            min_exp_ret=min_exp_ret,
            risk_fraction=risk_fraction,
            costs=costs,
        )
        sim["fold"] = fold
        equity_parts.append(sim)
        
        # Summary
        trades = int(sim["trade"].sum())
        end_equity = float(sim["equity"].iloc[-1])
        ret = (end_equity / float(sim["equity"].iloc[0])) - 1.0 if len(sim) else 0.0
        max_dd = float(sim["drawdown"].min()) if len(sim) else 0.0
        avg_r = float(sim.loc[sim["trade"] == 1, "net_ret"].mean()) if trades > 0 else 0.0
        
        fold_rows.append({
            "fold": fold,
            "train_start": train_df.index.min(),
            "train_end": train_df.index.max(),
            "test_start": test_df.index.min(),
            "test_end": test_df.index.max(),
            "trades": trades,
            "test_return": ret,
            "avg_trade_ret": avg_r,
            "max_drawdown": max_dd,
        })
        
        fold += 1
        cur = cur + pd.Timedelta(days=step_days)
        
    equity_curve = pd.concat(equity_parts).sort_index() if equity_parts else pd.DataFrame()
    fold_summaries = pd.DataFrame(fold_rows)
    return WalkforwardResult(fold_summaries=fold_summaries, equity_curve=equity_curve)