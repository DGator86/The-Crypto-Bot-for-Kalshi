from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from crypto15.backtest.sim import Costs, make_tradeable_labels, simulate_long_only
from crypto15.features.feature_set_v1 import build_features
from crypto15.model.xgb import fit_xgb_classifier, predict_proba

@dataclass(frozen=True)
class WalkforwardResult:
    fold_summaries: pd.DataFrame
    equity_curve: pd.DataFrame

def _slice_by_days(df: pd.DataFrame, start: pd.Timestamp, days: int) -> pd.DataFrame:
    end = start + pd.Timedelta(days=days)
    return df[(df.index >= start) & (df.index < end)]

def run_walkforward(
    ohlcv: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
    horizon_bars: int,
    no_trade_band_bps: float,
    p_enter: float,
    risk_fraction: float,
    costs: Costs,
) -> WalkforwardResult:
    """
    Walk-forward:
    - Build features once (causal).
    - Build labels on raw prices.
    - For each fold:
      fit scaler+model on TRAIN rows (dropping no-trade labels)
      predict probability on TEST rows
      simulate trades on TEST with raw ohlcv
    """
    ohlcv = ohlcv.sort_index()
    feats = build_features(ohlcv)
    y_all = make_tradeable_labels(ohlcv, horizon_bars=horizon_bars, no_trade_band_bps=no_trade_band_bps)
    
    # align to feature index
    y = y_all.loc[feats.index]
    raw = ohlcv.loc[feats.index]  # raw prices aligned
    
    start = feats.index.min().normalize()
    last = feats.index.max().normalize()
    
    fold_rows = []
    equity_parts = []
    
    cur = start
    fold = 0
    
    while cur + pd.Timedelta(days=train_days + test_days) <= last:
        train_df = _slice_by_days(feats, cur, train_days)
        test_df = _slice_by_days(feats, cur + pd.Timedelta(days=train_days), test_days)
        
        if len(train_df) < 500 or len(test_df) < 50:
            cur = cur + pd.Timedelta(days=step_days)
            continue
            
        y_train = y.loc[train_df.index].dropna()
        X_train = train_df.loc[y_train.index].to_numpy()
        y_train_np = y_train.to_numpy().astype(int)
        
        # Fit
        artifact = fit_xgb_classifier(X_train, y_train_np, feature_names=list(train_df.columns))
        
        # Predict on full test rows (including no-trade regions; strategy will abstain by p_enter)
        X_test = test_df.to_numpy()
        p_up = predict_proba(artifact, X_test)
        
        # Simulate on raw prices for the test range
        raw_test = raw.loc[test_df.index]
        sim = simulate_long_only(
            idx=test_df.index,
            ohlcv=raw_test,
            prob_up=p_up,
            horizon_bars=horizon_bars,
            p_enter=p_enter,
            risk_fraction=risk_fraction,
            costs=costs,
        )
        sim["fold"] = fold
        equity_parts.append(sim)
        
        # Fold summary
        trades = int(sim["trade"].sum())
        end_equity = float(sim["equity"].iloc[-1])
        ret = (end_equity / float(sim["equity"].iloc[0])) - 1.0 if len(sim) else 0.0
        max_dd = float(sim["drawdown"].min()) if len(sim) else 0.0
        avg_r = float(sim.loc[sim["trade"] == 1, "net_ret"].mean()) if trades > 0 else 0.0
        
        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_df.index.min(),
                "train_end": train_df.index.max(),
                "test_start": test_df.index.min(),
                "test_end": test_df.index.max(),
                "trades": trades,
                "test_return": ret,
                "avg_trade_ret": avg_r,
                "max_drawdown": max_dd,
            }
        )
        fold += 1
        cur = cur + pd.Timedelta(days=step_days)
        
    equity_curve = pd.concat(equity_parts).sort_index() if equity_parts else pd.DataFrame()
    fold_summaries = pd.DataFrame(fold_rows)
    
    return WalkforwardResult(fold_summaries=fold_summaries, equity_curve=equity_curve)