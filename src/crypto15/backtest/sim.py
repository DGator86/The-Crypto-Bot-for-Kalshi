from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from crypto15.util import bps_to_frac

@dataclass(frozen=True)
class Costs:
    fee_bps: float
    slippage_bps: float
    spread_bps: float
    
    @property
    def round_trip_cost_frac(self) -> float:
        # Conservative: treat these as total round-trip friction
        return bps_to_frac(self.fee_bps + self.slippage_bps + self.spread_bps)

def make_tradeable_labels(
    ohlcv: pd.DataFrame,
    horizon_bars: int,
    no_trade_band_bps: float,
) -> pd.Series:
    """
    Labels based on forward move from NEXT open to open at horizon (entry at next open after signal).
    Signal occurs at time t (close of bar t). Entry at open t+1.
    Forward return = open[t+1+h] / open[t+1] - 1.
    """
    o = ohlcv["open"].copy()
    entry = o.shift(-1)
    exit_ = o.shift(-(1 + horizon_bars))
    
    fwd_ret = (exit_ / entry) - 1.0
    band = bps_to_frac(no_trade_band_bps)
    
    y = pd.Series(index=ohlcv.index, dtype="float64")
    y[fwd_ret > band] = 1.0
    y[fwd_ret < -band] = 0.0
    # in-band stays NaN = no-trade sample (we drop for training)
    return y

def simulate_long_only(
    idx: pd.DatetimeIndex,
    ohlcv: pd.DataFrame,
    prob_up: np.ndarray,
    horizon_bars: int,
    p_enter: float,
    risk_fraction: float,
    costs: Costs,
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    """
    Long-only execution:
    - At time t, if prob_up[t] >= p_enter => enter long at open[t+1], exit at open[t+1+horizon]
    - Equity updated by allocating risk_fraction of current equity per trade.
    - Applies round-trip cost as a simple subtraction on return.
    """
    assert len(prob_up) == len(idx)
    
    o = ohlcv.loc[idx, "open"]
    entry = o.shift(-1)
    exit_ = o.shift(-(1 + horizon_bars))
    
    gross_ret = (exit_ / entry) - 1.0
    trade = (prob_up >= p_enter).astype(int)
    
    # last bars can’t be traded due to missing entry/exit
    trade = trade & entry.notna().values & exit_.notna().values
    
    net_ret = gross_ret.values - costs.round_trip_cost_frac
    net_ret = np.where(trade, net_ret, 0.0)
    
    equity = [initial_equity]
    for r in net_ret:
        eq = equity[-1]
        alloc = eq * risk_fraction
        eq = eq + alloc * r
        equity.append(eq)
        
    out = pd.DataFrame(
        {
            "equity": equity[1:],
            "trade": trade.astype(int),
            "net_ret": net_ret,
        },
        index=idx,
    )
    out["equity_peak"] = out["equity"].cummax()
    out["drawdown"] = (out["equity"] / out["equity_peak"]) - 1.0
    
    return out

def simulate_dual_model(
    idx: pd.DatetimeIndex,
    ohlcv: pd.DataFrame,
    prob_up: np.ndarray,
    exp_ret: np.ndarray,
    horizon_bars: int,
    p_enter: float,
    min_exp_ret: float,
    risk_fraction: float,
    costs: Costs,
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    """
    Long-only execution with dual model:
    - At time t, if prob_up[t] >= p_enter AND exp_ret[t] > min_exp_ret => enter long.
    """
    assert len(prob_up) == len(idx)
    assert len(exp_ret) == len(idx)
    
    o = ohlcv.loc[idx, "open"]
    entry = o.shift(-1)
    exit_ = o.shift(-(1 + horizon_bars))
    
    gross_ret = (exit_ / entry) - 1.0
    
    trade = ((prob_up >= p_enter) & (exp_ret > min_exp_ret)).astype(int)
    
    # last bars can’t be traded due to missing entry/exit
    trade = trade & entry.notna().values & exit_.notna().values
    
    net_ret = gross_ret.values - costs.round_trip_cost_frac
    net_ret = np.where(trade, net_ret, 0.0)
    
    equity = [initial_equity]
    for r in net_ret:
        eq = equity[-1]
        alloc = eq * risk_fraction
        eq = eq + alloc * r
        equity.append(eq)
        
    out = pd.DataFrame(
        {
            "equity": equity[1:],
            "trade": trade.astype(int),
            "net_ret": net_ret,
        },
        index=idx,
    )
    out["equity_peak"] = out["equity"].cummax()
    out["drawdown"] = (out["equity"] / out["equity_peak"]) - 1.0
    
    return out