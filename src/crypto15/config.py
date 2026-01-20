from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import yaml

@dataclass(frozen=True)
class LabelCfg:
    horizon_bars: int
    no_trade_band_bps: float
    min_move_for_positive_bps: float

@dataclass(frozen=True)
class CostsCfg:
    fee_bps: float
    slippage_bps: float
    spread_bps: float

@dataclass(frozen=True)
class WalkforwardCfg:
    train_days: int
    test_days: int
    step_days: int

@dataclass(frozen=True)
class StrategyCfg:
    p_enter: float
    long_only: bool
    risk_fraction: float

@dataclass(frozen=True)
class FeatureCfg:
    lookback_bars: int

@dataclass(frozen=True)
class AppCfg:
    exchange: str
    symbols: List[str]
    timeframe: str
    history_days: int
    feature: FeatureCfg
    label: LabelCfg
    costs: CostsCfg
    walkforward: WalkforwardCfg
    strategy: StrategyCfg

def load_config(path: str | Path) -> AppCfg:
    p = Path(path)
    d: Dict[str, Any] = yaml.safe_load(p.read_text())
    return AppCfg(
        exchange=d["exchange"],
        symbols=d["symbols"],
        timeframe=d["timeframe"],
        history_days=int(d["history_days"]),
        feature=FeatureCfg(**d["feature"]),
        label=LabelCfg(**d["label"]),
        costs=CostsCfg(**d["costs"]),
        walkforward=WalkforwardCfg(**d["walkforward"]),
        strategy=StrategyCfg(**d["strategy"]),
    )