"""Position sizing utilities for Kalshi contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SizingConfig:
    kelly_fraction: float = 0.25
    max_contracts_per_trade: int = 25
    max_dollars_per_trade: float = 25.0
    min_contracts: int = 1


def kelly_fraction_binary(p_win: float, net_odds: float) -> float:
    """Classic Kelly fraction for a binary outcome with net odds."""

    q = 1.0 - p_win
    if net_odds <= 0:
        return 0.0
    return max(0.0, (net_odds * p_win - q) / net_odds)


def size_contracts(bankroll: float, price: float, p_win: float, cfg: SizingConfig) -> int:
    if bankroll <= 0 or price <= 0:
        return 0

    net_odds = (1.0 - price) / price
    f_star = kelly_fraction_binary(p_win, net_odds) * cfg.kelly_fraction
    dollars = min(cfg.max_dollars_per_trade, bankroll * f_star)
    contracts = int(math.floor(dollars / price))
    contracts = min(contracts, cfg.max_contracts_per_trade)
    if contracts < cfg.min_contracts:
        return 0
    return contracts
