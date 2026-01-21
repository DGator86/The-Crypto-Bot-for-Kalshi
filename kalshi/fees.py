"""Kalshi fee helpers with proper rounding."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeSchedule:
    """Kalshi fee schedule (default retail taker/maker rates)."""

    taker_rate: float = 0.07
    maker_rate: float = 0.0175


def kalshi_fee_per_contract(price: float, *, is_maker: bool, schedule: FeeSchedule = FeeSchedule()) -> float:
    """Approximate Kalshi per-contract fee rounded up to the nearest cent.

    Kalshi's published formula (simplified):

        fee = ceil(rate * price * (1 - price) * 100) / 100

    where ``price`` is expressed in dollars (0.00-1.00). Rounding up is the key
    reason that 0.95-0.99 price levels are almost never profitable for takers.
    """

    if not 0.0 <= price <= 1.0:
        raise ValueError(f"Price must be within [0, 1], got {price!r}")

    rate = schedule.maker_rate if is_maker else schedule.taker_rate
    raw = rate * price * (1.0 - price)
    fee_cents = math.ceil(raw * 100.0)
    return fee_cents / 100.0
