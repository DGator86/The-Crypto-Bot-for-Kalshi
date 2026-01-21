"""Expected-value computation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kalshi.fees import FeeSchedule, kalshi_fee_per_contract

Side = Literal["YES", "NO"]


@dataclass(frozen=True)
class TradeEV:
    """Container describing EV characteristics for a candidate trade."""

    side: Side
    price: float
    fee: float
    ev: float
    edge_vs_price: float  # p_yes - price for YES, p_no - price for NO


def ev_buy_yes(p_yes: float, ask: float, *, is_maker: bool, schedule: FeeSchedule) -> TradeEV:
    fee = kalshi_fee_per_contract(ask, is_maker=is_maker, schedule=schedule)
    win = 1.0 - ask - fee
    lose = -(ask + fee)
    ev = p_yes * win + (1.0 - p_yes) * lose
    return TradeEV(side="YES", price=ask, fee=fee, ev=ev, edge_vs_price=p_yes - ask)


def ev_buy_no(p_yes: float, ask_no: float, *, is_maker: bool, schedule: FeeSchedule) -> TradeEV:
    p_no = 1.0 - p_yes
    fee = kalshi_fee_per_contract(ask_no, is_maker=is_maker, schedule=schedule)
    win = 1.0 - ask_no - fee
    lose = -(ask_no + fee)
    ev = p_no * win + (1.0 - p_no) * lose
    return TradeEV(side="NO", price=ask_no, fee=fee, ev=ev, edge_vs_price=p_no - ask_no)
