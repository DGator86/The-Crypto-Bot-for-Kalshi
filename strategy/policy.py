"""Decision policy combining EV gates with position sizing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from kalshi.fees import FeeSchedule
from kalshi.orderbook import TopOfBook
from strategy.ev import TradeEV, ev_buy_no, ev_buy_yes
from strategy.gates import GateConfig, GateDecision, check_book_sanity, check_ev, check_price_viability, check_time
from strategy.sizing import SizingConfig, size_contracts

Action = Literal["BUY_YES", "BUY_NO", "NO_TRADE"]


@dataclass(frozen=True)
class TradeDecision:
    action: Action
    contracts: int
    price: Optional[float]
    ev: Optional[TradeEV]
    reason: str


def _pick_best(candidates: list[TradeEV]) -> Optional[TradeEV]:
    if not candidates:
        return None
    return max(candidates, key=lambda ev: ev.ev)


def decide_trade(
    *,
    p_yes: float,
    book: TopOfBook,
    t_left_seconds: int,
    bankroll: float,
    is_maker: bool,
    fee_schedule: FeeSchedule,
    gate_cfg: GateConfig,
    size_cfg: SizingConfig,
) -> TradeDecision:
    # Time gate first
    g_time: GateDecision = check_time(t_left_seconds, gate_cfg)
    if not g_time.ok:
        return TradeDecision("NO_TRADE", 0, None, None, g_time.reason)

    candidates: list[TradeEV] = []

    # YES side
    g_book = check_book_sanity("YES", book, gate_cfg)
    if g_book.ok:
        g_price = check_price_viability(book.yes_ask, gate_cfg, "YES")
        if g_price.ok and book.yes_ask is not None:
            ev_yes = ev_buy_yes(p_yes, book.yes_ask, is_maker=is_maker, schedule=fee_schedule)
            g_ev = check_ev(ev_yes, gate_cfg)
            if g_ev.ok:
                candidates.append(ev_yes)

    # NO side
    g_book = check_book_sanity("NO", book, gate_cfg)
    if g_book.ok:
        g_price = check_price_viability(book.no_ask, gate_cfg, "NO")
        if g_price.ok and book.no_ask is not None:
            ev_no = ev_buy_no(p_yes, book.no_ask, is_maker=is_maker, schedule=fee_schedule)
            g_ev = check_ev(ev_no, gate_cfg)
            if g_ev.ok:
                candidates.append(ev_no)

    best = _pick_best(candidates)
    if best is None:
        return TradeDecision("NO_TRADE", 0, None, None, "no_candidate_passed_gates")

    if best.side == "YES":
        contracts = size_contracts(bankroll, best.price, p_yes, size_cfg)
        action: Action = "BUY_YES"
    else:
        contracts = size_contracts(bankroll, best.price, 1.0 - p_yes, size_cfg)
        action = "BUY_NO"

    if contracts <= 0:
        return TradeDecision("NO_TRADE", 0, best.price, best, "size_zero")

    return TradeDecision(action, contracts, best.price, best, "ok")
