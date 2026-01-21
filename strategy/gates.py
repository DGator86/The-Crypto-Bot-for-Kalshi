"""Trading gates to stop unprofitable Kalshi fills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kalshi.orderbook import TopOfBook
from strategy.ev import TradeEV


@dataclass(frozen=True)
class GateConfig:
    min_seconds_to_expiry: int = 60
    min_ask_price: float = 0.10
    max_ask_price: float = 0.90
    max_spread: float = 0.05
    min_ev_per_contract: float = 0.02
    min_edge_vs_price: float = 0.05


@dataclass(frozen=True)
class GateDecision:
    ok: bool
    reason: str


def check_time(t_left_seconds: int, cfg: GateConfig) -> GateDecision:
    if t_left_seconds < cfg.min_seconds_to_expiry:
        return GateDecision(False, "too_close_to_expiry")
    return GateDecision(True, "ok")


def _check_spread(spread: Optional[float], side: str, cfg: GateConfig) -> GateDecision:
    if spread is None:
        return GateDecision(False, f"missing_{side}_quotes")
    if spread > cfg.max_spread:
        return GateDecision(False, f"{side}_spread_too_wide")
    return GateDecision(True, "ok")


def check_book_sanity(side: str, book: TopOfBook, cfg: GateConfig) -> GateDecision:
    spread = book.spread_yes() if side == "YES" else book.spread_no()
    return _check_spread(spread, side.lower(), cfg)


def check_price_viability(ask: Optional[float], cfg: GateConfig, side: str) -> GateDecision:
    if ask is None:
        return GateDecision(False, f"missing_{side.lower()}_ask")
    if ask < cfg.min_ask_price:
        return GateDecision(False, "ask_too_low")
    if ask > cfg.max_ask_price:
        return GateDecision(False, "ask_too_high")
    return GateDecision(True, "ok")


def check_ev(ev: TradeEV, cfg: GateConfig) -> GateDecision:
    if ev.ev < cfg.min_ev_per_contract:
        return GateDecision(False, "ev_too_low")
    if ev.edge_vs_price < cfg.min_edge_vs_price:
        return GateDecision(False, "edge_too_low")
    return GateDecision(True, "ok")
