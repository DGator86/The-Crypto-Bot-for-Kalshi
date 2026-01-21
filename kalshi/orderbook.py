"""Normalized top-of-book representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TopOfBook:
    """Simple view of Kalshi top-of-book levels."""

    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]

    def spread_yes(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def spread_no(self) -> Optional[float]:
        if self.no_bid is None or self.no_ask is None:
            return None
        return self.no_ask - self.no_bid
