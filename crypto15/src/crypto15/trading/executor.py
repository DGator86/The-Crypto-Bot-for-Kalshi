"""
Trade execution module.

Handles order execution across different modes (live, paper, simulation).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .signals import TradingSignal, SignalAction

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enum."""
    LIVE = "live"
    PAPER = "paper"
    SIMULATION = "simulation"


@dataclass
class ExecutedTrade:
    """Record of an executed trade."""
    timestamp: datetime
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    order_id: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    mode: TradingMode = TradingMode.PAPER
    signal: Optional[TradingSignal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Current position in a symbol."""
    symbol: str
    side: str  # "long", "short", "flat"
    quantity: float
    entry_price: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def update_pnl(self, current_price: float):
        """Update unrealized P&L based on current price."""
        if self.side == "long":
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == "short":
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        else:
            self.unrealized_pnl = 0.0


@dataclass
class ExecutorConfig:
    """Configuration for trade executor."""
    mode: TradingMode = TradingMode.PAPER
    initial_capital: float = 10000.0
    max_position_value: float = 2000.0  # Maximum $ in a single position
    commission_rate: float = 0.0005  # 5 bps
    slippage_rate: float = 0.0002  # 2 bps
    stop_loss_pct: float = 0.03  # 3%
    take_profit_pct: float = 0.06  # 6%
    enable_stop_loss: bool = True
    enable_take_profit: bool = True


class BaseTradingExecutor(ABC):
    """Abstract base class for trade executors."""

    @abstractmethod
    def execute_signal(self, signal: TradingSignal) -> Optional[ExecutedTrade]:
        """Execute a trading signal."""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol."""
        pass

    @abstractmethod
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_balance(self) -> float:
        """Get available balance."""
        pass

    @abstractmethod
    def close_position(self, symbol: str, reason: str = "manual") -> Optional[ExecutedTrade]:
        """Close position for a symbol."""
        pass


class TradingExecutor(BaseTradingExecutor):
    """
    Trade executor that handles signal execution and position management.

    Supports:
    - Paper trading for testing
    - Stop-loss and take-profit orders
    - Position sizing based on signal confidence
    - Trade history tracking

    Example:
        executor = TradingExecutor(ExecutorConfig(
            mode=TradingMode.PAPER,
            initial_capital=10000.0,
            stop_loss_pct=0.03
        ))

        # Execute a signal
        signal = signal_generator.generate(predictions, features, "BTC/USDT")
        trade = executor.execute_signal(signal)

        # Check position
        position = executor.get_position("BTC/USDT")
    """

    def __init__(
        self,
        config: Optional[ExecutorConfig] = None,
        kalshi_client: Optional[Any] = None,
    ):
        self.config = config or ExecutorConfig()
        self.kalshi_client = kalshi_client

        # State
        self._balance = self.config.initial_capital
        self._positions: Dict[str, Position] = {}
        self._trade_history: List[ExecutedTrade] = []
        self._total_pnl = 0.0

        logger.info(
            "TradingExecutor initialized in %s mode with $%.2f",
            self.config.mode.value,
            self._balance
        )

    def get_balance(self) -> float:
        """Get available balance."""
        return self._balance

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self._positions.values())

    def get_trade_history(self) -> List[ExecutedTrade]:
        """Get all executed trades."""
        return self._trade_history.copy()

    def get_total_pnl(self) -> float:
        """Get total realized P&L."""
        return self._total_pnl

    def _calculate_order_size(self, signal: TradingSignal) -> float:
        """Calculate order size in dollars."""
        # Use signal's position size recommendation
        position_value = self._balance * signal.position_size

        # Cap at max position value
        position_value = min(position_value, self.config.max_position_value)

        return position_value

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to execution price."""
        slippage = price * self.config.slippage_rate
        if side == "buy":
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Receive less when selling

    def _calculate_commission(self, value: float) -> float:
        """Calculate commission for a trade."""
        return value * self.config.commission_rate

    def execute_signal(self, signal: TradingSignal) -> Optional[ExecutedTrade]:
        """
        Execute a trading signal.

        Returns:
            ExecutedTrade if a trade was executed, None otherwise.
        """
        symbol = signal.symbol
        current_position = self._positions.get(symbol)

        # Handle kill switch
        if signal.action == SignalAction.KILL:
            if current_position:
                logger.warning("Kill switch active - closing position in %s", symbol)
                return self.close_position(symbol, reason="kill_switch")
            return None

        # Handle FLAT signal
        if signal.action == SignalAction.FLAT:
            if current_position:
                # Only close if we've been flat for a while (to avoid churning)
                pass  # Keep position for now
            return None

        # Determine if we need to change position
        desired_side = "long" if signal.action == SignalAction.LONG else "short"

        if current_position:
            # Check if we need to flip or close
            if current_position.side != desired_side:
                # Close existing position first
                self.close_position(symbol, reason="flip_position")
            else:
                # Already in desired position
                return None

        # Calculate position size
        position_value = self._calculate_order_size(signal)

        if position_value < 10:  # Minimum $10 position
            logger.debug("Position value too small: $%.2f", position_value)
            return None

        # Calculate quantity
        quantity = position_value / signal.current_price

        # Execute trade
        side = "buy" if desired_side == "long" else "sell"
        exec_price = self._apply_slippage(signal.current_price, side)
        trade_value = quantity * exec_price
        commission = self._calculate_commission(trade_value)

        # Check balance
        if side == "buy" and trade_value + commission > self._balance:
            logger.warning("Insufficient balance for trade: need $%.2f, have $%.2f",
                          trade_value + commission, self._balance)
            return None

        # Update balance
        if side == "buy":
            self._balance -= (trade_value + commission)
        else:
            self._balance += (trade_value - commission)

        # Create position
        position = Position(
            symbol=symbol,
            side=desired_side,
            quantity=quantity,
            entry_price=exec_price,
            entry_time=datetime.now(timezone.utc),
        )
        self._positions[symbol] = position

        # Create trade record
        trade = ExecutedTrade(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=exec_price,
            commission=commission,
            slippage=abs(exec_price - signal.current_price),
            mode=self.config.mode,
            signal=signal,
            metadata={
                "position_value": position_value,
                "probability_up": signal.probability_up,
                "expected_return": signal.expected_return,
            }
        )
        self._trade_history.append(trade)

        logger.info(
            "Executed %s %s: %.4f @ $%.2f (value: $%.2f, commission: $%.4f)",
            side.upper(),
            symbol,
            quantity,
            exec_price,
            position_value,
            commission
        )

        return trade

    def close_position(self, symbol: str, reason: str = "manual") -> Optional[ExecutedTrade]:
        """Close position for a symbol."""
        position = self._positions.get(symbol)
        if not position:
            logger.debug("No position to close for %s", symbol)
            return None

        # Determine close price (use last known price + slippage)
        side = "sell" if position.side == "long" else "buy"
        # In real implementation, would fetch current market price
        close_price = position.entry_price * (1 + position.unrealized_pnl / (position.quantity * position.entry_price))
        exec_price = self._apply_slippage(close_price, side)

        trade_value = position.quantity * exec_price
        commission = self._calculate_commission(trade_value)

        # Calculate P&L
        if position.side == "long":
            pnl = (exec_price - position.entry_price) * position.quantity - commission
        else:
            pnl = (position.entry_price - exec_price) * position.quantity - commission

        # Update balance
        if side == "sell":
            self._balance += (trade_value - commission)
        else:
            self._balance -= (trade_value + commission)

        self._total_pnl += pnl
        position.realized_pnl = pnl

        # Remove position
        del self._positions[symbol]

        # Create trade record
        trade = ExecutedTrade(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            side=side,
            quantity=position.quantity,
            price=exec_price,
            commission=commission,
            slippage=0.0,
            mode=self.config.mode,
            metadata={
                "reason": reason,
                "pnl": pnl,
                "entry_price": position.entry_price,
                "hold_time": (datetime.now(timezone.utc) - position.entry_time).total_seconds(),
            }
        )
        self._trade_history.append(trade)

        logger.info(
            "Closed %s position in %s: P&L $%.2f (reason: %s)",
            position.side.upper(),
            symbol,
            pnl,
            reason
        )

        return trade

    def check_stop_loss_take_profit(self, symbol: str, current_price: float) -> Optional[ExecutedTrade]:
        """Check and execute stop-loss or take-profit orders."""
        position = self._positions.get(symbol)
        if not position:
            return None

        # Update unrealized P&L
        position.update_pnl(current_price)

        # Calculate return
        if position.side == "long":
            pct_return = (current_price - position.entry_price) / position.entry_price
        else:
            pct_return = (position.entry_price - current_price) / position.entry_price

        # Check stop-loss
        if self.config.enable_stop_loss and pct_return <= -self.config.stop_loss_pct:
            logger.warning(
                "Stop-loss triggered for %s: %.2f%% loss",
                symbol,
                pct_return * 100
            )
            return self.close_position(symbol, reason="stop_loss")

        # Check take-profit
        if self.config.enable_take_profit and pct_return >= self.config.take_profit_pct:
            logger.info(
                "Take-profit triggered for %s: %.2f%% profit",
                symbol,
                pct_return * 100
            )
            return self.close_position(symbol, reason="take_profit")

        return None

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio state."""
        total_position_value = sum(
            pos.quantity * pos.entry_price for pos in self._positions.values()
        )
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self._positions.values()
        )

        return {
            "balance": self._balance,
            "position_value": total_position_value,
            "total_value": self._balance + total_position_value,
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": self._total_pnl,
            "total_pnl": total_unrealized_pnl + self._total_pnl,
            "num_positions": len(self._positions),
            "num_trades": len(self._trade_history),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                }
                for pos in self._positions.values()
            ]
        }

    def reset(self):
        """Reset executor state."""
        self._balance = self.config.initial_capital
        self._positions.clear()
        self._trade_history.clear()
        self._total_pnl = 0.0
        logger.info("Executor reset to initial state")
