"""
Kalshi API integration module.

Provides both live trading and paper trading clients for Kalshi prediction markets.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

from .retry import (
    RetryConfig,
    with_retry,
    APIError,
    AuthenticationError,
    InsufficientFundsError,
    OrderError,
    RateLimitError,
    NetworkError,
)

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    """Order side enum."""
    YES = "yes"
    NO = "no"


class OrderType(Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"


class OrderAction(Enum):
    """Order action enum."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Position:
    """Represents a position in a market."""
    market_ticker: str
    yes_contracts: int = 0
    no_contracts: int = 0
    average_yes_price: float = 0.0
    average_no_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def net_position(self) -> int:
        """Net position (positive = long YES, negative = long NO)."""
        return self.yes_contracts - self.no_contracts


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    market_ticker: str
    side: OrderSide
    action: OrderAction
    order_type: OrderType
    price: Optional[float]  # In cents (1-99)
    quantity: int
    filled_quantity: int = 0
    status: str = "pending"
    created_at: Optional[datetime] = None

    @property
    def is_filled(self) -> bool:
        return self.status == "filled" or self.filled_quantity >= self.quantity


@dataclass
class MarketInfo:
    """Market information."""
    ticker: str
    title: str
    subtitle: str
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    no_bid: Optional[float] = None
    no_ask: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    status: str = "active"


@dataclass
class AccountBalance:
    """Account balance information."""
    available_balance: float  # In dollars
    portfolio_value: float
    total_deposits: float = 0.0
    total_withdrawals: float = 0.0


class BaseKalshiClient(ABC):
    """Abstract base class for Kalshi clients."""

    @abstractmethod
    def get_balance(self) -> AccountBalance:
        """Get account balance."""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    def get_position(self, market_ticker: str) -> Optional[Position]:
        """Get position for a specific market."""
        pass

    @abstractmethod
    def get_market(self, market_ticker: str) -> MarketInfo:
        """Get market information."""
        pass

    @abstractmethod
    def search_markets(self, query: str, limit: int = 10) -> List[MarketInfo]:
        """Search for markets."""
        pass

    @abstractmethod
    def place_order(
        self,
        market_ticker: str,
        side: OrderSide,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Order:
        """Place an order."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Get order status."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        pass


class KalshiClient(BaseKalshiClient):
    """
    Live Kalshi API client.

    Handles authentication, rate limiting, and all API operations.

    Example:
        client = KalshiClient(
            api_key="your_api_key",
            private_key="your_private_key"  # Or private_key_path for file
        )

        # Get balance
        balance = client.get_balance()
        print(f"Available: ${balance.available_balance}")

        # Search for BTC markets
        markets = client.search_markets("bitcoin")

        # Place an order
        order = client.place_order(
            market_ticker="BTCUSD-24JAN01",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            quantity=10,
            order_type=OrderType.LIMIT,
            price=55  # 55 cents
        )
    """

    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"

    def __init__(
        self,
        api_key: str,
        private_key: Optional[str] = None,
        private_key_path: Optional[str] = None,
        use_demo: bool = False,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize Kalshi client.

        Args:
            api_key: Your Kalshi API key.
            private_key: Your private key string (PEM format).
            private_key_path: Path to private key file (alternative to private_key).
            use_demo: Use demo/sandbox environment.
            retry_config: Configuration for retry behavior.
        """
        self.api_key = api_key
        self.base_url = self.DEMO_URL if use_demo else self.BASE_URL

        if private_key:
            self.private_key = private_key
        elif private_key_path:
            with open(private_key_path, 'r') as f:
                self.private_key = f.read()
        else:
            raise ValueError("Either private_key or private_key_path must be provided")

        self.retry_config = retry_config or RetryConfig(
            max_retries=4,
            base_delay=2.0,
            retryable_exceptions=(
                ConnectionError,
                TimeoutError,
                OSError,
                NetworkError,
                RateLimitError,
            )
        )

        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _get_timestamp(self) -> str:
        """Get current timestamp in milliseconds."""
        return str(int(time.time() * 1000))

    def _sign_request(self, method: str, path: str, timestamp: str, body: str = "") -> str:
        """
        Sign a request using HMAC-SHA256.

        The signature is: HMAC-SHA256(private_key, timestamp + method + path + body)
        """
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.private_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an authenticated API request."""
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        timestamp = self._get_timestamp()
        body = json.dumps(data) if data else ""

        # Parse path from URL for signing
        path = "/" + endpoint.lstrip("/")

        signature = self._sign_request(method.upper(), path, timestamp, body)

        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

        @with_retry(self.retry_config)
        def do_request() -> requests.Response:
            response = self._session.request(
                method=method.upper(),
                url=url,
                params=params,
                data=body if data else None,
                headers=headers,
                timeout=30,
            )
            return response

        try:
            response = do_request()
        except Exception as e:
            raise NetworkError(f"Network error: {str(e)}") from e

        # Handle response
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=float(retry_after) if retry_after else None
            )

        if response.status_code == 401:
            raise AuthenticationError("Authentication failed", status_code=401)

        if response.status_code == 403:
            raise AuthenticationError("Access forbidden", status_code=403)

        try:
            result = response.json()
        except json.JSONDecodeError:
            result = {"raw": response.text}

        if response.status_code >= 400:
            error_msg = result.get("error", result.get("message", str(result)))
            raise APIError(
                f"API error: {error_msg}",
                status_code=response.status_code,
                response=result
            )

        return result

    def get_balance(self) -> AccountBalance:
        """Get account balance."""
        result = self._request("GET", "/portfolio/balance")
        balance = result.get("balance", {})
        return AccountBalance(
            available_balance=balance.get("available_balance", 0) / 100,  # Convert cents to dollars
            portfolio_value=balance.get("portfolio_value", 0) / 100,
            total_deposits=balance.get("total_deposits", 0) / 100,
            total_withdrawals=balance.get("total_withdrawals", 0) / 100,
        )

    def get_positions(self) -> List[Position]:
        """Get all open positions."""
        result = self._request("GET", "/portfolio/positions")
        positions = []
        for pos in result.get("market_positions", []):
            positions.append(Position(
                market_ticker=pos.get("ticker", ""),
                yes_contracts=pos.get("position", 0) if pos.get("position", 0) > 0 else 0,
                no_contracts=abs(pos.get("position", 0)) if pos.get("position", 0) < 0 else 0,
                average_yes_price=pos.get("average_price", 0) / 100 if pos.get("position", 0) > 0 else 0,
                average_no_price=pos.get("average_price", 0) / 100 if pos.get("position", 0) < 0 else 0,
                realized_pnl=pos.get("realized_pnl", 0) / 100,
            ))
        return positions

    def get_position(self, market_ticker: str) -> Optional[Position]:
        """Get position for a specific market."""
        positions = self.get_positions()
        for pos in positions:
            if pos.market_ticker == market_ticker:
                return pos
        return None

    def get_market(self, market_ticker: str) -> MarketInfo:
        """Get market information."""
        result = self._request("GET", f"/markets/{market_ticker}")
        market = result.get("market", {})
        return MarketInfo(
            ticker=market.get("ticker", market_ticker),
            title=market.get("title", ""),
            subtitle=market.get("subtitle", ""),
            yes_bid=market.get("yes_bid", 0) / 100 if market.get("yes_bid") else None,
            yes_ask=market.get("yes_ask", 0) / 100 if market.get("yes_ask") else None,
            no_bid=market.get("no_bid", 0) / 100 if market.get("no_bid") else None,
            no_ask=market.get("no_ask", 0) / 100 if market.get("no_ask") else None,
            volume=market.get("volume", 0),
            open_interest=market.get("open_interest", 0),
            status=market.get("status", "unknown"),
        )

    def search_markets(self, query: str, limit: int = 10) -> List[MarketInfo]:
        """Search for markets."""
        result = self._request("GET", "/markets", params={"query": query, "limit": limit})
        markets = []
        for market in result.get("markets", []):
            markets.append(MarketInfo(
                ticker=market.get("ticker", ""),
                title=market.get("title", ""),
                subtitle=market.get("subtitle", ""),
                yes_bid=market.get("yes_bid", 0) / 100 if market.get("yes_bid") else None,
                yes_ask=market.get("yes_ask", 0) / 100 if market.get("yes_ask") else None,
                volume=market.get("volume", 0),
                open_interest=market.get("open_interest", 0),
                status=market.get("status", "unknown"),
            ))
        return markets

    def place_order(
        self,
        market_ticker: str,
        side: OrderSide,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Order:
        """
        Place an order.

        Args:
            market_ticker: Market ticker (e.g., "BTCUSD-24JAN01").
            side: YES or NO.
            action: BUY or SELL.
            quantity: Number of contracts.
            order_type: MARKET or LIMIT.
            price: Price in cents (1-99) for limit orders.

        Returns:
            Order object with order details.
        """
        order_data = {
            "ticker": market_ticker,
            "side": side.value,
            "action": action.value,
            "type": order_type.value,
            "count": quantity,
        }

        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("Price is required for limit orders")
            order_data["yes_price"] = int(price * 100)  # Convert to cents

        try:
            result = self._request("POST", "/portfolio/orders", data=order_data)
        except APIError as e:
            if "insufficient" in str(e).lower():
                raise InsufficientFundsError(str(e), status_code=e.status_code)
            raise OrderError(str(e), status_code=e.status_code)

        order = result.get("order", {})
        return Order(
            order_id=order.get("order_id", ""),
            market_ticker=market_ticker,
            side=side,
            action=action,
            order_type=order_type,
            price=price,
            quantity=quantity,
            filled_quantity=order.get("filled_count", 0),
            status=order.get("status", "pending"),
            created_at=datetime.now(timezone.utc),
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        try:
            self._request("DELETE", f"/portfolio/orders/{order_id}")
            return True
        except APIError:
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order status."""
        result = self._request("GET", f"/portfolio/orders/{order_id}")
        order = result.get("order", {})
        return Order(
            order_id=order.get("order_id", order_id),
            market_ticker=order.get("ticker", ""),
            side=OrderSide(order.get("side", "yes")),
            action=OrderAction(order.get("action", "buy")),
            order_type=OrderType(order.get("type", "market")),
            price=order.get("yes_price", 0) / 100 if order.get("yes_price") else None,
            quantity=order.get("count", 0),
            filled_quantity=order.get("filled_count", 0),
            status=order.get("status", "unknown"),
        )

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        result = self._request("GET", "/portfolio/orders", params={"status": "resting"})
        orders = []
        for order in result.get("orders", []):
            orders.append(Order(
                order_id=order.get("order_id", ""),
                market_ticker=order.get("ticker", ""),
                side=OrderSide(order.get("side", "yes")),
                action=OrderAction(order.get("action", "buy")),
                order_type=OrderType(order.get("type", "market")),
                price=order.get("yes_price", 0) / 100 if order.get("yes_price") else None,
                quantity=order.get("count", 0),
                filled_quantity=order.get("filled_count", 0),
                status=order.get("status", "resting"),
            ))
        return orders


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    timestamp: datetime
    market_ticker: str
    side: OrderSide
    action: OrderAction
    quantity: int
    price: float
    pnl: float = 0.0


class KalshiPaperClient(BaseKalshiClient):
    """
    Paper trading client for testing strategies without real money.

    Simulates Kalshi API behavior with virtual balance and positions.

    Example:
        client = KalshiPaperClient(initial_balance=10000.0)

        # Place simulated orders
        order = client.place_order(
            market_ticker="BTCUSD-TEST",
            side=OrderSide.YES,
            action=OrderAction.BUY,
            quantity=100,
            price=0.55
        )

        # Check paper balance
        balance = client.get_balance()

        # Get trade history
        trades = client.get_trade_history()
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission_rate: float = 0.0,  # Kalshi has no commission but we support it
    ):
        """
        Initialize paper trading client.

        Args:
            initial_balance: Starting balance in dollars.
            commission_rate: Commission rate (0.0 = no commission).
        """
        self.initial_balance = initial_balance
        self.available_balance = initial_balance
        self.commission_rate = commission_rate

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._trade_history: List[PaperTrade] = []
        self._order_counter = 0
        self._market_prices: Dict[str, Dict[str, float]] = {}

        logger.info("Paper trading client initialized with $%.2f", initial_balance)

    def set_market_price(self, market_ticker: str, yes_price: float, no_price: Optional[float] = None):
        """Set simulated market prices for a ticker."""
        if no_price is None:
            no_price = 1.0 - yes_price
        self._market_prices[market_ticker] = {
            "yes_bid": yes_price - 0.01,
            "yes_ask": yes_price + 0.01,
            "no_bid": no_price - 0.01,
            "no_ask": no_price + 0.01,
        }

    def get_balance(self) -> AccountBalance:
        """Get paper account balance."""
        portfolio_value = self._calculate_portfolio_value()
        return AccountBalance(
            available_balance=self.available_balance,
            portfolio_value=portfolio_value,
            total_deposits=self.initial_balance,
            total_withdrawals=0.0,
        )

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value including positions."""
        total = self.available_balance
        for pos in self._positions.values():
            prices = self._market_prices.get(pos.market_ticker, {})
            yes_price = prices.get("yes_bid", 0.5)
            no_price = prices.get("no_bid", 0.5)
            total += pos.yes_contracts * yes_price
            total += pos.no_contracts * no_price
        return total

    def get_positions(self) -> List[Position]:
        """Get all paper positions."""
        return list(self._positions.values())

    def get_position(self, market_ticker: str) -> Optional[Position]:
        """Get paper position for a specific market."""
        return self._positions.get(market_ticker)

    def get_market(self, market_ticker: str) -> MarketInfo:
        """Get simulated market information."""
        prices = self._market_prices.get(market_ticker, {})
        return MarketInfo(
            ticker=market_ticker,
            title=f"Paper Market: {market_ticker}",
            subtitle="Simulated",
            yes_bid=prices.get("yes_bid", 0.49),
            yes_ask=prices.get("yes_ask", 0.51),
            no_bid=prices.get("no_bid", 0.49),
            no_ask=prices.get("no_ask", 0.51),
            volume=0,
            open_interest=0,
            status="active",
        )

    def search_markets(self, query: str, limit: int = 10) -> List[MarketInfo]:
        """Search simulated markets."""
        # Return markets that have been set up
        markets = []
        for ticker in self._market_prices.keys():
            if query.lower() in ticker.lower():
                markets.append(self.get_market(ticker))
        return markets[:limit]

    def place_order(
        self,
        market_ticker: str,
        side: OrderSide,
        action: OrderAction,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
    ) -> Order:
        """Place a paper order (immediately filled for market orders)."""
        self._order_counter += 1
        order_id = f"PAPER-{self._order_counter:06d}"

        # Get execution price
        prices = self._market_prices.get(market_ticker, {})
        if order_type == OrderType.MARKET:
            if side == OrderSide.YES:
                exec_price = prices.get("yes_ask", 0.51) if action == OrderAction.BUY else prices.get("yes_bid", 0.49)
            else:
                exec_price = prices.get("no_ask", 0.51) if action == OrderAction.BUY else prices.get("no_bid", 0.49)
        else:
            exec_price = price if price else 0.50

        # Calculate cost
        cost = quantity * exec_price
        commission = cost * self.commission_rate

        # Check balance for buys
        if action == OrderAction.BUY:
            total_cost = cost + commission
            if total_cost > self.available_balance:
                raise InsufficientFundsError(
                    f"Insufficient funds. Need ${total_cost:.2f}, have ${self.available_balance:.2f}"
                )
            self.available_balance -= total_cost
        else:
            # For sells, add to balance (minus commission)
            self.available_balance += cost - commission

        # Update position
        if market_ticker not in self._positions:
            self._positions[market_ticker] = Position(market_ticker=market_ticker)

        pos = self._positions[market_ticker]
        if side == OrderSide.YES:
            if action == OrderAction.BUY:
                # Update average price
                total_value = pos.yes_contracts * pos.average_yes_price + quantity * exec_price
                pos.yes_contracts += quantity
                pos.average_yes_price = total_value / pos.yes_contracts if pos.yes_contracts > 0 else 0
            else:
                pos.yes_contracts = max(0, pos.yes_contracts - quantity)
        else:
            if action == OrderAction.BUY:
                total_value = pos.no_contracts * pos.average_no_price + quantity * exec_price
                pos.no_contracts += quantity
                pos.average_no_price = total_value / pos.no_contracts if pos.no_contracts > 0 else 0
            else:
                pos.no_contracts = max(0, pos.no_contracts - quantity)

        # Clean up empty positions
        if pos.yes_contracts == 0 and pos.no_contracts == 0:
            del self._positions[market_ticker]

        # Record trade
        trade = PaperTrade(
            timestamp=datetime.now(timezone.utc),
            market_ticker=market_ticker,
            side=side,
            action=action,
            quantity=quantity,
            price=exec_price,
        )
        self._trade_history.append(trade)

        # Create order (immediately filled for paper trading)
        order = Order(
            order_id=order_id,
            market_ticker=market_ticker,
            side=side,
            action=action,
            order_type=order_type,
            price=exec_price,
            quantity=quantity,
            filled_quantity=quantity,
            status="filled",
            created_at=datetime.now(timezone.utc),
        )
        self._orders[order_id] = order

        logger.info(
            "Paper trade: %s %s %d %s @ $%.2f (Balance: $%.2f)",
            action.value.upper(),
            side.value.upper(),
            quantity,
            market_ticker,
            exec_price,
            self.available_balance,
        )

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order (no-op for paper trading since orders are instant)."""
        return order_id in self._orders

    def get_order(self, order_id: str) -> Order:
        """Get paper order status."""
        if order_id not in self._orders:
            raise OrderError(f"Order not found: {order_id}")
        return self._orders[order_id]

    def get_open_orders(self) -> List[Order]:
        """Get open orders (always empty for paper trading)."""
        return []

    def get_trade_history(self) -> List[PaperTrade]:
        """Get paper trade history."""
        return self._trade_history.copy()

    def reset(self):
        """Reset paper trading state."""
        self.available_balance = self.initial_balance
        self._positions.clear()
        self._orders.clear()
        self._trade_history.clear()
        self._order_counter = 0
        logger.info("Paper trading state reset")
