"""
Live trading loop with scheduling.

Provides a continuous trading loop that:
- Fetches fresh market data
- Generates predictions
- Executes trades based on signals
- Monitors positions for stop-loss/take-profit
"""

from __future__ import annotations

import copy
import logging
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False

from ..config import get_data_config, get_feature_config, get_model_config
from ..data import build_lookahead_dataset
from ..features import create_features
from ..model import LookaheadModel
from .signals import SignalGenerator, SignalConfig, TradingSignal
from .executor import TradingExecutor, ExecutorConfig, TradingMode

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    """Configuration for the trading loop."""

    # Timing
    interval_minutes: int = 15  # Run every 15 minutes
    align_to_candle: bool = True  # Align to candle boundaries
    startup_delay_seconds: int = 10  # Delay before first run

    # Data
    lookback_hours: int = 96  # Hours of data to fetch
    symbol: str = "BTC/USDT"

    # Model
    model_path: Optional[str] = None

    # Trading
    trading_mode: TradingMode = TradingMode.PAPER
    initial_capital: float = 10000.0
    max_position_size: float = 0.20
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06

    # Risk
    max_daily_loss: float = 500.0  # Stop trading if daily loss exceeds this
    max_consecutive_losses: int = 5  # Stop trading after N consecutive losses

    # Callbacks
    on_signal: Optional[Callable[[TradingSignal], None]] = None
    on_trade: Optional[Callable[[Dict[str, Any]], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None

    # Persistence
    state_file: Optional[str] = None  # File to save/load state


@dataclass
class LoopState:
    """State of the trading loop."""
    is_running: bool = False
    last_run: Optional[datetime] = None
    last_signal: Optional[TradingSignal] = None
    run_count: int = 0
    error_count: int = 0
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    daily_reset_date: Optional[datetime] = None
    trading_halted: bool = False
    halt_reason: Optional[str] = None


class TradingLoop:
    """
    Live trading loop with scheduling.

    Runs on a fixed interval (default 15 minutes) to:
    1. Fetch fresh market data
    2. Generate features
    3. Run model predictions
    4. Generate and execute trading signals
    5. Monitor positions for stop-loss/take-profit

    Example:
        loop = TradingLoop(LoopConfig(
            interval_minutes=15,
            trading_mode=TradingMode.PAPER,
            initial_capital=10000.0,
            model_path="models/BTCUSDT_15m_lookahead.pkl"
        ))

        # Start the loop
        loop.start()

        # Run indefinitely (or until Ctrl+C)
        try:
            loop.wait()
        except KeyboardInterrupt:
            loop.stop()

        # Or run once for testing
        loop.run_once()
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        if not SCHEDULER_AVAILABLE:
            logger.warning(
                "APScheduler not available. Install with: pip install apscheduler. "
                "Manual loop mode will be used."
            )

        self.config = config or LoopConfig()
        self.state = LoopState()

        # Initialize components
        self._init_model()
        self._init_signal_generator()
        self._init_executor()
        self._init_scheduler()

        # Shutdown handling
        self._shutdown_event = threading.Event()
        self._setup_signal_handlers()

        logger.info(
            "TradingLoop initialized: %s mode, %d min interval, symbol=%s",
            self.config.trading_mode.value,
            self.config.interval_minutes,
            self.config.symbol
        )

    def _init_model(self):
        """Initialize the prediction model."""
        self.model: Optional[LookaheadModel] = None

        if self.config.model_path:
            try:
                self.model = LookaheadModel.load(self.config.model_path)
                logger.info("Loaded model from %s", self.config.model_path)
            except Exception as e:
                logger.error("Failed to load model: %s", e)
                raise

    def _init_signal_generator(self):
        """Initialize the signal generator."""
        model_config = get_model_config()
        thresholds = model_config.get('thresholds', {})

        signal_config = SignalConfig(
            probability_long=thresholds.get('probability_long', 0.55),
            probability_short=thresholds.get('probability_short', 0.45),
            min_expected_return=thresholds.get('min_expected_return', 0.0003),
            max_position_size=self.config.max_position_size,
        )
        self.signal_generator = SignalGenerator(signal_config)

    def _init_executor(self):
        """Initialize the trade executor."""
        executor_config = ExecutorConfig(
            mode=self.config.trading_mode,
            initial_capital=self.config.initial_capital,
            stop_loss_pct=self.config.stop_loss_pct,
            take_profit_pct=self.config.take_profit_pct,
        )
        self.executor = TradingExecutor(executor_config)

    def _init_scheduler(self):
        """Initialize the scheduler."""
        self.scheduler: Optional[Any] = None

        if SCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _fetch_data(self) -> Optional[pd.DataFrame]:
        """Fetch fresh market data."""
        try:
            data_config = get_data_config()

            # Override history for live trading
            live_history_days = max(2, math.ceil(self.config.lookback_hours / 24))
            live_data_config = copy.deepcopy(data_config)
            live_data_config['history_days'] = live_history_days

            dataset, metadata = build_lookahead_dataset(live_data_config)

            logger.info("Fetched %d rows of data", len(dataset))
            return dataset

        except Exception as e:
            logger.error("Failed to fetch data: %s", e)
            self.state.error_count += 1
            if self.config.on_error:
                self.config.on_error(e)
            return None

    def _generate_features(self, dataset: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate features from raw data."""
        try:
            feature_config = get_feature_config()
            features = create_features(dataset, config=feature_config)
            logger.debug("Generated %d features for %d rows", len(features.columns), len(features))
            return features

        except Exception as e:
            logger.error("Failed to generate features: %s", e)
            self.state.error_count += 1
            if self.config.on_error:
                self.config.on_error(e)
            return None

    def _generate_prediction(self, features: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate model predictions."""
        if self.model is None:
            logger.error("No model loaded")
            return None

        try:
            # Use only the latest row for prediction
            latest_features = features.iloc[[-1]]
            predictions = self.model.predict(latest_features)
            return predictions

        except Exception as e:
            logger.error("Failed to generate predictions: %s", e)
            self.state.error_count += 1
            if self.config.on_error:
                self.config.on_error(e)
            return None

    def _check_risk_limits(self) -> bool:
        """Check if trading should be halted due to risk limits."""
        # Reset daily P&L at midnight UTC
        now = datetime.now(timezone.utc)
        if self.state.daily_reset_date is None or now.date() > self.state.daily_reset_date.date():
            self.state.daily_pnl = 0.0
            self.state.daily_reset_date = now
            self.state.consecutive_losses = 0

        # Check daily loss limit
        if self.state.daily_pnl <= -self.config.max_daily_loss:
            self.state.trading_halted = True
            self.state.halt_reason = f"Daily loss limit exceeded: ${abs(self.state.daily_pnl):.2f}"
            logger.warning(self.state.halt_reason)
            return False

        # Check consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self.state.trading_halted = True
            self.state.halt_reason = f"Consecutive loss limit: {self.state.consecutive_losses} losses"
            logger.warning(self.state.halt_reason)
            return False

        return True

    def _run_iteration(self):
        """Run a single iteration of the trading loop."""
        iteration_start = datetime.now(timezone.utc)
        self.state.run_count += 1

        logger.info(
            "=== Trading Loop Iteration %d at %s ===",
            self.state.run_count,
            iteration_start.strftime("%Y-%m-%d %H:%M:%S UTC")
        )

        # Check risk limits
        if not self._check_risk_limits():
            logger.warning("Trading halted: %s", self.state.halt_reason)
            return

        # Step 1: Fetch data
        dataset = self._fetch_data()
        if dataset is None or dataset.empty:
            logger.warning("No data available, skipping iteration")
            return

        # Step 2: Generate features
        features = self._generate_features(dataset)
        if features is None or features.empty:
            logger.warning("Failed to generate features, skipping iteration")
            return

        # Step 3: Generate predictions
        predictions = self._generate_prediction(features)
        if predictions is None or predictions.empty:
            logger.warning("Failed to generate predictions, skipping iteration")
            return

        # Step 4: Generate signal
        try:
            signal = self.signal_generator.generate(
                predictions_df=predictions,
                features_df=features,
                symbol=self.config.symbol,
            )
            self.state.last_signal = signal

            if self.config.on_signal:
                self.config.on_signal(signal)

        except Exception as e:
            logger.error("Failed to generate signal: %s", e)
            self.state.error_count += 1
            return

        # Step 5: Execute trade
        if signal.is_trade:
            try:
                trade = self.executor.execute_signal(signal)

                if trade and self.config.on_trade:
                    self.config.on_trade({
                        "trade": trade,
                        "signal": signal.to_dict(),
                        "portfolio": self.executor.get_portfolio_summary(),
                    })

            except Exception as e:
                logger.error("Failed to execute trade: %s", e)
                self.state.error_count += 1

        # Step 6: Check stop-loss/take-profit for existing positions
        for position in self.executor.get_all_positions():
            try:
                # Get current price from latest features
                current_price = signal.current_price
                sl_tp_trade = self.executor.check_stop_loss_take_profit(
                    position.symbol, current_price
                )

                if sl_tp_trade:
                    pnl = sl_tp_trade.metadata.get('pnl', 0)
                    self.state.daily_pnl += pnl

                    if pnl < 0:
                        self.state.consecutive_losses += 1
                    else:
                        self.state.consecutive_losses = 0

            except Exception as e:
                logger.error("Failed to check SL/TP: %s", e)

        self.state.last_run = datetime.now(timezone.utc)

        # Log summary
        portfolio = self.executor.get_portfolio_summary()
        logger.info(
            "Iteration complete: Signal=%s, Balance=$%.2f, P&L=$%.2f, Positions=%d",
            signal.action.value,
            portfolio['balance'],
            portfolio['total_pnl'],
            portfolio['num_positions']
        )

    def run_once(self):
        """Run a single iteration of the loop (for testing)."""
        self._run_iteration()

    def start(self):
        """Start the trading loop."""
        if self.state.is_running:
            logger.warning("Trading loop is already running")
            return

        self.state.is_running = True
        self.state.trading_halted = False
        self.state.halt_reason = None

        if self.scheduler is not None:
            # Calculate next aligned run time
            if self.config.align_to_candle:
                # Align to candle boundaries (e.g., :00, :15, :30, :45 for 15m)
                trigger = CronTrigger(
                    minute=f"*/{self.config.interval_minutes}",
                    second=self.config.startup_delay_seconds % 60,
                )
            else:
                trigger = IntervalTrigger(minutes=self.config.interval_minutes)

            self.scheduler.add_job(
                self._run_iteration,
                trigger=trigger,
                id="trading_loop",
                name="Trading Loop",
                replace_existing=True,
            )
            self.scheduler.start()

            logger.info(
                "Trading loop started with %d minute interval (aligned=%s)",
                self.config.interval_minutes,
                self.config.align_to_candle
            )

            # Run first iteration after startup delay
            if self.config.startup_delay_seconds > 0:
                logger.info("First run in %d seconds...", self.config.startup_delay_seconds)
                time.sleep(self.config.startup_delay_seconds)
                self._run_iteration()
        else:
            # Manual loop mode (no APScheduler)
            logger.info("Running in manual loop mode (no APScheduler)")
            self._manual_loop()

    def _manual_loop(self):
        """Manual loop when APScheduler is not available."""
        interval_seconds = self.config.interval_minutes * 60

        while self.state.is_running and not self._shutdown_event.is_set():
            try:
                self._run_iteration()
            except Exception as e:
                logger.error("Error in trading loop: %s", e)
                self.state.error_count += 1

            # Wait for next interval
            self._shutdown_event.wait(interval_seconds)

    def stop(self):
        """Stop the trading loop."""
        logger.info("Stopping trading loop...")
        self.state.is_running = False
        self._shutdown_event.set()

        if self.scheduler is not None and self.scheduler.running:
            self.scheduler.shutdown(wait=False)

        # Close all positions on shutdown
        for position in self.executor.get_all_positions():
            self.executor.close_position(position.symbol, reason="shutdown")

        # Log final summary
        portfolio = self.executor.get_portfolio_summary()
        logger.info(
            "Trading loop stopped. Final balance: $%.2f, Total P&L: $%.2f, Trades: %d",
            portfolio['balance'],
            portfolio['total_pnl'],
            portfolio['num_trades']
        )

    def wait(self):
        """Wait for the trading loop to stop (blocking)."""
        if self.scheduler is not None:
            try:
                while self.state.is_running:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()
        else:
            self._shutdown_event.wait()

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the trading loop."""
        return {
            "is_running": self.state.is_running,
            "last_run": self.state.last_run.isoformat() if self.state.last_run else None,
            "run_count": self.state.run_count,
            "error_count": self.state.error_count,
            "trading_halted": self.state.trading_halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "consecutive_losses": self.state.consecutive_losses,
            "last_signal": self.state.last_signal.to_dict() if self.state.last_signal else None,
            "portfolio": self.executor.get_portfolio_summary(),
        }


def run_trading_bot(
    model_path: str,
    mode: str = "paper",
    interval_minutes: int = 15,
    initial_capital: float = 10000.0,
):
    """
    Convenience function to run the trading bot.

    Args:
        model_path: Path to the trained model file.
        mode: Trading mode ("paper" or "live").
        interval_minutes: Run interval in minutes.
        initial_capital: Initial capital for paper trading.
    """
    trading_mode = TradingMode.PAPER if mode == "paper" else TradingMode.LIVE

    config = LoopConfig(
        interval_minutes=interval_minutes,
        trading_mode=trading_mode,
        initial_capital=initial_capital,
        model_path=model_path,
    )

    loop = TradingLoop(config)

    print(f"Starting trading bot in {mode} mode...")
    print(f"Model: {model_path}")
    print(f"Interval: {interval_minutes} minutes")
    print(f"Initial capital: ${initial_capital}")
    print("Press Ctrl+C to stop\n")

    loop.start()
    loop.wait()
