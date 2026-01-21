#!/usr/bin/env python3
"""
Run the live trading bot.

Usage:
    # Paper trading (default)
    python run_trading_bot.py

    # Paper trading with custom capital
    python run_trading_bot.py --capital 5000

    # Live trading (requires Kalshi API credentials)
    python run_trading_bot.py --mode live

    # Run a single iteration (for testing)
    python run_trading_bot.py --once
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crypto15.config import get_data_config, load_config
from crypto15.trading import TradingLoop, LoopConfig, TradingMode
from crypto15.util import setup_logging

logger = setup_logging()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the crypto trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_trading_bot.py                    # Paper trading
  python run_trading_bot.py --mode live        # Live trading
  python run_trading_bot.py --capital 5000     # Custom capital
  python run_trading_bot.py --interval 5       # 5-minute interval
  python run_trading_bot.py --once             # Single iteration
        """
    )

    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Trading mode (default: paper)"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital for paper trading (default: 10000)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Trading interval in minutes (default: 15)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (auto-detected if not provided)"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol (default: from config)"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration and exit"
    )

    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.03,
        help="Stop-loss percentage (default: 0.03 = 3%%)"
    )

    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.06,
        help="Take-profit percentage (default: 0.06 = 6%%)"
    )

    parser.add_argument(
        "--max-position",
        type=float,
        default=0.20,
        help="Maximum position size as fraction of capital (default: 0.20 = 20%%)"
    )

    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=500.0,
        help="Maximum daily loss before halting (default: 500)"
    )

    return parser.parse_args()


def find_model_file(symbol_id: str, timeframe: str) -> str:
    """Find the model file for the given symbol and timeframe."""
    models_dir = Path(__file__).parent.parent / "models"

    # Try exact match first
    model_file = models_dir / f"{symbol_id}_{timeframe}_lookahead.pkl"
    if model_file.exists():
        return str(model_file)

    # Try finding any matching model
    for pattern in [f"{symbol_id}_*_lookahead.pkl", "*_lookahead.pkl"]:
        matches = list(models_dir.glob(pattern))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(
        f"No model file found in {models_dir}. "
        "Please train a model first with train_full_and_save.py"
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    data_config = get_data_config()
    full_config = load_config()

    # Determine symbol
    symbol = args.symbol or data_config.get('primary_symbol', 'BTC/USDT')
    symbol_id = data_config.get('primary_symbol_id', 'BTCUSDT')
    timeframe = data_config.get('target_timeframe', '15m')

    # Find model file
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = find_model_file(symbol_id, timeframe)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Crypto Trading Bot")
    logger.info("=" * 60)
    logger.info("Mode: %s", args.mode.upper())
    logger.info("Symbol: %s", symbol)
    logger.info("Timeframe: %s", timeframe)
    logger.info("Interval: %d minutes", args.interval)
    logger.info("Model: %s", model_path)
    logger.info("Initial Capital: $%.2f", args.capital)
    logger.info("Stop-Loss: %.1f%%", args.stop_loss * 100)
    logger.info("Take-Profit: %.1f%%", args.take_profit * 100)
    logger.info("Max Position: %.1f%%", args.max_position * 100)
    logger.info("Max Daily Loss: $%.2f", args.max_daily_loss)
    logger.info("=" * 60)

    # Create trading loop config
    trading_mode = TradingMode.LIVE if args.mode == "live" else TradingMode.PAPER

    loop_config = LoopConfig(
        interval_minutes=args.interval,
        symbol=symbol,
        model_path=model_path,
        trading_mode=trading_mode,
        initial_capital=args.capital,
        max_position_size=args.max_position,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_daily_loss=args.max_daily_loss,
        on_signal=lambda s: logger.info("Signal: %s", s.to_dict()),
        on_trade=lambda t: logger.info("Trade executed: %s", t),
        on_error=lambda e: logger.error("Error: %s", e),
    )

    # Create and run trading loop
    loop = TradingLoop(loop_config)

    if args.once:
        logger.info("Running single iteration...")
        loop.run_once()
        status = loop.get_status()
        logger.info("Status: %s", status)
    else:
        logger.info("Starting trading loop... (Press Ctrl+C to stop)")
        try:
            loop.start()
            loop.wait()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            loop.stop()

        # Print final summary
        status = loop.get_status()
        logger.info("=" * 60)
        logger.info("Final Summary")
        logger.info("=" * 60)
        logger.info("Total Iterations: %d", status['run_count'])
        logger.info("Total Errors: %d", status['error_count'])
        portfolio = status['portfolio']
        logger.info("Final Balance: $%.2f", portfolio['balance'])
        logger.info("Total P&L: $%.2f", portfolio['total_pnl'])
        logger.info("Total Trades: %d", portfolio['num_trades'])
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
