"""
Trading simulator for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Container for backtest results.
    """
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    
    def summary(self) -> str:
        """Get summary string of results."""
        lines = ["Backtest Results:"]
        for key, value in self.metrics.items():
            lines.append(f"  {key}: {value:.4f}")
        return "\n".join(lines)


class TradingSimulator:
    """
    Simple trading simulator for backtesting.
    """
    
    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_size: float = 0.1,
        stop_loss: float = 0.02,
        take_profit: float = 0.05,
        commission: float = 0.001,
        annualization_factor: float = 252
    ):
        """
        Initialize trading simulator.
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size as fraction of capital
            stop_loss: Stop loss as fraction (e.g., 0.02 = 2%)
            take_profit: Take profit as fraction
            commission: Trading commission as fraction
            annualization_factor: Factor for annualizing returns (252 for daily, 252*24 for hourly)
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.commission = commission
        self.annualization_factor = annualization_factor
    
    def simulate(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        price_col: str = 'close'
    ) -> BacktestResults:
        """
        Simulate trading based on signals.
        
        Args:
            df: DataFrame with price data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            price_col: Name of price column
        
        Returns:
            BacktestResults object
        """
        logger.info(f"Simulating trading on {len(df)} periods")
        
        capital = self.initial_capital
        position = 0.0
        entry_price = 0.0
        
        equity = []
        trades = []
        
        for i in range(len(df)):
            price = df[price_col].iloc[i]
            signal = signals.iloc[i]
            
            # Check stop loss and take profit
            if position != 0:
                pnl_pct = (price - entry_price) / entry_price * np.sign(position)
                
                if pnl_pct <= -self.stop_loss:
                    # Stop loss triggered
                    capital = capital + position * price * (1 - self.commission)
                    trades.append({
                        'index': i,
                        'type': 'stop_loss',
                        'price': price,
                        'position': -position,
                        'pnl': position * (price - entry_price) * (1 - self.commission)
                    })
                    position = 0.0
                    entry_price = 0.0
                
                elif pnl_pct >= self.take_profit:
                    # Take profit triggered
                    capital = capital + position * price * (1 - self.commission)
                    trades.append({
                        'index': i,
                        'type': 'take_profit',
                        'price': price,
                        'position': -position,
                        'pnl': position * (price - entry_price) * (1 - self.commission)
                    })
                    position = 0.0
                    entry_price = 0.0
            
            # Process signal
            if signal == 1 and position == 0:
                # Buy signal
                position_size = capital * self.max_position_size
                shares = position_size / price
                position = shares
                entry_price = price
                capital = capital - position_size * (1 + self.commission)
                
                trades.append({
                    'index': i,
                    'type': 'buy',
                    'price': price,
                    'position': shares,
                    'pnl': 0.0
                })
            
            elif signal == -1 and position != 0:
                # Sell signal
                capital = capital + position * price * (1 - self.commission)
                pnl = position * (price - entry_price) * (1 - self.commission)
                
                trades.append({
                    'index': i,
                    'type': 'sell',
                    'price': price,
                    'position': -position,
                    'pnl': pnl
                })
                
                position = 0.0
                entry_price = 0.0
            
            # Calculate equity
            current_equity = capital
            if position != 0:
                current_equity += position * price
            
            equity.append(current_equity)
        
        # Create results
        equity_curve = pd.Series(equity, index=df.index)
        trades_df = pd.DataFrame(trades)
        
        metrics = self.calculate_metrics(equity_curve, trades_df)
        
        return BacktestResults(
            equity_curve=equity_curve,
            trades=trades_df,
            metrics=metrics
        )
    
    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            equity_curve: Series of equity over time
            trades_df: DataFrame of trades
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Total return
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        metrics['total_return'] = total_return
        
        # Sharpe ratio (annualized)
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0:
            sharpe = np.sqrt(self.annualization_factor) * returns.mean() / returns.std() if returns.std() > 0 else 0
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Max drawdown
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        metrics['max_drawdown'] = drawdown.min()
        
        # Trade statistics
        if len(trades_df) > 0:
            metrics['num_trades'] = len(trades_df)
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            metrics['win_rate'] = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            metrics['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        else:
            metrics['num_trades'] = 0
            metrics['win_rate'] = 0.0
            metrics['avg_win'] = 0.0
            metrics['avg_loss'] = 0.0
        
        return metrics
