# Crypto15 - The Crypto Bot for Kalshi

A comprehensive cryptocurrency trading bot framework with data fetching, feature engineering, XGBoost modeling, and backtesting capabilities.

## Repository Structure

```
crypto15/
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration file
├── src/
│   └── crypto15/
│       ├── __init__.py          # Main package
│       ├── config.py            # Configuration management
│       ├── util.py              # Utility functions
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetch_ccxt.py    # CCXT data fetching
│       │   └── store.py         # Data storage
│       ├── features/
│       │   ├── __init__.py
│       │   └── feature_set_v1.py # Feature engineering
│       ├── model/
│       │   ├── __init__.py
│       │   └── xgb.py           # XGBoost model
│       └── backtest/
│           ├── __init__.py
│           ├── walkforward.py   # Walk-forward backtesting
│           └── sim.py           # Trading simulator
└── scripts/
    ├── fetch_history.py         # Fetch historical data
    ├── walkforward_backtest.py  # Run walk-forward backtest
    ├── train_full_and_save.py   # Train and save model
    └── live_predict.py          # Live predictions
```

## Installation

1. Install dependencies:
```bash
cd crypto15
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to customize:
- Exchange and trading pairs
- Timeframes and historical data period
- Feature engineering parameters
- Model hyperparameters
- Backtest settings
- Trading parameters (position size, stop loss, take profit)

## Usage

### 1. Fetch Historical Data

```bash
cd crypto15
python scripts/fetch_history.py
```

This will fetch historical OHLCV data for the symbols specified in `config.yaml`.

### 2. Run Walk-Forward Backtest

```bash
python scripts/walkforward_backtest.py
```

Performs walk-forward backtesting with multiple train/test splits.

### 3. Train Model on Full Dataset

```bash
python scripts/train_full_and_save.py
```

Trains an XGBoost model on the full dataset and saves it for later use.

### 4. Make Live Predictions

```bash
python scripts/live_predict.py
```

Fetches latest data and makes predictions using the saved model.

## Modules

### crypto15.data
- **fetch_ccxt**: Fetch OHLCV data from crypto exchanges using CCXT
- **store**: Save and load data in various formats (parquet, csv, pickle)

### crypto15.features
- **feature_set_v1**: Technical indicators and features:
  - Price returns
  - Rolling statistics (MA, STD)
  - Volume features
  - Momentum indicators (RSI, MACD)
  - Volatility features (ATR, High-Low range)

### crypto15.model
- **xgb**: XGBoost regression model for price prediction
  - Training and prediction
  - Feature importance
  - Model persistence

### crypto15.backtest
- **walkforward**: Walk-forward backtesting framework
- **sim**: Trading simulator with:
  - Position management
  - Stop loss and take profit
  - Performance metrics (returns, Sharpe ratio, max drawdown)

## Development

The package is structured for easy extension:
- Add new feature sets in `crypto15/features/`
- Add new models in `crypto15/model/`
- Add new backtesting strategies in `crypto15/backtest/`
- Create new scripts for different workflows

## License

MIT License
