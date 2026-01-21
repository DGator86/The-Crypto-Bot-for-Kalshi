# Crypto15 - The Crypto Bot for Kalshi

A comprehensive cryptocurrency trading bot framework with rich data ingestion, feature engineering, ensemble XGBoost modeling, and walk-forward evaluation tailored for 15-minute look-ahead signals.

## Key Enhancements (2026-01 Update)

- **Dataset builder** – automatically aggregates 1-minute CCXT candles into configurable 15-minute datasets, with optional funding-rate and open-interest enrichment.
- **Feature Set v2** – intraday seasonality, volatility regimes, cross-asset correlations, and derivative metrics to maximize short-horizon edge.
- **Dual-head LookaheadModel** – simultaneous probabilistic direction (classification) and expected return (regression) forecasts with configurable signal thresholds.
- **Purged walk-forward backtesting** – supports purge/embargo gaps, advanced diagnostics, and integrated trading simulator metrics.
- **Live prediction pipeline** – single command to fetch recent data, rebuild features, and emit actionable signals from the saved model.

## Repository Structure

```
crypto15/
├── requirements.txt              # Python dependencies
├── config.yaml                   # Configuration file (data/model/backtest/trading)
├── src/
│   └── crypto15/
│       ├── __init__.py          # Main package
│       ├── config.py            # Configuration management & helpers
│       ├── util.py              # Utility functions
│       ├── data/
│       │   ├── __init__.py
│       │   ├── fetch_ccxt.py    # CCXT data fetching & resampling helpers
│       │   ├── pipeline.py      # Look-ahead dataset builder
│       │   └── store.py         # Data storage helpers
│       ├── features/
│       │   ├── __init__.py
│       │   ├── feature_set_v1.py # Baseline feature set
│       │   └── feature_set_v2.py # Advanced 15m feature engineering
│       ├── model/
│       │   ├── __init__.py
│       │   ├── lookahead.py     # Dual-head XGBoost ensemble
│       │   └── xgb.py           # Legacy XGBoost utilities
│       └── backtest/
│           ├── __init__.py
│           ├── walkforward.py   # Purged walk-forward backtesting
│           └── sim.py           # Trading simulator
└── scripts/
    ├── fetch_history.py         # Build aggregated dataset + metadata
    ├── walkforward_backtest.py  # Run walk-forward backtest with diagnostics
    ├── train_full_and_save.py   # Train and persist look-ahead model
    └── live_predict.py          # Live predictions
```

## Installation

```bash
cd crypto15
pip install -r requirements.txt
```

## Configuration Overview

`config.yaml` is divided into sections:

- **data** – exchange, primary/context symbols, base/target timeframe, fetch limits, optional derivative data, live lookback horizon.
- **features** – choose feature version (`v1` or `v2`) and set lookback windows, volatility regimes, and target thresholds.
- **model** – hyperparameters for regression & classification heads, early-stopping/test split, and trading signal thresholds.
- **backtest** – walk-forward windows, purge/embargo ratios, commissions, and annualization factor.
- **trading** – simulator constraints such as position sizing, stops, and minimum signal confidence.

## Usage

1. **Fetch & Build Dataset**
   ```bash
   python scripts/fetch_history.py
   ```
   Generates an aggregated parquet dataset (e.g., `BTCUSDT_15m_dataset.parquet`) plus metadata.

2. **Run Walk-Forward Backtest**
   ```bash
   python scripts/walkforward_backtest.py
   ```
   Evaluates the look-ahead model with purge/embargo handling, ROC-AUC/precision/recall metrics, and simulated P&L.

3. **Train Look-Ahead Model**
   ```bash
   python scripts/train_full_and_save.py
   ```
   Trains the dual-head model, logs holdout metrics, and saves the artifact (`*_lookahead.pkl`) with feature importances.

4. **Generate Live Predictions**
   ```bash
   python scripts/live_predict.py
   ```
   Fetches the most recent data, rebuilds features, and prints the current expected return, probability, and signal.

## Modules

### crypto15.data
- `fetch_ccxt`: resilient CCXT OHLCV retrieval + resampling helpers.
- `pipeline`: build fully merged datasets for look-ahead modeling.
- `store`: save/load parquet/csv/pickle datasets.

### crypto15.features
- `feature_set_v1`: legacy hourly indicator baseline.
- `feature_set_v2`: advanced intraday-aware features (returns, volatility regimes, intraday seasonality, cross-asset correlations, derivative metrics).

### crypto15.model
- `lookahead`: dual-head XGBoost ensemble delivering probability + expected return with signal gating.
- `xgb`: legacy regression utilities for experimentation.

### crypto15.backtest
- `walkforward`: purged walk-forward backtesting with aggregated summaries.
- `sim`: trading simulator with position management, stops, and performance metrics.

## Development

The package is structured for easy extension:
- Add new feature sets in `crypto15/features/`
- Add new models in `crypto15/model/`
- Add new backtesting strategies in `crypto15/backtest/`
- Create scripts for additional workflows

## License

MIT License
