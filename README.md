# Crypto15 Trading Bot Baseline

This repository contains a leakage-proof baseline for crypto trading using XGBoost, walk-forward validation, and cost-aware labels.

## Project Structure

- `crypto15/`: Root directory
  - `config.yaml`: Configuration file (multi-asset support added)
  - `requirements.txt`: Python dependencies
  - `src/`: Source code
    - `crypto15/`: Main package
      - `data/`: Data fetching and storage
      - `features/`: Feature engineering (including cross-asset features)
      - `model/`: XGBoost classification and regression models
      - `backtest/`: Walk-forward backtesting engine
  - `scripts/`: Executable scripts
    - `fetch_history.py`: Fetch OHLCV data from Binance
    - `walkforward_backtest.py`: Run walk-forward backtest on panel data

## Key Features Implemented

1.  **Leakage-Free Design**: strict separation of train/test and causal features.
2.  **Multi-Asset Support**: `config.yaml` supports list of symbols (BTC, ETH, SOL).
3.  **Panel Data**: `panel.py` aligns data across assets.
4.  **Cross-Asset Features**: `feature_set_panel_v1.py` includes cross-asset returns and spreads.
5.  **Dual Model Strategy**: Uses both XGBoost Classifier (Direction) and Regressor (Expected Return).
6.  **Cost-Aware**: Simulates realistic trading costs (fees, slippage, spread) and abstains from trading if edge is insufficient.
7.  **Custom TA Library**: `ta_lib.py` implemented to remove dependency on `pandas_ta` (which had installation issues).

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    # Note: pandas_ta is replaced by local ta_lib.py
    ```

2.  **Fetch Data**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python3 scripts/fetch_history.py
    ```

3.  **Run Backtest**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    python3 scripts/walkforward_backtest.py
    ```

## Notes

- The default configuration uses 365 days of history for BTC/USDT, ETH/USDT, and SOL/USDT.
- The backtest runs a rolling walk-forward evaluation.
- Results are reported per-fold and aggregated per-asset.
