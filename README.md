# The Crypto Bot for Kalshi

This repository contains a 15-minute crypto direction modelling toolkit and
execution-focused policy helpers for Kalshi contracts. The latest update
introduces an extended training script (`crypto_ml_15m_extended.py`) plus a
fee-aware execution policy that blocks the common 0.95–0.99 price traps and
filters trades by expected value after Kalshi fees.

## Quick start

Install the Python dependencies (Python 3.10+ recommended):

```bash
pip install pandas numpy scikit-learn joblib requests
```

### Leakage-controlled training / evaluation

Training automatically performs purged + embargoed walk-forward CV and prints
metrics that serve as a leakage-resistant backtest.

Pull historical data from Binance (public REST) and train:

```bash
python crypto_ml_15m_extended.py \
  --mode train_binance \
  --symbols BTC,ETH,SOL \
  --model_dir models_15m_ext \
  --limit 3000
```

Or use your own CSV files (each must contain timestamp/open/high/low/close/volume):

```bash
python crypto_ml_15m_extended.py \
  --mode train_csv \
  --symbols BTC,ETH,SOL \
  --btc_csv data/BTC_15m.csv \
  --eth_csv data/ETH_15m.csv \
  --sol_csv data/SOL_15m.csv \
  --model_dir models_15m_ext
```

Each run saves per-asset models, thresholds and metrics to `models_15m_ext/`.

### Live probability + EV-gated decisioning

`MultiAssetML` still exposes `predict_and_signal(...)` for threshold-based
signals, but now also supports Kalshi-specific EV gating via
`decide_trade_with_policy(...)`. The new policy blocks trades that cannot beat
fees, enforces minimum time to expiry, and sizes positions with a fractional
Kelly cap.

Example usage in your execution loop:

```python
from kalshi.orderbook import TopOfBook
from crypto_ml_15m_extended import MultiAssetML, KalshiPolicyConfig

model = MultiAssetML.load("models_15m_ext", provider)
policy_cfg = KalshiPolicyConfig()  # tweak gate/sizing/fees here

decision = model.decide_trade_with_policy(
    symbol="BTC",
    book=TopOfBook(yes_bid=0.87, yes_ask=0.89, no_bid=0.11, no_ask=0.13),
    seconds_to_expiry=180,
    bankroll=250.0,
    limit=600,
    policy_cfg=policy_cfg,
)

if decision["policy_action"] == "BUY_YES":
    place_yes_order(price=decision["policy_price"], size=decision["policy_contracts"])
```

The helper returns both the model probabilities and the EV-based policy
decision (including EV/fee breakdown). Supply live order book data and bankroll
figures from your Kalshi execution environment.

### Heads-up

* Probabilities remain probabilistic forecasts – they are **not** profit
  guarantees.
* For production, wire secrets (API keys, tokens) via environment variables –
  never commit them to the repository.
* Future upgrades (regime ensembles, execution realism, CF benchmark alignment)
  can be layered on top of the current policy modules.
