import argparse
from crypto15.config import load_config
from crypto15.data.panel import load_panel
from crypto15.backtest.walkforward_panel import run_walkforward_panel
from crypto15.backtest.sim import Costs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    
    # Load panel
    print(f"Loading panel for {cfg.symbols}...")
    panel = load_panel(cfg.symbols, args.data_dir, cfg.timeframe)
    
    if not panel:
        print("No data loaded. Exiting.")
        return
        
    for symbol in cfg.symbols:
        if symbol not in panel:
            print(f"Skipping {symbol} (missing data)")
            continue
            
        print(f"\nRunning walkforward for {symbol}...")
        
        res = run_walkforward_panel(
            target_symbol=symbol,
            panel=panel,
            train_days=cfg.walkforward.train_days,
            test_days=cfg.walkforward.test_days,
            step_days=cfg.walkforward.step_days,
            horizon_bars=cfg.label.horizon_bars,
            no_trade_band_bps=cfg.label.no_trade_band_bps,
            p_enter=cfg.strategy.p_enter,
            risk_fraction=cfg.strategy.risk_fraction,
            costs=Costs(**cfg.costs.__dict__),
        )
        
        print(f"Fold summaries ({symbol}):")
        if not res.fold_summaries.empty:
            print(res.fold_summaries.to_string(index=False))
            
            # Aggregate stats
            if not res.equity_curve.empty:
                eq0 = float(res.equity_curve["equity"].iloc[0])
                eq1 = float(res.equity_curve["equity"].iloc[-1])
                total_ret = (eq1 / eq0) - 1.0
                max_dd = float(res.equity_curve["drawdown"].min())
                trades = int(res.equity_curve["trade"].sum())
                
                print(f"Aggregate ({symbol}):")
                print(f"  Trades: {trades}")
                print(f"  Total return: {total_ret:.2%}")
                print(f"  Max drawdown: {max_dd:.2%}")
        else:
            print("No folds run.")

if __name__ == "__main__":
    main()