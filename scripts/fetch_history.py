import argparse
import time
from pathlib import Path
from crypto15.config import load_config
from crypto15.data.fetch_ccxt import fetch_ohlcv_history
from crypto15.data.store import save_parquet
import ccxt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()
    
    cfg = load_config(args.config)
    
    # since = now - history_days
    ex = getattr(ccxt, cfg.exchange)({"enableRateLimit": True})
    now = ex.milliseconds()
    since = now - (cfg.history_days * 86400000)
    
    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    
    for symbol in cfg.symbols:
        print(f"Fetching {symbol}...")
        ohlcv = fetch_ohlcv_history(
            exchange=cfg.exchange,
            symbol=symbol,
            timeframe=cfg.timeframe,
            since_ms=since,
            limit=1000,
            sleep_s=0.25,
        )
        
        safe_sym = symbol.replace("/", "")
        fname = f"{safe_sym}_{cfg.timeframe}.parquet"
        out_path = Path(args.data_dir) / fname
        save_parquet(ohlcv.df, out_path)
        print(f"Saved {len(ohlcv.df)} rows to {out_path}")

if __name__ == "__main__":
    main()