"""High-level data acquisition and preparation pipeline for look-ahead modeling."""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd

from crypto15.config import sanitize_symbol

from .fetch_ccxt import (
    fetch_historical_data,
    get_exchange,
    resample_ohlcv,
    timeframe_to_timedelta,
    to_pandas_freq,
)

logger = logging.getLogger(__name__)


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    renamed = df.rename(columns={col: f"{prefix}_{col}" for col in df.columns})
    return renamed


def _resolve_perp_symbol(symbol: str, perp_suffix: str) -> str:
    if not perp_suffix:
        return symbol
    if ':' in symbol:
        return symbol
    if '/' not in symbol:
        return symbol
    base, quote = symbol.split('/')
    # Binance perpetuals follow BASE/QUOTE:QUOTE format by default
    if perp_suffix.startswith(':'):
        return f"{base}/{quote}{perp_suffix}"
    return f"{base}/{quote}{perp_suffix}"


def _fetch_funding_rates(exchange, symbol: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    if not hasattr(exchange, 'fetchFundingRateHistory'):
        logger.debug("Exchange %s does not support funding rate history", exchange.id)
        return pd.DataFrame()
    try:
        records = exchange.fetchFundingRateHistory(symbol, since=since_ms, limit=limit)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to fetch funding rates for %s: %s", symbol, exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    else:
        return pd.DataFrame()

    df.set_index('timestamp', inplace=True)
    columns = {}
    if 'fundingRate' in df.columns:
        columns['fundingRate'] = 'funding_rate'
    if 'fundingRateDaily' in df.columns:
        columns['fundingRateDaily'] = 'funding_rate_daily'
    if not columns:
        return pd.DataFrame()

    df = df[list(columns.keys())].rename(columns=columns)
    return df.sort_index()


def _fetch_open_interest(exchange, symbol: str, timeframe: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
    if not hasattr(exchange, 'fetchOpenInterestHistory'):
        logger.debug("Exchange %s does not support open interest history", exchange.id)
        return pd.DataFrame()
    try:
        records = exchange.fetchOpenInterestHistory(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Unable to fetch open interest for %s: %s", symbol, exc)
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    elif 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    else:
        return pd.DataFrame()

    df.set_index('timestamp', inplace=True)
    keep_cols = [col for col in ['openInterest', 'oi', 'value'] if col in df.columns]
    if not keep_cols:
        return pd.DataFrame()

    df = df[keep_cols]
    df = df.rename(columns={col: 'open_interest' for col in keep_cols})
    return df.sort_index()


def build_lookahead_dataset(data_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch and assemble a dataset ready for feature engineering.

    Returns a tuple of (dataset, metadata)
    """
    exchange_name = data_config.get('exchange', 'binance')
    primary_symbol = data_config.get('primary_symbol') or data_config.get('symbols', {}).get('primary')
    if not primary_symbol:
        raise ValueError("Primary symbol must be defined in configuration")

    context_symbols: List[str] = data_config.get('context_symbols', [])
    base_timeframe = data_config.get('base_timeframe', '1m')
    target_timeframe = data_config.get('target_timeframe', base_timeframe)
    history_days = data_config.get('history_days', 365)

    fetch_cfg = data_config.get('fetch', {})
    limit = fetch_cfg.get('limit', 1000)
    max_batches = fetch_cfg.get('max_batches', 5000)
    sleep_seconds = fetch_cfg.get('sleep_seconds', 0.0)

    alt_cfg = data_config.get('alt_data', {})

    logger.info(
        "Building dataset for %s (context: %s) with base TF %s -> target TF %s",
        primary_symbol,
        context_symbols,
        base_timeframe,
        target_timeframe,
    )

    exchange = get_exchange(exchange_name)

    # Fetch primary data
    primary_raw = fetch_historical_data(
        symbol=primary_symbol,
        timeframe=base_timeframe,
        days=history_days,
        exchange_name=exchange_name,
        limit=limit,
        max_batches=max_batches,
        sleep_seconds=sleep_seconds,
        exchange=exchange,
    )

    if primary_raw.empty:
        raise RuntimeError(f"No historical data retrieved for primary symbol {primary_symbol}")

    primary_resampled = resample_ohlcv(primary_raw, target_timeframe)
    dataset = _prefix_columns(primary_resampled, 'primary')

    # Fetch context symbols
    for ctx_symbol in context_symbols:
        ctx_raw = fetch_historical_data(
            symbol=ctx_symbol,
            timeframe=base_timeframe,
            days=history_days,
            exchange_name=exchange_name,
            limit=limit,
            max_batches=max_batches,
            sleep_seconds=sleep_seconds,
            exchange=exchange,
        )
        if ctx_raw.empty:
            logger.warning("No data returned for context symbol %s", ctx_symbol)
            continue

        ctx_resampled = resample_ohlcv(ctx_raw, target_timeframe)
        ctx_id = sanitize_symbol(ctx_symbol)
        ctx_prefixed = _prefix_columns(ctx_resampled, f"ctx_{ctx_id}")
        dataset = dataset.join(ctx_prefixed, how='outer')

    dataset.sort_index(inplace=True)
    dataset = dataset.ffill().dropna()

    # Fetch optional derivative metrics (funding/open interest)
    if alt_cfg.get('funding_rates') or alt_cfg.get('open_interest'):
        perp_suffix = alt_cfg.get('perp_suffix', ':USDT')
        perp_symbol = alt_cfg.get('primary_perp_symbol') or _resolve_perp_symbol(primary_symbol, perp_suffix)
        since_ms = int(dataset.index[0].timestamp() * 1000)
        target_freq = to_pandas_freq(target_timeframe)

        if alt_cfg.get('funding_rates'):
            funding_df = _fetch_funding_rates(exchange, perp_symbol, since_ms, limit=limit)
            if not funding_df.empty:
                funding_resampled = funding_df.resample(target_freq, label='right', closed='right').last()
                funding_resampled = funding_resampled.ffill()
                dataset = dataset.join(funding_resampled.add_prefix('primary_'), how='left')

        if alt_cfg.get('open_interest'):
            oi_df = _fetch_open_interest(exchange, perp_symbol, target_timeframe, since_ms, limit=limit)
            if not oi_df.empty:
                oi_resampled = oi_df.resample(target_freq, label='right', closed='right').last().ffill()
                dataset = dataset.join(oi_resampled.add_prefix('primary_'), how='left')

    dataset = dataset.dropna()

    metadata = {
        'primary_symbol': primary_symbol,
        'primary_symbol_id': sanitize_symbol(primary_symbol),
        'context_symbols': context_symbols,
        'context_symbol_ids': [sanitize_symbol(sym) for sym in context_symbols],
        'base_timeframe': base_timeframe,
        'target_timeframe': target_timeframe,
        'history_days': history_days,
    }

    return dataset, metadata
