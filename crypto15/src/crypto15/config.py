"""
Configuration management module with helper utilities.
"""

import re
import yaml
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, List, Optional

_SYMBOL_SANITIZE_RE = re.compile(r"[^A-Za-z0-9]+")


def sanitize_symbol(symbol: str) -> str:
    """Convert a trading symbol into a filesystem- and column-friendly token."""
    if symbol is None:
        return ""
    return _SYMBOL_SANITIZE_RE.sub("", symbol).upper()


def get_config_path() -> Path:
    """
    Get the path to the config.yaml file.
    
    Returns:
        Path to config.yaml
    """
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    return config_path


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml.
    
    Returns:
        Dictionary containing configuration settings
    """
    config_path = get_config_path()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    return config


def _copy_section(section: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a deep copy of a config section to avoid mutating cached state."""
    if section is None:
        return {}
    return deepcopy(section)


def _resolve_symbol_config(data_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize primary/context symbol entries for downstream consumption."""
    data_cfg = deepcopy(data_cfg)

    symbols_entry = data_cfg.get('symbols')
    primary_symbol: Optional[str] = data_cfg.get('primary_symbol')
    context_symbols: List[str] = data_cfg.get('context_symbols', [])

    if isinstance(symbols_entry, dict):
        primary_symbol = primary_symbol or symbols_entry.get('primary')
        context_entry = symbols_entry.get('context', [])
        if isinstance(context_entry, dict):
            context_symbols = list(context_entry.values())
        else:
            context_symbols = list(context_entry)
    elif isinstance(symbols_entry, list):
        if not primary_symbol and symbols_entry:
            primary_symbol = symbols_entry[0]
        if not context_symbols and len(symbols_entry) > 1:
            context_symbols = symbols_entry[1:]

    data_cfg['primary_symbol'] = primary_symbol
    data_cfg['context_symbols'] = context_symbols or []
    data_cfg['primary_symbol_id'] = sanitize_symbol(primary_symbol) if primary_symbol else None
    data_cfg['context_symbol_ids'] = [sanitize_symbol(sym) for sym in data_cfg['context_symbols']]

    return data_cfg


def get_data_config() -> Dict[str, Any]:
    """Get data-related configuration with normalized defaults."""
    config = load_config()
    data_cfg = _copy_section(config.get('data'))

    data_cfg = _resolve_symbol_config(data_cfg)

    timeframe = data_cfg.get('timeframe')
    data_cfg.setdefault('base_timeframe', timeframe or '1h')
    data_cfg.setdefault('target_timeframe', data_cfg.get('base_timeframe'))
    data_cfg.setdefault('history_days', 365)
    data_cfg.setdefault('fetch', {})
    data_cfg.setdefault('alt_data', {})
    data_cfg.setdefault('live', {})

    fetch_cfg = data_cfg['fetch']
    fetch_cfg.setdefault('limit', 1000)
    fetch_cfg.setdefault('max_batches', 5000)
    fetch_cfg.setdefault('sleep_seconds', 0.2)

    alt_cfg = data_cfg['alt_data']
    alt_cfg.setdefault('funding_rates', False)
    alt_cfg.setdefault('open_interest', False)
    alt_cfg.setdefault('perp_suffix', ':USDT')
    alt_cfg.setdefault('include_context', False)

    live_cfg = data_cfg['live']
    live_cfg.setdefault('history_hours', 72)
    live_cfg.setdefault('refresh_minutes', 15)

    return data_cfg


def get_feature_config() -> Dict[str, Any]:
    """Return feature engineering configuration with sensible defaults."""
    config = load_config()
    features_cfg = _copy_section(config.get('features'))

    features_cfg.setdefault('version', 'v1')
    features_cfg.setdefault('lookback_periods', [24, 168, 720])
    features_cfg.setdefault('lookback_bars', [1, 2, 4, 8, 16, 32, 64])
    features_cfg.setdefault('volatility_windows', [8, 16, 32])
    features_cfg.setdefault('volume_windows', [4, 16, 64])
    features_cfg.setdefault('context_correlation_windows', [8, 32])
    features_cfg.setdefault('include_intraday', True)
    features_cfg.setdefault('include_regime', True)
    features_cfg.setdefault('include_interactions', False)

    target_cfg = features_cfg.setdefault('target', {})
    target_cfg.setdefault('horizon_bars', 1)
    target_cfg.setdefault('classification_threshold', 0.0)
    target_cfg.setdefault('neutral_band', 0.0)

    return features_cfg


def get_model_config() -> Dict[str, Any]:
    """Get model-related configuration with defaults."""
    config = load_config()
    model_cfg = _copy_section(config.get('model'))

    model_cfg.setdefault('type', 'xgboost')
    model_cfg.setdefault('classification_params', {})
    model_cfg.setdefault('regression_params', {})
    model_cfg.setdefault('training', {})
    model_cfg.setdefault('thresholds', {})

    training_cfg = model_cfg['training']
    training_cfg.setdefault('test_size', 0.2)
    training_cfg.setdefault('early_stopping_rounds', 25)
    training_cfg.setdefault('eval_metric_classification', 'auc')
    training_cfg.setdefault('eval_metric_regression', 'rmse')
    training_cfg.setdefault('random_state', 42)

    thresholds_cfg = model_cfg['thresholds']
    thresholds_cfg.setdefault('probability_long', 0.55)
    thresholds_cfg.setdefault('probability_short', 0.45)
    thresholds_cfg.setdefault('min_expected_return', 0.0)
    thresholds_cfg.setdefault('neutral_absolute_return', 0.0)

    return model_cfg


def get_backtest_config() -> Dict[str, Any]:
    """Get backtest-related configuration with defaults."""
    config = load_config()
    backtest_cfg = _copy_section(config.get('backtest'))

    backtest_cfg.setdefault('initial_capital', 10000.0)
    backtest_cfg.setdefault('walkforward_windows', 12)
    backtest_cfg.setdefault('train_test_split', 0.8)
    backtest_cfg.setdefault('purge_ratio', 0.0)
    backtest_cfg.setdefault('embargo_ratio', 0.0)
    backtest_cfg.setdefault('commission', 0.0)
    backtest_cfg.setdefault('annualization_factor', 252)

    return backtest_cfg


def get_trading_config() -> Dict[str, Any]:
    """Get trading-related configuration with defaults."""
    config = load_config()
    trading_cfg = _copy_section(config.get('trading'))

    trading_cfg.setdefault('max_position_size', 0.1)
    trading_cfg.setdefault('stop_loss', 0.02)
    trading_cfg.setdefault('take_profit', 0.05)
    trading_cfg.setdefault('commission', 0.0)
    trading_cfg.setdefault('min_probability', 0.5)
    trading_cfg.setdefault('min_expected_return', 0.0)

    return trading_cfg
