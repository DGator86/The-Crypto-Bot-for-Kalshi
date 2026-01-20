"""
Configuration management module.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


def get_config_path() -> Path:
    """
    Get the path to the config.yaml file.
    
    Returns:
        Path to config.yaml
    """
    # Look for config.yaml in the project root
    current_dir = Path(__file__).parent
    config_path = current_dir.parent.parent.parent / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    return config_path


def load_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml.
    
    Returns:
        Dictionary containing configuration settings
    """
    config_path = get_config_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_config() -> Dict[str, Any]:
    """Get data-related configuration."""
    config = load_config()
    return config.get('data', {})


def get_model_config() -> Dict[str, Any]:
    """Get model-related configuration."""
    config = load_config()
    return config.get('model', {})


def get_backtest_config() -> Dict[str, Any]:
    """Get backtest-related configuration."""
    config = load_config()
    return config.get('backtest', {})


def get_trading_config() -> Dict[str, Any]:
    """Get trading-related configuration."""
    config = load_config()
    return config.get('trading', {})
