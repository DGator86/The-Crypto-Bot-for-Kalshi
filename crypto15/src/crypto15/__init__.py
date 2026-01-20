"""
The Crypto Bot for Kalshi - Main Package
"""

__version__ = "0.1.0"
__author__ = "Crypto Bot Team"

from .config import load_config
from .util import setup_logging

__all__ = ["load_config", "setup_logging"]
