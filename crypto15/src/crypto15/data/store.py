"""
Data storage module.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_data_dir() -> Path:
    """
    Get the data directory path.
    
    Returns:
        Path to data directory
    """
    # Create data directory in project root
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent.parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def save_data(
    df: pd.DataFrame,
    filename: str,
    format: str = "parquet"
) -> Path:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        filename: Name of the file (without extension)
        format: File format ('parquet', 'csv', 'pickle')
    
    Returns:
        Path to saved file
    """
    data_dir = get_data_dir()
    
    if format == "parquet":
        filepath = data_dir / f"{filename}.parquet"
        df.to_parquet(filepath)
    elif format == "csv":
        filepath = data_dir / f"{filename}.csv"
        df.to_csv(filepath)
    elif format == "pickle":
        filepath = data_dir / f"{filename}.pkl"
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved data to {filepath}")
    return filepath


def load_data(
    filename: str,
    format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load DataFrame from file.
    
    Args:
        filename: Name of the file (with or without extension)
        format: File format (auto-detected if None)
    
    Returns:
        Loaded DataFrame
    """
    data_dir = get_data_dir()
    
    # Auto-detect format if not provided
    if format is None:
        if filename.endswith('.parquet'):
            format = "parquet"
        elif filename.endswith('.csv'):
            format = "csv"
        elif filename.endswith('.pkl'):
            format = "pickle"
        else:
            # Try parquet first
            format = "parquet"
            filename = f"{filename}.parquet"
    else:
        # Add extension if not present
        if not filename.endswith(f'.{format}'):
            if format == "parquet":
                filename = f"{filename}.parquet"
            elif format == "csv":
                filename = f"{filename}.csv"
            elif format == "pickle":
                filename = f"{filename}.pkl"
    
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if format == "parquet":
        df = pd.read_parquet(filepath)
    elif format == "csv":
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == "pickle":
        df = pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Loaded data from {filepath}")
    return df
