"""
Walk-forward backtesting implementation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WalkForwardBacktest:
    """
    Walk-forward backtesting framework.
    """
    
    def __init__(
        self,
        n_splits: int = 12,
        train_ratio: float = 0.8,
        purge_ratio: float = 0.0,
        embargo_ratio: float = 0.0,
    ):
        """
        Initialize walk-forward backtest.
        
        Args:
            n_splits: Number of walk-forward splits
            train_ratio: Ratio of training data in each split
            purge_ratio: Fraction of training data removed from the end to avoid leakage
            embargo_ratio: Fraction of test data dropped from the start to provide embargo
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_ratio = purge_ratio
        self.embargo_ratio = embargo_ratio
    
    def create_splits(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create walk-forward splits.
        
        Args:
            df: Full dataset
        
        Returns:
            List of split dictionaries with train/test indices
        """
        total_samples = len(df)
        samples_per_split = total_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate split boundaries
            split_end = min((i + 1) * samples_per_split, total_samples)
            split_start = max(0, i * samples_per_split)
            
            # Training set: from start to train_ratio of current split
            train_size = int((split_end - split_start) * self.train_ratio)
            train_end = split_start + train_size
            
            # Test set: remaining data in current split
            test_start = train_end
            test_end = split_end
            
            if test_start >= test_end:
                logger.warning(f"Split {i}: Insufficient data for test set, skipping")
                continue
            
            splits.append({
                'split_id': i,
                'train_start': split_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
            })
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        train_func: Callable,
        predict_func: Callable,
        evaluate_func: Callable
    ) -> List[Dict[str, Any]]:
        """
        Run walk-forward backtest.
        
        Args:
            df: Full dataset with features and target
            train_func: Function to train model (df_train) -> model
            predict_func: Function to make predictions (model, df_test) -> predictions
            evaluate_func: Function to evaluate predictions (df_test, predictions) -> metrics
        
        Returns:
            List of results for each split
        """
        splits = self.create_splits(df)
        results = []
        
        for split in splits:
            logger.info(f"Processing split {split['split_id']}")
            
            # Get train and test data
            df_train = df.iloc[split['train_start']:split['train_end']]
            df_test = df.iloc[split['test_start']:split['test_end']]

            purge_count = int(len(df_train) * self.purge_ratio)
            if purge_count > 0 and len(df_train) > purge_count:
                df_train = df_train.iloc[:-purge_count]

            embargo_count = int(len(df_test) * self.embargo_ratio)
            if embargo_count > 0 and len(df_test) > embargo_count:
                df_test = df_test.iloc[embargo_count:]

            if df_train.empty or df_test.empty:
                logger.warning("Split %s skipped due to insufficient data after purge/embargo", split['split_id'])
                continue
            
            # Train model
            model = train_func(df_train)
            
            # Make predictions
            predictions = predict_func(model, df_test)
            
            # Evaluate
            metrics = evaluate_func(df_test, predictions)
            
            # Store results
            results.append({
                'split_id': split['split_id'],
                'train_samples': len(df_train),
                'test_samples': len(df_test),
                'metrics': metrics,
                'predictions': predictions,
                'train_indices': (split['train_start'], split['train_end']),
                'test_indices': (split['test_start'], split['test_end']),
            })
            
            logger.info(f"Split {split['split_id']} completed: {metrics}")
        
        return results
    
    def summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize backtest results across all splits.
        
        Args:
            results: List of results from run_backtest
        
        Returns:
            Summary statistics
        """
        # Extract metrics from all splits
        all_metrics = [r['metrics'] for r in results]
        
        # Calculate average metrics
        summary = {}
        if all_metrics:
            # Assuming metrics are dictionaries
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                values = [m[key] for m in all_metrics if key in m]
                if not values:
                    continue
                summary[f'{key}_mean'] = np.nanmean(values)
                summary[f'{key}_std'] = np.nanstd(values)
        
        summary['n_splits'] = len(results)
        
        return summary
