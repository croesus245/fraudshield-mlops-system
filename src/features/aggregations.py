"""
Aggregation features computed over time windows.

These are the features that catch fraud patterns:
- Sudden spike in transaction count
- Unusual total amount in short window
- Transactions from multiple locations
"""

from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger


class AggregationFeatures:
    """
    Compute rolling aggregation features per user.
    
    For each transaction, we compute statistics over recent history:
    - Transaction count in last N hours
    - Total amount in last N hours
    - Average amount in last N hours
    - Unique merchants in last N hours
    - etc.
    
    The brutal truth: These features are expensive to compute at scale.
    In production, you'd use a feature store (Feast, Tecton, etc.).
    This implementation is for learning and local evaluation.
    """
    
    def __init__(self, windows: Optional[list[int]] = None):
        """
        Args:
            windows: Window sizes in hours. Default: [1, 6, 24, 168]
        """
        self.windows = windows or [1, 6, 24, 168]  # 1h, 6h, 24h, 7d
        
        # Baseline statistics (learned during fit)
        self._user_stats: dict[str, dict] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "AggregationFeatures":
        """
        Learn baseline user statistics from training data.
        
        This helps with "cold start" users who have no recent history.
        """
        if "user_id" not in df.columns:
            logger.warning("No user_id column, skipping aggregation fit")
            return self
        
        # Compute per-user baselines
        user_groups = df.groupby("user_id")
        
        self._user_stats = {}
        for user_id, group in user_groups:
            self._user_stats[user_id] = {
                "txn_count": len(group),
                "avg_amount": group["amount"].mean() if "amount" in group.columns else 0,
                "std_amount": group["amount"].std() if "amount" in group.columns else 1,
            }
        
        # Global fallback for new users
        self._user_stats["__global__"] = {
            "txn_count": len(df),
            "avg_amount": df["amount"].mean() if "amount" in df.columns else 0,
            "std_amount": df["amount"].std() if "amount" in df.columns else 1,
        }
        
        self._is_fitted = True
        logger.info(f"Learned baselines for {len(self._user_stats)-1} users")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute aggregation features for each transaction.
        
        Note: This is O(n²) in the worst case. For production, use a proper
        feature store or streaming aggregation.
        """
        if "user_id" not in df.columns or "timestamp" not in df.columns:
            logger.warning("Missing user_id or timestamp, returning empty features")
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # Sort by timestamp for efficient windowing
        df_sorted = df.sort_values("timestamp").copy()
        df_sorted["timestamp"] = pd.to_datetime(df_sorted["timestamp"])
        
        # For each window size
        for window_hours in self.windows:
            window_td = pd.Timedelta(hours=window_hours)
            suffix = f"_{window_hours}h"
            
            # Initialize columns
            features[f"txn_count{suffix}"] = 0
            features[f"txn_amount{suffix}"] = 0.0
            features[f"avg_amount{suffix}"] = 0.0
            features[f"unique_merchants{suffix}"] = 0
            
            # Group by user and compute rolling stats
            for user_id, user_df in df_sorted.groupby("user_id"):
                user_indices = user_df.index
                
                for i, (idx, row) in enumerate(user_df.iterrows()):
                    current_time = row["timestamp"]
                    window_start = current_time - window_td
                    
                    # Get transactions in window (before current)
                    window_mask = (
                        (user_df["timestamp"] >= window_start) &
                        (user_df["timestamp"] < current_time)
                    )
                    window_df = user_df[window_mask]
                    
                    # Compute stats
                    txn_count = len(window_df)
                    txn_amount = window_df["amount"].sum() if "amount" in window_df.columns else 0
                    avg_amount = window_df["amount"].mean() if txn_count > 0 and "amount" in window_df.columns else 0
                    unique_merchants = window_df["merchant_id"].nunique() if "merchant_id" in window_df.columns else 0
                    
                    # Assign to features
                    features.loc[idx, f"txn_count{suffix}"] = txn_count
                    features.loc[idx, f"txn_amount{suffix}"] = txn_amount
                    features.loc[idx, f"avg_amount{suffix}"] = avg_amount
                    features.loc[idx, f"unique_merchants{suffix}"] = unique_merchants
        
        # Add ratio features
        if "txn_count_1h" in features.columns and "txn_count_24h" in features.columns:
            features["txn_count_ratio_1h_24h"] = (
                features["txn_count_1h"] / features["txn_count_24h"].replace(0, 1)
            )
        
        if "txn_amount_1h" in features.columns and "txn_amount_24h" in features.columns:
            features["txn_amount_ratio_1h_24h"] = (
                features["txn_amount_1h"] / features["txn_amount_24h"].replace(0, 1)
            )
        
        # Add user baseline deviation (if fitted)
        if self._is_fitted and "amount" in df.columns:
            features["amount_vs_user_avg"] = self._compute_user_deviation(df)
        
        return features
    
    def _compute_user_deviation(self, df: pd.DataFrame) -> pd.Series:
        """Compute how much each transaction deviates from user's baseline."""
        deviations = pd.Series(index=df.index, dtype=float)
        
        global_stats = self._user_stats.get("__global__", {"avg_amount": 0, "std_amount": 1})
        
        for idx, row in df.iterrows():
            user_id = row.get("user_id")
            amount = row.get("amount", 0)
            
            # Get user stats or fall back to global
            stats = self._user_stats.get(user_id, global_stats)
            
            # Z-score vs user baseline
            std = stats["std_amount"] if stats["std_amount"] > 0 else 1
            deviation = (amount - stats["avg_amount"]) / std
            deviations.loc[idx] = deviation
        
        return deviations


def compute_velocity_features(
    df: pd.DataFrame,
    windows_hours: list[int] = [1, 6, 24],
) -> pd.DataFrame:
    """
    Convenience function to compute velocity features.
    
    Velocity = how fast is the user transacting?
    """
    agg = AggregationFeatures(windows=windows_hours)
    return agg.transform(df)
