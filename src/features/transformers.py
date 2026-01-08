"""
Feature Transformers - Turn raw data into ML features.

WHY TRANSFORM?
Raw data is terrible for ML:
- Amount $50000 vs $50 → model thinks 1000x difference matters linearly
- Hour 23 vs 0 → model thinks midnight is far from 11pm
- Category "retail" → model can't use strings

TRANSFORMERS:
1. TimeFeatures → hour, day, weekend, cyclical encoding
2. AmountFeatures → log transform, z-score, percentiles
3. CategoricalEncoder → one-hot encoding
4. DeviceFeatures → risk flags from device signals
5. UserFeatures → risk flags from user history
6. VelocityFeatures → transaction frequency patterns
7. NetworkFeatures → shared device/merchant risk
"""

import pandas as pd
import numpy as np


class TimeFeatures:
    """
    Extract time patterns from timestamps.
    
    WHY CYCLICAL?
    Hour 23 and hour 0 are 1 hour apart, not 23.
    Sin/cos encoding captures this circular nature.
    """
    
    def transform(self, timestamps: pd.Series) -> pd.DataFrame:
        ts = pd.to_datetime(timestamps)
        f = pd.DataFrame(index=timestamps.index)
        
        # Basic time features
        f["hour"] = ts.dt.hour
        f["day_of_week"] = ts.dt.dayofweek
        f["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        f["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(int)  # Fraud peaks at night
        
        # Cyclical encoding (hour 23 is close to hour 0)
        f["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        f["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        
        return f


class AmountFeatures:
    """
    Transform amounts for ML.
    
    WHY LOG?
    $1M is not 1000x more suspicious than $1K.
    Log transform compresses the scale.
    """
    
    def __init__(self):
        self._mean = 0
        self._std = 1
        self._p95 = 0
        self._fitted = False
    
    def fit(self, amounts: pd.Series):
        self._mean = amounts.mean()
        self._std = amounts.std() or 1
        self._p95 = amounts.quantile(0.95)
        self._fitted = True
        return self
    
    def transform(self, amounts: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame(index=amounts.index)
        
        f["amount"] = amounts.clip(0, 1e6)
        f["amount_log"] = np.log1p(amounts.clip(0))  # log(1+x) handles zeros
        
        if self._fitted:
            f["amount_zscore"] = (amounts - self._mean) / self._std
            f["amount_high"] = (amounts > self._p95).astype(int)
        
        return f


class CategoricalEncoder:
    """
    One-hot encode categories.
    
    WHY ONE-HOT?
    ML can't understand "retail" vs "gaming".
    One-hot creates binary columns: is_retail=1, is_gaming=0
    """
    
    def __init__(self, max_cats: int = 20):
        self.max_cats = max_cats
        self._categories = {}
        self._fitted = False
    
    def fit(self, df: pd.DataFrame):
        for col in df.columns:
            top = df[col].value_counts().head(self.max_cats).index.tolist()
            self._categories[col] = top
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col not in self._categories:
                continue
            cats = self._categories[col]
            for cat in cats:
                result[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            result[f"{col}_other"] = (~df[col].isin(cats)).astype(int)
        
        return result


class DeviceFeatures:
    """
    Extract risk signals from device data.
    
    HIGH RISK SIGNALS:
    - VPN/proxy → hiding location
    - Emulator → not a real phone
    - New device → first time we see it
    - Rooted → bypassed security
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        
        # Direct risk flags
        for col in ["is_vpn", "is_new_device", "is_emulator", "is_rooted", "is_foreign"]:
            if col in df.columns:
                f[col] = df[col].astype(int)
        
        # Failed logins (suspicious)
        if "failed_logins_24h" in df.columns:
            f["failed_logins"] = df["failed_logins_24h"].fillna(0).clip(0, 20)
            f["has_failed_logins"] = (df["failed_logins_24h"] > 0).astype(int)
        
        # Composite risk score
        risk_cols = ["is_vpn", "is_new_device", "is_emulator", "is_rooted"]
        available = [c for c in risk_cols if c in df.columns]
        if available:
            f["device_risk"] = df[available].sum(axis=1) / len(available)
        
        return f


class UserFeatures:
    """
    Extract risk signals from user history.
    
    HIGH RISK SIGNALS:
    - New account → hasn't built trust
    - Unverified → identity not confirmed
    - Past fraud flags → repeat offender
    - Chargebacks → disputes transactions
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        
        # Account age (new accounts are risky)
        if "account_age_days" in df.columns:
            age = df["account_age_days"].fillna(0)
            f["account_age"] = age.clip(0, 3650)  # Cap at 10 years
            f["account_age_log"] = np.log1p(age)
            f["is_new_account"] = (age < 30).astype(int)
        
        # Verification
        if "is_verified" in df.columns:
            f["is_verified"] = df["is_verified"].astype(int)
        
        # Fraud history (critical signal)
        if "previous_fraud_flags" in df.columns:
            flags = df["previous_fraud_flags"].fillna(0)
            f["fraud_flags"] = flags.clip(0, 10)
            f["has_fraud_history"] = (flags > 0).astype(int)
        
        # Chargebacks/refunds (abuse indicators)
        for col in ["chargebacks_90d", "refunds_90d"]:
            if col in df.columns:
                f[col] = df[col].fillna(0).clip(0, 20)
        
        return f


class VelocityFeatures:
    """
    Extract transaction frequency patterns.
    
    WHY VELOCITY?
    Fraudsters work fast:
    - Drain account before victim notices
    - Test card limits with many small transactions
    - Cash out through multiple merchants
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        
        # Transaction counts per window
        for window in ["1h", "24h", "7d"]:
            col = f"txn_count_{window}"
            if col in df.columns:
                f[col] = df[col].fillna(0).clip(0, 500)
                f[f"{col}_log"] = np.log1p(df[col].fillna(0))
        
        # Amount sums
        for window in ["24h", "7d"]:
            col = f"amount_sum_{window}"
            if col in df.columns:
                f[col] = df[col].fillna(0).clip(0, 1e7)
        
        # Velocity breach flags
        if "txn_count_1h" in df.columns:
            f["velocity_breach_1h"] = (df["txn_count_1h"] > 5).astype(int)
        if "txn_count_24h" in df.columns:
            f["velocity_breach_24h"] = (df["txn_count_24h"] > 20).astype(int)
        
        return f


class NetworkFeatures:
    """
    Extract network/graph risk signals.
    
    WHY NETWORK?
    Fraud rings share resources:
    - Same device across multiple accounts
    - Same bank account
    - High-risk merchants
    """
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        
        # Shared devices (fraud ring indicator)
        if "shared_devices_count" in df.columns:
            count = df["shared_devices_count"].fillna(0)
            f["shared_devices"] = count.clip(0, 50)
            f["has_shared_device"] = (count > 0).astype(int)
        
        # Merchant risk
        if "merchant_fraud_rate" in df.columns:
            rate = df["merchant_fraud_rate"].fillna(0)
            f["merchant_fraud_rate"] = rate.clip(0, 1)
            f["high_risk_merchant"] = (rate > 0.1).astype(int)
        
        return f
