"""
Feature Store - Fast feature lookups for real-time inference.

WHY A FEATURE STORE?
- Training uses batch data (historical)
- Serving needs features in <10ms
- Features must be IDENTICAL in training and serving (training-serving skew kills models)

This implements:
1. In-memory store (for testing/dev)
2. Redis store (for production)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass, field
from loguru import logger

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class FeatureRecord:
    """A cached feature vector with metadata."""
    entity_id: str
    entity_type: str  # "user", "device", "merchant"
    features: dict[str, Any]
    timestamp: datetime
    ttl_seconds: int = 3600  # 1 hour default
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "features": self.features,
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "FeatureRecord":
        return cls(
            entity_id=data["entity_id"],
            entity_type=data["entity_type"],
            features=data["features"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            ttl_seconds=data["ttl_seconds"],
        )


class FeatureStore(ABC):
    """
    Abstract feature store interface.
    
    Implementations must provide fast get/set operations.
    """
    
    @abstractmethod
    def get(self, entity_type: str, entity_id: str) -> Optional[dict[str, Any]]:
        """Get features for an entity. Returns None if not found or expired."""
        pass
    
    @abstractmethod
    def set(self, entity_type: str, entity_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        """Store features for an entity."""
        pass
    
    @abstractmethod
    def get_many(self, entity_type: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch get features. Returns {entity_id: features} for found entities."""
        pass
    
    @abstractmethod
    def delete(self, entity_type: str, entity_id: str) -> None:
        """Delete features for an entity."""
        pass
    
    def _make_key(self, entity_type: str, entity_id: str) -> str:
        """Generate storage key."""
        return f"features:{entity_type}:{entity_id}"


class InMemoryFeatureStore(FeatureStore):
    """
    In-memory feature store for testing and development.
    
    NOT FOR PRODUCTION - no persistence, single process only.
    """
    
    def __init__(self):
        self._store: dict[str, FeatureRecord] = {}
        self._hits = 0
        self._misses = 0
    
    def get(self, entity_type: str, entity_id: str) -> Optional[dict[str, Any]]:
        key = self._make_key(entity_type, entity_id)
        record = self._store.get(key)
        
        if record is None:
            self._misses += 1
            return None
        
        if record.is_expired():
            del self._store[key]
            self._misses += 1
            return None
        
        self._hits += 1
        return record.features
    
    def set(self, entity_type: str, entity_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        key = self._make_key(entity_type, entity_id)
        self._store[key] = FeatureRecord(
            entity_id=entity_id,
            entity_type=entity_type,
            features=features,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
        )
    
    def get_many(self, entity_type: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        results = {}
        for entity_id in entity_ids:
            features = self.get(entity_type, entity_id)
            if features is not None:
                results[entity_id] = features
        return results
    
    def delete(self, entity_type: str, entity_id: str) -> None:
        key = self._make_key(entity_type, entity_id)
        self._store.pop(key, None)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "size": len(self._store),
        }


class RedisFeatureStore(FeatureStore):
    """
    Redis-backed feature store for production.
    
    WHY REDIS?
    - Sub-millisecond latency
    - Built-in TTL support
    - Atomic operations
    - Cluster mode for scaling
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        prefix: str = "fraudshield",
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Run: pip install redis")
        
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )
        self._prefix = prefix
        self._hits = 0
        self._misses = 0
        
        # Test connection
        try:
            self._client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _make_key(self, entity_type: str, entity_id: str) -> str:
        return f"{self._prefix}:features:{entity_type}:{entity_id}"
    
    def get(self, entity_type: str, entity_id: str) -> Optional[dict[str, Any]]:
        key = self._make_key(entity_type, entity_id)
        data = self._client.get(key)
        
        if data is None:
            self._misses += 1
            return None
        
        self._hits += 1
        return json.loads(data)
    
    def set(self, entity_type: str, entity_id: str, features: dict[str, Any], ttl: int = 3600) -> None:
        key = self._make_key(entity_type, entity_id)
        self._client.setex(key, ttl, json.dumps(features))
    
    def get_many(self, entity_type: str, entity_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not entity_ids:
            return {}
        
        keys = [self._make_key(entity_type, eid) for eid in entity_ids]
        values = self._client.mget(keys)
        
        results = {}
        for entity_id, value in zip(entity_ids, values):
            if value is not None:
                results[entity_id] = json.loads(value)
                self._hits += 1
            else:
                self._misses += 1
        
        return results
    
    def delete(self, entity_type: str, entity_id: str) -> None:
        key = self._make_key(entity_type, entity_id)
        self._client.delete(key)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> dict:
        info = self._client.info("memory")
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "memory_used_mb": info.get("used_memory", 0) / 1024 / 1024,
        }


# =============================================================================
# VELOCITY AGGREGATIONS
# =============================================================================

class VelocityStore:
    """
    Track velocity metrics in real-time.
    
    WHY VELOCITY?
    - Fraudsters move fast before getting caught
    - "5 transactions in 1 hour" is a strong signal
    - Must be computed in real-time, not batch
    
    Uses Redis sorted sets for time-windowed counts.
    """
    
    def __init__(self, feature_store: FeatureStore):
        self._store = feature_store
        self._windows = {
            "1h": 3600,
            "24h": 86400,
            "7d": 604800,
        }
    
    def record_transaction(
        self,
        user_id: str,
        device_id: str,
        merchant_id: str,
        amount: float,
        timestamp: datetime,
    ) -> None:
        """Record a transaction for velocity tracking."""
        ts = timestamp.timestamp()
        
        # Update user velocity
        user_features = self._store.get("user_velocity", user_id) or {
            "txn_timestamps": [],
            "txn_amounts": [],
            "merchants": [],
        }
        user_features["txn_timestamps"].append(ts)
        user_features["txn_amounts"].append(amount)
        user_features["merchants"].append(merchant_id)
        
        # Keep last 7 days only
        cutoff = ts - self._windows["7d"]
        user_features = self._prune_old(user_features, cutoff)
        self._store.set("user_velocity", user_id, user_features, ttl=self._windows["7d"])
        
        # Update device velocity
        device_features = self._store.get("device_velocity", device_id) or {
            "txn_timestamps": [],
            "users": [],
        }
        device_features["txn_timestamps"].append(ts)
        device_features["users"].append(user_id)
        device_features = self._prune_old(device_features, cutoff)
        self._store.set("device_velocity", device_id, device_features, ttl=self._windows["7d"])
    
    def get_velocity_features(
        self,
        user_id: str,
        device_id: str,
        timestamp: datetime,
    ) -> dict[str, float]:
        """Get velocity features for a transaction."""
        ts = timestamp.timestamp()
        features = {}
        
        # User velocity
        user_data = self._store.get("user_velocity", user_id) or {"txn_timestamps": [], "txn_amounts": [], "merchants": []}
        
        for window_name, window_seconds in self._windows.items():
            cutoff = ts - window_seconds
            recent_ts = [t for t in user_data["txn_timestamps"] if t > cutoff]
            recent_amounts = [a for t, a in zip(user_data["txn_timestamps"], user_data["txn_amounts"]) if t > cutoff]
            recent_merchants = [m for t, m in zip(user_data["txn_timestamps"], user_data["merchants"]) if t > cutoff]
            
            features[f"user_txn_count_{window_name}"] = len(recent_ts)
            features[f"user_txn_amount_{window_name}"] = sum(recent_amounts)
            features[f"user_unique_merchants_{window_name}"] = len(set(recent_merchants))
        
        # Device velocity
        device_data = self._store.get("device_velocity", device_id) or {"txn_timestamps": [], "users": []}
        
        for window_name, window_seconds in self._windows.items():
            cutoff = ts - window_seconds
            recent_ts = [t for t in device_data["txn_timestamps"] if t > cutoff]
            recent_users = [u for t, u in zip(device_data["txn_timestamps"], device_data["users"]) if t > cutoff]
            
            features[f"device_txn_count_{window_name}"] = len(recent_ts)
            features[f"device_unique_users_{window_name}"] = len(set(recent_users))
        
        return features
    
    def _prune_old(self, data: dict, cutoff: float) -> dict:
        """Remove entries older than cutoff."""
        if "txn_timestamps" not in data:
            return data
        
        timestamps = data["txn_timestamps"]
        mask = [t > cutoff for t in timestamps]
        
        result = {}
        for key, values in data.items():
            if isinstance(values, list) and len(values) == len(timestamps):
                result[key] = [v for v, m in zip(values, mask) if m]
            else:
                result[key] = values
        
        return result
