"""
Streaming infrastructure.

Real-time fraud detection requires:
- Feature store (Redis) for fast lookups
- Message queue (Kafka) for transaction streams
- Low-latency serving
"""

from .feature_store import FeatureStore, RedisFeatureStore
from .kafka_consumer import TransactionConsumer, TransactionProducer

__all__ = [
    "FeatureStore",
    "RedisFeatureStore", 
    "TransactionConsumer",
    "TransactionProducer",
]
