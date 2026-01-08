"""
Kafka Consumer/Producer for transaction streaming.

WHY KAFKA?
- High throughput (millions of events/sec)
- Durable (persisted to disk)
- Replay capability (reprocess historical data)
- Decoupled services (fraud service independent of payment service)

This module provides:
1. TransactionConsumer - Reads transactions from Kafka
2. TransactionProducer - Writes predictions back to Kafka
3. MockKafka - For testing without real Kafka
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
import json
import time
import threading
from queue import Queue, Empty
from loguru import logger

try:
    from kafka import KafkaConsumer, KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


@dataclass
class TransactionEvent:
    """A transaction event from the stream."""
    transaction_id: str
    user_id: str
    merchant_id: str
    merchant_category: str
    amount: float
    device_id: str
    device_type: str
    location_country: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "user_id": self.user_id,
            "merchant_id": self.merchant_id,
            "merchant_category": self.merchant_category,
            "amount": self.amount,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "location_country": self.location_country,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TransactionEvent":
        return cls(
            transaction_id=data["transaction_id"],
            user_id=data["user_id"],
            merchant_id=data["merchant_id"],
            merchant_category=data["merchant_category"],
            amount=data["amount"],
            device_id=data["device_id"],
            device_type=data["device_type"],
            location_country=data["location_country"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PredictionEvent:
    """A fraud prediction result."""
    transaction_id: str
    fraud_probability: float
    decision: str  # "allow", "block", "review"
    risk_level: str  # "low", "medium", "high"
    triggered_rules: list[str]
    model_version: str
    latency_ms: float
    timestamp: datetime
    
    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "fraud_probability": self.fraud_probability,
            "decision": self.decision,
            "risk_level": self.risk_level,
            "triggered_rules": self.triggered_rules,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class TransactionConsumer(ABC):
    """Abstract base for transaction consumers."""
    
    @abstractmethod
    def consume(self) -> Generator[TransactionEvent, None, None]:
        """Yield transaction events from the stream."""
        pass
    
    @abstractmethod
    def commit(self) -> None:
        """Commit current offset (acknowledge processing)."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the consumer."""
        pass


class TransactionProducer(ABC):
    """Abstract base for prediction producers."""
    
    @abstractmethod
    def send(self, prediction: PredictionEvent) -> None:
        """Send a prediction event."""
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush pending messages."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the producer."""
        pass


# =============================================================================
# MOCK IMPLEMENTATION (for testing without Kafka)
# =============================================================================

class MockTransactionConsumer(TransactionConsumer):
    """
    Mock consumer for testing.
    
    Generates synthetic transactions at a configurable rate.
    """
    
    def __init__(
        self,
        transactions_per_second: float = 10.0,
        total_transactions: Optional[int] = None,
    ):
        self._tps = transactions_per_second
        self._total = total_transactions
        self._count = 0
        self._running = True
        
        # Transaction generators
        self._merchants = ["merchant_" + str(i) for i in range(100)]
        self._categories = ["retail", "food", "travel", "gaming", "entertainment", "online", "services"]
        self._users = ["user_" + str(i) for i in range(1000)]
        self._devices = ["device_" + str(i) for i in range(500)]
        self._countries = ["US", "UK", "CA", "DE", "FR", "JP", "AU"]
    
    def consume(self) -> Generator[TransactionEvent, None, None]:
        import random
        
        while self._running:
            if self._total is not None and self._count >= self._total:
                break
            
            # Generate random transaction
            event = TransactionEvent(
                transaction_id=f"txn_{self._count:08d}",
                user_id=random.choice(self._users),
                merchant_id=random.choice(self._merchants),
                merchant_category=random.choice(self._categories),
                amount=round(random.expovariate(0.01), 2),  # Exponential distribution
                device_id=random.choice(self._devices),
                device_type=random.choice(["mobile", "desktop", "tablet"]),
                location_country=random.choice(self._countries),
                timestamp=datetime.now(),
            )
            
            self._count += 1
            yield event
            
            # Rate limiting
            time.sleep(1.0 / self._tps)
    
    def commit(self) -> None:
        pass  # No-op for mock
    
    def close(self) -> None:
        self._running = False


class MockTransactionProducer(TransactionProducer):
    """
    Mock producer for testing.
    
    Stores predictions in memory for inspection.
    """
    
    def __init__(self):
        self._predictions: list[PredictionEvent] = []
        self._lock = threading.Lock()
    
    def send(self, prediction: PredictionEvent) -> None:
        with self._lock:
            self._predictions.append(prediction)
    
    def flush(self) -> None:
        pass  # No-op for mock
    
    def close(self) -> None:
        pass
    
    @property
    def predictions(self) -> list[PredictionEvent]:
        with self._lock:
            return list(self._predictions)
    
    def clear(self) -> None:
        with self._lock:
            self._predictions.clear()


# =============================================================================
# REAL KAFKA IMPLEMENTATION
# =============================================================================

class KafkaTransactionConsumer(TransactionConsumer):
    """
    Real Kafka consumer for production.
    
    Reads from a Kafka topic and yields TransactionEvents.
    """
    
    def __init__(
        self,
        topic: str = "transactions",
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "fraud-detection",
        auto_offset_reset: str = "latest",
    ):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed. Run: pip install kafka-python")
        
        self._consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            auto_offset_reset=auto_offset_reset,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            enable_auto_commit=False,
        )
        self._topic = topic
        logger.info(f"Connected to Kafka topic: {topic}")
    
    def consume(self) -> Generator[TransactionEvent, None, None]:
        for message in self._consumer:
            try:
                event = TransactionEvent.from_dict(message.value)
                yield event
            except Exception as e:
                logger.error(f"Failed to parse message: {e}")
                continue
    
    def commit(self) -> None:
        self._consumer.commit()
    
    def close(self) -> None:
        self._consumer.close()


class KafkaTransactionProducer(TransactionProducer):
    """
    Real Kafka producer for production.
    
    Sends PredictionEvents to a Kafka topic.
    """
    
    def __init__(
        self,
        topic: str = "predictions",
        bootstrap_servers: str = "localhost:9092",
    ):
        if not KAFKA_AVAILABLE:
            raise ImportError("kafka-python not installed. Run: pip install kafka-python")
        
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda m: json.dumps(m).encode("utf-8"),
        )
        self._topic = topic
        logger.info(f"Connected to Kafka topic: {topic}")
    
    def send(self, prediction: PredictionEvent) -> None:
        self._producer.send(self._topic, prediction.to_dict())
    
    def flush(self) -> None:
        self._producer.flush()
    
    def close(self) -> None:
        self._producer.flush()
        self._producer.close()


# =============================================================================
# STREAMING PROCESSOR
# =============================================================================

class StreamingFraudProcessor:
    """
    Main streaming processor.
    
    Orchestrates:
    1. Consuming transactions
    2. Computing features (with feature store)
    3. Running ML model
    4. Applying rules
    5. Producing predictions
    """
    
    def __init__(
        self,
        consumer: TransactionConsumer,
        producer: TransactionProducer,
        predictor: Any,  # EnhancedPredictor
        feature_store: Any = None,  # FeatureStore
        velocity_store: Any = None,  # VelocityStore
        batch_size: int = 1,
        model_version: str = "v1",
    ):
        self._consumer = consumer
        self._producer = producer
        self._predictor = predictor
        self._feature_store = feature_store
        self._velocity_store = velocity_store
        self._batch_size = batch_size
        self._model_version = model_version
        
        self._running = False
        self._processed = 0
        self._total_latency_ms = 0.0
    
    def process_one(self, event: TransactionEvent) -> PredictionEvent:
        """Process a single transaction."""
        start_time = time.perf_counter()
        
        # Build transaction dict
        txn = event.to_dict()
        
        # Add velocity features if available
        if self._velocity_store:
            velocity_features = self._velocity_store.get_velocity_features(
                event.user_id,
                event.device_id,
                event.timestamp,
            )
            txn.update(velocity_features)
        
        # Run prediction
        result = self._predictor.check(txn)
        
        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record transaction for velocity
        if self._velocity_store:
            self._velocity_store.record_transaction(
                event.user_id,
                event.device_id,
                event.merchant_id,
                event.amount,
                event.timestamp,
            )
        
        prediction = PredictionEvent(
            transaction_id=event.transaction_id,
            fraud_probability=result.fraud_probability,
            decision=result.action,
            risk_level=result.risk_level,
            triggered_rules=[r.rule_name for r in result.rule_results if r.triggered],
            model_version=self._model_version,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
        )
        
        return prediction
    
    def run(self, max_events: Optional[int] = None) -> None:
        """Run the streaming processor."""
        self._running = True
        self._processed = 0
        self._total_latency_ms = 0.0
        
        logger.info("Starting streaming processor...")
        
        try:
            for event in self._consumer.consume():
                if not self._running:
                    break
                
                if max_events and self._processed >= max_events:
                    break
                
                prediction = self.process_one(event)
                self._producer.send(prediction)
                
                self._processed += 1
                self._total_latency_ms += prediction.latency_ms
                
                # Log progress
                if self._processed % 100 == 0:
                    avg_latency = self._total_latency_ms / self._processed
                    logger.info(f"Processed {self._processed} transactions, avg latency: {avg_latency:.2f}ms")
                
                # Commit periodically
                if self._processed % self._batch_size == 0:
                    self._consumer.commit()
                    self._producer.flush()
        
        finally:
            self._consumer.close()
            self._producer.close()
            logger.info(f"Processor stopped. Total processed: {self._processed}")
    
    def stop(self) -> None:
        """Stop the processor."""
        self._running = False
    
    def stats(self) -> dict:
        """Get processing statistics."""
        avg_latency = self._total_latency_ms / self._processed if self._processed > 0 else 0
        return {
            "processed": self._processed,
            "avg_latency_ms": avg_latency,
            "total_latency_ms": self._total_latency_ms,
        }
