"""
Benchmark script for latency and throughput measurements.

Measures:
- p50, p95, p99 latency for inference
- Transactions per second (TPS)
- Feature lookup latency
- End-to-end latency (feature lookup + inference)
"""

import time
import statistics
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import numpy as np
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    benchmark_name: str
    timestamp: str
    num_requests: int
    
    # Latency (milliseconds)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    
    # Throughput
    total_time_seconds: float
    requests_per_second: float
    
    # Errors
    error_count: int
    error_rate: float
    
    # Metadata
    concurrency: int = 1
    warmup_requests: int = 0
    notes: str = ""


class InferenceBenchmark:
    """
    Benchmark inference latency and throughput.
    """
    
    def __init__(
        self,
        model_fn: callable,
        warmup_requests: int = 100,
    ):
        """
        Args:
            model_fn: Function that takes a sample and returns prediction
            warmup_requests: Number of requests to warm up the model
        """
        self.model_fn = model_fn
        self.warmup_requests = warmup_requests
    
    def run(
        self,
        samples: list,
        num_requests: int = 1000,
        concurrency: int = 1,
    ) -> BenchmarkResult:
        """
        Run the benchmark.
        
        Args:
            samples: List of input samples
            num_requests: Total requests to make
            concurrency: Number of concurrent workers
        
        Returns:
            BenchmarkResult with latency and throughput metrics
        """
        logger.info(f"Running benchmark: {num_requests} requests, {concurrency} concurrent")
        
        # Warmup
        logger.info(f"Warming up with {self.warmup_requests} requests...")
        for i in range(self.warmup_requests):
            sample = samples[i % len(samples)]
            try:
                self.model_fn(sample)
            except Exception:
                pass
        
        # Benchmark
        latencies = []
        errors = 0
        
        start_time = time.perf_counter()
        
        if concurrency == 1:
            # Single-threaded
            for i in range(num_requests):
                sample = samples[i % len(samples)]
                req_start = time.perf_counter()
                try:
                    self.model_fn(sample)
                    latencies.append((time.perf_counter() - req_start) * 1000)
                except Exception as e:
                    errors += 1
                    logger.debug(f"Request error: {e}")
        else:
            # Multi-threaded
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                for i in range(num_requests):
                    sample = samples[i % len(samples)]
                    future = executor.submit(self._timed_request, sample)
                    futures.append(future)
                
                for future in as_completed(futures):
                    latency, error = future.result()
                    if error:
                        errors += 1
                    else:
                        latencies.append(latency)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        if latencies:
            latencies_sorted = sorted(latencies)
            result = BenchmarkResult(
                benchmark_name="inference",
                timestamp=datetime.now().isoformat(),
                num_requests=num_requests,
                latency_p50_ms=self._percentile(latencies_sorted, 50),
                latency_p95_ms=self._percentile(latencies_sorted, 95),
                latency_p99_ms=self._percentile(latencies_sorted, 99),
                latency_mean_ms=statistics.mean(latencies),
                latency_min_ms=min(latencies),
                latency_max_ms=max(latencies),
                total_time_seconds=total_time,
                requests_per_second=num_requests / total_time,
                error_count=errors,
                error_rate=errors / num_requests,
                concurrency=concurrency,
                warmup_requests=self.warmup_requests,
            )
        else:
            result = BenchmarkResult(
                benchmark_name="inference",
                timestamp=datetime.now().isoformat(),
                num_requests=num_requests,
                latency_p50_ms=0,
                latency_p95_ms=0,
                latency_p99_ms=0,
                latency_mean_ms=0,
                latency_min_ms=0,
                latency_max_ms=0,
                total_time_seconds=total_time,
                requests_per_second=0,
                error_count=errors,
                error_rate=1.0,
                concurrency=concurrency,
                warmup_requests=self.warmup_requests,
            )
        
        return result
    
    def _timed_request(self, sample) -> tuple[float, bool]:
        """Make a timed request. Returns (latency_ms, had_error)."""
        start = time.perf_counter()
        try:
            self.model_fn(sample)
            return (time.perf_counter() - start) * 1000, False
        except Exception:
            return 0, True
    
    @staticmethod
    def _percentile(sorted_data: list, percentile: float) -> float:
        """Calculate percentile from sorted data."""
        if not sorted_data:
            return 0
        index = int(len(sorted_data) * percentile / 100)
        index = min(index, len(sorted_data) - 1)
        return sorted_data[index]


class APIConcurrencyBenchmark:
    """
    Benchmark API endpoint with concurrent load.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def run(
        self,
        endpoint: str,
        payloads: list[dict],
        num_requests: int = 1000,
        concurrency: int = 10,
        warmup_requests: int = 50,
    ) -> BenchmarkResult:
        """
        Run API benchmark.
        
        Args:
            endpoint: API endpoint path (e.g., "/predict")
            payloads: List of request payloads
            num_requests: Total requests to make
            concurrency: Number of concurrent workers
            warmup_requests: Number of warmup requests
        
        Returns:
            BenchmarkResult
        """
        import requests
        
        url = f"{self.base_url}{endpoint}"
        logger.info(f"Benchmarking {url}: {num_requests} requests, {concurrency} concurrent")
        
        # Warmup
        logger.info(f"Warming up with {warmup_requests} requests...")
        for i in range(warmup_requests):
            payload = payloads[i % len(payloads)]
            try:
                requests.post(url, json=payload, timeout=10)
            except Exception:
                pass
        
        # Benchmark
        latencies = []
        errors = 0
        
        def make_request(payload):
            start = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    return (time.perf_counter() - start) * 1000, False
                else:
                    return 0, True
            except Exception:
                return 0, True
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                payload = payloads[i % len(payloads)]
                future = executor.submit(make_request, payload)
                futures.append(future)
            
            for future in as_completed(futures):
                latency, error = future.result()
                if error:
                    errors += 1
                else:
                    latencies.append(latency)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Calculate statistics
        if latencies:
            latencies_sorted = sorted(latencies)
            result = BenchmarkResult(
                benchmark_name=f"api_{endpoint.strip('/')}",
                timestamp=datetime.now().isoformat(),
                num_requests=num_requests,
                latency_p50_ms=InferenceBenchmark._percentile(latencies_sorted, 50),
                latency_p95_ms=InferenceBenchmark._percentile(latencies_sorted, 95),
                latency_p99_ms=InferenceBenchmark._percentile(latencies_sorted, 99),
                latency_mean_ms=statistics.mean(latencies),
                latency_min_ms=min(latencies),
                latency_max_ms=max(latencies),
                total_time_seconds=total_time,
                requests_per_second=num_requests / total_time,
                error_count=errors,
                error_rate=errors / num_requests,
                concurrency=concurrency,
                warmup_requests=warmup_requests,
            )
        else:
            result = BenchmarkResult(
                benchmark_name=f"api_{endpoint.strip('/')}",
                timestamp=datetime.now().isoformat(),
                num_requests=num_requests,
                latency_p50_ms=0,
                latency_p95_ms=0,
                latency_p99_ms=0,
                latency_mean_ms=0,
                latency_min_ms=0,
                latency_max_ms=0,
                total_time_seconds=total_time,
                requests_per_second=0,
                error_count=errors,
                error_rate=1.0,
                concurrency=concurrency,
                warmup_requests=warmup_requests,
            )
        
        return result


def print_benchmark_result(result: BenchmarkResult) -> None:
    """Pretty print benchmark results."""
    print("\n" + "=" * 60)
    print(f"BENCHMARK: {result.benchmark_name}")
    print("=" * 60)
    print(f"Timestamp: {result.timestamp}")
    print(f"Requests: {result.num_requests:,}")
    print(f"Concurrency: {result.concurrency}")
    print()
    print("LATENCY")
    print("-" * 40)
    print(f"  p50:  {result.latency_p50_ms:>8.2f} ms")
    print(f"  p95:  {result.latency_p95_ms:>8.2f} ms")
    print(f"  p99:  {result.latency_p99_ms:>8.2f} ms")
    print(f"  mean: {result.latency_mean_ms:>8.2f} ms")
    print(f"  min:  {result.latency_min_ms:>8.2f} ms")
    print(f"  max:  {result.latency_max_ms:>8.2f} ms")
    print()
    print("THROUGHPUT")
    print("-" * 40)
    print(f"  Total time:  {result.total_time_seconds:>8.2f} seconds")
    print(f"  TPS:         {result.requests_per_second:>8.1f} req/s")
    print()
    print("ERRORS")
    print("-" * 40)
    print(f"  Count: {result.error_count}")
    print(f"  Rate:  {result.error_rate*100:.2f}%")
    print("=" * 60)


def save_benchmark_result(result: BenchmarkResult, output_dir: str) -> str:
    """Save benchmark result to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = f"benchmark_{result.benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)
    
    logger.info(f"Saved benchmark result to {filepath}")
    return str(filepath)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    parser = argparse.ArgumentParser(description="Run FraudShield benchmarks")
    parser.add_argument("--mode", choices=["model", "api"], default="model",
                       help="Benchmark mode: model (direct inference) or api (HTTP endpoint)")
    parser.add_argument("--requests", type=int, default=1000, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--output", type=str, default="artifacts/benchmarks", help="Output directory")
    args = parser.parse_args()
    
    if args.mode == "model":
        # Benchmark direct model inference
        from src.models.trainer import ModelTrainer
        
        # Load model
        model_path = Path("artifacts/models/fraud_model")
        trainer = ModelTrainer.load(str(model_path))
        
        # Generate synthetic samples
        np.random.seed(42)
        num_features = 20  # Adjust based on your model
        samples = [np.random.randn(1, num_features) for _ in range(100)]
        
        def predict_fn(sample):
            return trainer.predict_proba(sample)
        
        benchmark = InferenceBenchmark(predict_fn, warmup_requests=100)
        result = benchmark.run(
            samples=samples,
            num_requests=args.requests,
            concurrency=args.concurrency,
        )
        
        print_benchmark_result(result)
        save_benchmark_result(result, args.output)
        
    elif args.mode == "api":
        # Benchmark API endpoint
        benchmark = APIConcurrencyBenchmark()
        
        # Generate sample payloads
        payloads = [
            {
                "transaction_id": f"bench_{i}",
                "amount": float(np.random.uniform(10, 1000)),
                "merchant_category": np.random.choice(["retail", "grocery", "travel", "entertainment"]),
                "card_present": np.random.choice([True, False]),
                "hour_of_day": int(np.random.randint(0, 24)),
                "day_of_week": int(np.random.randint(0, 7)),
            }
            for i in range(100)
        ]
        
        result = benchmark.run(
            endpoint="/predict",
            payloads=payloads,
            num_requests=args.requests,
            concurrency=args.concurrency,
            warmup_requests=50,
        )
        
        print_benchmark_result(result)
        save_benchmark_result(result, args.output)
