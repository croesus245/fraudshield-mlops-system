# FraudShield Cost & Latency Report

**Date:** December 2024  
**Environment:** Dev load test (local machine)  
**Model:** XGBoost v1.0.0

---

## Latency Profile

### Summary

| Percentile | Latency | Notes |
|------------|---------|-------|
| p50 | ~12ms | Median response time |
| p95 | ~45ms | Within SLA |
| p99 | ~78ms | Acceptable tail |
| Max | ~120ms | Cold start / GC |

### Measurement Conditions

- **Machine:** Local dev (8-core, 16GB RAM)
- **Workers:** uvicorn --workers=4
- **Requests:** 10,000 sequential
- **Payload:** Single transaction (~1KB JSON)

### Breakdown

| Component | Time |
|-----------|------|
| Feature engineering | ~3ms |
| Model inference | ~6ms |
| JSON serialization | ~2ms |
| Network overhead | ~1ms |
| **Total (p50)** | **~12ms** |

---

## Throughput

### Single Instance

| Metric | Value |
|--------|-------|
| TPS (sustained) | ~1,200 |
| TPS (burst) | ~1,500 |
| Concurrent connections | 50 |

### Scaling Projection

| Instances | Est. TPS | Notes |
|-----------|----------|-------|
| 1 | 1,200 | Baseline |
| 2 | 2,200 | Near-linear |
| 4 | 4,000 | Some overhead |
| 8 | 7,000 | Load balancer bottleneck |

---

## Infrastructure Costs (Estimated)

### AWS Pricing (us-east-1)

| Component | Spec | Monthly Cost |
|-----------|------|--------------|
| EC2 (c5.xlarge) | 4 vCPU, 8GB | ~$125 |
| ECS Fargate | 2 vCPU, 4GB | ~$70 |
| Load Balancer | ALB | ~$20 |
| CloudWatch | Logs + metrics | ~$15 |
| **Total (single instance)** | | **~$230/month** |

### Cost per Million Requests

| Setup | Cost |
|-------|------|
| EC2 (c5.xlarge) | ~$0.07 |
| Fargate | ~$0.10 |
| Lambda (cold start issues) | ~$0.20 |

---

## Optimization Notes

### What's Optimized

- XGBoost uses pre-compiled inference
- Features computed in NumPy (vectorized)
- No database calls in hot path
- Model loaded once at startup

### Potential Improvements

| Improvement | Est. Gain | Effort |
|-------------|-----------|--------|
| Batch scoring | 3-5x for batches | Low |
| ONNX export | 10-20% | Medium |
| Feature caching | 20-30% for repeat users | Medium |
| GPU inference | 2-3x | High |

---

## Recommendations

1. **Start with 2 instances** behind load balancer for redundancy
2. **Use Fargate** for simplicity unless cost-constrained
3. **Add batch endpoint** for bulk scoring use cases
4. **Monitor p99 latency** - degradation indicates scaling need
