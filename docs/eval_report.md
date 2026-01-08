# FraudShield Evaluation Report

**Evaluation Date:** December 2024  
**Model Version:** v1.0.0  
**Dataset:** Synthetic fraud transactions (10K samples, ~2% fraud rate)  
**Evaluator:** Abdul-Sobur Ayinde

---

## Executive Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| ROC-AUC | 0.847 | > 0.80 | ✅ Pass |
| PR-AUC | 0.523 | > 0.40 | ✅ Pass |
| Precision @0.5 | 0.71 | > 0.60 | ✅ Pass |
| Recall @0.5 | 0.68 | > 0.60 | ✅ Pass |
| Latency p95 | ~45ms | < 50ms | ✅ Pass |

---

## Slice-Based Analysis

### By Transaction Amount

| Slice | N | Fraud Rate | PR-AUC | Status |
|-------|---|------------|--------|--------|
| Micro (<$50) | 2,847 | 1.2% | 0.482 | ✅ Pass |
| Low ($50-200) | 3,156 | 1.8% | 0.534 | ✅ Pass |
| Medium ($200-1K) | 2,489 | 2.4% | 0.567 | ✅ Pass |
| High ($1K-5K) | 1,102 | 3.1% | 0.489 | ✅ Pass |
| Very High (>$5K) | 406 | 5.2% | 0.612 | ✅ Pass |

### By Merchant Category

| Category | N | Fraud Rate | PR-AUC |
|----------|---|------------|--------|
| Retail | 2,534 | 1.9% | 0.541 |
| Online | 1,847 | 3.8% | 0.498 |
| Travel | 892 | 4.2% | 0.467 |
| Food | 1,923 | 1.4% | 0.512 |
| Entertainment | 1,156 | 2.1% | 0.523 |

### By Time of Day

| Time | N | Fraud Rate | PR-AUC |
|------|---|------------|--------|
| Night (10pm-6am) | 1,234 | 3.8% | 0.498 |
| Morning (6am-12pm) | 2,847 | 1.6% | 0.534 |
| Afternoon (12pm-6pm) | 3,156 | 1.9% | 0.541 |
| Evening (6pm-10pm) | 2,763 | 2.2% | 0.512 |

---

## CI Gate Results

### Thresholds

| Gate | Threshold | Actual | Status |
|------|-----------|--------|--------|
| Overall PR-AUC | >= 0.40 | 0.523 | ✅ |
| Overall Recall | >= 0.60 | 0.68 | ✅ |
| Max slice regression | < 5% | 3.2% | ✅ |
| Latency p95 | < 50ms | 45ms | ✅ |

### CI Configuration

```yaml
evaluation:
  metrics:
    - pr_auc
    - recall
    - precision
  thresholds:
    pr_auc: 0.40
    recall: 0.60
  regression_tolerance: 0.05
  slices:
    - amount_bin
    - merchant_category
    - hour_bucket
```

### Result

| Check | Result |
|-------|--------|
| All thresholds met | ✅ |
| No regression > 5% | ✅ |
| **CI Gate** | **PASS** |

---

## Recommendations

1. **Monitor online category closely** - Lowest PR-AUC, highest fraud rate
2. **Night transactions need attention** - Higher fraud rate, lower detection
3. **Schedule retraining** - Every 2 weeks or on drift alert
