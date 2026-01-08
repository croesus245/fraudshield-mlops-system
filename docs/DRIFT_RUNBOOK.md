# Drift Response Runbook

## Overview

This runbook provides step-by-step procedures for responding to data drift alerts in the FraudShield system.

**When to use this runbook:**
- Drift alert triggered in monitoring
- Performance degradation observed
- Scheduled drift review

---

## Quick Reference

| Severity | Response Time | Escalation |
|----------|---------------|------------|
| Warning  | 24 hours      | ML Engineer |
| Critical | 4 hours       | ML Lead + On-call |

---

## Step 1: Assess the Alert

### 1.1 Gather Information

```bash
# Check recent drift alerts
python scripts/evaluate.py --config configs/config.yaml --run-stress-tests

# View drift metrics
cat artifacts/reports/drift_*.yaml
```

### 1.2 Identify Affected Features

Look for features with:
- PSI > 0.2 (critical drift)
- PSI 0.1-0.2 (moderate drift)
- KS p-value < 0.05

### 1.3 Determine Scope

- Is drift in one feature or many?
- Is drift gradual or sudden?
- Does drift correlate with external events?

---

## Step 2: Root Cause Analysis

### 2.1 Common Causes

| Cause | Indicators | Response |
|-------|------------|----------|
| **Seasonal change** | Gradual shift, time-correlated | Retrain with recent data |
| **Data pipeline bug** | Sudden shift, specific features | Fix pipeline, backfill |
| **Fraud pattern change** | Prediction drift, new patterns | Urgent retrain + review |
| **Business change** | Coincides with product change | Update features, retrain |
| **Upstream data change** | External data source shifted | Contact data provider |

### 2.2 Investigation Queries

```python
# Check feature distributions over time
import pandas as pd
from src.monitoring.drift import DriftDetector

# Load recent data
recent_data = pd.read_parquet("data/transactions.parquet")

# Compare weekly windows
week_1 = recent_data[recent_data["timestamp"] < "2024-01-08"]
week_2 = recent_data[recent_data["timestamp"] >= "2024-01-08"]

detector = DriftDetector()
detector.set_reference(week_1)
results = detector.detect(week_2)

for r in results:
    if r.is_drifted:
        print(f"{r.feature}: PSI={r.statistic:.3f}")
```

---

## Step 3: Immediate Actions

### 3.1 For Warning-Level Drift

1. **Document** the drift in the incident log
2. **Monitor** for 24-48 hours
3. **Schedule** evaluation if drift persists
4. **Consider** retraining with recent data

### 3.2 For Critical-Level Drift

1. **Alert** the ML team immediately
2. **Check** model predictions for anomalies
3. **Evaluate** model on recent labeled data (if available)
4. **Consider** rollback if performance degraded

```bash
# Quick performance check
python scripts/ci_checks.py --strict
```

### 3.3 Emergency Rollback

If model is producing bad predictions:

```bash
# Switch to previous model version
cp artifacts/models/xgboost_model_backup.json artifacts/models/xgboost_model.json
cp artifacts/models/feature_pipeline_backup.pkl artifacts/models/feature_pipeline.pkl

# Restart serving
python scripts/serve.py --port 8000
```

---

## Step 4: Remediation

### 4.1 Short-term Fix

**Retrain with recent data:**

```bash
# Generate fresh data (or use production data)
python scripts/generate_data.py --n-transactions 50000

# Retrain
python scripts/train.py --config configs/config.yaml

# Evaluate
python scripts/evaluate.py --run-stress-tests
```

### 4.2 Long-term Fix

1. **Update monitoring thresholds** if too sensitive
2. **Add new features** if patterns changed
3. **Adjust model architecture** if needed
4. **Update data contracts** for new distributions

---

## Step 5: Post-Incident

### 5.1 Update Documentation

- Add to incident log (see `docs/INCIDENT_POSTMORTEM.md`)
- Update thresholds in `configs/config.yaml`
- Document root cause and fix

### 5.2 Improve Monitoring

- Add specific alerts for this drift type
- Update dashboards
- Consider adding automated retraining

### 5.3 Review Checklist

- [ ] Incident documented
- [ ] Root cause identified
- [ ] Fix deployed
- [ ] Monitoring updated
- [ ] Runbook updated if needed
- [ ] Team notified of changes

---

## Reference: Drift Metrics

### Population Stability Index (PSI)

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

| PSI Value | Interpretation |
|-----------|----------------|
| < 0.10    | No significant shift |
| 0.10 - 0.20 | Moderate shift - monitor |
| > 0.20    | Significant shift - action required |

### Kolmogorov-Smirnov Test

- Compares cumulative distributions
- p-value < 0.05 suggests different distributions
- More sensitive to local differences than PSI

---

## Contacts

| Role | Contact |
|------|---------|
| ML Team Lead | ml-lead@company.com |
| On-call Engineer | oncall@company.com |
| Data Platform | data-platform@company.com |

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2024-01-15 | Initial version | ML Team |
