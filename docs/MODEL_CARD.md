# Model Card: FraudShield

## Model Details

### Basic Information

| Field | Value |
|-------|-------|
| **Model Name** | FraudShield v1.0 |
| **Model Type** | Binary Classification (XGBoost) |
| **Task** | Real-time fraud detection |
| **Version** | 1.0.0 |
| **Date** | January 2024 |
| **Author** | Abdul-Sobur Ayinde |

### Model Architecture

- **Algorithm**: XGBoost (Gradient Boosted Trees)
- **Objective**: Binary logistic regression
- **Trees**: 100
- **Max Depth**: 6
- **Learning Rate**: 0.1
- **Class Imbalance Handling**: scale_pos_weight=10

---

## Intended Use

### Primary Use Cases

1. **Real-time fraud scoring** for payment transactions
2. **Risk tiering** for manual review prioritization
3. **Automated blocking** of high-risk transactions

### Users

- Fraud operations teams
- Payment processing systems
- Risk management dashboards

### Out-of-Scope Uses

- ❌ Credit scoring or lending decisions
- ❌ Identity verification
- ❌ Law enforcement or legal determinations
- ❌ Decisions affecting non-financial aspects of users

---

## Training Data

### Dataset Description

| Aspect | Detail |
|--------|--------|
| **Source** | Synthetic data simulating real transaction patterns |
| **Size** | 10,000 transactions |
| **Fraud Rate** | ~2% (typical for financial services) |
| **Time Range** | 30 days |

### Features

| Feature Category | Features |
|-----------------|----------|
| **Transaction** | amount, timestamp |
| **User** | user_id, device_type |
| **Merchant** | merchant_id, merchant_category |
| **Contextual** | is_foreign |
| **Engineered** | hour, day_of_week, is_weekend, amount_log, amount_zscore |

### Data Splits

- **Training**: 80% (time-based split)
- **Test**: 20% (most recent data)
- **Validation**: Built into XGBoost early stopping

---

## Evaluation

### Overall Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **ROC-AUC** | 0.85+ | Area under ROC curve |
| **PR-AUC** | 0.50+ | Precision-Recall AUC (important for imbalanced data) |
| **F1 Score** | 0.40+ | At default threshold |
| **Precision** | Varies | Depends on threshold |
| **Recall** | Varies | Depends on threshold |

### Threshold Trade-offs

| Threshold | Precision | Recall | Use Case |
|-----------|-----------|--------|----------|
| 0.3 | Lower | Higher | Catch more fraud, more false positives |
| 0.5 | Balanced | Balanced | Default production setting |
| 0.7 | Higher | Lower | High-confidence alerts only |

### Disaggregated Metrics

Performance varies by subgroup:

| Merchant Category | ROC-AUC | Notes |
|-------------------|---------|-------|
| Retail | ~0.85 | Strong performance |
| Food | ~0.83 | Good performance |
| Travel | ~0.80 | Slightly lower - more fraud patterns |
| Online | ~0.78 | Lower - complex patterns |

### Calibration

- **ECE (Expected Calibration Error)**: < 0.10
- Model probabilities are well-calibrated for risk scoring

---

## Limitations

### Known Limitations

1. **Delayed labels**: Model trained on data where labels arrive 7+ days late. Recent transactions have incomplete feedback.

2. **Synthetic data**: Current model trained on synthetic data. Production deployment requires retraining on real data.

3. **Concept drift**: Fraud patterns change rapidly. Model performance degrades without regular retraining.

4. **Cold start**: New users/merchants have limited history for aggregation features.

5. **Feature availability**: Some features may be missing in real-time inference.

### Failure Modes

| Scenario | Risk | Mitigation |
|----------|------|------------|
| New fraud pattern | Miss fraud | Regular retraining, pattern monitoring |
| Data pipeline failure | Missing features | Graceful degradation, alerts |
| High latency | Timeout | Fallback rules, caching |
| Extreme drift | Performance drop | Drift monitoring, automated alerts |

---

## Ethical Considerations

### Fairness

- Model does not use protected attributes (race, gender, age) directly
- Risk: Proxy discrimination through correlated features (e.g., merchant location)
- Mitigation: Regular fairness audits, slice-based evaluation

### Privacy

- Model uses transaction data only
- No PII stored in model artifacts
- Prediction logs anonymized

### Transparency

- Model is interpretable (tree-based)
- Feature importance available
- Explanations can be provided per-prediction

### Human Oversight

- High-risk predictions flagged for human review
- Model does not make autonomous blocking decisions above threshold
- Regular model review by fraud team

---

## Deployment

### Infrastructure

- **Serving**: FastAPI + Uvicorn
- **Latency Target**: < 50ms p99
- **Throughput**: 100+ predictions/second (single instance)

### Monitoring

- Prediction distribution drift
- Feature drift (PSI, KS tests)
- Latency metrics
- Error rates

### Update Cadence

| Component | Frequency |
|-----------|-----------|
| Model retraining | Weekly |
| Feature pipeline | Monthly |
| Full evaluation | Bi-weekly |
| Drift monitoring | Continuous |

---

## Maintenance

### Owners

| Role | Responsibility |
|------|----------------|
| ML Engineer | Model development, training |
| Data Engineer | Feature pipeline, data quality |
| Fraud Analyst | Label quality, pattern review |
| MLOps | Deployment, monitoring |

### Feedback

To report issues or provide feedback:
- Email: abdulsobur245@gmail.com
- Create an issue in the project repository

---

## Citation

```
@misc{fraudshield2024,
  title={FraudShield: Real-time Fraud Detection with Delayed Labels},
  author={Abdul-Sobur Ayinde},
  year={2024},
  note={Portfolio Project}
}
```

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 2024 | Initial release |

---

*This model card follows best practices from [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993) and [Google's Model Cards](https://modelcards.withgoogle.com/).*
