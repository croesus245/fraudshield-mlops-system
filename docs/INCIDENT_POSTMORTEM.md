# Incident Postmortem: [TEMPLATE]

> **This is a template.** Copy and fill in for actual incidents.

---

## Incident Summary

| Field | Value |
|-------|-------|
| **Incident ID** | INC-XXXX |
| **Date** | YYYY-MM-DD |
| **Duration** | X hours |
| **Severity** | Critical / High / Medium / Low |
| **Status** | Resolved / Monitoring |

### One-line Summary

_[Brief description of what happened]_

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert triggered |
| HH:MM | Engineer acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Fix deployed |
| HH:MM | Incident resolved |

---

## Impact

### Business Impact

- **Transactions affected**: X,XXX
- **False positives**: XXX (legitimate transactions blocked)
- **False negatives**: XX (fraud not caught)
- **Revenue impact**: $XXX

### User Impact

- Users affected: XXX
- User complaints: XX
- Support tickets: XX

---

## Root Cause

### What Happened

_[Detailed technical explanation of what went wrong]_

### Why It Happened

_[The deeper cause - process, design, or operational failure]_

### Contributing Factors

1. _[Factor 1]_
2. _[Factor 2]_
3. _[Factor 3]_

---

## Detection

### How Was It Detected?

- [ ] Automated monitoring alert
- [ ] Manual observation
- [ ] User report
- [ ] Downstream system alert

### Detection Gap

_[If detection was delayed, explain why and how to improve]_

---

## Response

### Immediate Actions

1. _[Action taken]_
2. _[Action taken]_

### Resolution

_[How was the incident resolved?]_

---

## Lessons Learned

### What Went Well

- _[Positive observation]_
- _[Positive observation]_

### What Went Wrong

- _[Negative observation]_
- _[Negative observation]_

### Where We Got Lucky

- _[Things that could have been worse]_

---

## Action Items

| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
| P0 | Fix immediate issue | @name | YYYY-MM-DD | Done |
| P1 | Add monitoring | @name | YYYY-MM-DD | In Progress |
| P2 | Update runbook | @name | YYYY-MM-DD | Not Started |

---

## Supporting Data

### Metrics

```
# Insert relevant metrics, graphs, or data
```

### Logs

```
# Insert relevant log snippets
```

---

## Review

| Reviewer | Date | Sign-off |
|----------|------|----------|
| ML Lead | | |
| Engineering Manager | | |
| Product Owner | | |

---

---

# Example Postmortem: Drift-Induced False Positive Spike

## Incident Summary

| Field | Value |
|-------|-------|
| **Incident ID** | INC-2024-001 |
| **Date** | 2024-01-15 |
| **Duration** | 6 hours |
| **Severity** | High |
| **Status** | Resolved |

### One-line Summary

Sudden increase in false positives due to undetected drift in transaction amount distribution after Black Friday sales.

---

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 08:00 | Black Friday sale starts |
| 10:30 | Transaction volume increases 3x |
| 12:00 | Average transaction amount increases 40% |
| 14:00 | Customer complaints about blocked transactions |
| 14:30 | Alert acknowledged by on-call |
| 15:00 | Root cause identified: amount drift |
| 15:30 | Emergency threshold adjustment deployed |
| 16:00 | False positive rate back to normal |

---

## Impact

### Business Impact

- **Transactions affected**: 15,000
- **False positives**: 2,100 (14% of flagged transactions vs normal 5%)
- **Revenue impact**: ~$50,000 in failed sales

### User Impact

- Users affected: 1,800
- Support tickets: 340

---

## Root Cause

### What Happened

The fraud model was trained on normal transaction patterns. During Black Friday, average transaction amounts increased significantly (users buying more expensive items). The model's amount-based features interpreted these higher amounts as fraud indicators.

### Why It Happened

1. **Drift monitoring was daily, not hourly** - didn't catch rapid shift
2. **No seasonal adjustment** - model had no concept of promotional periods
3. **Alert thresholds too relaxed** - PSI of 0.15 didn't trigger alert

### Contributing Factors

1. First Black Friday with this model version
2. No pre-event threshold adjustment
3. On-call didn't have runbook for this scenario

---

## Detection

### How Was It Detected?

- [ ] Automated monitoring alert
- [x] Manual observation (customer complaints)
- [x] User report
- [ ] Downstream system alert

### Detection Gap

Drift monitoring was configured for daily checks. The shift happened within hours and wasn't caught until customer impact was significant.

---

## Response

### Immediate Actions

1. Identified amount-related features as source of false positives
2. Temporarily raised fraud threshold from 0.5 to 0.7
3. Manually reviewed and approved blocked transactions

### Resolution

1. Deployed threshold adjustment
2. Added hourly drift monitoring for high-volume periods
3. Created runbook for promotional events

---

## Lessons Learned

### What Went Well

- Quick root cause identification once escalated
- Threshold adjustment was effective
- Team coordinated well during response

### What Went Wrong

- Detection took too long (4 hours)
- No proactive preparation for known event
- Runbook didn't cover this scenario

### Where We Got Lucky

- Incident happened during business hours
- No actual fraud increase during the period

---

## Action Items

| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
| P0 | Hourly drift monitoring | @mlops | 2024-01-20 | Done |
| P1 | Promotional event runbook | @ml-lead | 2024-01-25 | Done |
| P1 | Pre-event checklist | @fraud-team | 2024-01-30 | In Progress |
| P2 | Seasonal feature engineering | @ml-eng | 2024-02-15 | Not Started |
| P2 | Auto-scaling thresholds | @mlops | 2024-03-01 | Not Started |

---

## Supporting Data

### Drift Metrics

```
Feature: amount
  Normal PSI: 0.02
  Black Friday PSI: 0.35 (critical drift)
  
Feature: amount_zscore
  Normal PSI: 0.01
  Black Friday PSI: 0.42 (critical drift)
```

### False Positive Rate

```
Date           FP Rate
2024-01-14     4.8%
2024-01-15     14.2%  <- Incident
2024-01-16     5.1%   <- After fix
```

---

## Review

| Reviewer | Date | Sign-off |
|----------|------|----------|
| ML Lead | 2024-01-18 | ✓ |
| Engineering Manager | 2024-01-18 | ✓ |
| Product Owner | 2024-01-19 | ✓ |
