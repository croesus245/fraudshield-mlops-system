"""
Generate synthetic fraud data.

WHY SYNTHETIC?
- Real fraud data is sensitive/confidential
- But synthetic data shows the same patterns
- Lets us demonstrate the system without real PII

WHAT GETS GENERATED:
1. Users → account_age, verified status, fraud history
2. Merchants → category, fraud rate
3. Transactions → with realistic fraud signals

HOW FRAUD IS DIFFERENT:
- Happens more at night (fraudsters work when you sleep)
- Uses VPNs, emulators, new devices
- Higher amounts or tiny "test" amounts
- Comes from accounts with past fraud flags
"""

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import random
import hashlib
import pandas as pd
import numpy as np
from loguru import logger


# ============================================================
# DATA GENERATORS
# ============================================================

def make_users(n: int) -> pd.DataFrame:
    """Generate user profiles with risk signals."""
    users = []
    for i in range(n):
        risky = random.random() < 0.05  # 5% are risky
        users.append({
            "user_id": f"user_{i:05d}",
            "account_age_days": random.randint(1, 30) if risky else random.randint(30, 1000),
            "is_verified": random.random() > 0.6 if risky else random.random() > 0.1,
            "previous_fraud_flags": random.choices([0, 1, 2], [0.9, 0.07, 0.03])[0] if not risky else random.choices([0, 1, 2, 3], [0.4, 0.3, 0.2, 0.1])[0],
            "chargebacks_90d": random.choices([0, 1, 2], [0.95, 0.04, 0.01])[0],
        })
    return pd.DataFrame(users)


def make_merchants(n: int) -> pd.DataFrame:
    """Generate merchant profiles."""
    categories = ["retail", "food", "travel", "entertainment", "services", "online", "gaming"]
    merchants = []
    for i in range(n):
        risky = random.random() < 0.03
        merchants.append({
            "merchant_id": f"merchant_{i:04d}",
            "merchant_category": random.choice(categories),
            "merchant_fraud_rate": random.uniform(0.05, 0.2) if risky else random.uniform(0, 0.02),
        })
    return pd.DataFrame(merchants)


def make_transactions(n: int, fraud_rate: float, users: pd.DataFrame, merchants: pd.DataFrame) -> pd.DataFrame:
    """
    Generate transactions with realistic fraud patterns.
    
    FRAUD SIGNALS ENCODED:
    - VPN usage (40% of fraud vs 5% legit)
    - New device (50% of fraud vs 15% legit)
    - Night transactions (fraud peaks 2-5am)
    - High velocity (many txns in short time)
    """
    txns = []
    user_ids = users["user_id"].tolist()
    merchant_ids = merchants["merchant_id"].tolist()
    start = datetime.now() - timedelta(days=30)

    for i in range(n):
        is_fraud = random.random() < fraud_rate
        user_id = random.choice(user_ids[:50]) if is_fraud and random.random() < 0.3 else random.choice(user_ids)
        merchant = merchants[merchants["merchant_id"] == random.choice(merchant_ids)].iloc[0]

        # Amount: fraud is bimodal (tiny test OR large theft)
        if is_fraud:
            amount = random.uniform(1, 10) if random.random() < 0.3 else random.lognormvariate(6, 1)
        else:
            amount = random.lognormvariate(3.5, 1.2)

        # Time: fraud peaks at night
        hour = random.choices(range(24), weights=[3,3,3,2,2,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3])[0] if is_fraud else random.choices(range(24), weights=[1,1,1,1,1,2,3,4,5,6,6,6,6,6,5,5,5,4,4,3,2,2,1,1])[0]
        timestamp = start + timedelta(days=random.uniform(0, 30), hours=hour)

        # Device signals: fraud uses VPNs, emulators, new devices
        is_vpn = random.random() < 0.4 if is_fraud else random.random() < 0.05
        is_new_device = random.random() < 0.5 if is_fraud else random.random() < 0.15
        is_emulator = random.random() < 0.2 if is_fraud else random.random() < 0.01

        # Velocity: fraud has more txns in short windows
        txn_count_1h = random.choices([1,2,3,5,10], [0.3,0.2,0.2,0.2,0.1])[0] if is_fraud else random.choices([1,2,3], [0.7,0.2,0.1])[0]
        txn_count_24h = random.choices([1,5,10,20], [0.2,0.3,0.3,0.2])[0] if is_fraud else random.choices([1,2,5], [0.6,0.3,0.1])[0]

        txns.append({
            "transaction_id": f"txn_{i:08d}",
            "user_id": user_id,
            "merchant_id": merchant["merchant_id"],
            "merchant_category": merchant["merchant_category"],
            "amount": round(amount, 2),
            "timestamp": timestamp,
            "is_vpn": is_vpn,
            "is_new_device": is_new_device,
            "is_emulator": is_emulator,
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
            "txn_count_1h": txn_count_1h,
            "txn_count_24h": txn_count_24h,
            "amount_sum_24h": round(amount * txn_count_24h * random.uniform(0.5, 1.5), 2),
            "merchant_fraud_rate": merchant["merchant_fraud_rate"],
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    return df.merge(users, on="user_id", how="left")


def make_delayed_labels(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate delayed label arrival.
    
    WHY DELAYED?
    In production, you don't know immediately if a transaction is fraud.
    Labels come days/weeks later via:
    - Chargebacks (customer disputes)
    - Investigations
    - User reports
    """
    labels = []
    for _, t in txns.iterrows():
        if random.random() > 0.95:  # 5% never get labeled
            continue
        delay = random.uniform(1, 14) if t["is_fraud"] else random.uniform(5, 10)
        labels.append({
            "transaction_id": t["transaction_id"],
            "is_fraud": t["is_fraud"],
            "label_timestamp": t["timestamp"] + timedelta(days=delay),
            "label_source": random.choice(["chargeback", "investigation", "user_report", "auto"]),
        })
    return pd.DataFrame(labels)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10000, help="Number of transactions")
    parser.add_argument("--fraud-rate", type=float, default=0.02)
    parser.add_argument("--output", default="data")
    args = parser.parse_args()

    # Generate
    users = make_users(1000)
    merchants = make_merchants(200)
    txns = make_transactions(args.n, args.fraud_rate, users, merchants)
    labels = make_delayed_labels(txns)

    # Save
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    
    # transactions.parquet = what you see in production (NO labels)
    # ground_truth.parquet = for training only (HAS labels)
    # labels.parquet = labels arriving with delay
    txns.drop(columns=["is_fraud"]).to_parquet(out / "transactions.parquet")
    txns.to_parquet(out / "ground_truth.parquet")
    labels.to_parquet(out / "labels.parquet")

    print(f"\n{'='*40}")
    print(f"Generated {len(txns):,} transactions")
    print(f"Fraud rate: {txns['is_fraud'].mean():.1%}")
    print(f"Saved to {out}/")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
