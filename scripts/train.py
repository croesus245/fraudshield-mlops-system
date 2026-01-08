"""
Train the fraud detection model.

HOW IT WORKS:
1. Load data with labels (ground_truth.parquet)
2. Split by TIME (not random!) to avoid data leakage
3. Engineer features from raw transactions
4. Train XGBoost model
5. Evaluate and save artifacts

WHY TIME-BASED SPLIT?
- Random splits leak future info into training
- In production, you can't see the future
- Always train on past, test on future
"""

import argparse
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
from loguru import logger

from src.features.pipeline import FeaturePipeline
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ===== 1. LOAD DATA =====
    # ground_truth.parquet has the is_fraud label (we know truth for training)
    data = pd.read_parquet(Path(args.data_dir) / "ground_truth.parquet")
    logger.info(f"Loaded {len(data):,} transactions, {data['is_fraud'].mean():.1%} fraud")

    # ===== 2. TIME-BASED SPLIT =====
    # Train on first 80% of time, test on last 20%
    split_date = data["timestamp"].quantile(0.8)
    train = data[data["timestamp"] <= split_date]
    test = data[data["timestamp"] > split_date]
    logger.info(f"Train: {len(train):,} | Test: {len(test):,}")

    # ===== 3. FEATURE ENGINEERING =====
    # Converts raw data → ML-ready features (log amounts, time encoding, etc.)
    pipeline = FeaturePipeline()
    X_train = pipeline.fit_transform(train)
    X_test = pipeline.transform(test)
    y_train, y_test = train["is_fraud"].values, test["is_fraud"].values

    # Remove non-feature columns
    drop_cols = ["transaction_id"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    logger.info(f"Features: {X_train.shape[1]}")

    # ===== 4. TRAIN MODEL =====
    model_config = config.get("model", {})
    trainer = ModelTrainer(
        model_type=model_config.get("type", "xgboost"),
        model_params=model_config.get("params"),
        threshold=model_config.get("threshold", 0.5),
    )
    trainer.fit(X_train, y_train, eval_set=(X_test, y_test))

    # ===== 5. EVALUATE =====
    y_proba = trainer.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_pred_proba=y_proba, threshold=model_config.get("threshold", 0.5))
    
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print("=" * 40)

    # ===== 6. SAVE ARTIFACTS =====
    out = Path(args.output_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)

    trainer.save(out / "models" / "xgboost_model.json")
    pipeline.save(out / "models" / "feature_pipeline.pkl")
    
    print(f"\nSaved to {out}/models/")


if __name__ == "__main__":
    main()
