"""
Fraud Predictor - The brain that makes decisions.

HOW IT WORKS:
1. Receive transaction data
2. Transform into features (feature pipeline)
3. Get risk score from ML model (0-1 probability)
4. Apply rules (blocklists, velocity limits)
5. Decide action: ALLOW / CHALLENGE / BLOCK
6. Return explanation (why this decision?)

RISK TIERS:
- low (0-30%): Let it through
- medium (30-60%): Watch closely
- high (60-85%): Probably fraud
- critical (85-100%): Almost certainly fraud
"""

from typing import Optional, Union
from pathlib import Path
from datetime import datetime
import time
import pandas as pd
import yaml
from loguru import logger

from ..features.pipeline import FeaturePipeline
from ..data.schemas import (
    FraudCheckRequest, FraudCheckResponse, PredictionRequest, PredictionResponse,
    ActionType, ChallengeType, RiskReason,
)
from .trainer import ModelTrainer


class FraudPredictor:
    """
    Production fraud predictor.
    
    Takes transaction → returns risk score + action.
    """
    
    # Score → Risk tier mapping
    TIERS = {"low": (0, 0.3), "medium": (0.3, 0.6), "high": (0.6, 0.85), "critical": (0.85, 1)}
    
    def __init__(self, model: ModelTrainer, pipeline: FeaturePipeline, version: str = "v1"):
        self.model = model
        self.pipeline = pipeline
        self.version = version
        self._stats = {"served": 0, "total_ms": 0}
    
    @classmethod
    def load(cls, model_path: str, pipeline_path: str, **kw) -> "FraudPredictor":
        """Load from saved files."""
        return cls(
            model=ModelTrainer.load(model_path),
            pipeline=FeaturePipeline.load(pipeline_path),
            **kw
        )
    
    def predict(self, txn: Union[dict, PredictionRequest]) -> PredictionResponse:
        """
        Predict fraud risk for one transaction.
        
        Returns: PredictionResponse with probability and risk tier
        """
        start = time.perf_counter()
        
        # Handle both dict and PredictionRequest
        if hasattr(txn, "model_dump"):
            txn = txn.model_dump()
        
        # Transform to features
        df = pd.DataFrame([txn])
        features = self.pipeline.transform(df)
        X = features.drop(columns=["transaction_id"], errors="ignore")
        
        # Get probability
        prob = float(self.model.predict_proba(X)[0])
        tier = self._get_tier(prob)
        
        # Stats
        ms = (time.perf_counter() - start) * 1000
        self._stats["served"] += 1
        self._stats["total_ms"] += ms
        
        return PredictionResponse(
            transaction_id=txn.get("transaction_id", "unknown"),
            fraud_probability=round(prob, 4),
            risk_tier=tier,
            model_version=self.version,
            latency_ms=round(ms, 2),
        )
    
    def _get_tier(self, prob: float) -> str:
        """Map probability to risk tier."""
        for tier, (lo, hi) in self.TIERS.items():
            if lo <= prob < hi:
                return tier
        return "critical"
    
    def get_health(self) -> dict:
        """Health check stats."""
        avg = self._stats["total_ms"] / max(self._stats["served"], 1)
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": self.version,
            "total_predictions": self._stats["served"],
            "avg_latency_ms": round(avg, 2),
        }


class EnhancedPredictor(FraudPredictor):
    """
    Full predictor with rules + explanations.
    
    FLOW:
    1. Pre-ML rules (blocklist check) → instant BLOCK if matched
    2. ML score
    3. Post-ML rules → decide ALLOW / CHALLENGE / BLOCK
    4. Generate explanation (why?)
    """
    
    # Thresholds for action decisions
    THRESHOLDS = {"allow": 0.30, "challenge": 0.70, "block": 0.90}
    
    def __init__(self, model, pipeline, config: dict = None, **kw):
        super().__init__(model, pipeline, **kw)
        self.config = config or {}
        self.rules = self._build_rules()
    
    @classmethod
    def load(cls, model_path, pipeline_path, config_path=None, **kw):
        """Load with optional config."""
        config = {}
        if config_path:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        return cls(
            model=ModelTrainer.load(model_path),
            pipeline=FeaturePipeline.load(pipeline_path),
            config=config,
            **kw
        )
    
    def check(self, req: Union[FraudCheckRequest, dict]) -> FraudCheckResponse:
        """
        Full fraud check: rules + ML + explanation.
        
        This is the main production endpoint.
        """
        start = time.perf_counter()
        
        # Parse request
        if isinstance(req, dict):
            req = FraudCheckRequest(**req)
        
        txn_id = req.transaction.transaction_id
        ctx = self._build_context(req)
        
        # STEP 1: Pre-ML rules (blocklist, hard limits)
        blocked = self._check_blocklist(ctx)
        if blocked:
            return self._response(txn_id, 1.0, ActionType.BLOCK, blocked, start)
        
        # STEP 2: ML prediction
        features = self._prepare_features(req)
        X = features.drop(columns=["transaction_id"], errors="ignore")
        score = float(self.model.predict_proba(X)[0])
        
        # STEP 3: Post-ML rules (decide action)
        action, challenge, reason = self._decide_action(score, ctx)
        
        # STEP 4: Explanation
        reasons = self._explain(X.iloc[0].to_dict(), score) if req.return_explanation else []
        
        return self._response(txn_id, score, action, reason, start, challenge, reasons)
    
    # ========== RULES ==========
    
    def _build_rules(self) -> list:
        """
        Simple rules that override ML.
        
        WHY RULES?
        - Blocklists: known bad actors, skip ML entirely
        - Velocity: too many txns = suspicious
        - Device: VPN + high amount = risky
        """
        return [
            # Blocklist rules (instant block)
            {"name": "blocked_user", "check": lambda c: c.get("user_is_blocked"), "action": "block"},
            {"name": "blocked_device", "check": lambda c: c.get("device_is_blocked"), "action": "block"},
            {"name": "repeat_fraudster", "check": lambda c: c.get("previous_fraud_flags", 0) >= 2, "action": "block"},
            
            # Velocity rules
            {"name": "velocity_1h", "check": lambda c: c.get("txn_count_1h", 0) > 20, "action": "block"},
            {"name": "velocity_10m", "check": lambda c: c.get("txn_count_10m", 0) > 5, "action": "challenge"},
            
            # Device risk rules
            {"name": "vpn_high_amount", "check": lambda c: c.get("is_vpn") and c.get("amount", 0) > 50000, "action": "challenge"},
            {"name": "emulator", "check": lambda c: c.get("is_emulator"), "action": "challenge"},
        ]
    
    def _check_blocklist(self, ctx: dict) -> Optional[str]:
        """Check pre-ML rules. Returns reason if blocked, None otherwise."""
        for rule in self.rules:
            if rule["action"] == "block" and rule["check"](ctx):
                return f"Rule triggered: {rule['name']}"
        return None
    
    def _decide_action(self, score: float, ctx: dict) -> tuple:
        """
        Decide action based on score + rules.
        
        LOGIC:
        - score >= 0.90 → BLOCK
        - score >= 0.70 → CHALLENGE (verify identity)
        - score >= 0.30 → ALLOW (but watch)
        - score < 0.30 → ALLOW (low risk)
        """
        # Check challenge rules first
        for rule in self.rules:
            if rule["action"] == "challenge" and rule["check"](ctx):
                return ActionType.CHALLENGE, ChallengeType.OTP, f"Rule: {rule['name']}"
        
        # Score-based decision
        if score >= self.THRESHOLDS["block"]:
            return ActionType.BLOCK, None, "Risk score too high"
        elif score >= self.THRESHOLDS["challenge"]:
            return ActionType.CHALLENGE, ChallengeType.OTP, "Elevated risk - verification needed"
        else:
            return ActionType.ALLOW, None, "Risk acceptable"
    
    # ========== HELPERS ==========
    
    def _build_context(self, req: FraudCheckRequest) -> dict:
        """Flatten request into context dict for rules."""
        ctx = req.transaction.model_dump() if hasattr(req.transaction, 'model_dump') else dict(req.transaction)
        for signals in [req.user_signals, req.device_signals, req.velocity_signals, req.network_signals]:
            if signals:
                ctx.update(signals.model_dump() if hasattr(signals, 'model_dump') else dict(signals))
        return ctx
    
    def _prepare_features(self, req: FraudCheckRequest) -> pd.DataFrame:
        """Convert request to feature DataFrame."""
        ctx = self._build_context(req)
        return self.pipeline.transform(pd.DataFrame([ctx]))
    
    def _explain(self, features: dict, score: float) -> list:
        """
        Generate human-readable explanations.
        
        WHY?
        Humans need to understand why a transaction was flagged.
        "Transaction blocked" isn't helpful.
        "Blocked because: VPN detected + new device + 10 txns in 1 hour" is.
        """
        reasons = []
        
        # Check each risky feature
        explanations = {
            "is_vpn": ("VPN/proxy detected", 0.2),
            "is_new_device": ("First transaction from this device", 0.15),
            "is_emulator": ("Running on emulator", 0.25),
            "txn_count_1h": ("High transaction frequency", 0.1),
            "previous_fraud_flags": ("Account has fraud history", 0.3),
            "account_age_days": ("New account", 0.1),
        }
        
        for feat, (desc, weight) in explanations.items():
            val = features.get(feat)
            if val and (val == True or val == 1 or (isinstance(val, (int, float)) and val > 3)):
                if feat == "account_age_days" and val > 30:
                    continue  # Old account is fine
                reasons.append(RiskReason(
                    feature=feat,
                    description=desc,
                    contribution=weight,
                    severity="high" if weight > 0.15 else "medium",
                ))
        
        return reasons[:5]  # Top 5 reasons
    
    def _response(self, txn_id, score, action, reason, start, challenge=None, reasons=None) -> FraudCheckResponse:
        """Build response object."""
        tier = self._get_tier(score)
        ms = (time.perf_counter() - start) * 1000
        self._stats["served"] += 1
        self._stats["total_ms"] += ms
        
        return FraudCheckResponse(
            transaction_id=txn_id,
            risk_score=round(score, 4),
            risk_tier=tier,
            action=action,
            challenge_type=challenge,
            action_reason=reason,
            risk_reasons=reasons or [],
            model_version=self.version,
            processing_time_ms=round(ms, 2),
        )
