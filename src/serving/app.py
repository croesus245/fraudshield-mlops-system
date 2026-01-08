"""
FraudShield API - REST endpoints for fraud detection.

ENDPOINTS:
- GET  /health → Is the service running?
- POST /predict → Get fraud probability (simple)
- POST /check → Full fraud check with action + explanation (recommended)

HOW TO RUN:
    python -m scripts.serve
    # Then: curl http://localhost:8000/health
"""

from pathlib import Path
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ..data.schemas import (
    PredictionRequest, PredictionResponse, HealthResponse,
    FraudCheckRequest, FraudCheckResponse, ActionType, RiskReason
)
from ..models.predictor import FraudPredictor, EnhancedPredictor


# Global predictor (loaded on startup)
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global predictor
    
    logger.info("Starting FraudShield API...")
    
    # Use base name - trainer adds .model/.meta suffixes automatically
    model_path = Path("artifacts/models/xgboost_model")
    pipeline_path = Path("artifacts/models/feature_pipeline.pkl")
    
    if model_path.with_suffix(".model").exists() and pipeline_path.exists():
        predictor = EnhancedPredictor.load(str(model_path), str(pipeline_path))
        logger.info("Model loaded ✓")
    else:
        logger.warning("Model not found - running in degraded mode")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="FraudShield API",
    description="Real-time fraud detection with ML + rules",
    version="2.0",
    lifespan=lifespan,
)

# Allow CORS (configure properly in production)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """
    Health check.
    
    Returns: status, model_loaded, avg_latency
    """
    if predictor:
        return HealthResponse(**predictor.get_health())
    return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    """
    Simple prediction - just get the fraud probability.
    
    Use /check for full analysis with action + explanation.
    """
    if not predictor:
        return PredictionResponse(
            transaction_id=req.transaction_id,
            fraud_probability=0.5,
            risk_tier="medium",
            model_version="degraded",
            latency_ms=0,
        )
    
    return predictor.predict(req.model_dump())


@app.post("/check", response_model=FraudCheckResponse)
async def check(req: FraudCheckRequest):
    """
    Full fraud check with action + explanation.
    
    RECOMMENDED ENDPOINT.
    
    Returns:
    - risk_score: 0-1 probability
    - action: ALLOW / CHALLENGE / BLOCK
    - challenge_type: OTP / SELFIE / etc (if challenged)
    - risk_reasons: Why this decision
    
    Example:
        curl -X POST http://localhost:8000/check -H "Content-Type: application/json" -d '{
            "transaction": {
                "transaction_id": "txn_001",
                "user_id": "user_123",
                "amount": 50000,
                "merchant_id": "m_001",
                "merchant_category": "electronics",
                "timestamp": "2024-01-15T14:30:00Z"
            },
            "device_signals": {"is_vpn": true, "is_new_device": true}
        }'
    """
    if not predictor:
        return FraudCheckResponse(
            transaction_id=req.transaction.transaction_id,
            risk_score=0.5,
            risk_tier="medium",
            action=ActionType.CHALLENGE,
            action_reason="System in degraded mode",
            risk_reasons=[RiskReason(feature="system", description="Model not loaded", contribution=0.5)],
        )
    
    return predictor.check(req)


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "FraudShield API",
        "version": "2.0",
        "endpoints": ["/health", "/predict", "/check"],
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
