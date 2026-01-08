"""
Data schemas using Pydantic for runtime validation.

These schemas validate individual records (e.g., API requests).
For batch validation, see contracts.py.
"""

from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    """Types of transactions we monitor."""
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    REFUND = "refund"
    ORDER = "order"
    TOP_UP = "top_up"


class VerificationLevel(str, Enum):
    """User verification levels."""
    NONE = "none"
    BASIC = "basic"           # Email/phone verified
    STANDARD = "standard"     # ID uploaded
    FULL = "full"             # Full KYC complete


class RiskTier(str, Enum):
    """Risk classification tiers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """Actions to take based on risk."""
    ALLOW = "allow"
    CHALLENGE = "challenge"
    BLOCK = "block"
    MANUAL_REVIEW = "manual_review"


class ChallengeType(str, Enum):
    """Types of verification challenges."""
    OTP = "otp"
    SECURITY_QUESTION = "security_question"
    SELFIE = "selfie_verification"
    MANUAL_REVIEW = "manual_review"
    HOLD_FUNDS = "hold_funds"


# ============================================================
# USER SIGNALS
# ============================================================
class UserSignals(BaseModel):
    """Signals about the user's account and history."""
    
    user_id: str = Field(..., description="User identifier")
    account_age_days: int = Field(0, ge=0, description="Days since account created")
    is_verified: bool = Field(False, description="KYC completed")
    verification_level: VerificationLevel = Field(VerificationLevel.NONE)
    previous_fraud_flags: int = Field(0, ge=0, description="Past fraud incidents")
    is_blocked: bool = Field(False, description="User on blocklist")
    
    # Transaction history
    total_lifetime_transactions: int = Field(0, ge=0)
    total_lifetime_amount: float = Field(0, ge=0)
    days_since_last_transaction: Optional[int] = Field(None, ge=0)
    avg_transaction_amount: Optional[float] = Field(None, ge=0)
    transaction_frequency_weekly: Optional[float] = Field(None, ge=0)
    
    # Risk indicators
    chargebacks_90d: int = Field(0, ge=0)
    refunds_90d: int = Field(0, ge=0)
    disputes_90d: int = Field(0, ge=0)


# ============================================================
# DEVICE SIGNALS
# ============================================================
class DeviceSignals(BaseModel):
    """Signals about the device and session."""
    
    device_id: Optional[str] = Field(None, description="Device fingerprint")
    is_new_device: bool = Field(False, description="First time seeing this device")
    device_age_days: Optional[int] = Field(None, ge=0)
    device_type: str = Field("unknown", description="mobile/desktop/tablet")
    
    # Security indicators
    is_vpn: bool = Field(False, description="VPN/proxy detected")
    is_emulator: bool = Field(False, description="Running in emulator")
    is_rooted: bool = Field(False, description="Rooted/jailbroken")
    is_tor: bool = Field(False, description="Tor exit node")
    
    # Session info
    ip_address: Optional[str] = Field(None)
    ip_country: Optional[str] = Field(None)
    ip_country_mismatch: bool = Field(False, description="IP country != account country")
    
    # Login patterns
    login_hour_deviation: Optional[float] = Field(None, description="Std devs from usual login time")
    failed_logins_24h: int = Field(0, ge=0)
    session_duration_minutes: Optional[float] = Field(None, ge=0)
    
    # Device sharing
    devices_last_30d: int = Field(1, ge=1, description="Unique devices in 30 days")


# ============================================================
# NETWORK SIGNALS
# ============================================================
class NetworkSignals(BaseModel):
    """Signals about relationships and network patterns."""
    
    # Sharing patterns (fraud rings share resources)
    shared_bank_accounts: int = Field(0, ge=0, description="Users sharing this bank account")
    shared_devices_count: int = Field(0, ge=0, description="Accounts on this device")
    shared_ip_count: int = Field(0, ge=0, description="Accounts from this IP")
    shared_phone_count: int = Field(0, ge=0, description="Accounts with this phone")
    
    # Merchant risk
    merchant_age_days: Optional[int] = Field(None, ge=0)
    merchant_transaction_count: Optional[int] = Field(None, ge=0)
    merchant_complaint_rate: Optional[float] = Field(None, ge=0, le=1)
    merchant_fraud_rate: Optional[float] = Field(None, ge=0, le=1)
    merchant_chargeback_rate: Optional[float] = Field(None, ge=0, le=1)
    
    # Recipient risk (for transfers)
    recipient_fraud_flags: int = Field(0, ge=0)
    recipient_is_new: bool = Field(False)
    
    # Computed scores
    refund_abuse_score: Optional[float] = Field(None, ge=0, le=1)
    network_risk_score: Optional[float] = Field(None, ge=0, le=1)


# ============================================================
# VELOCITY SIGNALS
# ============================================================
class VelocitySignals(BaseModel):
    """Transaction velocity and aggregation signals."""
    
    # Transaction counts
    txn_count_10m: int = Field(0, ge=0)
    txn_count_1h: int = Field(0, ge=0)
    txn_count_6h: int = Field(0, ge=0)
    txn_count_24h: int = Field(0, ge=0)
    txn_count_7d: int = Field(0, ge=0)
    txn_count_30d: int = Field(0, ge=0)
    
    # Amount sums
    amount_sum_10m: float = Field(0, ge=0)
    amount_sum_1h: float = Field(0, ge=0)
    amount_sum_24h: float = Field(0, ge=0)
    amount_sum_7d: float = Field(0, ge=0)
    
    # Unique counts
    unique_merchants_24h: int = Field(0, ge=0)
    unique_recipients_24h: int = Field(0, ge=0)
    unique_ips_24h: int = Field(0, ge=0)
    
    # Failures
    failed_attempts_1h: int = Field(0, ge=0)
    declined_count_24h: int = Field(0, ge=0)


# ============================================================
# TRANSACTION SCHEMA (ENHANCED)
# ============================================================
class TransactionSchema(BaseModel):
    """Schema for a single transaction record."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    amount: float = Field(..., ge=0, le=10000000, description="Transaction amount")
    currency: str = Field("NGN", description="Currency code")
    
    # Transaction details
    transaction_type: TransactionType = Field(TransactionType.PAYMENT)
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code")
    channel: str = Field("app", description="app/web/ussd/pos")
    
    # User & device
    user_id: str = Field(..., description="User identifier")
    device_type: Optional[str] = Field(None, description="Device type")
    location_country: Optional[str] = Field(None, description="Country code")
    
    # Optional: recipient (for transfers)
    recipient_id: Optional[str] = Field(None, description="Transfer recipient")
    
    # Derived features (optional, can be computed)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    is_weekend: Optional[bool] = None
    is_night: Optional[bool] = None
    
    @field_validator("merchant_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {"retail", "travel", "entertainment", "groceries", "other", 
                   "electronics", "restaurants", "utilities", "healthcare",
                   "food", "services", "online", "gaming"}
        if v.lower() not in allowed:
            return "other"
        return v.lower()
    
    @field_validator("device_type")
    @classmethod
    def validate_device(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        allowed = {"mobile", "desktop", "tablet", "unknown", "pos"}
        if v.lower() not in allowed:
            return "unknown"
        return v.lower()


class LabelSchema(BaseModel):
    """Schema for fraud labels (arrive with delay)."""
    
    transaction_id: str = Field(..., description="Transaction to label")
    is_fraud: int = Field(..., ge=0, le=1, description="1 if fraud, 0 otherwise")
    label_timestamp: datetime = Field(..., description="When the label was determined")
    label_source: Optional[str] = Field(None, description="How label was determined")
    fraud_type: Optional[str] = Field(None, description="Type of fraud if applicable")
    
    @field_validator("label_source")
    @classmethod
    def validate_source(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "unknown"
        allowed = {"chargeback", "manual_review", "user_report", "automated", 
                   "investigation", "auto_confirm", "unknown"}
        if v.lower() not in allowed:
            return "unknown"
        return v.lower()


# ============================================================
# API REQUEST/RESPONSE SCHEMAS
# ============================================================
class FraudCheckRequest(BaseModel):
    """Full fraud check request with all signals."""
    
    transaction: TransactionSchema
    user_signals: Optional[UserSignals] = None
    device_signals: Optional[DeviceSignals] = None
    network_signals: Optional[NetworkSignals] = None
    velocity_signals: Optional[VelocitySignals] = None
    
    # Options
    return_explanation: bool = Field(True, description="Return risk reasons")
    return_action: bool = Field(True, description="Return recommended action")


class RiskReason(BaseModel):
    """A single reason contributing to risk score."""
    
    feature: str = Field(..., description="Feature name")
    description: str = Field(..., description="Human-readable explanation")
    contribution: float = Field(..., description="Contribution to risk score")
    value: Optional[str] = Field(None, description="Actual value")
    severity: Optional[str] = Field("medium", description="low/medium/high/critical")


class FraudCheckResponse(BaseModel):
    """Full fraud check response."""
    
    transaction_id: str
    
    # Risk assessment
    risk_score: float = Field(..., ge=0, le=1, description="Fraud probability 0-1")
    risk_tier: str = Field(..., description="low/medium/high/critical")
    
    # Action
    action: ActionType = Field(..., description="allow/challenge/block")
    challenge_type: Optional[ChallengeType] = Field(None, description="If challenged, how")
    action_reason: Optional[str] = Field(None, description="Why this action")
    
    # Explanation
    risk_reasons: List[RiskReason] = Field(default_factory=list)
    
    # Metadata
    model_version: str = Field("unknown")
    processing_time_ms: float = Field(0.0)
    rule_triggered: Optional[str] = Field(None, description="If a rule overrode ML")


# Legacy schemas for backward compatibility
class PredictionRequest(BaseModel):
    """Legacy schema for prediction API requests."""
    
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    amount: float = Field(..., ge=0)
    merchant_id: str = Field(...)
    merchant_category: str = Field(...)
    user_id: str = Field(...)
    device_type: Optional[str] = Field(None)
    is_foreign: bool = Field(False)


class PredictionResponse(BaseModel):
    """Legacy schema for prediction API responses."""
    
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    risk_tier: str = Field(...)
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    """Schema for health check responses."""
    
    status: str = Field(..., description="healthy/degraded/unhealthy")
    model_loaded: bool
    model_version: Optional[str] = None
    total_predictions: int = Field(0)
    avg_latency_ms: Optional[float] = None
