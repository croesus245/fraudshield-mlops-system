"""
Rule Engine - Business logic that ML can't capture.

WHY RULES + ML?
- ML gives probability (0-1)
- Rules decide ACTION (allow/challenge/block)
- Rules handle edge cases ML might miss

RULE TYPES:
1. PRE-ML: Check BEFORE ML (blocklists) → instant decision
2. POST-ML: Check AFTER ML → combine score + context

EXAMPLE:
- User on blocklist → BLOCK (skip ML entirely)
- Score=0.85 → CHALLENGE (verify via OTP)
- Score=0.95 → BLOCK (too risky)
"""

from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum
from loguru import logger

from ..data.schemas import ActionType, ChallengeType


class RulePriority(Enum):
    """Higher priority = evaluated first."""
    CRITICAL = 1  # Blocklists
    HIGH = 2      # Velocity limits
    MEDIUM = 3    # Business rules
    LOW = 4       # Soft rules


@dataclass
class Rule:
    """A single rule."""
    name: str
    condition: Callable[[dict], bool]  # Returns True if rule fires
    action: ActionType
    priority: RulePriority = RulePriority.MEDIUM
    challenge_type: Optional[ChallengeType] = None
    description: str = ""


@dataclass
class RuleResult:
    """Result of rule evaluation."""
    triggered: bool
    rule_name: str = None
    action: ActionType = None
    challenge_type: ChallengeType = None
    reason: str = None


class RuleEngine:
    """
    Evaluate rules against transaction context.
    
    Usage:
        engine = RuleEngine()
        
        # Before ML
        result = engine.check_pre_ml(context)
        if result.triggered:
            return result.action  # Skip ML
        
        # After ML
        score = model.predict(features)
        action = engine.decide_action(score, context)
    """
    
    # Score thresholds
    THRESHOLDS = {
        "payment": {"allow": 0.30, "challenge": 0.70, "block": 0.90},
        "transfer": {"allow": 0.20, "challenge": 0.60, "block": 0.85},  # Stricter
        "withdrawal": {"allow": 0.20, "challenge": 0.60, "block": 0.85},  # Stricter
    }
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.rules = self._default_rules()
    
    def _default_rules(self) -> list:
        """
        Built-in rules.
        
        BLOCKLIST RULES → Instant block, skip ML
        VELOCITY RULES → Rate limiting
        DEVICE RULES → Suspicious devices
        USER RULES → Suspicious users
        """
        return [
            # ===== BLOCKLIST (instant block) =====
            Rule("blocked_user", lambda c: c.get("user_is_blocked", False),
                 ActionType.BLOCK, RulePriority.CRITICAL, description="User on blocklist"),
            
            Rule("blocked_device", lambda c: c.get("device_is_blocked", False),
                 ActionType.BLOCK, RulePriority.CRITICAL, description="Device on blocklist"),
            
            Rule("repeat_fraudster", lambda c: c.get("previous_fraud_flags", 0) >= 2,
                 ActionType.BLOCK, RulePriority.CRITICAL, description="Multiple fraud flags"),
            
            # ===== VELOCITY (rate limits) =====
            Rule("velocity_1h", lambda c: c.get("txn_count_1h", 0) > 20,
                 ActionType.BLOCK, RulePriority.HIGH, description=">20 txns in 1 hour"),
            
            Rule("velocity_10m", lambda c: c.get("txn_count_10m", 0) > 5,
                 ActionType.CHALLENGE, RulePriority.HIGH, ChallengeType.OTP, ">5 txns in 10 min"),
            
            Rule("amount_24h", lambda c: c.get("amount_sum_24h", 0) > 1000000,
                 ActionType.CHALLENGE, RulePriority.HIGH, ChallengeType.MANUAL_REVIEW, "Daily limit exceeded"),
            
            # ===== DEVICE RISK =====
            Rule("vpn_high_amount",
                 lambda c: c.get("is_vpn", False) and c.get("amount", 0) > 50000,
                 ActionType.CHALLENGE, RulePriority.HIGH, ChallengeType.OTP, "VPN + high amount"),
            
            Rule("emulator", lambda c: c.get("is_emulator", False),
                 ActionType.CHALLENGE, RulePriority.HIGH, ChallengeType.SELFIE, "Emulator detected"),
            
            Rule("new_device_high", 
                 lambda c: c.get("is_new_device", False) and c.get("amount", 0) > 100000,
                 ActionType.CHALLENGE, RulePriority.MEDIUM, ChallengeType.OTP, "New device + high amount"),
            
            # ===== USER RISK =====
            Rule("new_user_high",
                 lambda c: c.get("account_age_days", 999) < 3 and c.get("amount", 0) > 50000,
                 ActionType.CHALLENGE, RulePriority.MEDIUM, ChallengeType.MANUAL_REVIEW, "New account + high amount"),
            
            Rule("unverified_high",
                 lambda c: not c.get("is_verified", True) and c.get("amount", 0) > 100000,
                 ActionType.CHALLENGE, RulePriority.MEDIUM, ChallengeType.SELFIE, "Unverified + high amount"),
        ]
    
    def add_rule(self, rule: Rule):
        """Add custom rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority.value)
    
    def check_pre_ml(self, context: dict) -> RuleResult:
        """
        Check blocklist rules BEFORE ML.
        If any fires → skip ML entirely.
        """
        for rule in self.rules:
            if rule.priority == RulePriority.CRITICAL:
                try:
                    if rule.condition(context):
                        return RuleResult(True, rule.name, rule.action, reason=rule.description)
                except Exception as e:
                    logger.warning(f"Rule {rule.name} failed: {e}")
        
        return RuleResult(False)
    
    def decide_action(self, score: float, context: dict, txn_type: str = "payment") -> RuleResult:
        """
        Decide action based on ML score + rules.
        
        LOGIC:
        1. Check challenge rules (velocity, device risk)
        2. Apply score thresholds
        """
        # Get thresholds for this transaction type
        thresh = self.THRESHOLDS.get(txn_type, self.THRESHOLDS["payment"])
        
        # Check non-critical rules that might override
        for rule in self.rules:
            if rule.priority != RulePriority.CRITICAL and rule.action == ActionType.CHALLENGE:
                try:
                    if rule.condition(context):
                        return RuleResult(True, rule.name, rule.action, rule.challenge_type, rule.description)
                except:
                    pass
        
        # Score-based decision
        if score >= thresh["block"]:
            return RuleResult(True, "score_block", ActionType.BLOCK, reason="Risk score too high")
        elif score >= thresh["challenge"]:
            challenge = ChallengeType.OTP if score < 0.85 else ChallengeType.SELFIE
            return RuleResult(True, "score_challenge", ActionType.CHALLENGE, challenge, "Elevated risk")
        else:
            return RuleResult(False, action=ActionType.ALLOW, reason="Risk acceptable")
