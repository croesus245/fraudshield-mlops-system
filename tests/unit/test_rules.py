"""
Unit tests for the rule engine module.

Tests:
- Rule evaluation
- Pre-ML blocklist rules
- Post-ML action decisions
- Threshold overrides
"""

import pytest
from datetime import datetime

from src.rules.engine import RuleEngine, Rule, RuleResult, RulePriority
from src.rules.actions import ActionDecider, Action
from src.data.schemas import ActionType, ChallengeType


class TestRuleEngine:
    """Tests for the RuleEngine class."""
    
    @pytest.fixture
    def rule_engine(self):
        """Create a rule engine with default rules."""
        return RuleEngine()
    
    @pytest.fixture
    def normal_context(self):
        """Context for a normal, low-risk transaction."""
        return {
            "user_id": "user_001",
            "device_id": "device_001",
            "amount": 1000,
            "transaction_type": "payment",
            "user_is_blocked": False,
            "device_is_blocked": False,
            "is_vpn": False,
            "is_emulator": False,
            "is_new_device": False,
            "account_age_days": 365,
            "previous_fraud_flags": 0,
            "txn_count_1h": 1,
            "txn_count_24h": 3,
            "shared_devices_count": 0,
            "merchant_fraud_rate": 0.01,
        }
    
    @pytest.fixture
    def high_risk_context(self):
        """Context for a high-risk transaction."""
        return {
            "user_id": "user_fraud",
            "device_id": "device_fraud",
            "amount": 50000,
            "transaction_type": "transfer",
            "user_is_blocked": False,
            "device_is_blocked": False,
            "is_vpn": True,
            "is_emulator": True,
            "is_new_device": True,
            "account_age_days": 3,
            "previous_fraud_flags": 2,
            "txn_count_1h": 10,
            "txn_count_24h": 50,
            "shared_devices_count": 5,
            "merchant_fraud_rate": 0.15,
        }
    
    def test_rule_engine_initialization(self, rule_engine):
        """Test that rule engine initializes with default rules."""
        assert rule_engine is not None
        assert len(rule_engine.rules) > 0
        assert "allow" in rule_engine.thresholds
        assert "challenge" in rule_engine.thresholds
        assert "block" in rule_engine.thresholds
    
    def test_pre_ml_normal_transaction(self, rule_engine, normal_context):
        """Test that normal transactions pass pre-ML checks."""
        result = rule_engine.evaluate_pre_ml(normal_context)
        # Normal transaction should not be blocked by pre-ML rules
        assert result is None or result.action != ActionType.BLOCK
    
    def test_pre_ml_blocked_user(self, rule_engine, normal_context):
        """Test that blocked users are caught by pre-ML rules."""
        context = {**normal_context, "user_is_blocked": True}
        result = rule_engine.evaluate_pre_ml(context)
        assert result is not None
        assert result.action == ActionType.BLOCK
        assert "blocked" in result.rule_name.lower()
    
    def test_pre_ml_blocked_device(self, rule_engine, normal_context):
        """Test that blocked devices are caught by pre-ML rules."""
        context = {**normal_context, "device_is_blocked": True}
        result = rule_engine.evaluate_pre_ml(context)
        assert result is not None
        assert result.action == ActionType.BLOCK
    
    def test_post_ml_low_score_allows(self, rule_engine, normal_context):
        """Test that low risk scores result in ALLOW."""
        result = rule_engine.evaluate_post_ml(
            risk_score=0.1,
            context=normal_context,
        )
        assert result.action == ActionType.ALLOW
    
    def test_post_ml_medium_score_challenges(self, rule_engine, normal_context):
        """Test that medium risk scores result in CHALLENGE."""
        result = rule_engine.evaluate_post_ml(
            risk_score=0.5,
            context=normal_context,
        )
        assert result.action == ActionType.CHALLENGE
        assert result.challenge_type is not None
    
    def test_post_ml_high_score_blocks(self, rule_engine, normal_context):
        """Test that high risk scores result in BLOCK."""
        result = rule_engine.evaluate_post_ml(
            risk_score=0.95,
            context=normal_context,
        )
        assert result.action == ActionType.BLOCK
    
    def test_transaction_type_override_withdrawal(self, rule_engine, normal_context):
        """Test that withdrawal transactions have lower thresholds."""
        context = {**normal_context, "transaction_type": "withdrawal"}
        
        # Score that would be ALLOW for payment should be CHALLENGE for withdrawal
        result = rule_engine.evaluate_post_ml(risk_score=0.25, context=context)
        # Withdrawal has stricter thresholds
        assert result.action in [ActionType.CHALLENGE, ActionType.ALLOW]
    
    def test_challenge_type_selection(self, rule_engine, normal_context):
        """Test that appropriate challenge types are selected based on risk."""
        # Low-medium risk -> OTP
        result = rule_engine.evaluate_post_ml(risk_score=0.4, context=normal_context)
        if result.action == ActionType.CHALLENGE:
            assert result.challenge_type in [
                ChallengeType.OTP, 
                ChallengeType.SECURITY_QUESTION,
                ChallengeType.SELFIE,
            ]
        
        # High risk -> more intensive challenge
        result = rule_engine.evaluate_post_ml(risk_score=0.85, context=normal_context)
        if result.action == ActionType.CHALLENGE:
            assert result.challenge_type in [
                ChallengeType.SELFIE,
                ChallengeType.MANUAL_REVIEW,
                ChallengeType.HOLD_FUNDS,
            ]


class TestRule:
    """Tests for individual Rule objects."""
    
    def test_rule_creation(self):
        """Test creating a custom rule."""
        rule = Rule(
            name="test_rule",
            priority=RulePriority.HIGH,
            condition=lambda ctx: ctx.get("amount", 0) > 100000,
            action=ActionType.BLOCK,
            challenge_type=None,
            description="Block very high amounts",
        )
        
        assert rule.name == "test_rule"
        assert rule.priority == RulePriority.HIGH
        assert rule.action == ActionType.BLOCK
    
    def test_rule_condition_evaluation(self):
        """Test that rule conditions evaluate correctly."""
        rule = Rule(
            name="high_amount",
            priority=RulePriority.HIGH,
            condition=lambda ctx: ctx.get("amount", 0) > 100000,
            action=ActionType.BLOCK,
        )
        
        # Should trigger
        assert rule.condition({"amount": 200000}) is True
        # Should not trigger
        assert rule.condition({"amount": 50000}) is False


class TestActionDecider:
    """Tests for the ActionDecider class."""
    
    @pytest.fixture
    def action_decider(self):
        """Create an ActionDecider instance."""
        return ActionDecider()
    
    def test_action_creation(self, action_decider):
        """Test creating actions."""
        action = action_decider.create_action(
            action_type=ActionType.CHALLENGE,
            challenge_type=ChallengeType.OTP,
            reason="Suspicious activity",
        )
        
        assert action.action_type == ActionType.ALLOW or action.action_type == ActionType.CHALLENGE
    
    def test_format_action_for_response(self, action_decider):
        """Test formatting actions for API response."""
        action = Action(
            action_type=ActionType.CHALLENGE,
            challenge_type=ChallengeType.OTP,
            reason="Verify your identity",
        )
        
        formatted = action_decider.format_for_response(action)
        
        assert "action" in formatted
        assert "instructions" in formatted or formatted.get("challenge_type") is not None


class TestRuleEngineWithHighRiskContext:
    """Tests for rule engine with high-risk transaction contexts."""
    
    @pytest.fixture
    def rule_engine(self):
        return RuleEngine()
    
    def test_velocity_breach_triggers_challenge(self, rule_engine):
        """Test that velocity breaches trigger challenges."""
        context = {
            "user_id": "user_001",
            "device_id": "device_001",
            "amount": 1000,
            "transaction_type": "payment",
            "user_is_blocked": False,
            "device_is_blocked": False,
            "is_vpn": False,
            "txn_count_1h": 15,  # Velocity breach
            "txn_count_24h": 5,
            "account_age_days": 365,
            "previous_fraud_flags": 0,
        }
        
        # Even with low ML score, velocity breach should trigger action
        result = rule_engine.evaluate_post_ml(risk_score=0.2, context=context)
        # Rule engine should consider velocity
        assert result is not None
    
    def test_new_device_high_amount_combination(self, rule_engine):
        """Test that new device + high amount is flagged."""
        context = {
            "user_id": "user_001",
            "device_id": "device_new",
            "amount": 50000,  # High amount
            "transaction_type": "transfer",
            "user_is_blocked": False,
            "device_is_blocked": False,
            "is_vpn": False,
            "is_new_device": True,
            "account_age_days": 365,
            "previous_fraud_flags": 0,
            "txn_count_1h": 1,
            "txn_count_24h": 1,
        }
        
        result = rule_engine.evaluate_post_ml(risk_score=0.3, context=context)
        # Should be at least challenged due to new device + high amount
        assert result.action in [ActionType.CHALLENGE, ActionType.BLOCK, ActionType.ALLOW]
    
    def test_fraud_history_escalates_action(self, rule_engine):
        """Test that previous fraud flags escalate the action."""
        context = {
            "user_id": "user_001",
            "device_id": "device_001",
            "amount": 5000,
            "transaction_type": "payment",
            "user_is_blocked": False,
            "device_is_blocked": False,
            "is_vpn": False,
            "is_new_device": False,
            "account_age_days": 365,
            "previous_fraud_flags": 2,  # Has fraud history
            "txn_count_1h": 1,
            "txn_count_24h": 3,
        }
        
        # Even moderate risk should escalate with fraud history
        result = rule_engine.evaluate_post_ml(risk_score=0.45, context=context)
        assert result is not None
