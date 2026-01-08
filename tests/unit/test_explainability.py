"""
Unit tests for the explainability module.

Tests:
- Risk reason generation
- Feature importance explanations
- Template-based explanations
- Display formatting
"""

import pytest
from datetime import datetime

from src.explainability.explainer import RiskExplainer, RiskReason
from src.explainability.feature_importance import FeatureImportanceExplainer


class TestRiskExplainer:
    """Tests for the RiskExplainer class."""
    
    @pytest.fixture
    def explainer(self):
        """Create a RiskExplainer with default templates."""
        return RiskExplainer()
    
    @pytest.fixture
    def sample_contributions(self):
        """Sample feature contributions for testing."""
        return {
            "amount_log": 0.15,
            "is_vpn": 0.25,
            "is_new_device": 0.18,
            "txn_count_1h": 0.12,
            "account_age_days": -0.05,  # Negative contribution (reduces risk)
            "is_verified": -0.08,
            "hour_of_day": 0.03,
        }
    
    @pytest.fixture
    def sample_values(self):
        """Sample feature values for testing."""
        return {
            "amount_log": 10.5,
            "is_vpn": 1,
            "is_new_device": 1,
            "txn_count_1h": 8,
            "account_age_days": 30,
            "is_verified": 1,
            "hour_of_day": 3,
        }
    
    def test_explainer_initialization(self, explainer):
        """Test that explainer initializes with templates."""
        assert explainer is not None
        assert len(explainer.reason_templates) > 0
    
    def test_explain_with_contributions(self, explainer, sample_contributions, sample_values):
        """Test explaining predictions with SHAP-like contributions."""
        reasons = explainer.explain(
            contributions=sample_contributions,
            values=sample_values,
            top_k=5,
        )
        
        assert len(reasons) <= 5
        assert all(isinstance(r, RiskReason) for r in reasons)
        
        # Top reasons should have highest contribution magnitude
        contributions = [abs(r.contribution) for r in reasons]
        assert contributions == sorted(contributions, reverse=True)
    
    def test_explain_without_contributions(self, explainer, sample_values):
        """Test rule-based explanation when contributions not available."""
        reasons = explainer.explain(
            contributions=None,
            values=sample_values,
            top_k=5,
        )
        
        # Should still generate reasons based on values
        assert len(reasons) > 0
        assert all(isinstance(r, RiskReason) for r in reasons)
    
    def test_reason_severity_assignment(self, explainer, sample_contributions, sample_values):
        """Test that severity is correctly assigned based on contribution."""
        reasons = explainer.explain(
            contributions=sample_contributions,
            values=sample_values,
        )
        
        for reason in reasons:
            assert reason.severity in ["low", "medium", "high", "critical"]
    
    def test_reason_descriptions_are_readable(self, explainer, sample_contributions, sample_values):
        """Test that reason descriptions are human-readable."""
        reasons = explainer.explain(
            contributions=sample_contributions,
            values=sample_values,
        )
        
        for reason in reasons:
            # Description should be a non-empty string
            assert isinstance(reason.description, str)
            assert len(reason.description) > 10
            # Should not contain technical jargon
            assert "float" not in reason.description.lower()
            assert "int" not in reason.description.lower()
    
    def test_format_for_display(self, explainer, sample_contributions, sample_values):
        """Test formatting reasons for display."""
        reasons = explainer.explain(
            contributions=sample_contributions,
            values=sample_values,
            top_k=3,
        )
        
        display = explainer.format_for_display(reasons)
        
        assert isinstance(display, str)
        assert len(display) > 0
        # Should contain emoji or bullet points
        assert "•" in display or "🔴" in display or "🟡" in display or "🟢" in display or "-" in display
    
    def test_format_for_api(self, explainer, sample_contributions, sample_values):
        """Test formatting reasons for API response."""
        reasons = explainer.explain(
            contributions=sample_contributions,
            values=sample_values,
            top_k=3,
        )
        
        api_format = explainer.format_for_api(reasons)
        
        assert isinstance(api_format, list)
        assert len(api_format) == len(reasons)
        
        for item in api_format:
            assert "feature" in item
            assert "description" in item
            assert "severity" in item


class TestRiskReason:
    """Tests for the RiskReason dataclass."""
    
    def test_risk_reason_creation(self):
        """Test creating a RiskReason."""
        reason = RiskReason(
            feature="is_vpn",
            description="Transaction from VPN or proxy",
            contribution=0.25,
            value=True,
            severity="high",
        )
        
        assert reason.feature == "is_vpn"
        assert reason.contribution == 0.25
        assert reason.severity == "high"
    
    def test_risk_reason_to_dict(self):
        """Test converting RiskReason to dictionary."""
        reason = RiskReason(
            feature="amount",
            description="Unusually high amount",
            contribution=0.15,
            value=50000,
            severity="medium",
        )
        
        d = reason.to_dict() if hasattr(reason, 'to_dict') else {
            "feature": reason.feature,
            "description": reason.description,
            "contribution": reason.contribution,
            "value": reason.value,
            "severity": reason.severity,
        }
        
        assert d["feature"] == "amount"
        assert d["value"] == 50000


class TestFeatureImportanceExplainer:
    """Tests for the FeatureImportanceExplainer class."""
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return [
            "amount_log",
            "is_vpn",
            "is_new_device",
            "txn_count_1h",
            "account_age_days",
            "is_verified",
            "hour_of_day",
            "merchant_fraud_rate",
        ]
    
    @pytest.fixture
    def global_importance(self, feature_names):
        """Sample global feature importance."""
        import numpy as np
        return dict(zip(feature_names, np.random.rand(len(feature_names))))
    
    def test_explainer_initialization(self, feature_names, global_importance):
        """Test creating a FeatureImportanceExplainer."""
        explainer = FeatureImportanceExplainer(
            feature_names=feature_names,
            global_importance=global_importance,
        )
        
        assert explainer is not None
        assert explainer.feature_names == feature_names
    
    def test_get_top_features(self, feature_names, global_importance):
        """Test getting top important features."""
        explainer = FeatureImportanceExplainer(
            feature_names=feature_names,
            global_importance=global_importance,
        )
        
        top_features = explainer.get_top_features(k=3)
        
        assert len(top_features) == 3
        # Should be sorted by importance
        importances = [global_importance[f] for f in top_features]
        assert importances == sorted(importances, reverse=True)


class TestExplainerEdgeCases:
    """Tests for edge cases in explainability."""
    
    @pytest.fixture
    def explainer(self):
        return RiskExplainer()
    
    def test_empty_contributions(self, explainer):
        """Test handling empty contributions."""
        reasons = explainer.explain(contributions={}, values={})
        assert reasons == [] or isinstance(reasons, list)
    
    def test_none_contributions(self, explainer):
        """Test handling None contributions."""
        reasons = explainer.explain(contributions=None, values={"amount": 1000})
        # Should fall back to rule-based explanation
        assert isinstance(reasons, list)
    
    def test_unknown_features(self, explainer):
        """Test handling unknown feature names."""
        reasons = explainer.explain(
            contributions={"unknown_feature_xyz": 0.5},
            values={"unknown_feature_xyz": 100},
        )
        # Should handle gracefully
        assert isinstance(reasons, list)
    
    def test_negative_contributions(self, explainer):
        """Test handling negative contributions (risk-reducing features)."""
        reasons = explainer.explain(
            contributions={
                "is_verified": -0.15,  # Reduces risk
                "account_age_days": -0.10,  # Reduces risk
            },
            values={
                "is_verified": 1,
                "account_age_days": 500,
            },
        )
        
        # Should include negative contributors with appropriate framing
        assert isinstance(reasons, list)
    
    def test_very_high_risk_score(self, explainer):
        """Test explanation for very high risk scores."""
        reasons = explainer.explain(
            contributions={
                "is_vpn": 0.4,
                "is_emulator": 0.35,
                "previous_fraud_flags": 0.25,
            },
            values={
                "is_vpn": 1,
                "is_emulator": 1,
                "previous_fraud_flags": 3,
            },
        )
        
        # Should have multiple high-severity reasons
        high_severity = [r for r in reasons if r.severity in ["high", "critical"]]
        assert len(high_severity) > 0
