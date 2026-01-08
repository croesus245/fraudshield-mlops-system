"""
Unit tests for monitoring module.
"""

import pytest
import pandas as pd
import numpy as np

from src.monitoring.drift import (
    compute_psi,
    compute_ks_statistic,
    DriftDetector,
    DriftSeverity,
)
from src.monitoring.alerts import (
    Alert,
    AlertManager,
    AlertSeverity,
)


class TestPSI:
    """Tests for PSI computation."""
    
    def test_identical_distributions(self):
        """Test PSI is 0 for identical distributions."""
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        
        assert psi < 0.01
    
    def test_shifted_distribution(self):
        """Test PSI detects shifted distribution."""
        reference = np.random.normal(0, 1, 1000)
        shifted = np.random.normal(2, 1, 1000)  # Mean shifted by 2
        
        psi = compute_psi(reference, shifted)
        
        assert psi > 0.1  # Should detect shift
    
    def test_psi_non_negative(self):
        """Test PSI is always non-negative."""
        for _ in range(10):
            ref = np.random.random(100)
            cur = np.random.random(100)
            psi = compute_psi(ref, cur)
            
            assert psi >= 0


class TestKS:
    """Tests for KS statistic."""
    
    def test_identical_distributions(self):
        """Test KS for identical distributions."""
        data = np.random.normal(0, 1, 1000)
        stat, pvalue = compute_ks_statistic(data, data)
        
        assert pvalue > 0.05  # Can't reject same distribution
    
    def test_different_distributions(self):
        """Test KS detects different distributions."""
        ref = np.random.normal(0, 1, 1000)
        diff = np.random.normal(5, 1, 1000)  # Very different
        
        stat, pvalue = compute_ks_statistic(ref, diff)
        
        assert pvalue < 0.05  # Should reject same distribution


class TestDriftDetector:
    """Tests for DriftDetector."""
    
    def test_no_drift_detected(self):
        """Test no drift with similar distributions."""
        ref_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000),
        })
        
        cur_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
        })
        
        detector = DriftDetector()
        detector.set_reference(ref_data)
        results = detector.detect(cur_data)
        
        # Most results should be no drift
        critical = [r for r in results if r.severity == DriftSeverity.CRITICAL]
        assert len(critical) == 0
    
    def test_drift_detected(self):
        """Test drift detection with shifted data."""
        ref_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
        })
        
        cur_data = pd.DataFrame({
            "feature1": np.random.normal(5, 1, 100),  # Shifted mean
        })
        
        detector = DriftDetector(psi_warning=0.1, psi_critical=0.2)
        detector.set_reference(ref_data)
        results = detector.detect(cur_data)
        
        # Should detect drift
        assert any(r.is_drifted for r in results)
    
    def test_reference_not_set_raises(self):
        """Test error when reference not set."""
        detector = DriftDetector()
        
        with pytest.raises(RuntimeError):
            detector.detect(pd.DataFrame({"a": [1, 2, 3]}))


class TestAlerts:
    """Tests for alert management."""
    
    def test_alert_creation(self):
        """Test Alert dataclass."""
        from datetime import datetime
        
        alert = Alert(
            name="test_alert",
            severity=AlertSeverity.WARNING,
            message="Test message",
            timestamp=datetime.now(),
        )
        
        assert alert.name == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
    
    def test_alert_to_dict(self):
        """Test Alert serialization."""
        from datetime import datetime
        
        alert = Alert(
            name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            timestamp=datetime.now(),
        )
        
        d = alert.to_dict()
        
        assert "name" in d
        assert "severity" in d
        assert d["severity"] == "info"
    
    def test_alert_manager_filtering(self):
        """Test AlertManager severity filtering."""
        from datetime import datetime
        
        manager = AlertManager(min_severity=AlertSeverity.WARNING)
        
        info_alert = Alert(
            name="info",
            severity=AlertSeverity.INFO,
            message="Info",
            timestamp=datetime.now(),
        )
        
        warning_alert = Alert(
            name="warning",
            severity=AlertSeverity.WARNING,
            message="Warning",
            timestamp=datetime.now(),
        )
        
        # Info should be filtered
        assert not manager.send(info_alert)
        
        # Warning should go through
        assert manager.send(warning_alert)
    
    def test_create_drift_alert(self):
        """Test drift alert creation."""
        manager = AlertManager()
        
        alert = manager.create_drift_alert(
            feature="amount",
            drift_value=0.25,
            threshold=0.2,
            is_critical=True,
        )
        
        assert alert.severity == AlertSeverity.CRITICAL
        assert "amount" in alert.message
