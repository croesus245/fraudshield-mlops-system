"""
Cost Calculator for Fraud Detection System.

Estimates:
- Cost per 1M predictions
- Infrastructure costs (compute, storage, network)
- ROI based on fraud prevented vs model cost
- Break-even analysis
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    ON_PREMISE = "on_premise"


@dataclass
class InfrastructureConfig:
    """Infrastructure configuration for cost estimation."""
    # Compute
    instance_type: str = "c5.xlarge"  # 4 vCPU, 8GB RAM
    num_instances: int = 2
    instance_hourly_cost: float = 0.17  # USD
    
    # GPU (optional)
    gpu_instance_type: Optional[str] = None
    num_gpu_instances: int = 0
    gpu_hourly_cost: float = 0.0
    
    # Storage
    model_storage_gb: float = 1.0
    feature_store_gb: float = 10.0
    logs_storage_gb: float = 50.0
    storage_cost_per_gb: float = 0.023  # USD/GB/month
    
    # Network
    egress_gb_per_month: float = 100.0
    egress_cost_per_gb: float = 0.09  # USD/GB
    
    # Cache (Redis)
    cache_instance_type: str = "cache.t3.medium"
    cache_hourly_cost: float = 0.034
    
    # Message Queue (Kafka)
    mq_hourly_cost: float = 0.05  # Managed Kafka per broker-hour
    num_mq_brokers: int = 3


@dataclass
class ModelMetrics:
    """Model performance metrics for ROI calculation."""
    precision: float
    recall: float
    avg_fraud_amount: float = 500.0  # Average fraud transaction value
    manual_review_cost: float = 5.0  # Cost to manually review a flagged transaction
    fraud_loss_rate: float = 1.0  # % of fraud amount lost if not caught (chargebacks, etc.)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    # Monthly costs
    compute_monthly: float
    storage_monthly: float
    network_monthly: float
    cache_monthly: float
    message_queue_monthly: float
    
    # Per-request costs
    cost_per_prediction: float
    cost_per_1m_predictions: float
    
    # Totals
    total_monthly: float
    total_yearly: float


@dataclass
class ROIAnalysis:
    """Return on Investment analysis."""
    # Savings
    fraud_prevented_monthly: float  # $ value of caught fraud
    fraud_missed_monthly: float  # $ value of missed fraud (FN)
    false_alarm_cost_monthly: float  # Cost of manual review for FP
    
    # Net
    gross_savings_monthly: float
    net_savings_monthly: float  # After subtracting infra costs
    
    # ROI
    roi_percentage: float
    break_even_predictions_per_month: int


class CostCalculator:
    """
    Calculate costs and ROI for the fraud detection system.
    """
    
    def __init__(
        self,
        infra_config: Optional[InfrastructureConfig] = None,
        provider: CloudProvider = CloudProvider.AWS,
    ):
        self.infra = infra_config or InfrastructureConfig()
        self.provider = provider
        
        # Provider-specific adjustments
        self._apply_provider_adjustments()
    
    def _apply_provider_adjustments(self) -> None:
        """Apply cloud provider specific cost adjustments."""
        adjustments = {
            CloudProvider.AWS: 1.0,
            CloudProvider.GCP: 0.95,  # Generally slightly cheaper
            CloudProvider.AZURE: 1.02,  # Slightly more expensive
            CloudProvider.ON_PREMISE: 0.7,  # Cheaper but need capital
        }
        self._cost_factor = adjustments.get(self.provider, 1.0)
    
    def calculate_monthly_costs(self) -> CostBreakdown:
        """
        Calculate monthly infrastructure costs.
        
        Returns:
            CostBreakdown with all cost details
        """
        hours_per_month = 730
        
        # Compute (CPU instances)
        compute = (
            self.infra.num_instances *
            self.infra.instance_hourly_cost *
            hours_per_month
        )
        
        # GPU compute (if applicable)
        compute += (
            self.infra.num_gpu_instances *
            self.infra.gpu_hourly_cost *
            hours_per_month
        )
        
        # Storage
        total_storage = (
            self.infra.model_storage_gb +
            self.infra.feature_store_gb +
            self.infra.logs_storage_gb
        )
        storage = total_storage * self.infra.storage_cost_per_gb
        
        # Network egress
        network = self.infra.egress_gb_per_month * self.infra.egress_cost_per_gb
        
        # Cache (Redis)
        cache = self.infra.cache_hourly_cost * hours_per_month
        
        # Message Queue (Kafka)
        mq = (
            self.infra.mq_hourly_cost *
            self.infra.num_mq_brokers *
            hours_per_month
        )
        
        # Apply provider factor
        compute *= self._cost_factor
        storage *= self._cost_factor
        network *= self._cost_factor
        cache *= self._cost_factor
        mq *= self._cost_factor
        
        total_monthly = compute + storage + network + cache + mq
        
        return CostBreakdown(
            compute_monthly=round(compute, 2),
            storage_monthly=round(storage, 2),
            network_monthly=round(network, 2),
            cache_monthly=round(cache, 2),
            message_queue_monthly=round(mq, 2),
            cost_per_prediction=0,  # Set below
            cost_per_1m_predictions=0,  # Set below
            total_monthly=round(total_monthly, 2),
            total_yearly=round(total_monthly * 12, 2),
        )
    
    def calculate_per_prediction_cost(
        self,
        predictions_per_month: int,
    ) -> CostBreakdown:
        """
        Calculate cost per prediction based on volume.
        
        Args:
            predictions_per_month: Expected monthly prediction volume
        
        Returns:
            CostBreakdown with per-prediction costs
        """
        breakdown = self.calculate_monthly_costs()
        
        if predictions_per_month > 0:
            cost_per_pred = breakdown.total_monthly / predictions_per_month
            cost_per_1m = cost_per_pred * 1_000_000
        else:
            cost_per_pred = 0
            cost_per_1m = 0
        
        breakdown.cost_per_prediction = cost_per_pred
        breakdown.cost_per_1m_predictions = round(cost_per_1m, 2)
        
        return breakdown
    
    def calculate_roi(
        self,
        model_metrics: ModelMetrics,
        transactions_per_month: int,
        fraud_rate: float = 0.02,
    ) -> ROIAnalysis:
        """
        Calculate ROI of the fraud detection system.
        
        Args:
            model_metrics: Model performance metrics
            transactions_per_month: Monthly transaction volume
            fraud_rate: Expected fraud rate in transactions
        
        Returns:
            ROIAnalysis with savings and ROI metrics
        """
        # Expected counts
        total_fraud = transactions_per_month * fraud_rate
        total_legit = transactions_per_month * (1 - fraud_rate)
        
        # Model outcomes
        # True positives (caught fraud)
        tp = total_fraud * model_metrics.recall
        # False negatives (missed fraud)
        fn = total_fraud * (1 - model_metrics.recall)
        # False positives (legitimate flagged as fraud)
        # Precision = TP / (TP + FP) => FP = TP * (1 - precision) / precision
        if model_metrics.precision > 0:
            fp = tp * (1 - model_metrics.precision) / model_metrics.precision
        else:
            fp = 0
        
        # Dollar values
        fraud_prevented = tp * model_metrics.avg_fraud_amount * model_metrics.fraud_loss_rate
        fraud_missed = fn * model_metrics.avg_fraud_amount * model_metrics.fraud_loss_rate
        false_alarm_cost = fp * model_metrics.manual_review_cost
        
        # Infrastructure costs
        cost_breakdown = self.calculate_per_prediction_cost(transactions_per_month)
        
        # Net savings
        gross_savings = fraud_prevented
        net_savings = gross_savings - false_alarm_cost - cost_breakdown.total_monthly
        
        # ROI
        if cost_breakdown.total_monthly > 0:
            roi = (net_savings / cost_breakdown.total_monthly) * 100
        else:
            roi = float('inf') if net_savings > 0 else 0
        
        # Break-even analysis
        # At what volume does savings = costs?
        # savings_per_tx = fraud_rate * recall * avg_fraud - (1-precision)/precision * manual_cost
        # break_even = fixed_cost / savings_per_tx
        savings_per_tx = (
            fraud_rate * model_metrics.recall * model_metrics.avg_fraud_amount -
            (1 - model_metrics.precision) / model_metrics.precision * model_metrics.manual_review_cost
            if model_metrics.precision > 0 else 0
        )
        
        if savings_per_tx > 0:
            fixed_costs = cost_breakdown.total_monthly
            break_even = int(fixed_costs / savings_per_tx)
        else:
            break_even = -1  # Not profitable at any volume
        
        return ROIAnalysis(
            fraud_prevented_monthly=round(fraud_prevented, 2),
            fraud_missed_monthly=round(fraud_missed, 2),
            false_alarm_cost_monthly=round(false_alarm_cost, 2),
            gross_savings_monthly=round(gross_savings, 2),
            net_savings_monthly=round(net_savings, 2),
            roi_percentage=round(roi, 1),
            break_even_predictions_per_month=break_even,
        )


def print_cost_breakdown(breakdown: CostBreakdown) -> None:
    """Pretty print cost breakdown."""
    print("\n" + "=" * 50)
    print("COST BREAKDOWN")
    print("=" * 50)
    print("\nMonthly Infrastructure Costs:")
    print("-" * 40)
    print(f"  Compute:       ${breakdown.compute_monthly:>10,.2f}")
    print(f"  Storage:       ${breakdown.storage_monthly:>10,.2f}")
    print(f"  Network:       ${breakdown.network_monthly:>10,.2f}")
    print(f"  Cache (Redis): ${breakdown.cache_monthly:>10,.2f}")
    print(f"  Message Queue: ${breakdown.message_queue_monthly:>10,.2f}")
    print("-" * 40)
    print(f"  TOTAL MONTHLY: ${breakdown.total_monthly:>10,.2f}")
    print(f"  TOTAL YEARLY:  ${breakdown.total_yearly:>10,.2f}")
    print()
    if breakdown.cost_per_1m_predictions > 0:
        print("Per-Request Costs:")
        print("-" * 40)
        print(f"  Per prediction:     ${breakdown.cost_per_prediction:.6f}")
        print(f"  Per 1M predictions: ${breakdown.cost_per_1m_predictions:,.2f}")
    print("=" * 50)


def print_roi_analysis(roi: ROIAnalysis) -> None:
    """Pretty print ROI analysis."""
    print("\n" + "=" * 50)
    print("ROI ANALYSIS")
    print("=" * 50)
    print("\nMonthly Impact:")
    print("-" * 40)
    print(f"  Fraud Prevented:   ${roi.fraud_prevented_monthly:>12,.2f}")
    print(f"  Fraud Missed:      ${roi.fraud_missed_monthly:>12,.2f}")
    print(f"  False Alarm Costs: ${roi.false_alarm_cost_monthly:>12,.2f}")
    print()
    print("Net Result:")
    print("-" * 40)
    print(f"  Gross Savings:     ${roi.gross_savings_monthly:>12,.2f}")
    print(f"  Net Savings:       ${roi.net_savings_monthly:>12,.2f}")
    print()
    print("ROI Metrics:")
    print("-" * 40)
    print(f"  ROI:               {roi.roi_percentage:>12.1f}%")
    if roi.break_even_predictions_per_month > 0:
        print(f"  Break-even volume: {roi.break_even_predictions_per_month:>12,} tx/month")
    else:
        print(f"  Break-even volume: Not profitable")
    print("=" * 50)


if __name__ == "__main__":
    # Example usage
    
    # Configure infrastructure
    infra = InfrastructureConfig(
        num_instances=2,
        instance_hourly_cost=0.17,
        feature_store_gb=10,
        num_mq_brokers=3,
    )
    
    # Model metrics (from evaluation)
    metrics = ModelMetrics(
        precision=0.667,  # 66.7% precision
        recall=0.828,  # 82.8% recall
        avg_fraud_amount=500,
        manual_review_cost=5,
    )
    
    # Calculate costs
    calculator = CostCalculator(infra, CloudProvider.AWS)
    
    # Monthly costs
    breakdown = calculator.calculate_per_prediction_cost(
        predictions_per_month=10_000_000  # 10M predictions/month
    )
    print_cost_breakdown(breakdown)
    
    # ROI analysis
    roi = calculator.calculate_roi(
        model_metrics=metrics,
        transactions_per_month=10_000_000,
        fraud_rate=0.02,  # 2% fraud rate
    )
    print_roi_analysis(roi)
