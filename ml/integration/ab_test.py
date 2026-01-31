"""
A/B Testing Framework

Enables gradual rollout and experimentation with ML models.
"""

from typing import Dict, Any, Optional, List
import time
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)


class Variant:
    """A/B test variant."""
    
    def __init__(
        self,
        name: str,
        traffic_percentage: float,
        use_ml: bool,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.traffic_percentage = traffic_percentage
        self.use_ml = use_ml
        self.config = config or {}
        
        # Metrics
        self.impressions = 0
        self.successes = 0
        self.failures = 0
        self.total_latency_ms = 0.0


class ABTestFramework:
    """
    A/B testing framework for ML models.
    
    Enables gradual rollout:
    - Control group: No ML (deterministic only)
    - Treatment group: ML-optimized
    
    Tracks metrics for both groups to validate ML improvements.
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Variant]] = {}
        
        # User assignments: {user_id: variant_name}
        self.user_assignments: Dict[str, str] = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        control_config: Optional[Dict[str, Any]] = None,
        treatment_config: Optional[Dict[str, Any]] = None,
        traffic_split: float = 0.5
    ) -> None:
        """
        Create A/B experiment.
        
        Args:
            experiment_name: Experiment name
            control_config: Control group config (no ML)
            treatment_config: Treatment group config (ML-optimized)
            traffic_split: Fraction to treatment (0.0-1.0)
        """
        control = Variant(
            name='control',
            traffic_percentage=1.0 - traffic_split,
            use_ml=False,
            config=control_config
        )
        
        treatment = Variant(
            name='treatment',
            traffic_percentage=traffic_split,
            use_ml=True,
            config=treatment_config
        )
        
        self.experiments[experiment_name] = {
            'control': control,
            'treatment': treatment,
        }
        
        logger.info(
            f"Created experiment '{experiment_name}' "
            f"(control: {control.traffic_percentage:.1%}, "
            f"treatment: {treatment.traffic_percentage:.1%})"
        )
    
    def get_variant(
        self,
        experiment_name: str,
        user_id: str
    ) -> Optional[Variant]:
        """
        Get variant for user.
        
        Uses consistent hashing to ensure same user gets same variant.
        
        Args:
            experiment_name: Experiment name
            user_id: User/request ID
            
        Returns:
            Assigned variant
        """
        if experiment_name not in self.experiments:
            logger.warning(f"Experiment '{experiment_name}' not found")
            return None
        
        # Check if user already assigned
        assignment_key = f"{experiment_name}:{user_id}"
        if assignment_key in self.user_assignments:
            variant_name = self.user_assignments[assignment_key]
            return self.experiments[experiment_name][variant_name]
        
        # Assign variant based on hash
        variants = self.experiments[experiment_name]
        
        # Use hash to deterministically assign
        hash_val = hash(assignment_key) % 100
        
        treatment = variants['treatment']
        threshold = treatment.traffic_percentage * 100
        
        if hash_val < threshold:
            assigned_variant = treatment
        else:
            assigned_variant = variants['control']
        
        # Cache assignment
        self.user_assignments[assignment_key] = assigned_variant.name
        
        return assigned_variant
    
    def record_impression(
        self,
        experiment_name: str,
        user_id: str
    ) -> None:
        """Record experiment impression."""
        variant = self.get_variant(experiment_name, user_id)
        if variant:
            variant.impressions += 1
    
    def record_result(
        self,
        experiment_name: str,
        user_id: str,
        success: bool,
        latency_ms: float
    ) -> None:
        """
        Record experiment result.
        
        Args:
            experiment_name: Experiment name
            user_id: User/request ID
            success: Whether operation succeeded
            latency_ms: Operation latency
        """
        variant = self.get_variant(experiment_name, user_id)
        if not variant:
            return
        
        if success:
            variant.successes += 1
        else:
            variant.failures += 1
        
        variant.total_latency_ms += latency_ms
    
    def get_results(
        self,
        experiment_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get experiment results.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Results for each variant
        """
        if experiment_name not in self.experiments:
            return {}
        
        variants = self.experiments[experiment_name]
        results = {}
        
        for variant_name, variant in variants.items():
            total_events = variant.successes + variant.failures
            success_rate = (
                variant.successes / total_events
                if total_events > 0 else 0.0
            )
            avg_latency = (
                variant.total_latency_ms / total_events
                if total_events > 0 else 0.0
            )
            
            results[variant_name] = {
                'name': variant_name,
                'use_ml': variant.use_ml,
                'traffic_percentage': variant.traffic_percentage,
                'impressions': variant.impressions,
                'total_events': total_events,
                'successes': variant.successes,
                'failures': variant.failures,
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
            }
        
        return results
    
    def compute_statistical_significance(
        self,
        experiment_name: str,
        metric: str = 'success_rate'
    ) -> Dict[str, Any]:
        """
        Compute statistical significance of results.
        
        Args:
            experiment_name: Experiment name
            metric: Metric to compare ('success_rate' or 'avg_latency_ms')
            
        Returns:
            Statistical test results
        """
        results = self.get_results(experiment_name)
        
        if 'control' not in results or 'treatment' not in results:
            return {}
        
        control = results['control']
        treatment = results['treatment']
        
        # Extract metric values
        control_value = control[metric]
        treatment_value = treatment[metric]
        
        # Compute lift
        if control_value > 0:
            lift = (treatment_value - control_value) / control_value
        else:
            lift = 0.0
        
        # Simple significance test (would use proper stats in production)
        control_events = control['total_events']
        treatment_events = treatment['total_events']
        
        # Require minimum sample size
        min_sample_size = 100
        is_significant = (
            control_events >= min_sample_size and
            treatment_events >= min_sample_size and
            abs(lift) > 0.05  # 5% threshold
        )
        
        return {
            'experiment': experiment_name,
            'metric': metric,
            'control_value': control_value,
            'treatment_value': treatment_value,
            'lift': lift,
            'lift_percentage': lift * 100,
            'is_significant': is_significant,
            'control_sample_size': control_events,
            'treatment_sample_size': treatment_events,
        }
    
    def get_recommendation(
        self,
        experiment_name: str
    ) -> str:
        """
        Get recommendation for experiment.
        
        Args:
            experiment_name: Experiment name
            
        Returns:
            Recommendation ('rollout', 'rollback', 'continue')
        """
        # Check both success rate and latency
        success_sig = self.compute_statistical_significance(
            experiment_name,
            'success_rate'
        )
        latency_sig = self.compute_statistical_significance(
            experiment_name,
            'avg_latency_ms'
        )
        
        if not success_sig or not latency_sig:
            return 'continue'
        
        # Rollout if both metrics improved
        if (
            success_sig['is_significant'] and
            success_sig['lift'] > 0 and
            latency_sig['is_significant'] and
            latency_sig['lift'] < 0  # Lower latency is better
        ):
            return 'rollout'
        
        # Rollback if metrics degraded
        if (
            success_sig['is_significant'] and
            success_sig['lift'] < -0.05  # >5% degradation
        ):
            return 'rollback'
        
        # Continue testing
        return 'continue'
    
    def get_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all experiments."""
        summary = []
        
        for exp_name in self.experiments:
            results = self.get_results(exp_name)
            recommendation = self.get_recommendation(exp_name)
            
            summary.append({
                'experiment': exp_name,
                'results': results,
                'recommendation': recommendation,
            })
        
        return summary
