"""
Model Monitor

Monitors model performance in production.
"""

from typing import Dict, Any, List, Optional
import time
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects model prediction metrics.
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Recent predictions: [(success, latency_ms, timestamp)]
        self.recent_predictions = deque(maxlen=window_size)
        
        # Counters
        self.total_predictions = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_latency_ms = 0.0
    
    def record(
        self,
        success: bool,
        latency_ms: float
    ) -> None:
        """Record prediction result."""
        timestamp = time.time()
        
        self.recent_predictions.append((success, latency_ms, timestamp))
        
        self.total_predictions += 1
        if success:
            self.total_successes += 1
        else:
            self.total_failures += 1
        
        self.total_latency_ms += latency_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'success_rate': 0.0,
                'avg_latency_ms': 0.0,
            }
        
        # Overall metrics
        success_rate = self.total_successes / self.total_predictions
        avg_latency = self.total_latency_ms / self.total_predictions
        
        # Recent window metrics
        if self.recent_predictions:
            recent_successes = sum(1 for s, _, _ in self.recent_predictions if s)
            recent_success_rate = recent_successes / len(self.recent_predictions)
            
            recent_latency = sum(l for _, l, _ in self.recent_predictions)
            recent_avg_latency = recent_latency / len(self.recent_predictions)
        else:
            recent_success_rate = 0.0
            recent_avg_latency = 0.0
        
        return {
            'total_predictions': self.total_predictions,
            'total_successes': self.total_successes,
            'total_failures': self.total_failures,
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
            'recent_success_rate': recent_success_rate,
            'recent_avg_latency_ms': recent_avg_latency,
        }


class ModelMonitor:
    """
    Monitors model performance.
    
    Tracks:
    - Prediction accuracy/success rate
    - Latency (p50, p95, p99)
    - Error rates
    - Throughput (predictions/second)
    """
    
    def __init__(self):
        # Per-model metrics
        self.collectors: Dict[str, MetricsCollector] = defaultdict(MetricsCollector)
        
        # Alerts
        self.alerts: List[Dict[str, Any]] = []
        
        # Thresholds
        self.latency_threshold_ms = 100.0
        self.error_rate_threshold = 0.05
    
    def record_prediction(
        self,
        model_name: str,
        success: bool,
        latency_ms: float
    ) -> None:
        """
        Record prediction result.
        
        Args:
            model_name: Model name
            success: Whether prediction succeeded
            latency_ms: Prediction latency
        """
        collector = self.collectors[model_name]
        collector.record(success, latency_ms)
        
        # Check for alerts
        self._check_alerts(model_name, latency_ms, success)
    
    def get_metrics(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Get metrics for model.
        
        Args:
            model_name: Model name
            
        Returns:
            Metrics dict
        """
        collector = self.collectors.get(model_name)
        if not collector:
            return {}
        
        metrics = collector.get_metrics()
        
        # Add latency percentiles
        if collector.recent_predictions:
            latencies = [l for _, l, _ in collector.recent_predictions]
            latencies.sort()
            
            n = len(latencies)
            p50 = latencies[int(n * 0.50)] if n > 0 else 0.0
            p95 = latencies[int(n * 0.95)] if n > 0 else 0.0
            p99 = latencies[int(n * 0.99)] if n > 0 else 0.0
            
            metrics['latency_p50'] = p50
            metrics['latency_p95'] = p95
            metrics['latency_p99'] = p99
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all models."""
        return {
            model_name: self.get_metrics(model_name)
            for model_name in self.collectors
        }
    
    def _check_alerts(
        self,
        model_name: str,
        latency_ms: float,
        success: bool
    ) -> None:
        """Check if alert should be raised."""
        # High latency alert
        if latency_ms > self.latency_threshold_ms:
            self._raise_alert(
                model_name,
                'high_latency',
                f"Latency {latency_ms:.1f}ms exceeds threshold {self.latency_threshold_ms}ms"
            )
        
        # Error rate alert
        collector = self.collectors[model_name]
        metrics = collector.get_metrics()
        
        if metrics['recent_success_rate'] < (1.0 - self.error_rate_threshold):
            self._raise_alert(
                model_name,
                'high_error_rate',
                f"Error rate {1.0 - metrics['recent_success_rate']:.2%} "
                f"exceeds threshold {self.error_rate_threshold:.2%}"
            )
    
    def _raise_alert(
        self,
        model_name: str,
        alert_type: str,
        message: str
    ) -> None:
        """Raise alert."""
        alert = {
            'model': model_name,
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
        }
        
        self.alerts.append(alert)
        
        logger.warning(f"Alert: {message}")
    
    def get_alerts(
        self,
        model_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            model_name: Filter by model (or all if None)
            limit: Max alerts to return
            
        Returns:
            List of alerts (newest first)
        """
        alerts = self.alerts[-limit:]
        
        if model_name:
            alerts = [a for a in alerts if a['model'] == model_name]
        
        return list(reversed(alerts))
    
    def clear_alerts(self, model_name: Optional[str] = None) -> None:
        """Clear alerts."""
        if model_name:
            self.alerts = [a for a in self.alerts if a['model'] != model_name]
        else:
            self.alerts.clear()
    
    def export_prometheus_metrics(self) -> str:
        """
        Export metrics in Prometheus format.
        
        Returns:
            Prometheus metrics string
        """
        lines = []
        
        for model_name, collector in self.collectors.items():
            metrics = collector.get_metrics()
            
            # Predictions counter
            lines.append(
                f'pakit_ml_predictions_total{{model="{model_name}"}} '
                f'{metrics["total_predictions"]}'
            )
            
            # Success rate
            lines.append(
                f'pakit_ml_success_rate{{model="{model_name}"}} '
                f'{metrics["success_rate"]}'
            )
            
            # Average latency
            lines.append(
                f'pakit_ml_latency_ms{{model="{model_name}",quantile="avg"}} '
                f'{metrics["avg_latency_ms"]}'
            )
            
            # Percentiles
            if 'latency_p50' in metrics:
                lines.append(
                    f'pakit_ml_latency_ms{{model="{model_name}",quantile="0.5"}} '
                    f'{metrics["latency_p50"]}'
                )
                lines.append(
                    f'pakit_ml_latency_ms{{model="{model_name}",quantile="0.95"}} '
                    f'{metrics["latency_p95"]}'
                )
                lines.append(
                    f'pakit_ml_latency_ms{{model="{model_name}",quantile="0.99"}} '
                    f'{metrics["latency_p99"]}'
                )
        
        return '\n'.join(lines)
