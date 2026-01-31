"""
Fallback Manager

Manages graceful degradation when ML models fail or underperform.
"""

from typing import Dict, Any
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern for ML model failures.
    
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        success_threshold: int = 2
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
        # Recent failures for tracking
        self.recent_failures = deque(maxlen=100)
    
    def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result of func or raises exception
        """
        if self.state == 'OPEN':
            # Check if timeout elapsed
            if time.time() - self.last_failure_time >= self.timeout_seconds:
                logger.info("Circuit breaker entering HALF_OPEN state")
                self.state = 'HALF_OPEN'
                self.success_count = 0
            else:
                raise Exception("Circuit breaker OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure()
            raise e
    
    def _record_success(self) -> None:
        """Record successful call."""
        if self.state == 'HALF_OPEN':
            self.success_count += 1
            
            if self.success_count >= self.success_threshold:
                logger.info("Circuit breaker entering CLOSED state")
                self.state = 'CLOSED'
                self.failure_count = 0
        
        elif self.state == 'CLOSED':
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def _record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.recent_failures.append(time.time())
        
        if self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker entering OPEN state "
                f"({self.failure_count} failures)"
            )
            self.state = 'OPEN'
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        self.state = 'CLOSED'
        self.failure_count = 0
        self.success_count = 0
        self.recent_failures.clear()


class FallbackManager:
    """
    Manages fallback behavior for ML models.
    
    Tracks model health and automatically disables unhealthy models.
    """
    
    def __init__(self):
        # Circuit breakers per model
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Performance tracking
        self.model_stats: Dict[str, Dict[str, Any]] = {}
    
    def get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for model."""
        if model_name not in self.circuit_breakers:
            self.circuit_breakers[model_name] = CircuitBreaker(
                failure_threshold=5,
                timeout_seconds=60,
                success_threshold=2
            )
        
        return self.circuit_breakers[model_name]
    
    def is_healthy(self, model_name: str) -> bool:
        """
        Check if model is healthy (circuit closed).
        
        Args:
            model_name: Model name
            
        Returns:
            True if healthy, False otherwise
        """
        cb = self.get_circuit_breaker(model_name)
        return cb.state == 'CLOSED'
    
    def record_success(self, model_name: str) -> None:
        """Record successful model prediction."""
        cb = self.get_circuit_breaker(model_name)
        cb._record_success()
        
        # Update stats
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'successes': 0,
                'failures': 0,
                'last_success': None,
            }
        
        self.model_stats[model_name]['successes'] += 1
        self.model_stats[model_name]['last_success'] = time.time()
    
    def record_failure(self, model_name: str) -> None:
        """Record failed model prediction."""
        cb = self.get_circuit_breaker(model_name)
        cb._record_failure()
        
        # Update stats
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {
                'successes': 0,
                'failures': 0,
                'last_failure': None,
            }
        
        self.model_stats[model_name]['failures'] += 1
        self.model_stats[model_name]['last_failure'] = time.time()
    
    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """
        Get health metrics for model.
        
        Args:
            model_name: Model name
            
        Returns:
            Health metrics
        """
        cb = self.get_circuit_breaker(model_name)
        stats = self.model_stats.get(model_name, {
            'successes': 0,
            'failures': 0,
        })
        
        total = stats['successes'] + stats['failures']
        success_rate = stats['successes'] / total if total > 0 else 1.0
        
        return {
            'model_name': model_name,
            'state': cb.state,
            'success_rate': success_rate,
            'total_calls': total,
            'successes': stats['successes'],
            'failures': stats['failures'],
            'recent_failures': len(cb.recent_failures),
        }
    
    def reset_circuit_breaker(self, model_name: str) -> None:
        """Manually reset circuit breaker."""
        if model_name in self.circuit_breakers:
            self.circuit_breakers[model_name].reset()
            logger.info(f"Reset circuit breaker for {model_name}")
    
    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health for all models."""
        return {
            model_name: self.get_model_health(model_name)
            for model_name in self.circuit_breakers
        }


class FallbackStrategy:
    """
    Defines fallback strategies when ML unavailable.
    """
    
    @staticmethod
    def compression_fallback(block_size: int) -> str:
        """
        Fallback compression algorithm selection.
        
        Simple heuristic:
        - Small blocks (<1KB): No compression
        - Medium blocks (1KB-100KB): LZ4 (fast)
        - Large blocks (>100KB): ZSTD (high ratio)
        
        Args:
            block_size: Block size in bytes
            
        Returns:
            Algorithm name
        """
        if block_size < 1024:
            return 'none'
        elif block_size < 102400:
            return 'lz4'
        else:
            return 'zstd'
    
    @staticmethod
    def peer_fallback(available_peers: list) -> str:
        """
        Fallback peer selection.
        
        Simply use random selection.
        
        Args:
            available_peers: Available peer IDs
            
        Returns:
            Selected peer
        """
        import random
        return random.choice(available_peers) if available_peers else ""
    
    @staticmethod
    def prefetch_fallback() -> list:
        """
        Fallback prefetch strategy.
        
        Don't prefetch anything (conservative).
        
        Returns:
            Empty list
        """
        return []
    
    @staticmethod
    def network_params_fallback() -> Dict[str, int]:
        """
        Fallback network parameters.
        
        Use safe defaults.
        
        Returns:
            {'fanout': 6, 'ttl': 10}
        """
        return {'fanout': 6, 'ttl': 10}
