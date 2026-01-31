"""
ML-DAG Integration

Integrates ML models into the deterministic DAG backend.
ML provides optimization hints, deterministic core makes final decisions.
"""

from typing import Dict, Any, Optional, List
import logging
import time

logger = logging.getLogger(__name__)

from pakit.ml.registry import ModelRegistry
from pakit.ml.integration.fallback import FallbackManager


class MLDAGIntegration:
    """
    Integrates ML optimization layer with DAG backend.
    
    Architecture:
        User Request → ML Predictions (hints) → DAG Backend (final decision) → Result
        
    ML NEVER compromises cryptographic determinism or security.
    """
    
    def __init__(self, enable_ml: bool = True):
        self.enable_ml = enable_ml
        self.registry = ModelRegistry()
        self.fallback_manager = FallbackManager()
        
        # Performance tracking
        self.ml_suggestions_used = 0
        self.ml_suggestions_ignored = 0
        self.fallback_activations = 0
    
    def get_compression_hint(
        self,
        block_hash: str,
        block_size: int,
        block_depth: int,
        content_entropy: float
    ) -> Optional[str]:
        """
        Get ML hint for compression algorithm.
        
        DAG backend makes final decision based on hint + validation.
        
        Args:
            block_hash: Block hash
            block_size: Block size in bytes
            block_depth: Block depth in DAG
            content_entropy: Content entropy (0-8)
            
        Returns:
            Suggested algorithm ('zstd', 'lz4', 'snappy', 'none') or None if ML unavailable
        """
        if not self.enable_ml:
            return None
        
        # Get compression predictor
        model = self.registry.get('compression_predictor')
        if not model or not model.enabled:
            self.fallback_activations += 1
            return None
        
        # Check if model is healthy
        if not self.fallback_manager.is_healthy('compression_predictor'):
            logger.warning("Compression predictor unhealthy, using fallback")
            self.fallback_activations += 1
            return None
        
        try:
            # Extract features
            import math
            features = {
                'block_size_log': math.log10(max(block_size, 1)),
                'block_depth_log': math.log10(max(block_depth, 1)),
                'parent_count': 1,  # Would come from DAG
                'hash_entropy': content_entropy,
                'is_leaf': block_depth == 0,
                'is_merge': False,
                'content_type': [1, 0, 0, 0],  # Binary (placeholder)
                'previous_compression': [0, 0],
            }
            
            # Get prediction
            algorithm = model.predict(features)
            
            # Record success
            self.fallback_manager.record_success('compression_predictor')
            self.ml_suggestions_used += 1
            
            logger.debug(f"ML suggests {algorithm} for block {block_hash[:8]}")
            
            return algorithm
            
        except Exception as e:
            logger.error(f"Compression prediction failed: {e}")
            self.fallback_manager.record_failure('compression_predictor')
            self.fallback_activations += 1
            return None
    
    def get_deduplication_candidates(
        self,
        block_hash: str,
        block_size: int
    ) -> List[str]:
        """
        Get ML hint for potential duplicate blocks.
        
        DAG backend verifies cryptographic hash before deduplication.
        
        Args:
            block_hash: Block to check
            block_size: Block size
            
        Returns:
            List of candidate block hashes (may be duplicates)
        """
        if not self.enable_ml:
            return []
        
        model = self.registry.get('dedup_optimizer')
        if not model or not model.enabled:
            return []
        
        if not self.fallback_manager.is_healthy('dedup_optimizer'):
            return []
        
        try:
            features = {
                'block_hash': block_hash,
                'block_size': block_size,
            }
            
            # Get similar blocks
            result = model.predict(features)
            
            self.fallback_manager.record_success('dedup_optimizer')
            
            if result:
                self.ml_suggestions_used += 1
                logger.debug(f"ML found dedup candidate: {result}")
                return [result]
            
            return []
            
        except Exception as e:
            logger.error(f"Deduplication prediction failed: {e}")
            self.fallback_manager.record_failure('dedup_optimizer')
            return []
    
    def get_prefetch_blocks(
        self,
        recent_accesses: List[str]
    ) -> List[str]:
        """
        Get ML hint for blocks to prefetch.
        
        DAG backend decides whether to actually prefetch based on cache space.
        
        Args:
            recent_accesses: Recently accessed block hashes
            
        Returns:
            List of block hashes to prefetch
        """
        if not self.enable_ml:
            return []
        
        model = self.registry.get('prefetch_engine')
        if not model or not model.enabled:
            return []
        
        if not self.fallback_manager.is_healthy('prefetch_engine'):
            return []
        
        try:
            features = {
                'recent_accesses': recent_accesses,
            }
            
            predictions = model.predict(features)
            
            self.fallback_manager.record_success('prefetch_engine')
            
            if predictions:
                self.ml_suggestions_used += 1
                logger.debug(f"ML suggests prefetching {len(predictions)} blocks")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Prefetch prediction failed: {e}")
            self.fallback_manager.record_failure('prefetch_engine')
            return []
    
    def get_best_peer(
        self,
        available_peers: List[str],
        block_size: int,
        context: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Get ML hint for best peer to contact.
        
        DAG backend tries ML suggestion first, falls back if fails.
        
        Args:
            available_peers: Available peer IDs
            block_size: Block size to retrieve
            context: Optional context features
            
        Returns:
            Suggested peer ID or None
        """
        if not self.enable_ml or not available_peers:
            return None
        
        model = self.registry.get('peer_selector')
        if not model or not model.enabled:
            return None
        
        if not self.fallback_manager.is_healthy('peer_selector'):
            return None
        
        try:
            import math
            
            features = {
                'available_peers': available_peers,
                'context': context or {
                    'block_size_log': math.log10(max(block_size, 1)),
                    'block_depth': 0.0,
                    'time_of_day': time.time() % 86400 / 86400,
                    'network_load': 0.5,
                },
            }
            
            peer = model.predict(features)
            
            self.fallback_manager.record_success('peer_selector')
            
            if peer:
                self.ml_suggestions_used += 1
                logger.debug(f"ML suggests peer {peer}")
            
            return peer
            
        except Exception as e:
            logger.error(f"Peer selection failed: {e}")
            self.fallback_manager.record_failure('peer_selector')
            return None
    
    def get_network_params(
        self,
        avg_latency_ms: float,
        message_loss_rate: float
    ) -> Dict[str, int]:
        """
        Get ML hint for network protocol parameters.
        
        DAG backend validates params are within safe ranges.
        
        Args:
            avg_latency_ms: Average network latency
            message_loss_rate: Message loss rate
            
        Returns:
            {'fanout': int, 'ttl': int}
        """
        if not self.enable_ml:
            return {'fanout': 6, 'ttl': 10}  # Defaults
        
        model = self.registry.get('network_optimizer')
        if not model or not model.enabled:
            return {'fanout': 6, 'ttl': 10}
        
        if not self.fallback_manager.is_healthy('network_optimizer'):
            return {'fanout': 6, 'ttl': 10}
        
        try:
            features = {
                'avg_latency_ms': avg_latency_ms,
                'message_loss_rate': message_loss_rate,
            }
            
            params = model.predict(features)
            
            self.fallback_manager.record_success('network_optimizer')
            self.ml_suggestions_used += 1
            
            # Validate params are safe
            fanout = max(4, min(8, params['fanout']))
            ttl = max(8, min(12, params['ttl']))
            
            return {'fanout': fanout, 'ttl': ttl}
            
        except Exception as e:
            logger.error(f"Network optimization failed: {e}")
            self.fallback_manager.record_failure('network_optimizer')
            return {'fanout': 6, 'ttl': 10}
    
    def record_compression_result(
        self,
        block_hash: str,
        suggested_algorithm: Optional[str],
        actual_algorithm: str,
        compression_ratio: float
    ) -> None:
        """
        Record compression result for model improvement.
        
        Args:
            block_hash: Block that was compressed
            suggested_algorithm: What ML suggested
            actual_algorithm: What was actually used
            compression_ratio: Achieved ratio
        """
        if suggested_algorithm == actual_algorithm:
            logger.debug(f"ML suggestion used: {actual_algorithm}")
        else:
            self.ml_suggestions_ignored += 1
            logger.debug(
                f"ML suggestion ignored: {suggested_algorithm} -> {actual_algorithm}"
            )
    
    def record_peer_result(
        self,
        peer_id: str,
        success: bool,
        latency_ms: float
    ) -> None:
        """
        Record peer interaction result for learning.
        
        Args:
            peer_id: Peer that was contacted
            success: Whether request succeeded
            latency_ms: Request latency
        """
        model = self.registry.get('peer_selector')
        if model:
            try:
                model.record_result(peer_id, success, latency_ms)
            except Exception as e:
                logger.error(f"Failed to record peer result: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        total_suggestions = self.ml_suggestions_used + self.ml_suggestions_ignored
        
        return {
            'ml_enabled': self.enable_ml,
            'ml_suggestions_used': self.ml_suggestions_used,
            'ml_suggestions_ignored': self.ml_suggestions_ignored,
            'ml_acceptance_rate': (
                self.ml_suggestions_used / total_suggestions
                if total_suggestions > 0 else 0.0
            ),
            'fallback_activations': self.fallback_activations,
            'models_loaded': len(self.registry.models),
            'models_enabled': len([m for m in self.registry.models.values() if m.enabled]),
        }
