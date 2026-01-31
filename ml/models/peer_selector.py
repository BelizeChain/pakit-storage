"""
Intelligent Peer Selector

Multi-armed bandit for optimal peer selection.
Uses Thompson Sampling with contextual features.
"""

from typing import Dict, Any, List, Optional
import time
import logging
import random

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available")

from pakit.ml.base_model import PakitMLModel, ModelConfig


class ThompsonBandit:
    """
    Thompson Sampling multi-armed bandit.
    
    Each arm (peer) has Beta distribution for success probability.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        # Prior parameters
        self.alpha_prior = alpha
        self.beta_prior = beta
        
        # Arm statistics: {arm_id: (alpha, beta)}
        self.arms: Dict[str, tuple] = {}
    
    def select_arm(self, available_arms: List[str]) -> str:
        """
        Select best arm using Thompson Sampling.
        
        Args:
            available_arms: List of arm IDs (peer IDs)
            
        Returns:
            Selected arm ID
        """
        if not NUMPY_AVAILABLE:
            return random.choice(available_arms)
        
        # Ensure all arms exist
        for arm in available_arms:
            if arm not in self.arms:
                self.arms[arm] = (self.alpha_prior, self.beta_prior)
        
        # Sample from Beta distribution for each arm
        samples = {}
        for arm in available_arms:
            alpha, beta = self.arms[arm]
            samples[arm] = np.random.beta(alpha, beta)
        
        # Select arm with highest sample
        best_arm = max(samples.items(), key=lambda x: x[1])[0]
        return best_arm
    
    def update(self, arm_id: str, reward: float) -> None:
        """
        Update arm statistics after observing reward.
        
        Args:
            arm_id: Arm that was selected
            reward: Reward received (0.0-1.0, higher is better)
        """
        if arm_id not in self.arms:
            self.arms[arm_id] = (self.alpha_prior, self.beta_prior)
        
        alpha, beta = self.arms[arm_id]
        
        # Bayesian update (assume Bernoulli reward)
        if reward > 0.5:
            # Success
            alpha += 1
        else:
            # Failure
            beta += 1
        
        self.arms[arm_id] = (alpha, beta)
    
    def get_arm_stats(self, arm_id: str) -> Dict[str, float]:
        """Get statistics for an arm."""
        if arm_id not in self.arms:
            return {'mean': 0.5, 'alpha': self.alpha_prior, 'beta': self.beta_prior}
        
        alpha, beta = self.arms[arm_id]
        mean = alpha / (alpha + beta)
        
        return {'mean': mean, 'alpha': alpha, 'beta': beta}


class ContextualBandit:
    """
    Contextual bandit with linear model.
    
    Reward prediction: r = θ^T * x (features)
    """
    
    def __init__(self, feature_dim: int = 4):
        self.feature_dim = feature_dim
        
        # Arm models: {arm_id: (weights, covariance)}
        self.arms: Dict[str, tuple] = {}
    
    def select_arm(
        self,
        available_arms: List[str],
        context: Dict[str, float]
    ) -> str:
        """
        Select arm based on context.
        
        Args:
            available_arms: Available arms
            context: Context features
            
        Returns:
            Selected arm
        """
        if not NUMPY_AVAILABLE:
            return random.choice(available_arms)
        
        # Convert context to vector
        x = self._context_to_vector(context)
        
        # Ensure all arms exist
        for arm in available_arms:
            if arm not in self.arms:
                self.arms[arm] = self._init_arm()
        
        # Compute UCB for each arm
        scores = {}
        for arm in available_arms:
            weights, cov = self.arms[arm]
            
            # Expected reward
            expected = np.dot(weights, x)
            
            # Uncertainty bonus
            uncertainty = np.sqrt(np.dot(x, np.dot(cov, x)))
            
            scores[arm] = expected + uncertainty
        
        # Select best
        best_arm = max(scores.items(), key=lambda x: x[1])[0]
        return best_arm
    
    def update(self, arm_id: str, context: Dict[str, float], reward: float) -> None:
        """Update arm model."""
        if not NUMPY_AVAILABLE:
            return
        
        x = self._context_to_vector(context)
        
        if arm_id not in self.arms:
            self.arms[arm_id] = self._init_arm()
        
        weights, cov = self.arms[arm_id]
        
        # Ridge regression update
        cov_inv = np.linalg.inv(cov)
        cov_inv += np.outer(x, x)
        cov = np.linalg.inv(cov_inv)
        
        weights = np.dot(cov, np.dot(cov_inv, weights) + reward * x)
        
        self.arms[arm_id] = (weights, cov)
    
    def _init_arm(self) -> tuple:
        """Initialize arm with prior."""
        weights = np.zeros(self.feature_dim)
        cov = np.eye(self.feature_dim)
        return (weights, cov)
    
    def _context_to_vector(self, context: Dict[str, float]) -> np.ndarray:
        """Convert context dict to vector."""
        # Extract standard features
        return np.array([
            context.get('block_size_log', 0.0),
            context.get('block_depth', 0.0),
            context.get('time_of_day', 0.0),
            context.get('network_load', 0.0),
        ])


class PeerSelector(PakitMLModel):
    """
    Intelligent peer selector using contextual bandits.
    
    Target: 10-20% latency reduction, >95% success rate
    """
    
    def __init__(self, config: ModelConfig, use_context: bool = True):
        super().__init__(config)
        
        self.use_context = use_context
        
        if use_context:
            self.bandit = ContextualBandit(feature_dim=4)
        else:
            self.bandit = ThompsonBandit()
        
        # Performance tracking
        self.peer_latencies: Dict[str, List[float]] = {}
        self.peer_successes: Dict[str, int] = {}
        self.peer_failures: Dict[str, int] = {}
    
    def predict(self, features: Dict[str, Any]) -> str:
        """
        Select best peer for request.
        
        Args:
            features: Must include 'available_peers', optionally 'context'
            
        Returns:
            Selected peer ID
        """
        start = time.time()
        
        available_peers = features.get('available_peers', [])
        if not available_peers:
            return ""
        
        if self.use_context and 'context' in features:
            selected_peer = self.bandit.select_arm(
                available_peers,
                features['context']
            )
        else:
            selected_peer = self.bandit.select_arm(available_peers)
        
        self._track_inference(time.time() - start)
        
        return selected_peer
    
    def record_result(
        self,
        peer_id: str,
        success: bool,
        latency_ms: float,
        context: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record peer performance.
        
        Args:
            peer_id: Peer that was used
            success: Whether request succeeded
            latency_ms: Request latency
            context: Optional context features
        """
        # Track statistics
        if peer_id not in self.peer_latencies:
            self.peer_latencies[peer_id] = []
            self.peer_successes[peer_id] = 0
            self.peer_failures[peer_id] = 0
        
        self.peer_latencies[peer_id].append(latency_ms)
        
        if success:
            self.peer_successes[peer_id] += 1
        else:
            self.peer_failures[peer_id] += 1
        
        # Compute reward (success + low latency)
        if success:
            # Normalize latency to 0-1 (assume 1000ms is bad)
            latency_score = max(0.0, 1.0 - latency_ms / 1000.0)
            reward = 0.5 + 0.5 * latency_score  # 0.5-1.0 range
        else:
            reward = 0.0
        
        # Update bandit
        if self.use_context and context:
            self.bandit.update(peer_id, context, reward)
        else:
            self.bandit.update(peer_id, reward)
    
    def get_peer_stats(self, peer_id: str) -> Dict[str, Any]:
        """Get performance statistics for peer."""
        if peer_id not in self.peer_successes:
            return {}
        
        total_requests = self.peer_successes[peer_id] + self.peer_failures[peer_id]
        success_rate = self.peer_successes[peer_id] / total_requests if total_requests > 0 else 0.0
        
        latencies = self.peer_latencies.get(peer_id, [])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        return {
            'peer_id': peer_id,
            'total_requests': total_requests,
            'successes': self.peer_successes[peer_id],
            'failures': self.peer_failures[peer_id],
            'success_rate': success_rate,
            'avg_latency_ms': avg_latency,
        }
    
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train from historical peer interactions.
        
        Args:
            dataset: Historical peer performance data
            validation_split: Unused (online learning)
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.config.name} from historical data")
        
        trained_samples = 0
        for sample in dataset:
            peer_id = sample.get('peer_id')
            success = sample.get('success')
            latency = sample.get('latency_ms')
            context = sample.get('context')
            
            if peer_id:
                self.record_result(peer_id, success, latency, context)
                trained_samples += 1
        
        logger.info(f"Trained on {trained_samples} samples, {len(self.bandit.arms)} peers")
        
        return {'samples': trained_samples, 'peers': len(self.bandit.arms)}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle
        
        data = {
            'bandit_arms': self.bandit.arms,
            'use_context': self.use_context,
            'peer_latencies': self.peer_latencies,
            'peer_successes': self.peer_successes,
            'peer_failures': self.peer_failures,
            'config': self.config.to_dict(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {self.config.name} to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.bandit.arms = data['bandit_arms']
        self.use_context = data['use_context']
        self.peer_latencies = data['peer_latencies']
        self.peer_successes = data['peer_successes']
        self.peer_failures = data['peer_failures']
        
        logger.info(f"Loaded {self.config.name} from {path} ({len(self.bandit.arms)} peers)")
