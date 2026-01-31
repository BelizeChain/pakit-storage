"""
Network Optimizer

Q-Learning agent for adaptive network protocol parameters.
Dynamically tunes gossip fanout and TTL based on network conditions.
"""

from typing import Dict, Any, List, Tuple, Optional
import time
import logging
import random
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available")

from pakit.ml.base_model import PakitMLModel, ModelConfig


class QLearningAgent:
    """
    Q-Learning agent for network optimization.
    
    State: (network_size, avg_latency, partition_detected, message_loss)
    Actions: (fanout, TTL) pairs
    Reward: (propagation_coverage / bandwidth_used)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table: {(state, action): Q-value}
        self.q_table: Dict[Tuple, float] = defaultdict(float)
        
        # Action space
        self.fanout_values = [4, 6, 8]
        self.ttl_values = [8, 10, 12]
        self.actions = [
            (f, t) for f in self.fanout_values for t in self.ttl_values
        ]
    
    def select_action(self, state: Tuple, explore: bool = True) -> Tuple[int, int]:
        """
        Select action using epsilon-greedy.
        
        Args:
            state: Current state tuple
            explore: Whether to explore (or always exploit)
            
        Returns:
            (fanout, ttl) tuple
        """
        if explore and random.random() < self.epsilon:
            # Explore
            return random.choice(self.actions)
        
        # Exploit - select best action
        q_values = {
            action: self.q_table[(state, action)]
            for action in self.actions
        }
        
        best_action = max(q_values.items(), key=lambda x: x[1])[0]
        return best_action
    
    def update(
        self,
        state: Tuple,
        action: Tuple[int, int],
        reward: float,
        next_state: Tuple
    ) -> None:
        """
        Q-learning update.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Get current Q-value
        current_q = self.q_table[(state, action)]
        
        # Get max Q-value for next state
        next_q_values = [
            self.q_table[(next_state, a)] for a in self.actions
        ]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[(state, action)] = new_q
    
    def decay_epsilon(self, decay_rate: float = 0.995) -> None:
        """Decay exploration rate."""
        self.epsilon = max(0.01, self.epsilon * decay_rate)


class NetworkEstimator:
    """
    Estimates current network conditions.
    """
    
    def __init__(self):
        self.network_size = 100
        self.recent_latencies: List[float] = []
        self.recent_message_losses: List[float] = []
        self.partition_detected = False
    
    def update_metrics(
        self,
        latency_ms: Optional[float] = None,
        message_loss_rate: Optional[float] = None,
        partition: bool = False
    ) -> None:
        """Update network metrics."""
        if latency_ms is not None:
            self.recent_latencies.append(latency_ms)
            if len(self.recent_latencies) > 100:
                self.recent_latencies.pop(0)
        
        if message_loss_rate is not None:
            self.recent_message_losses.append(message_loss_rate)
            if len(self.recent_message_losses) > 100:
                self.recent_message_losses.pop(0)
        
        self.partition_detected = partition
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Get discretized state.
        
        Returns:
            (network_size_bucket, latency_bucket, partition, loss_bucket)
        """
        # Network size buckets: small/medium/large
        if self.network_size < 50:
            size_bucket = 0
        elif self.network_size < 200:
            size_bucket = 1
        else:
            size_bucket = 2
        
        # Latency buckets: low/medium/high
        avg_latency = (
            sum(self.recent_latencies) / len(self.recent_latencies)
            if self.recent_latencies else 50.0
        )
        
        if avg_latency < 100:
            latency_bucket = 0
        elif avg_latency < 500:
            latency_bucket = 1
        else:
            latency_bucket = 2
        
        # Partition: binary
        partition_bucket = 1 if self.partition_detected else 0
        
        # Message loss buckets: low/medium/high
        avg_loss = (
            sum(self.recent_message_losses) / len(self.recent_message_losses)
            if self.recent_message_losses else 0.01
        )
        
        if avg_loss < 0.05:
            loss_bucket = 0
        elif avg_loss < 0.15:
            loss_bucket = 1
        else:
            loss_bucket = 2
        
        return (size_bucket, latency_bucket, partition_bucket, loss_bucket)


class NetworkOptimizer(PakitMLModel):
    """
    Network optimizer using Q-Learning.
    
    Target: 15-30% bandwidth reduction, >99.5% coverage
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.agent = QLearningAgent(
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1
        )
        
        self.estimator = NetworkEstimator()
        
        # Current protocol parameters
        self.current_fanout = 6
        self.current_ttl = 10
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.coverage_history: List[float] = []
        self.bandwidth_history: List[float] = []
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, int]:
        """
        Predict optimal network parameters.
        
        Args:
            features: Network metrics
            
        Returns:
            {'fanout': int, 'ttl': int}
        """
        start = time.time()
        
        # Update estimator
        self.estimator.update_metrics(
            latency_ms=features.get('avg_latency_ms'),
            message_loss_rate=features.get('message_loss_rate'),
            partition=features.get('partition_detected', False)
        )
        
        # Get current state
        state = self.estimator.get_state()
        
        # Select action (exploit in production)
        action = self.agent.select_action(state, explore=False)
        fanout, ttl = action
        
        self.current_fanout = fanout
        self.current_ttl = ttl
        
        self._track_inference(time.time() - start)
        
        return {'fanout': fanout, 'ttl': ttl}
    
    def record_performance(
        self,
        coverage: float,
        bandwidth_used: float,
        messages_sent: int
    ) -> None:
        """
        Record network performance for learning.
        
        Args:
            coverage: Propagation coverage (0.0-1.0)
            bandwidth_used: Bandwidth in MB
            messages_sent: Total messages sent
        """
        # Compute reward
        # Maximize coverage, minimize bandwidth
        reward = coverage / max(bandwidth_used, 0.1)
        
        # Get current state
        state = self.estimator.get_state()
        
        # Current action
        action = (self.current_fanout, self.current_ttl)
        
        # Next state (same for now - would update after time passes)
        next_state = state
        
        # Update Q-table
        self.agent.update(state, action, reward, next_state)
        
        # Track metrics
        self.episode_rewards.append(reward)
        self.coverage_history.append(coverage)
        self.bandwidth_history.append(bandwidth_used)
        
        logger.debug(
            f"Reward: {reward:.3f}, Coverage: {coverage:.3f}, "
            f"Bandwidth: {bandwidth_used:.1f}MB, Action: {action}"
        )
    
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train from historical network performance.
        
        Args:
            dataset: Historical network episodes
            validation_split: Unused (RL training)
            
        Returns:
            Training metrics
        """
        logger.info(f"Training {self.config.name} from episodes")
        
        episodes = 0
        total_reward = 0.0
        
        for episode in dataset:
            state = episode.get('state')
            action = tuple(episode.get('action'))
            reward = episode.get('reward')
            next_state = episode.get('next_state')
            
            self.agent.update(state, action, reward, next_state)
            
            total_reward += reward
            episodes += 1
        
        avg_reward = total_reward / episodes if episodes > 0 else 0.0
        
        logger.info(f"Trained on {episodes} episodes, avg_reward={avg_reward:.3f}")
        
        return {
            'episodes': episodes,
            'avg_reward': avg_reward,
            'q_table_size': len(self.agent.q_table),
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        import pickle
        
        data = {
            'q_table': dict(self.agent.q_table),
            'epsilon': self.agent.epsilon,
            'episode_rewards': self.episode_rewards,
            'coverage_history': self.coverage_history,
            'bandwidth_history': self.bandwidth_history,
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
        
        self.agent.q_table = defaultdict(float, data['q_table'])
        self.agent.epsilon = data['epsilon']
        self.episode_rewards = data['episode_rewards']
        self.coverage_history = data['coverage_history']
        self.bandwidth_history = data['bandwidth_history']
        
        logger.info(
            f"Loaded {self.config.name} from {path} "
            f"({len(self.agent.q_table)} Q-values)"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-100:]
        recent_coverage = self.coverage_history[-100:]
        recent_bandwidth = self.bandwidth_history[-100:]
        
        return {
            'avg_reward': sum(recent_rewards) / len(recent_rewards),
            'avg_coverage': sum(recent_coverage) / len(recent_coverage),
            'avg_bandwidth_mb': sum(recent_bandwidth) / len(recent_bandwidth),
            'total_episodes': len(self.episode_rewards),
            'exploration_rate': self.agent.epsilon,
        }
