"""
Base ML Model Classes

Provides abstract base class and configuration for all Pakit ML models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    
    name: str
    version: str
    model_type: str  # "compression", "dedup", "prefetch", "peer", "network"
    
    # Training config
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.1
    
    # Inference config
    device: str = "cpu"  # "cpu", "cuda", "rocm"
    quantized: bool = False
    max_inference_time_ms: float = 10.0
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    trained_samples: int = 0
    accuracy: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'model_type': self.model_type,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'device': self.device,
            'quantized': self.quantized,
            'max_inference_time_ms': self.max_inference_time_ms,
            'created_at': self.created_at,
            'trained_samples': self.trained_samples,
            'accuracy': self.accuracy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**data)


class PakitMLModel(ABC):
    """
    Abstract base class for all Pakit ML models.
    
    All models must implement:
    - predict(): Make predictions on input data
    - train(): Train the model on dataset
    - save(): Save model to disk
    - load(): Load model from disk
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self._inference_count = 0
        self._total_inference_time = 0.0
        
        logger.info(
            f"Initialized {config.model_type} model: {config.name} v{config.version}"
        )
    
    @abstractmethod
    def predict(self, features: Dict[str, Any]) -> Any:
        """
        Make prediction on input features.
        
        Args:
            features: Input feature dictionary
            
        Returns:
            Prediction (type depends on model)
        """
        pass
    
    @abstractmethod
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train the model on dataset.
        
        Args:
            dataset: Training dataset
            validation_split: Fraction for validation (0.0-1.0)
            
        Returns:
            Training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: File path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: File path to load model from
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics."""
        avg_inference_time = (
            self._total_inference_time / self._inference_count
            if self._inference_count > 0
            else 0.0
        )
        
        return {
            'name': self.config.name,
            'version': self.config.version,
            'type': self.config.model_type,
            'device': self.config.device,
            'quantized': self.config.quantized,
            'inference_count': self._inference_count,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'trained_samples': self.config.trained_samples,
            'accuracy': self.config.accuracy,
        }
    
    def _track_inference(self, inference_time: float) -> None:
        """Track inference timing."""
        self._inference_count += 1
        self._total_inference_time += inference_time
        
        # Warn if inference is slow
        if inference_time * 1000 > self.config.max_inference_time_ms:
            logger.warning(
                f"Slow inference: {inference_time*1000:.2f}ms "
                f"(max: {self.config.max_inference_time_ms}ms)"
            )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.config.name}, "
            f"version={self.config.version}, "
            f"type={self.config.model_type})"
        )


class DummyModel(PakitMLModel):
    """
    Dummy model for testing/fallback.
    Always returns default predictions.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.default_prediction = None
    
    def predict(self, features: Dict[str, Any]) -> Any:
        """Return default prediction."""
        start = time.time()
        result = self.default_prediction
        self._track_inference(time.time() - start)
        return result
    
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """No training for dummy model."""
        return {'loss': 0.0, 'accuracy': 0.0}
    
    def save(self, path: str) -> None:
        """No-op for dummy model."""
        logger.info(f"Dummy model {self.config.name} - save skipped")
    
    def load(self, path: str) -> None:
        """No-op for dummy model."""
        logger.info(f"Dummy model {self.config.name} - load skipped")
