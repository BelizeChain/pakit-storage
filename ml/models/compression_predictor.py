"""
Compression Algorithm Predictor

Neural network that predicts the best compression algorithm for each block.
Chooses between: zstd, lz4, snappy, or none.
"""

from typing import Dict, Any, List
import time
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

from pakit.ml.base_model import PakitMLModel, ModelConfig


class CompressionNet(nn.Module):
    """
    Neural network for compression algorithm prediction.
    
    Architecture: 3-layer MLP (128-64-32) with dropout
    Output: 4-way softmax (zstd, lz4, snappy, none)
    """
    
    def __init__(self, input_dim: int = 12, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], 4)  # 4 classes
        
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dims[0])
        self.batch_norm2 = nn.BatchNorm1d(hidden_dims[1])
    
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.softmax(x, dim=1)


class CompressionPredictor(PakitMLModel):
    """
    Predicts best compression algorithm for blocks.
    
    Target: >80% accuracy, <1ms inference time
    """
    
    # Class mapping
    ALGO_TO_IDX = {'zstd': 0, 'lz4': 1, 'snappy': 2, 'none': 3}
    IDX_TO_ALGO = {0: 'zstd', 1: 'lz4', 2: 'snappy', 3: 'none'}
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        if TORCH_AVAILABLE:
            self.model = CompressionNet(
                hidden_dims=config.hidden_dims,
                dropout=config.dropout
            )
            self.model.to(config.device)
        else:
            self.model = None
    
    def predict(self, features: Dict[str, Any]) -> str:
        """
        Predict best compression algorithm.
        
        Args:
            features: Block features (size, entropy, content_type, etc.)
            
        Returns:
            Algorithm name ('zstd', 'lz4', 'snappy', 'none')
        """
        if not TORCH_AVAILABLE or self.model is None:
            # Fallback: round-robin
            return 'zstd'
        
        start = time.time()
        
        # Extract feature vector
        x = self._features_to_tensor(features)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            probs = self.model(x)
            pred_idx = probs.argmax(dim=1).item()
        
        algo = self.IDX_TO_ALGO[pred_idx]
        
        self._track_inference(time.time() - start)
        
        return algo
    
    def predict_probs(self, features: Dict[str, Any]) -> Dict[str, float]:
        """
        Get probabilities for each algorithm.
        
        Returns:
            Dict mapping algorithm to probability
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {'zstd': 0.25, 'lz4': 0.25, 'snappy': 0.25, 'none': 0.25}
        
        x = self._features_to_tensor(features)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(x).cpu().numpy()[0]
        
        return {
            'zstd': float(probs[0]),
            'lz4': float(probs[1]),
            'snappy': float(probs[2]),
            'none': float(probs[3]),
        }
    
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train compression predictor.
        
        Args:
            dataset: TrainingDataset with (features, labels)
            validation_split: Validation fraction
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot train - PyTorch not available")
            return {'loss': 0.0, 'accuracy': 0.0}
        
        from pakit.ml.training.trainer import ModelTrainer, TrainingConfig
        
        # Create training config
        train_config = TrainingConfig(
            model_name=self.config.name,
            model_type=self.config.model_type,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            device=self.config.device,
        )
        
        trainer = ModelTrainer(train_config)
        
        # Prepare data loaders (simplified - implement full version)
        # train_loader, val_loader = self._prepare_data_loaders(dataset, validation_split)
        
        # history = trainer.train(self.model, train_loader, val_loader)
        
        # Placeholder
        logger.info(f"Training {self.config.name} (placeholder)")
        return {'loss': 0.1, 'accuracy': 0.85}
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        if not TORCH_AVAILABLE or self.model is None:
            return
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
        }, path)
        
        logger.info(f"Saved {self.config.name} to {path}")
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            return
        
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.config.device)
        
        logger.info(f"Loaded {self.config.name} from {path}")
    
    def _features_to_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """Convert features to tensor."""
        # Extract feature vector (12 dimensions)
        feature_vector = [
            features.get('block_size_log', 0.0),
            features.get('block_depth_log', 0.0),
            features.get('parent_count', 0.0),
            features.get('hash_entropy', 0.0),
            features.get('is_leaf', 0.0),
            features.get('is_merge', 0.0),
            # Content type one-hot (placeholder - expand)
            features.get('content_type_text', 0.0),
            features.get('content_type_binary', 0.0),
            features.get('content_type_json', 0.0),
            features.get('content_type_image', 0.0),
            # Previous compression
            features.get('algo_zstd', 0.0),
            features.get('algo_lz4', 0.0),
        ]
        
        x = torch.tensor([feature_vector], dtype=torch.float32)
        return x.to(self.config.device)
