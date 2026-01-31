"""
Training Dataset

Manages training datasets for ML models.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TrainingDataset:
    """
    Training dataset for ML models.
    
    Stores features and labels, supports train/val/test splits.
    """
    
    def __init__(self, name: str = "pakit_dataset"):
        self.name = name
        self.features: List[Dict[str, Any]] = []
        self.labels: List[Any] = []
        self._metadata = {}
    
    def add_sample(self, features: Dict[str, Any], label: Any) -> None:
        """
        Add a training sample.
        
        Args:
            features: Feature dictionary
            label: Target label
        """
        self.features.append(features)
        self.labels.append(label)
    
    def add_batch(
        self,
        features: List[Dict[str, Any]],
        labels: List[Any]
    ) -> None:
        """
        Add multiple samples.
        
        Args:
            features: List of feature dicts
            labels: List of labels
        """
        if len(features) != len(labels):
            raise ValueError("Features and labels must have same length")
        
        self.features.extend(features)
        self.labels.extend(labels)
    
    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple['TrainingDataset', 'TrainingDataset', 'TrainingDataset']:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            train_ratio: Fraction for training (0.0-1.0)
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            shuffle: Whether to shuffle before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        n_samples = len(self.features)
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        # Create datasets
        train_ds = TrainingDataset(f"{self.name}_train")
        val_ds = TrainingDataset(f"{self.name}_val")
        test_ds = TrainingDataset(f"{self.name}_test")
        
        # Split data
        for i in indices[:train_end]:
            train_ds.add_sample(self.features[i], self.labels[i])
        
        for i in indices[train_end:val_end]:
            val_ds.add_sample(self.features[i], self.labels[i])
        
        for i in indices[val_end:]:
            test_ds.add_sample(self.features[i], self.labels[i])
        
        logger.info(
            f"Split {n_samples} samples: "
            f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )
        
        return train_ds, val_ds, test_ds
    
    def get_numpy_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to NumPy arrays.
        
        Returns:
            (features_array, labels_array)
        """
        # Convert features to array (assumes all features have same keys)
        if not self.features:
            return np.array([]), np.array([])
        
        # Get feature keys from first sample
        feature_keys = sorted(self.features[0].keys())
        
        # Build feature matrix
        X = []
        for feature_dict in self.features:
            row = [feature_dict[key] for key in feature_keys]
            X.append(row)
        
        return np.array(X), np.array(self.labels)
    
    def save(self, path: str) -> None:
        """
        Save dataset to disk.
        
        Args:
            path: File path to save to
        """
        data = {
            'name': self.name,
            'features': self.features,
            'labels': self.labels,
            'metadata': self._metadata,
        }
        
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved dataset: {path} ({len(self)} samples)")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingDataset':
        """
        Load dataset from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            Loaded dataset
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        dataset = cls(data['name'])
        dataset.features = data['features']
        dataset.labels = data['labels']
        dataset._metadata = data.get('metadata', {})
        
        logger.info(f"Loaded dataset: {path} ({len(dataset)} samples)")
        return dataset
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata field."""
        self._metadata[key] = value
    
    def get_metadata(self, key: str) -> Optional[Any]:
        """Get metadata field."""
        return self._metadata.get(key)
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.features)
    
    def __repr__(self) -> str:
        return f"TrainingDataset(name={self.name}, samples={len(self)})"
