"""
Model Registry

Central registry for managing all ML models in Pakit.
Handles model registration, retrieval, and lifecycle management.
"""

from typing import Dict, Optional, List
import logging
from pathlib import Path

from pakit.ml.base_model import PakitMLModel, ModelConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing ML models.
    
    Singleton pattern - only one registry per process.
    """
    
    _instance: Optional['ModelRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._models: Dict[str, PakitMLModel] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._enabled: Dict[str, bool] = {}
        self._initialized = True
        
        logger.info("Model registry initialized")
    
    def register(
        self,
        name: str,
        model: PakitMLModel,
        enabled: bool = True
    ) -> None:
        """
        Register a model in the registry.
        
        Args:
            name: Unique model name
            model: Model instance
            enabled: Whether model is enabled for inference
        """
        if name in self._models:
            logger.warning(f"Model {name} already registered, replacing")
        
        self._models[name] = model
        self._configs[name] = model.config
        self._enabled[name] = enabled
        
        logger.info(
            f"Registered model: {name} "
            f"(type={model.config.model_type}, enabled={enabled})"
        )
    
    def get(self, name: str) -> Optional[PakitMLModel]:
        """
        Get model by name.
        
        Args:
            name: Model name
            
        Returns:
            Model instance or None if not found
        """
        return self._models.get(name)
    
    def get_by_type(self, model_type: str) -> List[PakitMLModel]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Model type (compression, dedup, prefetch, etc.)
            
        Returns:
            List of models matching type
        """
        return [
            model for model in self._models.values()
            if model.config.model_type == model_type
        ]
    
    def is_enabled(self, name: str) -> bool:
        """
        Check if model is enabled.
        
        Args:
            name: Model name
            
        Returns:
            True if enabled, False otherwise
        """
        return self._enabled.get(name, False)
    
    def enable(self, name: str) -> None:
        """Enable a model."""
        if name not in self._models:
            raise ValueError(f"Model {name} not registered")
        
        self._enabled[name] = True
        logger.info(f"Enabled model: {name}")
    
    def disable(self, name: str) -> None:
        """Disable a model."""
        if name not in self._models:
            raise ValueError(f"Model {name} not registered")
        
        self._enabled[name] = False
        logger.info(f"Disabled model: {name}")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a model.
        
        Args:
            name: Model name
        """
        if name in self._models:
            del self._models[name]
            del self._configs[name]
            del self._enabled[name]
            logger.info(f"Unregistered model: {name}")
    
    def list_models(self) -> List[str]:
        """Get list of all registered model names."""
        return list(self._models.keys())
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all models."""
        return {
            name: model.get_stats()
            for name, model in self._models.items()
        }
    
    def load_all(self, model_dir: Path) -> int:
        """
        Load all models from directory.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            Number of models loaded
        """
        if not model_dir.exists():
            logger.warning(f"Model directory {model_dir} does not exist")
            return 0
        
        loaded = 0
        for model_file in model_dir.glob("*.pt"):
            try:
                name = model_file.stem
                if name in self._models:
                    self._models[name].load(str(model_file))
                    loaded += 1
                    logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.error(f"Failed to load {model_file}: {e}")
        
        return loaded
    
    def save_all(self, model_dir: Path) -> int:
        """
        Save all models to directory.
        
        Args:
            model_dir: Directory to save models
            
        Returns:
            Number of models saved
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        
        saved = 0
        for name, model in self._models.items():
            try:
                model_file = model_dir / f"{name}.pt"
                model.save(str(model_file))
                saved += 1
                logger.info(f"Saved model: {name}")
            except Exception as e:
                logger.error(f"Failed to save {name}: {e}")
        
        return saved
    
    def clear(self) -> None:
        """Clear all registered models."""
        self._models.clear()
        self._configs.clear()
        self._enabled.clear()
        logger.info("Cleared model registry")


# Global registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry."""
    return _registry
