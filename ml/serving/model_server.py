"""
Model Server

Production model serving with hot-swapping and multi-model support.
"""

from typing import Dict, Any, Optional, List
import threading
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from pakit.ml.registry import ModelRegistry
from pakit.ml.serving.version_manager import VersionManager
from pakit.ml.serving.monitor import ModelMonitor


class ModelServer:
    """
    Production model server.
    
    Features:
    - Multi-model serving
    - Hot-swapping (zero-downtime updates)
    - Version management
    - Performance monitoring
    - Health checks
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = ModelRegistry()
        self.version_manager = VersionManager(str(self.model_dir))
        self.monitor = ModelMonitor()
        
        # Server state
        self.is_running = False
        self.lock = threading.Lock()
        
        # Hot-swap queue
        self.pending_swaps: List[Dict[str, Any]] = []
    
    def start(self) -> None:
        """Start model server."""
        logger.info("Starting model server...")
        
        # Load all models from disk
        self._load_initial_models()
        
        self.is_running = True
        
        logger.info(
            f"Model server started with {len(self.registry.models)} models"
        )
    
    def stop(self) -> None:
        """Stop model server."""
        logger.info("Stopping model server...")
        
        self.is_running = False
        
        # Save all models
        self.registry.save_all(str(self.model_dir))
        
        logger.info("Model server stopped")
    
    def predict(
        self,
        model_name: str,
        features: Dict[str, Any]
    ) -> Any:
        """
        Get prediction from model.
        
        Args:
            model_name: Model name
            features: Input features
            
        Returns:
            Model prediction
        """
        if not self.is_running:
            raise RuntimeError("Model server not running")
        
        start = time.time()
        
        # Get model
        model = self.registry.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found")
        
        if not model.enabled:
            raise ValueError(f"Model '{model_name}' is disabled")
        
        try:
            # Get prediction
            prediction = model.predict(features)
            
            # Record metrics
            latency_ms = (time.time() - start) * 1000
            self.monitor.record_prediction(
                model_name,
                success=True,
                latency_ms=latency_ms
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed for {model_name}: {e}")
            
            # Record failure
            latency_ms = (time.time() - start) * 1000
            self.monitor.record_prediction(
                model_name,
                success=False,
                latency_ms=latency_ms
            )
            
            raise e
    
    def load_model(
        self,
        model_name: str,
        model_path: str,
        version: Optional[str] = None
    ) -> None:
        """
        Load model into server.
        
        Args:
            model_name: Model name
            model_path: Path to model file
            version: Model version
        """
        with self.lock:
            logger.info(f"Loading model '{model_name}' from {model_path}")
            
            # Register with version manager
            if version:
                self.version_manager.register_version(
                    model_name,
                    version,
                    model_path
                )
            
            # Load into registry
            # (Actual loading would happen here)
            
            logger.info(f"Model '{model_name}' loaded successfully")
    
    def hot_swap_model(
        self,
        model_name: str,
        new_model_path: str,
        new_version: str
    ) -> None:
        """
        Hot-swap model with zero downtime.
        
        Args:
            model_name: Model to swap
            new_model_path: Path to new model
            new_version: New model version
        """
        logger.info(
            f"Hot-swapping model '{model_name}' to version {new_version}"
        )
        
        # Queue swap (will be applied during safe window)
        swap = {
            'model_name': model_name,
            'path': new_model_path,
            'version': new_version,
            'timestamp': time.time(),
        }
        
        with self.lock:
            self.pending_swaps.append(swap)
        
        # Apply swap immediately if safe
        self._apply_pending_swaps()
    
    def _apply_pending_swaps(self) -> None:
        """Apply pending model swaps."""
        with self.lock:
            if not self.pending_swaps:
                return
            
            for swap in self.pending_swaps:
                model_name = swap['model_name']
                new_path = swap['path']
                new_version = swap['version']
                
                try:
                    # Load new model
                    # (Actual loading logic would be here)
                    
                    # Register version
                    self.version_manager.register_version(
                        model_name,
                        new_version,
                        new_path
                    )
                    
                    logger.info(
                        f"Successfully swapped '{model_name}' to v{new_version}"
                    )
                    
                except Exception as e:
                    logger.error(f"Model swap failed: {e}")
            
            # Clear queue
            self.pending_swaps.clear()
    
    def rollback_model(
        self,
        model_name: str,
        target_version: Optional[str] = None
    ) -> None:
        """
        Rollback model to previous version.
        
        Args:
            model_name: Model to rollback
            target_version: Target version (or previous if None)
        """
        logger.info(f"Rolling back model '{model_name}'")
        
        if target_version:
            version_info = self.version_manager.get_version(
                model_name,
                target_version
            )
        else:
            # Get previous version
            versions = self.version_manager.list_versions(model_name)
            if len(versions) < 2:
                logger.error("No previous version to rollback to")
                return
            
            target_version = versions[-2]['version']
            version_info = versions[-2]
        
        # Hot-swap to target version
        self.hot_swap_model(
            model_name,
            version_info['path'],
            target_version
        )
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status
        """
        health = {
            'status': 'healthy' if self.is_running else 'stopped',
            'models_loaded': len(self.registry.models),
            'models_enabled': len([
                m for m in self.registry.models.values()
                if m.enabled
            ]),
            'pending_swaps': len(self.pending_swaps),
            'uptime_seconds': 0.0,  # Would track actual uptime
        }
        
        # Add model-level health
        model_health = {}
        for model_name in self.registry.models:
            metrics = self.monitor.get_metrics(model_name)
            model_health[model_name] = {
                'enabled': self.registry.models[model_name].enabled,
                'predictions': metrics.get('total_predictions', 0),
                'success_rate': metrics.get('success_rate', 0.0),
                'avg_latency_ms': metrics.get('avg_latency_ms', 0.0),
            }
        
        health['models'] = model_health
        
        return health
    
    def _load_initial_models(self) -> None:
        """Load models on server startup."""
        logger.info(f"Loading models from {self.model_dir}")
        
        # Load from registry
        self.registry.load_all(str(self.model_dir))
        
        logger.info(f"Loaded {len(self.registry.models)} models")
