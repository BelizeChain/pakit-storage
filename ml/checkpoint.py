"""
Model Checkpointing

Utilities for saving and loading model checkpoints.
Supports versioning, metadata, and GPU/CPU compatibility.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Try to import PyTorch, fall back gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - checkpointing disabled")


class ModelCheckpoint:
    """
    Model checkpoint manager.
    
    Handles saving/loading of model state, config, and metadata.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./pakit_models",
        auto_save: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_save = auto_save
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Checkpoint manager initialized: {checkpoint_dir}")
    
    def save(
        self,
        model_state: Dict[str, Any],
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "model",
        version: str = "latest"
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model_state: Model state dict (PyTorch state_dict or custom)
            config: Model configuration
            metadata: Optional metadata (training metrics, etc.)
            name: Model name
            version: Model version
            
        Returns:
            Path to saved checkpoint
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot save checkpoint - PyTorch not available")
            return ""
        
        checkpoint = {
            'model_state': model_state,
            'config': config,
            'metadata': metadata or {},
            'version': version,
        }
        
        # Create filename
        filename = f"{name}_v{version}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        try:
            torch.save(checkpoint, filepath)
            logger.info(f"Saved checkpoint: {filepath}")
            
            # Save metadata as JSON for easy inspection
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump({
                    'config': config,
                    'metadata': checkpoint['metadata'],
                    'version': version,
                }, f, indent=2)
            
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return ""
    
    def load(
        self,
        name: str,
        version: str = "latest",
        map_location: str = "cpu"
    ) -> Optional[Dict[str, Any]]:
        """
        Load model checkpoint.
        
        Args:
            name: Model name
            version: Model version
            map_location: Device to load to ("cpu", "cuda", etc.)
            
        Returns:
            Checkpoint dict or None if not found
        """
        if not TORCH_AVAILABLE:
            logger.error("Cannot load checkpoint - PyTorch not available")
            return None
        
        filename = f"{name}_v{version}.pt"
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            logger.error(f"Checkpoint not found: {filepath}")
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location=map_location)
            logger.info(f"Loaded checkpoint: {filepath}")
            return checkpoint
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def list_checkpoints(self) -> Dict[str, list]:
        """
        List all available checkpoints.
        
        Returns:
            Dict mapping model names to versions
        """
        checkpoints = {}
        
        for filepath in self.checkpoint_dir.glob("*.pt"):
            # Parse filename: model_v1.0.0.pt -> (model, 1.0.0)
            stem = filepath.stem
            if '_v' in stem:
                name, version = stem.rsplit('_v', 1)
                if name not in checkpoints:
                    checkpoints[name] = []
                checkpoints[name].append(version)
        
        return checkpoints
    
    def delete(self, name: str, version: str = "latest") -> bool:
        """
        Delete a checkpoint.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if deleted, False otherwise
        """
        filename = f"{name}_v{version}.pt"
        filepath = self.checkpoint_dir / filename
        
        if filepath.exists():
            filepath.unlink()
            # Also delete metadata
            metadata_file = filepath.with_suffix('.json')
            if metadata_file.exists():
                metadata_file.unlink()
            
            logger.info(f"Deleted checkpoint: {filepath}")
            return True
        
        return False


def save_checkpoint(
    model_state: Dict[str, Any],
    config: Dict[str, Any],
    path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Quick save checkpoint to specific path.
    
    Args:
        model_state: Model state dict
        config: Model configuration
        path: File path to save to
        metadata: Optional metadata
        
    Returns:
        True if saved successfully
    """
    if not TORCH_AVAILABLE:
        logger.error("Cannot save - PyTorch not available")
        return False
    
    checkpoint = {
        'model_state': model_state,
        'config': config,
        'metadata': metadata or {},
    }
    
    try:
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save: {e}")
        return False


def load_checkpoint(
    path: str,
    map_location: str = "cpu"
) -> Optional[Dict[str, Any]]:
    """
    Quick load checkpoint from specific path.
    
    Args:
        path: File path to load from
        map_location: Device to load to
        
        
    Returns:
        Checkpoint dict or None
    """
    if not TORCH_AVAILABLE:
        logger.error("Cannot load - PyTorch not available")
        return None
    
    if not os.path.exists(path):
        logger.error(f"Checkpoint not found: {path}")
        return None
    
    try:
        checkpoint = torch.load(path, map_location=map_location)
        logger.info(f"Loaded checkpoint: {path}")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load: {e}")
        return None


def detect_device() -> str:
    """
    Detect best available device (CUDA, ROCm, or CPU).
    
    Returns:
        Device string: "cuda", "rocm", or "cpu"
    """
    if not TORCH_AVAILABLE:
        return "cpu"
    
    # Check CUDA
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {device_name}")
        return device
    
    # Check ROCm (AMD GPUs)
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        device = "rocm"
        logger.info("ROCm GPU detected")
        return device
    
    logger.info("No GPU detected, using CPU")
    return "cpu"


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Returns:
        Dict with device details
    """
    if not TORCH_AVAILABLE:
        return {
            'available': False,
            'device': 'cpu',
            'cuda_available': False,
            'device_count': 0,
        }
    
    info = {
        'available': True,
        'device': detect_device(),
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info
