"""
Pakit ML Optimization Layer

Machine learning models for intelligent optimization of Pakit storage operations.
Provides compression prediction, deduplication optimization, prefetching, and more.

Architecture:
    ML Layer (predictions/recommendations) → Deterministic Core (final decisions)
    
Models never compromise cryptographic determinism - they only provide hints.
"""

from pakit.ml.base_model import PakitMLModel, ModelConfig
from pakit.ml.registry import ModelRegistry
from pakit.ml.checkpoint import ModelCheckpoint, load_checkpoint, save_checkpoint

__all__ = [
    'PakitMLModel',
    'ModelConfig',
    'ModelRegistry',
    'ModelCheckpoint',
    'load_checkpoint',
    'save_checkpoint',
]

__version__ = '0.3.0-alpha'
