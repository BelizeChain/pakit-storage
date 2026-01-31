"""
Pakit ML Models

Intelligent optimization models for Pakit storage.
"""

from pakit.ml.models.compression_predictor import CompressionPredictor
from pakit.ml.models.dedup_optimizer import DeduplicationOptimizer

__all__ = [
    'CompressionPredictor',
    'DeduplicationOptimizer',
]
