"""
Pakit ML Telemetry System

Collects training data from live Pakit operations for ML model training.
Privacy-preserving - never leaks actual block content.
"""

from pakit.ml.telemetry.collector import TelemetryCollector, BlockEvent
from pakit.ml.telemetry.dataset import TrainingDataset
from pakit.ml.telemetry.features import FeatureExtractor
from pakit.ml.telemetry.privacy import PrivacyFilter

__all__ = [
    'TelemetryCollector',
    'BlockEvent',
    'TrainingDataset',
    'FeatureExtractor',
    'PrivacyFilter',
]
