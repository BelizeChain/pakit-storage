"""
ML-DAG Integration

Connects ML models to deterministic DAG backend.
"""

from pakit.ml.integration.dag_integration import MLDAGIntegration
from pakit.ml.integration.fallback import FallbackManager
from pakit.ml.integration.ab_test import ABTestFramework

__all__ = [
    'MLDAGIntegration',
    'FallbackManager',
    'ABTestFramework',
]
