"""
Block Discovery Service

DHT-based peer and block discovery.
"""

from .block_discovery import (
    BlockDiscoveryService,
    BlockProvider,
    BlockMetadata,
    MAX_PROVIDERS,
    PROVIDER_TIMEOUT,
    ANNOUNCE_INTERVAL
)

__all__ = [
    "BlockDiscoveryService",
    "BlockProvider",
    "BlockMetadata",
    "MAX_PROVIDERS",
    "PROVIDER_TIMEOUT",
    "ANNOUNCE_INTERVAL"
]
