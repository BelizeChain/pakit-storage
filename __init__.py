"""
Pakit - Data Farm Killer

Maximum storage efficiency through intelligent compression, deduplication, and content addressing.

Vision: By 2030, eliminate 90% of data farm waste worldwide.

Quick Start:
    >>> from pakit import Pakit
    >>> 
    >>> # Initialize storage engine
    >>> pakit = Pakit(storage_dir="./my_storage")
    >>> 
    >>> # Store data (automatic compression + deduplication)
    >>> data = b"Hello, Belize!"
    >>> content_id = pakit.store(data)
    >>> 
    >>> # Retrieve data
    >>> retrieved = pakit.retrieve(content_id)
    >>> 
    >>> # Get efficiency stats
    >>> stats = pakit.get_stats()
    >>> print(f"Efficiency: {stats.efficiency_percent:.2f}%")

Features:
    - Multi-algorithm compression (ZSTD, LZ4, Brotli, LZMA, etc.)
    - Content-addressed storage (hash-based deduplication)
    - Three-tier storage (hot/warm/cold)
    - Quantum compression experiments (via Kinich)
    - ML optimization (via Nawal)
    - Blockchain proofs (via BelizeChain)
    - Decentralized backends (IPFS, Arweave)
"""

from .core.storage_engine import PakitStorageEngine, Pakit, StorageTier
from .core.compression import (
    CompressionEngine,
    CompressionAlgorithm,
    CompressionResult
)
from .core.deduplication import DeduplicationEngine
from .core.content_addressing import ContentAddressingEngine, ContentID

__version__ = "0.1.0"

__all__ = [
    # Main API
    "Pakit",
    "PakitStorageEngine",
    
    # Storage tiers
    "StorageTier",
    
    # Compression
    "CompressionEngine",
    "CompressionAlgorithm",
    "CompressionResult",
    
    # Deduplication
    "DeduplicationEngine",
    
    # Content addressing
    "ContentAddressingEngine",
    "ContentID",
]

__version__ = "0.1.0"
__author__ = "BelizeChain Team"

from .core.storage_engine import PakitStorageEngine, Pakit
from .core.compression import CompressionEngine, CompressionAlgorithm
from .core.deduplication import DeduplicationEngine
from .core.content_addressing import ContentAddressingEngine

__all__ = [
    "Pakit",
    "PakitStorageEngine",
    "CompressionEngine",
    "CompressionAlgorithm",
    "DeduplicationEngine",
    "ContentAddressingEngine",
]
