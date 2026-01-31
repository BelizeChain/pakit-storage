"""
Pakit Core Module

Core functionality for quantum-enhanced storage:
- Compression algorithms (ZSTD, LZ4, Brotli, Quantum)
- Content addressing (SHA-256, BLAKE3)
- Deduplication (content-based chunking)
- Storage engine (unified interface)
"""

from pakit.core.compression import CompressionEngine, CompressionAlgorithm, CompressionResult
from pakit.core.content_addressing import ContentAddressingEngine, ContentID
from pakit.core.deduplication import DeduplicationEngine, DeduplicationStats
from pakit.core.storage_engine import PakitStorageEngine, StorageTier, StorageMetadata, StorageStats

__all__ = [
    "CompressionEngine",
    "CompressionAlgorithm",
    "CompressionResult",
    "ContentAddressingEngine",
    "ContentID",
    "DeduplicationEngine",
    "DeduplicationStats",
    "PakitStorageEngine",
    "StorageTier",
    "StorageMetadata",
    "StorageStats",
]
