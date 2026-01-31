"""
Core compression engine with multi-algorithm support.

Automatically selects the best compression algorithm for maximum efficiency.
"""

import zlib
import lzma
import bz2
from enum import Enum
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Try to import optional high-performance compressors
try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logger.warning("zstandard not available. Install with: pip install zstandard")

try:
    import lz4.frame as lz4
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    logger.warning("lz4 not available. Install with: pip install lz4")

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logger.warning("brotli not available. Install with: pip install brotli")


class CompressionAlgorithm(Enum):
    """Supported compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"  # Standard library, good general purpose
    LZMA = "lzma"  # Standard library, best compression but slow
    BZ2 = "bz2"    # Standard library, good for text
    ZSTD = "zstd"  # Best general-purpose (requires install)
    LZ4 = "lz4"    # Ultra-fast, lower ratio (requires install)
    BROTLI = "brotli"  # Excellent for text/web (requires install)
    AUTO = "auto"  # Try all and pick best
    QUANTUM = "quantum"  # Experimental quantum compression


@dataclass
class CompressionResult:
    """Result of compression operation."""
    algorithm: CompressionAlgorithm
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    
    @property
    def efficiency_score(self) -> float:
        """
        Calculate efficiency score balancing ratio and speed.
        
        Higher is better. Considers both compression ratio and speed.
        """
        # Ratio weight: 70%, Speed weight: 30%
        # Normalize speed: <10ms = 1.0, >1000ms = 0.0
        speed_score = max(0.0, 1.0 - (self.compression_time_ms / 1000.0))
        return (self.compression_ratio * 0.7) + (speed_score * 0.3)


class CompressionEngine:
    """
    Multi-algorithm compression engine.
    
    Automatically selects the best compression algorithm for maximum
    storage efficiency. Tracks metrics and learns from usage patterns.
    """
    
    def __init__(
        self,
        default_algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO,
        zstd_level: int = 3,  # 1-22, 3 is good balance
        lzma_preset: int = 6,  # 0-9, 6 is default
    ):
        """
        Initialize compression engine.
        
        Args:
            default_algorithm: Default compression algorithm
            zstd_level: Zstandard compression level (1-22)
            lzma_preset: LZMA compression preset (0-9)
        """
        self.default_algorithm = default_algorithm
        self.zstd_level = zstd_level
        self.lzma_preset = lzma_preset
        
        # Compression statistics
        self.stats: Dict[str, Any] = {
            "total_compressed": 0,
            "total_original_bytes": 0,
            "total_compressed_bytes": 0,
            "algorithm_usage": {},
            "best_ratios": {},
        }
        
        # Initialize compressors
        self._init_compressors()
    
    def _init_compressors(self):
        """Initialize available compressors."""
        self.compressors = {
            CompressionAlgorithm.ZLIB: self._compress_zlib,
            CompressionAlgorithm.LZMA: self._compress_lzma,
            CompressionAlgorithm.BZ2: self._compress_bz2,
        }
        
        if ZSTD_AVAILABLE:
            self.compressors[CompressionAlgorithm.ZSTD] = self._compress_zstd
        
        if LZ4_AVAILABLE:
            self.compressors[CompressionAlgorithm.LZ4] = self._compress_lz4
        
        if BROTLI_AVAILABLE:
            self.compressors[CompressionAlgorithm.BROTLI] = self._compress_brotli
        
        logger.info(f"Initialized compression engine with {len(self.compressors)} algorithms")
    
    def compress(
        self,
        data: bytes,
        algorithm: Optional[CompressionAlgorithm] = None
    ) -> CompressionResult:
        """
        Compress data using specified or default algorithm.
        
        Args:
            data: Raw data to compress
            algorithm: Compression algorithm (None = use default)
        
        Returns:
            CompressionResult with compressed data and metrics
        """
        if algorithm is None:
            algorithm = self.default_algorithm
        
        if algorithm == CompressionAlgorithm.AUTO:
            return self._compress_auto(data)
        elif algorithm == CompressionAlgorithm.NONE:
            return self._compress_none(data)
        elif algorithm == CompressionAlgorithm.QUANTUM:
            return self._compress_quantum(data)
        else:
            return self._compress_with_algorithm(data, algorithm)
    
    def decompress(
        self,
        compressed_data: bytes,
        algorithm: CompressionAlgorithm
    ) -> bytes:
        """
        Decompress data.
        
        Args:
            compressed_data: Compressed data
            algorithm: Algorithm used for compression
        
        Returns:
            Original decompressed data
        """
        if algorithm == CompressionAlgorithm.NONE:
            return compressed_data
        elif algorithm == CompressionAlgorithm.QUANTUM:
            return self._decompress_quantum(compressed_data)
        
        decompressor = {
            CompressionAlgorithm.ZLIB: zlib.decompress,
            CompressionAlgorithm.LZMA: lzma.decompress,
            CompressionAlgorithm.BZ2: bz2.decompress,
        }
        
        if ZSTD_AVAILABLE:
            decompressor[CompressionAlgorithm.ZSTD] = lambda d: zstd.ZstdDecompressor().decompress(d)
        
        if LZ4_AVAILABLE:
            decompressor[CompressionAlgorithm.LZ4] = lz4.decompress
        
        if BROTLI_AVAILABLE:
            decompressor[CompressionAlgorithm.BROTLI] = brotli.decompress
        
        if algorithm not in decompressor:
            raise ValueError(f"Unsupported decompression algorithm: {algorithm}")
        
        return decompressor[algorithm](compressed_data)
    
    def _compress_auto(self, data: bytes) -> CompressionResult:
        """
        Try all available algorithms and select the best.
        
        Selection criteria:
        1. Best compression ratio (primary)
        2. Reasonable compression time (secondary)
        """
        import time
        
        original_size = len(data)
        results = []
        
        # Try all available algorithms
        for algorithm in self.compressors.keys():
            try:
                start = time.perf_counter()
                result = self._compress_with_algorithm(data, algorithm)
                end = time.perf_counter()
                
                result.compression_time_ms = (end - start) * 1000
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to compress with {algorithm}: {e}")
        
        if not results:
            # Fallback to no compression
            return self._compress_none(data)
        
        # Select best by efficiency score
        best = max(results, key=lambda r: r.efficiency_score)
        
        logger.debug(
            f"AUTO compression: {best.algorithm.value} selected "
            f"(ratio: {best.compression_ratio:.2f}x, "
            f"time: {best.compression_time_ms:.2f}ms)"
        )
        
        return best
    
    def _compress_with_algorithm(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm
    ) -> CompressionResult:
        """Compress data with specific algorithm."""
        import time
        
        original_size = len(data)
        
        if algorithm not in self.compressors:
            raise ValueError(f"Algorithm not available: {algorithm}")
        
        start = time.perf_counter()
        compressed_data = self.compressors[algorithm](data)
        end = time.perf_counter()
        
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        compression_time_ms = (end - start) * 1000
        
        # Update stats
        self._update_stats(algorithm, original_size, compressed_size)
        
        return CompressionResult(
            algorithm=algorithm,
            compressed_data=compressed_data,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compression_time_ms=compression_time_ms,
        )
    
    def _compress_none(self, data: bytes) -> CompressionResult:
        """No compression (passthrough)."""
        size = len(data)
        return CompressionResult(
            algorithm=CompressionAlgorithm.NONE,
            compressed_data=data,
            original_size=size,
            compressed_size=size,
            compression_ratio=1.0,
            compression_time_ms=0.0,
        )
    
    def _compress_zlib(self, data: bytes) -> bytes:
        """Compress with zlib (level 6)."""
        return zlib.compress(data, level=6)
    
    def _compress_lzma(self, data: bytes) -> bytes:
        """Compress with LZMA."""
        return lzma.compress(data, preset=self.lzma_preset)
    
    def _compress_bz2(self, data: bytes) -> bytes:
        """Compress with BZ2."""
        return bz2.compress(data, compresslevel=9)
    
    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress with Zstandard."""
        compressor = zstd.ZstdCompressor(level=self.zstd_level)
        return compressor.compress(data)
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress with LZ4."""
        return lz4.compress(data)
    
    def _compress_brotli(self, data: bytes) -> bytes:
        """Compress with Brotli."""
        return brotli.compress(data, quality=11)
    
    def _compress_quantum(self, data: bytes) -> CompressionResult:
        """
        Experimental quantum compression (placeholder).
        
        In production, this would submit a job to Kinich quantum network.
        For now, uses best classical algorithm.
        """
        logger.warning(
            "Quantum compression not yet implemented. "
            "Falling back to AUTO compression."
        )
        return self._compress_auto(data)
    
    def _decompress_quantum(self, compressed_data: bytes) -> bytes:
        """Decompress quantum-compressed data (placeholder)."""
        raise NotImplementedError(
            "Quantum decompression not yet implemented. "
            "Submit quantum compression jobs through Kinich integration."
        )
    
    def _update_stats(
        self,
        algorithm: CompressionAlgorithm,
        original_size: int,
        compressed_size: int
    ):
        """Update compression statistics."""
        self.stats["total_compressed"] += 1
        self.stats["total_original_bytes"] += original_size
        self.stats["total_compressed_bytes"] += compressed_size
        
        algo_name = algorithm.value
        if algo_name not in self.stats["algorithm_usage"]:
            self.stats["algorithm_usage"][algo_name] = 0
        self.stats["algorithm_usage"][algo_name] += 1
        
        ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        if algo_name not in self.stats["best_ratios"]:
            self.stats["best_ratios"][algo_name] = ratio
        else:
            self.stats["best_ratios"][algo_name] = max(
                self.stats["best_ratios"][algo_name],
                ratio
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        overall_ratio = (
            self.stats["total_original_bytes"] / self.stats["total_compressed_bytes"]
            if self.stats["total_compressed_bytes"] > 0
            else 1.0
        )
        
        return {
            **self.stats,
            "overall_compression_ratio": overall_ratio,
            "space_saved_bytes": self.stats["total_original_bytes"] - self.stats["total_compressed_bytes"],
            "space_saved_percent": (1 - 1/overall_ratio) * 100 if overall_ratio > 1 else 0,
        }
    
    def get_recommended_algorithm(self, data_type: str = "generic") -> CompressionAlgorithm:
        """
        Get recommended algorithm for data type.
        
        Args:
            data_type: Type of data (text, binary, media, etc.)
        
        Returns:
            Recommended compression algorithm
        """
        recommendations = {
            "text": CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
            "json": CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
            "binary": CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.LZMA,
            "image": CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.NONE,
            "video": CompressionAlgorithm.NONE,  # Already compressed
            "audio": CompressionAlgorithm.NONE,  # Already compressed
            "generic": CompressionAlgorithm.AUTO,
        }
        
        return recommendations.get(data_type.lower(), CompressionAlgorithm.AUTO)
