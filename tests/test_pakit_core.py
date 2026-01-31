"""
Comprehensive test suite for Pakit core functionality.

Tests compression, deduplication, content addressing, and storage engine.

Author: BelizeChain Team
License: MIT
"""

import pytest
import tempfile
import shutil
from pathlib import Path

# Module imports
from pakit.core.compression import CompressionEngine, CompressionAlgorithm, CompressionResult
from pakit.core.content_addressing import ContentAddressingEngine, ContentID
from pakit.core.deduplication import DeduplicationEngine
from pakit.core.storage_engine import PakitStorageEngine, StorageTier, StorageMetadata, StorageStats


# ===== FIXTURES =====

@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp(prefix="pakit_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def compression_engine():
    """Provide a compression engine instance."""
    return CompressionEngine()


@pytest.fixture
def content_addressing_engine():
    """Provide a content addressing engine instance."""
    return ContentAddressingEngine()


@pytest.fixture
def deduplication_engine():
    """Provide a deduplication engine instance."""
    return DeduplicationEngine()


@pytest.fixture
def storage_engine(temp_storage_dir):
    """Provide a Pakit storage engine instance."""
    return PakitStorageEngine(storage_dir=temp_storage_dir)


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return b"Hello, Belize! This is sample data for testing. " * 100


@pytest.fixture
def large_data():
    """Provide large data for compression testing."""
    return b"Large dataset " * 10000  # ~140KB


@pytest.fixture
def duplicate_data():
    """Provide duplicate data for deduplication testing."""
    data = b"Duplicate content for deduplication testing."
    return [data, data, data]  # Same data 3 times


# ===== COMPRESSION TESTS =====

@pytest.mark.unit
class TestCompression:
    """Test compression engine functionality."""
    
    def test_zlib_compression(self, compression_engine, sample_data):
        """Test ZLIB compression."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.ZLIB)
        
        assert result.algorithm == CompressionAlgorithm.ZLIB
        assert result.original_size == len(sample_data)
        assert result.compressed_size < result.original_size
        assert result.compression_ratio > 0
    
    def test_zstd_compression(self, compression_engine, sample_data):
        """Test Zstandard compression."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.ZSTD)
        
        assert result.algorithm == CompressionAlgorithm.ZSTD
        assert result.compressed_size < result.original_size
    
    def test_lz4_compression(self, compression_engine, sample_data):
        """Test LZ4 compression."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.LZ4)
        
        assert result.algorithm == CompressionAlgorithm.LZ4
        assert result.compressed_size <= result.original_size
    
    def test_brotli_compression(self, compression_engine, sample_data):
        """Test Brotli compression."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.BROTLI)
        
        assert result.algorithm == CompressionAlgorithm.BROTLI
        assert result.compressed_size < result.original_size
    
    def test_auto_compression_selection(self, compression_engine, sample_data):
        """Test automatic algorithm selection."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.AUTO)
        
        assert result.algorithm in [
            CompressionAlgorithm.ZLIB,
            CompressionAlgorithm.ZSTD,
            CompressionAlgorithm.LZ4,
            CompressionAlgorithm.BROTLI
        ]
        assert result.efficiency_score > 0
    
    def test_decompression(self, compression_engine, sample_data):
        """Test decompression returns original data."""
        # Test each algorithm
        for algo in [CompressionAlgorithm.ZLIB, CompressionAlgorithm.ZSTD]:
            result = compression_engine.compress(sample_data, algo)
            decompressed = compression_engine.decompress(result.compressed_data, result.algorithm)
            assert decompressed == sample_data
    
    def test_compression_ratio_calculation(self, compression_engine, sample_data):
        """Test compression ratio calculation."""
        result = compression_engine.compress(sample_data, CompressionAlgorithm.ZLIB)
        
        # Compression ratio is calculated as original_size / compressed_size (a multiplier)
        # For repetitive data, should achieve good compression (ratio > 1)
        assert result.compression_ratio > 1.0
        assert result.compressed_size < result.original_size
    
    def test_large_data_compression(self, compression_engine, large_data):
        """Test compression of large data."""
        result = compression_engine.compress(large_data, CompressionAlgorithm.ZSTD)
        
        assert result.compressed_size < result.original_size
        assert result.compression_ratio > 0.5  # Should achieve >50% compression
    
    def test_empty_data_compression(self, compression_engine):
        """Test compression of empty data."""
        empty_data = b""
        result = compression_engine.compress(empty_data, CompressionAlgorithm.ZLIB)
        
        assert result.original_size == 0
    
    def test_uncompressible_data(self, compression_engine):
        """Test compression of random (uncompressible) data."""
        import os
        random_data = os.urandom(1000)  # Random bytes don't compress well
        
        result = compression_engine.compress(random_data, CompressionAlgorithm.ZLIB)
        
        # Random data typically doesn't compress much, but algorithm still succeeds
        assert result.compressed_data is not None
        # Compressed size might be similar or even larger than original
        assert result.original_size == 1000


# ===== CONTENT ADDRESSING TESTS =====

@pytest.mark.unit
class TestContentAddressing:
    """Test content addressing functionality."""
    
    def test_compute_content_id(self, content_addressing_engine, sample_data):
        """Test content ID generation."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        assert isinstance(content_id, ContentID)
        assert content_id.hash_algorithm == "sha256"
        assert len(content_id.hash_value) > 0
    
    def test_content_id_deterministic(self, content_addressing_engine, sample_data):
        """Test that same data produces same content ID."""
        id1 = content_addressing_engine.compute_content_id(sample_data)
        id2 = content_addressing_engine.compute_content_id(sample_data)
        
        assert id1.hex == id2.hex
    
    def test_different_data_different_ids(self, content_addressing_engine):
        """Test that different data produces different IDs."""
        data1 = b"Content A"
        data2 = b"Content B"
        
        id1 = content_addressing_engine.compute_content_id(data1)
        id2 = content_addressing_engine.compute_content_id(data2)
        
        assert id1.hex != id2.hex
    
    def test_verify_content(self, content_addressing_engine, sample_data):
        """Test content ID verification."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        is_valid = content_addressing_engine.verify_content(sample_data, content_id)
        assert is_valid is True
    
    def test_verify_content_invalid(self, content_addressing_engine, sample_data):
        """Test content ID verification with wrong data."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        wrong_data = b"Wrong data"
        
        is_valid = content_addressing_engine.verify_content(wrong_data, content_id)
        assert is_valid is False


# ===== DEDUPLICATION TESTS =====

@pytest.mark.unit
class TestDeduplication:
    """Test deduplication functionality."""
    
    def test_check_exists_first_time(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test checking existence for first-time data."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        exists = deduplication_engine.check_exists(content_id)
        assert exists is False
    
    def test_add_reference_new_content(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test adding reference to new content."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        is_unique = deduplication_engine.add_reference(content_id, len(sample_data))
        
        assert is_unique is True
        assert deduplication_engine.check_exists(content_id) is True
    
    def test_add_reference_duplicate_content(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test adding reference to duplicate content."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        # First add - should be unique
        is_unique1 = deduplication_engine.add_reference(content_id, len(sample_data))
        assert is_unique1 is True
        
        # Second add - should be duplicate
        is_unique2 = deduplication_engine.add_reference(content_id, len(sample_data))
        assert is_unique2 is False
    
    def test_get_reference_count(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test getting reference count."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        # Add multiple references
        deduplication_engine.add_reference(content_id, len(sample_data))
        deduplication_engine.add_reference(content_id, len(sample_data))
        deduplication_engine.add_reference(content_id, len(sample_data))
        
        count = deduplication_engine.get_reference_count(content_id)
        assert count == 3
    
    def test_remove_reference(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test removing reference."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        # Add references
        deduplication_engine.add_reference(content_id, len(sample_data))
        deduplication_engine.add_reference(content_id, len(sample_data))
        
        # Remove one reference
        can_delete = deduplication_engine.remove_reference(content_id)
        assert can_delete is False  # Still has 1 reference
        
        # Remove last reference
        can_delete = deduplication_engine.remove_reference(content_id)
        assert can_delete is True  # Can delete now
    
    def test_deduplication_stats(self, deduplication_engine, content_addressing_engine, sample_data):
        """Test deduplication statistics."""
        content_id = content_addressing_engine.compute_content_id(sample_data)
        
        # Add same content 3 times
        deduplication_engine.add_reference(content_id, len(sample_data))
        deduplication_engine.add_reference(content_id, len(sample_data))
        deduplication_engine.add_reference(content_id, len(sample_data))
        
        stats = deduplication_engine.get_stats()
        assert stats.total_stores == 3
        assert stats.unique_content == 1
        assert stats.duplicate_saves == 2
        assert stats.bytes_saved == len(sample_data) * 2


# ===== STORAGE ENGINE TESTS =====

@pytest.mark.unit
class TestStorageEngine:
    """Test main storage engine functionality."""
    
    def test_storage_engine_initialization(self, temp_storage_dir):
        """Test storage engine initialization."""
        engine = PakitStorageEngine(storage_dir=temp_storage_dir)
        
        assert engine is not None
        assert engine.storage_dir == temp_storage_dir
    
    def test_store_data(self, storage_engine, sample_data):
        """Test storing data."""
        content_id = storage_engine.store(sample_data)
        
        assert content_id is not None
        assert isinstance(content_id, ContentID)
        assert len(content_id.hex) > 0
    
    def test_retrieve_data(self, storage_engine, sample_data):
        """Test retrieving stored data."""
        content_id = storage_engine.store(sample_data)
        retrieved_data = storage_engine.retrieve(content_id)
        
        assert retrieved_data == sample_data
    
    def test_store_and_retrieve_multiple(self, storage_engine):
        """Test storing and retrieving multiple items."""
        data_items = [
            b"Data item 1",
            b"Data item 2",
            b"Data item 3"
        ]
        
        content_ids = [storage_engine.store(data) for data in data_items]
        
        for content_id, original_data in zip(content_ids, data_items):
            retrieved = storage_engine.retrieve(content_id)
            assert retrieved == original_data
    
    def test_deduplication_in_storage(self, storage_engine, duplicate_data):
        """Test that duplicate data is deduplicated."""
        content_ids = [storage_engine.store(data) for data in duplicate_data]
        
        # All duplicates should have the same content ID
        assert content_ids[0] == content_ids[1] == content_ids[2]
    
    def test_get_metadata(self, storage_engine, sample_data):
        """Test retrieving storage metadata."""
        content_id = storage_engine.store(sample_data)
        metadata = storage_engine.get_metadata(content_id)
        
        assert metadata is not None
        assert metadata.original_size == len(sample_data)
        assert metadata.compression_ratio > 0
    
    def test_compression_in_storage(self, storage_engine, sample_data):
        """Test that data is compressed when stored."""
        content_id = storage_engine.store(sample_data)
        metadata = storage_engine.get_metadata(content_id)
        
        # Compressed size should be less than original
        assert metadata.compressed_size < metadata.original_size
    
    def test_storage_stats(self, storage_engine, sample_data, large_data):
        """Test storage statistics calculation."""
        storage_engine.store(sample_data)
        storage_engine.store(large_data)
        
        stats = storage_engine.get_stats()
        
        assert stats.total_items >= 2
        assert stats.total_original_bytes > 0
        assert stats.total_compressed_bytes > 0
        assert stats.overall_efficiency > 0
    
    def test_delete_content(self, storage_engine, sample_data):
        """Test deleting content."""
        content_id = storage_engine.store(sample_data)
        
        result = storage_engine.delete(content_id)
        assert result is True
        
        # After delete, metadata should be None
        metadata = storage_engine.get_metadata(content_id)
        assert metadata is None
    
    def test_content_exists_check(self, storage_engine, sample_data):
        """Test checking if content exists via metadata."""
        content_id = storage_engine.store(sample_data)
        
        # Should have metadata
        metadata = storage_engine.get_metadata(content_id)
        assert metadata is not None
        
        # Delete and check again
        storage_engine.delete(content_id)
        metadata = storage_engine.get_metadata(content_id)
        assert metadata is None


# ===== INTEGRATION TESTS =====

@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow(self, storage_engine, sample_data):
        """Test complete store-retrieve-delete workflow."""
        # Store
        content_id = storage_engine.store(sample_data)
        assert content_id is not None
        
        # Verify has metadata
        metadata = storage_engine.get_metadata(content_id)
        assert metadata is not None
        
        # Retrieve
        retrieved = storage_engine.retrieve(content_id)
        assert retrieved == sample_data
        
        # Get metadata
        metadata = storage_engine.get_metadata(content_id)
        assert metadata.original_size == len(sample_data)
        
        # Delete
        storage_engine.delete(content_id)
        metadata = storage_engine.get_metadata(content_id)
        assert metadata is None
    
    def test_multiple_stores_with_dedup(self, storage_engine, duplicate_data):
        """Test multiple stores with automatic deduplication."""
        content_ids = [storage_engine.store(data) for data in duplicate_data]
        
        # All duplicates should have the same content ID (same hex)
        hex_ids = [cid.hex for cid in content_ids]
        assert hex_ids[0] == hex_ids[1] == hex_ids[2]
        
        # Stats should reflect deduplication savings
        stats = storage_engine.get_stats()
        assert stats.total_deduplication_saves > 0
    
    def test_compression_efficiency(self, storage_engine, large_data):
        """Test compression efficiency on large data."""
        content_id = storage_engine.store(large_data)
        metadata = storage_engine.get_metadata(content_id)
        
        # Should achieve significant compression
        assert metadata.compression_ratio > 0.5  # >50% compression
        
        # Data should be retrievable
        retrieved = storage_engine.retrieve(content_id)
        assert retrieved == large_data


# ===== PERFORMANCE TESTS =====

@pytest.mark.slow
class TestPerformance:
    """Performance tests for storage operations."""
    
    def test_store_performance(self, storage_engine):
        """Test storage performance with many items."""
        import time
        
        num_items = 100
        data_items = [f"Data item {i}".encode() * 10 for i in range(num_items)]
        
        start_time = time.time()
        for data in data_items:
            storage_engine.store(data)
        end_time = time.time()
        
        elapsed = end_time - start_time
        ops_per_second = num_items / elapsed
        
        # Should be reasonably fast
        assert ops_per_second > 10  # At least 10 ops/second
    
    def test_retrieve_performance(self, storage_engine, sample_data):
        """Test retrieval performance."""
        import time
        
        # Store data
        content_id = storage_engine.store(sample_data)
        
        # Retrieve multiple times
        num_retrievals = 100
        start_time = time.time()
        for _ in range(num_retrievals):
            storage_engine.retrieve(content_id)
        end_time = time.time()
        
        elapsed = end_time - start_time
        ops_per_second = num_retrievals / elapsed
        
        # Retrieval should be fast
        assert ops_per_second > 50  # At least 50 retrievals/second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
