#!/usr/bin/env python3
"""
Quick integration test for Pakit core functionality.

Tests compression, deduplication, and storage operations.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Import Pakit modules (pip installed package)
from pakit.core.compression import CompressionEngine, CompressionAlgorithm, CompressionResult
from pakit.core.content_addressing import ContentAddressingEngine, ContentID
from pakit.core.deduplication import DeduplicationEngine
from pakit.core.storage_engine import Pakit, StorageTier


def test_compression():
    """Test compression engine."""
    print("Testing compression engine...")
    
    engine = CompressionEngine()
    
    # Test data
    data = b"Hello, Belize! " * 1000
    
    # Test ZLIB
    result = engine.compress(data, CompressionAlgorithm.ZLIB)
    assert result.algorithm == CompressionAlgorithm.ZLIB
    assert result.compressed_size < result.original_size
    assert result.compression_ratio > 0.9
    
    # Test decompression
    decompressed = engine.decompress(result.compressed_data, CompressionAlgorithm.ZLIB)
    assert decompressed == data
    
    # Test AUTO mode
    auto_result = engine.compress(data, CompressionAlgorithm.AUTO)
    assert auto_result.efficiency_score > 0
    
    print(f"  ✅ Compression works (ratio: {auto_result.compression_ratio:.2%})")


def test_content_addressing():
    """Test content addressing."""
    print("Testing content addressing...")
    
    engine = ContentAddressingEngine()
    
    # Test content ID generation
    data1 = b"Test data 1"
    data2 = b"Test data 2"
    data3 = b"Test data 1"  # Same as data1
    
    cid1 = engine.compute_content_id(data1)
    cid2 = engine.compute_content_id(data2)
    cid3 = engine.compute_content_id(data3)
    
    # Same data should have same hash
    assert cid1.hex == cid3.hex
    
    # Different data should have different hash
    assert cid1.hex != cid2.hex
    
    # Test verification
    assert engine.verify_content(data1, cid1)
    assert not engine.verify_content(data2, cid1)
    
    # Test chunking
    large_data = b"X" * (512 * 1024)  # 512KB
    chunks = engine.chunk_data(large_data, chunk_size=256*1024)
    assert len(chunks) == 2
    
    print(f"  ✅ Content addressing works (CID: {cid1.hex[:32]}...)")


def test_deduplication():
    """Test deduplication engine."""
    print("Testing deduplication...")
    
    engine = DeduplicationEngine()
    
    data = b"Important data"
    size = len(data)
    
    # Generate content ID
    ca_engine = ContentAddressingEngine()
    cid = ca_engine.compute_content_id(data)
    
    # First store (unique)
    is_unique = engine.add_reference(cid, size)
    assert is_unique == True
    assert engine.get_reference_count(cid) == 1
    
    # Second store (duplicate)
    is_unique = engine.add_reference(cid, size)
    assert is_unique == False
    assert engine.get_reference_count(cid) == 2
    
    # Remove one reference
    can_delete = engine.remove_reference(cid)
    assert can_delete == False
    assert engine.get_reference_count(cid) == 1
    
    # Remove last reference
    can_delete = engine.remove_reference(cid)
    assert can_delete == True
    assert engine.get_reference_count(cid) == 0
    
    # Get stats
    stats = engine.get_stats()
    assert stats.duplicate_saves == 1
    assert stats.bytes_saved > 0
    
    print(f"  ✅ Deduplication works (saved {stats.bytes_saved} bytes)")


def test_storage_engine():
    """Test full storage engine."""
    print("Testing storage engine...")
    
    # Create temporary storage directory
    temp_dir = tempfile.mkdtemp(prefix="pakit_test_")
    
    try:
        # Initialize Pakit
        pakit = Pakit(
            storage_dir=Path(temp_dir),
            compression_algorithm=CompressionAlgorithm.AUTO,
            enable_deduplication=True,
        )
        
        # Test data
        data1 = b"Document 1: Important information. " * 100
        data2 = b"Document 2: Different content. " * 100
        data3 = b"Document 1: Important information. " * 100  # Duplicate of data1
        
        # Store first document
        cid1 = pakit.store(data1, tier=StorageTier.HOT)
        assert cid1 is not None
        
        # Store second document
        cid2 = pakit.store(data2, tier=StorageTier.WARM)
        assert cid2 is not None
        assert cid2.hex != cid1.hex
        
        # Store duplicate (should deduplicate)
        cid3 = pakit.store(data3)
        assert cid3.hex == cid1.hex  # Same content ID
        
        # Retrieve documents
        retrieved1 = pakit.retrieve(cid1)
        assert retrieved1 == data1
        
        retrieved2 = pakit.retrieve(cid2)
        assert retrieved2 == data2
        
        # Check stats
        stats = pakit.get_stats()
        assert stats.total_items == 2  # Only 2 unique items
        assert stats.total_deduplication_saves == 1  # 1 duplicate detected
        assert stats.efficiency_percent > 0
        
        # Get efficiency report
        report = pakit.get_efficiency_report()
        assert 'storage' in report
        assert 'savings' in report
        assert 'efficiency' in report
        assert 'data_farm_comparison' in report
        
        # Delete document (with reference counting)
        deleted = pakit.delete(cid1)
        assert deleted == False  # Still has reference from cid3
        
        deleted = pakit.delete(cid3)
        assert deleted == True  # Last reference removed
        
        # Should not be able to retrieve anymore
        retrieved = pakit.retrieve(cid1)
        assert retrieved is None
        
        print(f"  ✅ Storage engine works (efficiency: {stats.efficiency_percent:.2f}%)")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_efficiency_report():
    """Test efficiency reporting."""
    print("Testing efficiency reporting...")
    
    temp_dir = tempfile.mkdtemp(prefix="pakit_test_")
    
    try:
        pakit = Pakit(storage_dir=Path(temp_dir))
        
        # Store 10 files with 50% duplication
        unique_data = [
            b"File A: " + b"A" * 1000,
            b"File B: " + b"B" * 1000,
            b"File C: " + b"C" * 1000,
            b"File D: " + b"D" * 1000,
            b"File E: " + b"E" * 1000,
        ]
        
        for i in range(10):
            data = unique_data[i % 5]  # 50% duplication
            pakit.store(data)
        
        # Get report
        report = pakit.get_efficiency_report()
        
        # Verify report structure
        assert 'storage' in report
        assert report['storage']['total_items'] == 5
        assert report['storage']['dedup_saves'] == 5
        
        assert 'savings' in report
        assert report['savings']['deduplication_bytes'] > 0
        assert report['savings']['compression_bytes'] > 0
        
        assert 'efficiency' in report
        assert report['efficiency']['overall_percent'] > 0
        
        assert 'data_farm_comparison' in report
        multiplier = report['data_farm_comparison']['efficiency_multiplier']
        assert multiplier > 1.0  # Should be more efficient
        
        print(f"  ✅ Efficiency reporting works (multiplier: {multiplier:.2f}x)")
        
    finally:
        shutil.rmtree(temp_dir)


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("           PAKIT INTEGRATION TEST SUITE")
    print("=" * 80 + "\n")
    
    try:
        test_compression()
        test_content_addressing()
        test_deduplication()
        test_storage_engine()
        test_efficiency_report()
        
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
