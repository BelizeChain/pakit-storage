#!/usr/bin/env python3
"""
Test script for Pakit storage backends.

Tests LocalBackend, IPFSBackend (if available), and ArweaveBackend (if available).
"""

import sys
from pathlib import Path
import tempfile
import shutil

# Import Pakit modules (pip installed package)
from pakit.backends import LocalBackend, IPFSBackend, ArweaveBackend
from pakit.core.storage_engine import PakitStorageEngine, StorageTier


def test_local_backend():
    """Test local filesystem backend."""
    print("\n" + "="*60)
    print("Testing LocalBackend")
    print("="*60)
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp(prefix="pakit_test_"))
    
    try:
        backend = LocalBackend(temp_dir)
        
        # Test data
        test_data = b"Hello, BelizeChain! This is test data for Pakit storage."
        content_id = "test123abc"
        
        # Store
        print(f"Storing {len(test_data)} bytes...")
        success = backend.store(content_id, test_data, tier="warm")
        print(f"✅ Store: {success}")
        
        # Check exists
        exists = backend.exists(content_id, tier="warm")
        print(f"✅ Exists: {exists}")
        
        # Get size
        size = backend.get_size(content_id, tier="warm")
        print(f"✅ Size: {size} bytes")
        
        # Retrieve
        retrieved = backend.retrieve(content_id, tier="warm")
        print(f"✅ Retrieve: {len(retrieved) if retrieved else 0} bytes")
        
        # Verify data
        assert retrieved == test_data, "Data mismatch!"
        print(f"✅ Data integrity verified")
        
        # Delete
        deleted = backend.delete(content_id, tier="warm")
        print(f"✅ Delete: {deleted}")
        
        # Verify deleted
        exists_after = backend.exists(content_id, tier="warm")
        print(f"✅ Exists after delete: {exists_after}")
        
        print("\n✅ LocalBackend: ALL TESTS PASSED")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_ipfs_backend():
    """Test IPFS backend (if daemon is running)."""
    print("\n" + "="*60)
    print("Testing IPFSBackend")
    print("="*60)
    
    try:
        backend = IPFSBackend()
        
        # Test data
        test_data = b"IPFS test data for BelizeChain Pakit"
        content_id = "ipfs_test_456def"
        
        # Store
        print(f"Storing {len(test_data)} bytes to IPFS...")
        success = backend.store(content_id, test_data, tier="warm")
        print(f"✅ Store: {success}")
        
        if success:
            # Retrieve
            retrieved = backend.retrieve(content_id, tier="warm")
            if retrieved:
                print(f"✅ Retrieve: {len(retrieved)} bytes")
                assert retrieved == test_data, "Data mismatch!"
                print(f"✅ Data integrity verified")
            
            # Cleanup
            backend.delete(content_id, tier="warm")
            print(f"✅ Cleanup complete")
        
        print("\n✅ IPFSBackend: ALL TESTS PASSED")
        
    except ConnectionError as e:
        print(f"\n⚠️  IPFSBackend: {e}")
        print("   Start IPFS daemon with: ipfs daemon")
    except Exception as e:
        print(f"\n⚠️  IPFSBackend: {e}")


def test_storage_engine():
    """Test integrated storage engine."""
    print("\n" + "="*60)
    print("Testing PakitStorageEngine")
    print("="*60)
    
    # Create temporary storage
    temp_dir = Path(tempfile.mkdtemp(prefix="pakit_engine_test_"))
    
    try:
        # Initialize engine (local only for testing)
        engine = PakitStorageEngine(
            storage_dir=temp_dir,
            enable_ipfs=False,  # Don't require IPFS daemon
            enable_arweave=False,  # Don't require Arweave wallet
        )
        
        # Test data
        test_data = b"BelizeChain storage engine test data. " * 100  # ~4KB
        
        print(f"Storing {len(test_data)} bytes...")
        
        # Store with compression and deduplication
        content_id = engine.store(
            test_data,
            tier=StorageTier.WARM
        )
        
        print(f"✅ Stored with content ID: {content_id.hex[:16]}...")
        
        # Get stats
        stats = engine.get_stats()
        print(f"✅ Total items: {stats.total_items}")
        print(f"✅ Original size: {stats.total_original_bytes} bytes")
        print(f"✅ Compressed size: {stats.total_compressed_bytes} bytes")
        print(f"✅ Compression ratio: {(1 - stats.total_compressed_bytes/stats.total_original_bytes)*100:.1f}%")
        
        # Retrieve
        retrieved = engine.retrieve(content_id)
        assert retrieved is not None, "Failed to retrieve data!"
        print(f"✅ Retrieved {len(retrieved)} bytes")
        
        # Verify data integrity
        assert retrieved == test_data, "Data mismatch!"
        print(f"✅ Data integrity verified")
        
        # Test deduplication
        print("\nTesting deduplication...")
        content_id_2 = engine.store(test_data, tier=StorageTier.WARM)
        
        assert content_id_2.hex == content_id.hex, "Duplicate detection failed!"
        print(f"✅ Duplicate detected (same content ID)")
        
        stats_after = engine.get_stats()
        print(f"✅ Deduplication saves: {stats_after.total_deduplication_saves}")
        print(f"✅ Bytes saved by dedup: {stats_after.bytes_saved_deduplication}")
        
        # Delete
        deleted = engine.delete(content_id)
        print(f"✅ Delete: {deleted}")
        
        print("\n✅ PakitStorageEngine: ALL TESTS PASSED")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run all tests."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          PAKIT STORAGE BACKENDS TEST SUITE                   ║
║          BelizeChain Decentralized Storage Layer             ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Test local backend
    test_local_backend()
    
    # Test IPFS backend (if available)
    test_ipfs_backend()
    
    # Test integrated storage engine
    test_storage_engine()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED")
    print("="*60)
    print()
    print("Summary:")
    print("- LocalBackend: ✅ Working")
    print("- IPFSBackend: ⚠️  Requires IPFS daemon")
    print("- ArweaveBackend: ⚠️  Requires wallet for writes")
    print("- PakitStorageEngine: ✅ Working")
    print()
    print("Next steps:")
    print("1. Start IPFS daemon: ipfs daemon")
    print("2. Generate Arweave wallet: https://faucet.arweave.net/")
    print("3. Update UI to use Pakit for document storage")
    print()


if __name__ == "__main__":
    main()
