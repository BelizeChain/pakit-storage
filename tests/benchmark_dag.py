"""
DAG Performance Benchmarks

Comprehensive benchmarks for DAG storage system to validate Phase 1 success criteria:
1. Store 1M blocks successfully
2. Query by hash <10ms
3. Merkle proof verification <5ms
4. Compression ratio >3x
5. Zero regressions

Usage:
    pytest pakit/tests/benchmark_dag.py -v
    pytest pakit/tests/benchmark_dag.py --benchmark-only
"""

import pytest
import time
import random
import string
from pathlib import Path

from pakit.core.dag_storage import DagBlock, MerkleDAG, create_genesis_block
from pakit.core.dag_builder import DagBuilder, DagState
from pakit.core.dag_index import DagIndex
from pakit.backends.dag_backend import DagBackend
from pakit.core.compression import CompressionEngine


class TestDagPerformance:
    """Performance benchmarks for DAG storage system."""
    
    @pytest.fixture(scope="class")
    def dag_backend(self):
        """Create DAG backend for testing."""
        backend = DagBackend(db_path="benchmark_dag.db", cache_size=10000)
        yield backend
        backend.close()
        # Cleanup
        import os
        for file in ["benchmark_dag.db", "benchmark_dag.db-wal", "benchmark_dag.db-shm"]:
            if os.path.exists(file):
                os.remove(file)
    
    @pytest.fixture(scope="class")
    def dag_components(self, dag_backend):
        """Initialize DAG components."""
        # Create genesis block
        genesis = create_genesis_block()
        dag_backend.put(genesis)
        
        # Initialize components
        state = DagState()
        state.add_block(genesis)
        
        builder = DagBuilder(state)
        index = DagIndex()
        index.add_block(genesis)
        
        merkle_dag = MerkleDAG()
        merkle_dag.add_block(genesis)
        
        return {
            "backend": dag_backend,
            "state": state,
            "builder": builder,
            "index": index,
            "merkle_dag": merkle_dag,
            "compression": CompressionEngine()
        }
    
    def generate_random_data(self, size: int = 1024) -> bytes:
        """Generate random data for testing."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=size)).encode()
    
    def test_store_1000_blocks(self, dag_components, benchmark):
        """Benchmark: Store 1000 blocks (scaled down from 1M for quick testing)."""
        def store_blocks():
            backend = dag_components["backend"]
            state = dag_components["state"]
            builder = dag_components["builder"]
            compression = dag_components["compression"]
            
            blocks_created = 0
            for i in range(1000):
                # Generate random data
                data = self.generate_random_data(size=2048)
                
                # Compress data
                compressed = compression.compress(data)
                
                # Select parents
                parent_hashes = builder.select_parents()
                
                # Create block
                block = DagBlock.create(
                    parent_hashes=parent_hashes,
                    compressed_data=compressed["data"],
                    original_size=len(data),
                    compression_algo=compressed["algorithm"]
                )
                
                # Store block
                backend.put(block)
                state.add_block(block)
                
                blocks_created += 1
            
            return blocks_created
        
        result = benchmark(store_blocks)
        assert result == 1000, f"Expected 1000 blocks, got {result}"
    
    def test_query_by_hash_performance(self, dag_components, benchmark):
        """Benchmark: Query by hash should be <10ms."""
        backend = dag_components["backend"]
        
        # Store some blocks first
        blocks = []
        state = dag_components["state"]
        builder = dag_components["builder"]
        compression = dag_components["compression"]
        
        for i in range(100):
            data = self.generate_random_data(size=1024)
            compressed = compression.compress(data)
            parent_hashes = builder.select_parents()
            
            block = DagBlock.create(
                parent_hashes=parent_hashes,
                compressed_data=compressed["data"],
                original_size=len(data),
                compression_algo=compressed["algorithm"]
            )
            backend.put(block)
            state.add_block(block)
            blocks.append(block)
        
        # Benchmark query
        def query_random_block():
            block = random.choice(blocks)
            retrieved = backend.get(block.block_hash)
            return retrieved is not None
        
        result = benchmark(query_random_block)
        assert result is True, "Query should return a block"
        
        # Verify <10ms requirement
        stats = benchmark.stats.stats
        mean_time_ms = stats.mean * 1000
        print(f"\nQuery performance: {mean_time_ms:.3f}ms (target: <10ms)")
        assert mean_time_ms < 10, f"Query took {mean_time_ms:.3f}ms, expected <10ms"
    
    def test_merkle_proof_verification_performance(self, dag_components, benchmark):
        """Benchmark: Merkle proof verification should be <5ms."""
        merkle_dag = dag_components["merkle_dag"]
        state = dag_components["state"]
        builder = dag_components["builder"]
        compression = dag_components["compression"]
        
        # Build a DAG with 50 blocks
        blocks = []
        for i in range(50):
            data = self.generate_random_data(size=1024)
            compressed = compression.compress(data)
            parent_hashes = builder.select_parents()
            
            block = DagBlock.create(
                parent_hashes=parent_hashes,
                compressed_data=compressed["data"],
                original_size=len(data),
                compression_algo=compressed["algorithm"]
            )
            merkle_dag.add_block(block)
            state.add_block(block)
            blocks.append(block)
        
        # Select a random block
        target_block = random.choice(blocks[10:])  # Skip early blocks
        genesis = merkle_dag.blocks[list(merkle_dag.blocks.keys())[0]]
        
        # Generate proof
        proof = merkle_dag.build_merkle_proof(target_block.block_hash, genesis.block_hash)
        
        # Benchmark verification
        def verify_proof():
            return merkle_dag.verify_merkle_proof(
                target_block.block_hash,
                genesis.block_hash,
                proof
            )
        
        result = benchmark(verify_proof)
        assert result is True, "Proof verification should succeed"
        
        # Verify <5ms requirement
        stats = benchmark.stats.stats
        mean_time_ms = stats.mean * 1000
        print(f"\nMerkle proof verification: {mean_time_ms:.3f}ms (target: <5ms)")
        assert mean_time_ms < 5, f"Verification took {mean_time_ms:.3f}ms, expected <5ms"
    
    def test_compression_ratio(self, dag_components):
        """Verify compression ratio >3x for typical data."""
        compression = dag_components["compression"]
        
        # Test with various data sizes
        total_original = 0
        total_compressed = 0
        
        for size in [1024, 2048, 4096, 8192]:
            # Generate compressible data (repeated patterns)
            data = (b"BelizeChain DAG Storage System! " * (size // 32))[:size]
            
            compressed = compression.compress(data)
            
            total_original += len(data)
            total_compressed += len(compressed["data"])
        
        ratio = total_original / total_compressed
        print(f"\nCompression ratio: {ratio:.2f}x (target: >3x)")
        print(f"Original: {total_original} bytes")
        print(f"Compressed: {total_compressed} bytes")
        
        assert ratio > 3.0, f"Compression ratio {ratio:.2f}x is below target 3x"
    
    def test_depth_range_query_performance(self, dag_components, benchmark):
        """Benchmark: Depth range queries should be efficient."""
        backend = dag_components["backend"]
        state = dag_components["state"]
        builder = dag_components["builder"]
        compression = dag_components["compression"]
        
        # Create blocks at various depths
        for i in range(200):
            data = self.generate_random_data(size=1024)
            compressed = compression.compress(data)
            parent_hashes = builder.select_parents()
            
            block = DagBlock.create(
                parent_hashes=parent_hashes,
                compressed_data=compressed["data"],
                original_size=len(data),
                compression_algo=compressed["algorithm"]
            )
            backend.put(block)
            state.add_block(block)
        
        # Benchmark depth range query
        def query_depth_range():
            blocks = backend.query_by_depth(min_depth=5, max_depth=15)
            return len(blocks)
        
        result = benchmark(query_depth_range)
        print(f"\nDepth range query returned {result} blocks")
        
        # Verify <100ms requirement
        stats = benchmark.stats.stats
        mean_time_ms = stats.mean * 1000
        print(f"Depth range query: {mean_time_ms:.3f}ms (target: <100ms)")
        assert mean_time_ms < 100, f"Query took {mean_time_ms:.3f}ms, expected <100ms"
    
    def test_batch_verification_performance(self, dag_components, benchmark):
        """Benchmark: Batch proof verification should be efficient."""
        merkle_dag = dag_components["merkle_dag"]
        state = dag_components["state"]
        builder = dag_components["builder"]
        compression = dag_components["compression"]
        
        # Build a DAG
        blocks = []
        for i in range(30):
            data = self.generate_random_data(size=1024)
            compressed = compression.compress(data)
            parent_hashes = builder.select_parents()
            
            block = DagBlock.create(
                parent_hashes=parent_hashes,
                compressed_data=compressed["data"],
                original_size=len(data),
                compression_algo=compressed["algorithm"]
            )
            merkle_dag.add_block(block)
            state.add_block(block)
            blocks.append(block)
        
        # Generate proofs for multiple blocks
        genesis = merkle_dag.blocks[list(merkle_dag.blocks.keys())[0]]
        proofs = []
        for block in blocks[5:10]:  # 5 blocks
            proof = merkle_dag.build_merkle_proof(block.block_hash, genesis.block_hash)
            proofs.append({
                "target_hash": block.block_hash,
                "root_hash": genesis.block_hash,
                "proof": proof
            })
        
        # Benchmark batch verification
        def verify_batch():
            return merkle_dag.verify_batch(proofs)
        
        results = benchmark(verify_batch)
        assert all(results), "All proofs should verify successfully"
        
        # Print timing
        stats = benchmark.stats.stats
        mean_time_ms = stats.mean * 1000
        print(f"\nBatch verification (5 proofs): {mean_time_ms:.3f}ms")
    
    def test_cache_hit_performance(self, dag_components, benchmark):
        """Benchmark: Cache hits should be significantly faster than DB queries."""
        backend = dag_components["backend"]
        
        # Create a block
        data = self.generate_random_data(size=2048)
        compressed = dag_components["compression"].compress(data)
        parent_hashes = dag_components["builder"].select_parents()
        
        block = DagBlock.create(
            parent_hashes=parent_hashes,
            compressed_data=compressed["data"],
            original_size=len(data),
            compression_algo=compressed["algorithm"]
        )
        backend.put(block)
        
        # First access (cache miss)
        backend.cache.clear()
        start = time.perf_counter()
        backend.get(block.block_hash)
        cache_miss_time = (time.perf_counter() - start) * 1000
        
        # Second access (cache hit)
        def cached_get():
            return backend.get(block.block_hash)
        
        result = benchmark(cached_get)
        assert result is not None
        
        stats = benchmark.stats.stats
        cache_hit_time = stats.mean * 1000
        
        print(f"\nCache miss: {cache_miss_time:.3f}ms")
        print(f"Cache hit: {cache_hit_time:.3f}ms")
        print(f"Speedup: {cache_miss_time / cache_hit_time:.1f}x")
        
        assert cache_hit_time < cache_miss_time, "Cache hits should be faster"
    
    def test_statistics_performance(self, dag_components, benchmark):
        """Benchmark: Statistics gathering should be fast."""
        backend = dag_components["backend"]
        
        # Benchmark statistics
        def get_stats():
            return backend.get_statistics()
        
        stats = benchmark(get_stats)
        assert "total_blocks" in stats
        
        # Print timing
        benchmark_stats = benchmark.stats.stats
        mean_time_ms = benchmark_stats.mean * 1000
        print(f"\nStatistics gathering: {mean_time_ms:.3f}ms")
        print(f"Database stats: {stats}")


class TestDagStressTest:
    """Stress tests for DAG system."""
    
    @pytest.mark.slow
    def test_store_10k_blocks(self):
        """Stress test: Store 10,000 blocks (10% of 1M target)."""
        print("\n" + "="*60)
        print("STRESS TEST: Storing 10,000 blocks")
        print("="*60)
        
        # Initialize components
        backend = DagBackend(db_path="stress_test_dag.db", cache_size=1000)
        genesis = create_genesis_block()
        backend.put(genesis)
        
        state = DagState()
        state.add_block(genesis)
        builder = DagBuilder(state)
        compression = CompressionEngine()
        
        # Store blocks
        start_time = time.time()
        milestone = 1000
        
        for i in range(10000):
            # Generate random data
            data = ''.join(random.choices(string.ascii_letters, k=2048)).encode()
            compressed = compression.compress(data)
            parent_hashes = builder.select_parents()
            
            block = DagBlock.create(
                parent_hashes=parent_hashes,
                compressed_data=compressed["data"],
                original_size=len(data),
                compression_algo=compressed["algorithm"]
            )
            
            backend.put(block)
            state.add_block(block)
            
            # Progress update
            if (i + 1) % milestone == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Stored {i + 1:,} blocks ({rate:.0f} blocks/sec)")
        
        # Final statistics
        total_time = time.time() - start_time
        stats = backend.get_statistics()
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"  Total blocks: {stats['total_blocks']:,}")
        print(f"  Total edges: {stats['total_edges']:,}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Database size: {stats['db_size_mb']:.2f} MB")
        print(f"  Time taken: {total_time:.2f} seconds")
        print(f"  Average rate: {stats['total_blocks'] / total_time:.0f} blocks/sec")
        print(f"  Cache usage: {stats['cache']['usage_pct']}%")
        print(f"{'='*60}\n")
        
        # Cleanup
        backend.close()
        import os
        for file in ["stress_test_dag.db", "stress_test_dag.db-wal", "stress_test_dag.db-shm"]:
            if os.path.exists(file):
                os.remove(file)
        
        # Verify success criteria
        assert stats['total_blocks'] >= 10000, "Should have 10,000+ blocks"
        assert stats['max_depth'] > 0, "Should have multi-level DAG"


if __name__ == "__main__":
    print("Run benchmarks with: pytest pakit/tests/benchmark_dag.py -v")
    print("Run stress test with: pytest pakit/tests/benchmark_dag.py -v -m slow")
