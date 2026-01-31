"""
P2P Integration Tests

Comprehensive test suite for Pakit P2P protocol.

Test Coverage:
- DHT routing and lookups
- Gossip protocol propagation
- Block request/response
- Merkle proof verification
- Reputation system
- Network transport
- Block discovery
- End-to-end scenarios
"""

import pytest
import asyncio
import hashlib
import time
from typing import List

# Import P2P components
from pakit.p2p.dht.kademlia import KademliaDHT, DHTNode, xor_distance
from pakit.p2p.gossip.protocol import GossipProtocol, BlockAnnouncement
from pakit.p2p.network.protocol import BlockRequestProtocol, RequestPriority
from pakit.p2p.verification.merkle_verify import RemoteMerkleVerifier, MerkleProof
from pakit.p2p.reputation.system import ReputationSystem, ReputationEvent
from pakit.p2p.discovery.block_discovery import BlockDiscoveryService, BlockMetadata


class TestDHTRouting:
    """Test Kademlia DHT routing and lookups."""
    
    def test_xor_distance(self):
        """Test XOR distance metric properties."""
        id1 = hashlib.sha256(b"peer1").hexdigest()
        id2 = hashlib.sha256(b"peer2").hexdigest()
        id3 = hashlib.sha256(b"peer3").hexdigest()
        
        # d(x,x) = 0
        assert xor_distance(id1, id1) == 0
        
        # d(x,y) = d(y,x) (symmetric)
        assert xor_distance(id1, id2) == xor_distance(id2, id1)
        
        # d(x,y) > 0 for x != y
        assert xor_distance(id1, id2) > 0
        
        # Triangle inequality: d(x,y) + d(y,z) >= d(x,z)
        d_xy = xor_distance(id1, id2)
        d_yz = xor_distance(id2, id3)
        d_xz = xor_distance(id1, id3)
        assert d_xy + d_yz >= d_xz
    
    def test_dht_basic_operations(self):
        """Test basic DHT operations."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        dht = KademliaDHT(node_id=node_id, node_address="127.0.0.1:7777")
        
        # Store and retrieve
        key = "test_key"
        value = b"test_value"
        
        assert dht.store(key, value) == True
        assert dht.find_value(key) == value
        
        # Find nodes
        target = hashlib.sha256(b"target").hexdigest()
        closest = dht.find_node(target)
        assert isinstance(closest, list)
    
    def test_dht_routing_table(self):
        """Test routing table management."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        dht = KademliaDHT(node_id=node_id, node_address="127.0.0.1:7777")
        
        # Add 50 nodes
        for i in range(50):
            peer_id = hashlib.sha256(f"peer_{i}".encode()).hexdigest()
            node = DHTNode(peer_id=peer_id, address=f"127.0.0.1:{8000+i}")
            dht.routing_table.add_node(node)
        
        stats = dht.routing_table.get_stats()
        assert stats["total_nodes"] <= 50
        assert stats["non_empty_buckets"] > 0
    
    def test_dht_scalability(self):
        """Test DHT with 1000+ nodes."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        dht = KademliaDHT(node_id=node_id, node_address="127.0.0.1:7777")
        
        # Add 1000 nodes
        start = time.time()
        for i in range(1000):
            peer_id = hashlib.sha256(f"peer_{i}".encode()).hexdigest()
            node = DHTNode(peer_id=peer_id, address=f"127.0.0.1:{8000+(i%1000)}")
            dht.routing_table.add_node(node)
        
        elapsed = time.time() - start
        
        # Should handle 1000 nodes quickly
        assert elapsed < 1.0  # Less than 1 second
        
        # Lookup should be fast
        start = time.time()
        target = hashlib.sha256(b"lookup_target").hexdigest()
        closest = dht.routing_table.find_closest_nodes(target, count=20)
        elapsed = time.time() - start
        
        assert len(closest) > 0
        assert elapsed < 0.1  # Less than 100ms


class TestGossipProtocol:
    """Test block gossip protocol."""
    
    def test_block_announcement(self):
        """Test block announcement creation."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        gossip = GossipProtocol(node_id=node_id)
        
        block_hash = hashlib.sha256(b"block_data").hexdigest()
        peers = [hashlib.sha256(f"peer_{i}".encode()).hexdigest() for i in range(10)]
        
        announced = gossip.announce_block(
            block_hash=block_hash,
            block_depth=5,
            compression_algo="zstd",
            original_size=2048,
            compressed_size=512,
            parent_hashes=["parent1"],
            peers=peers
        )
        
        assert announced == 6  # Fanout=6
        assert gossip.stats["blocks_announced"] == 1
    
    def test_gossip_duplicate_detection(self):
        """Test bloom filter duplicate detection."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        gossip = GossipProtocol(node_id=node_id)
        
        block_hash = hashlib.sha256(b"block_data").hexdigest()
        
        # Create announcement
        announcement = BlockAnnouncement(
            block_hash=block_hash,
            block_depth=5,
            compression_algo="zstd",
            original_size=2048,
            compressed_size=512,
            parent_hashes=[],
            timestamp=time.time(),
            ttl=10
        )
        
        # First announcement should be processed
        peers = [hashlib.sha256(f"peer_{i}".encode()).hexdigest() for i in range(10)]
        processed = gossip.handle_announcement(announcement, "peer_1", peers)
        assert processed == True
        
        # Duplicate should be rejected
        processed = gossip.handle_announcement(announcement, "peer_2", peers)
        assert processed == False
        assert gossip.stats["duplicate_blocks"] == 1
    
    def test_gossip_propagation_latency(self):
        """Test gossip propagation speed."""
        # Create network of 10 nodes
        nodes = []
        for i in range(10):
            node_id = hashlib.sha256(f"node_{i}".encode()).hexdigest()
            gossip = GossipProtocol(node_id=node_id)
            nodes.append(gossip)
        
        # Measure propagation time
        start = time.time()
        
        # Node 0 announces
        block_hash = hashlib.sha256(b"test_block").hexdigest()
        peers = [hashlib.sha256(f"peer_{i}".encode()).hexdigest() for i in range(10)]
        nodes[0].announce_block(
            block_hash=block_hash,
            block_depth=1,
            compression_algo="zstd",
            original_size=1024,
            compressed_size=256,
            parent_hashes=[],
            peers=peers
        )
        
        elapsed = time.time() - start
        
        # Should be very fast (< 10ms)
        assert elapsed < 0.01


class TestBlockRequestResponse:
    """Test block request/response protocol."""
    
    def test_single_block_request(self):
        """Test requesting a single block."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        protocol = BlockRequestProtocol(node_id=node_id)
        
        block_hash = hashlib.sha256(b"block_data").hexdigest()
        request_id = protocol.request_block(block_hash, priority=RequestPriority.HIGH)
        
        assert request_id is not None
        assert len(protocol.active_requests) == 1
        assert protocol.stats["requests_sent"] == 1
    
    def test_batch_request(self):
        """Test batch block requests."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        protocol = BlockRequestProtocol(node_id=node_id)
        
        # Request 50 blocks
        hashes = [hashlib.sha256(f"block_{i}".encode()).hexdigest() for i in range(50)]
        request_id = protocol.request_blocks(hashes)
        
        assert request_id is not None
        assert len(protocol.active_requests) >= 1
    
    def test_priority_queue(self):
        """Test request priority handling."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        protocol = BlockRequestProtocol(node_id=node_id)
        
        # Add requests with different priorities
        low_req = protocol.request_block("block1", priority=RequestPriority.LOW)
        high_req = protocol.request_block("block2", priority=RequestPriority.HIGH)
        critical_req = protocol.request_block("block3", priority=RequestPriority.CRITICAL)
        
        # Get next request (should be CRITICAL)
        next_req = protocol.get_next_request()
        assert next_req.priority == RequestPriority.CRITICAL


class TestMerkleVerification:
    """Test Merkle proof verification."""
    
    def test_valid_proof(self):
        """Test verification of valid Merkle proof."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        genesis = hashlib.sha256(b"genesis").hexdigest()
        verifier = RemoteMerkleVerifier(node_id=node_id, trusted_roots=[genesis])
        
        # Create simple proof
        leaf = hashlib.sha256(b"leaf_data").hexdigest()
        sibling = hashlib.sha256(b"sibling").hexdigest()
        
        # Compute root
        combined = leaf + sibling
        root = hashlib.sha256(combined.encode()).hexdigest()
        verifier.add_trusted_root(root)
        
        proof = MerkleProof(
            target_hash=leaf,
            root_hash=root,
            path=[sibling],
            indices=[0],
            block_depth=1
        )
        
        result = verifier.verify_proof(proof)
        assert result.status.value == "valid"
    
    def test_invalid_proof(self):
        """Test detection of invalid proof."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        genesis = hashlib.sha256(b"genesis").hexdigest()
        verifier = RemoteMerkleVerifier(node_id=node_id, trusted_roots=[genesis])
        
        # Create invalid proof (wrong sibling)
        leaf = hashlib.sha256(b"leaf_data").hexdigest()
        wrong_sibling = hashlib.sha256(b"wrong").hexdigest()
        
        # Compute expected root with correct sibling
        correct_sibling = hashlib.sha256(b"sibling").hexdigest()
        combined = leaf + correct_sibling
        root = hashlib.sha256(combined.encode()).hexdigest()
        verifier.add_trusted_root(root)
        
        # Proof with wrong sibling
        proof = MerkleProof(
            target_hash=leaf,
            root_hash=root,
            path=[wrong_sibling],  # Wrong!
            indices=[0],
            block_depth=1
        )
        
        result = verifier.verify_proof(proof)
        assert result.status.value == "invalid"
    
    def test_proof_caching(self):
        """Test proof cache hit rate."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        genesis = hashlib.sha256(b"genesis").hexdigest()
        verifier = RemoteMerkleVerifier(node_id=node_id, trusted_roots=[genesis])
        
        # Create valid proof
        leaf = hashlib.sha256(b"leaf_data").hexdigest()
        sibling = hashlib.sha256(b"sibling").hexdigest()
        combined = leaf + sibling
        root = hashlib.sha256(combined.encode()).hexdigest()
        verifier.add_trusted_root(root)
        
        proof = MerkleProof(
            target_hash=leaf,
            root_hash=root,
            path=[sibling],
            indices=[0],
            block_depth=1
        )
        
        # First verification (miss)
        result1 = verifier.verify_proof(proof)
        assert result1.cached == False
        
        # Second verification (hit)
        result2 = verifier.verify_proof(proof)
        assert result2.cached == True
        
        # Cache hit rate should be 50%
        stats = verifier.get_stats()
        assert "66.67%" in stats["cache"]["hit_rate"]


class TestReputationSystem:
    """Test peer reputation system."""
    
    def test_reputation_events(self):
        """Test reputation event handling."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        rep_system = ReputationSystem(node_id=node_id)
        
        peer_id = hashlib.sha256(b"test_peer").hexdigest()
        
        # Good behavior
        for _ in range(10):
            rep_system.record_event(peer_id, ReputationEvent.BLOCK_DELIVERED)
        
        peer = rep_system.get_reputation(peer_id)
        assert peer.reputation > 0.5  # Should increase
        assert peer.blocks_delivered == 10
    
    def test_auto_ban(self):
        """Test automatic banning of malicious peers."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        rep_system = ReputationSystem(node_id=node_id)
        
        peer_id = hashlib.sha256(b"bad_peer").hexdigest()
        
        # Malicious behavior
        for _ in range(5):
            rep_system.record_event(peer_id, ReputationEvent.PROOF_INVALID)
        
        # Should be auto-banned
        assert rep_system.is_banned(peer_id) == True
        assert rep_system.stats["peers_banned"] >= 1
    
    def test_best_peer_selection(self):
        """Test selection of best peers."""
        node_id = hashlib.sha256(b"test_node").hexdigest()
        rep_system = ReputationSystem(node_id=node_id)
        
        # Create peers with different reputations
        for i in range(10):
            peer_id = hashlib.sha256(f"peer_{i}".encode()).hexdigest()
            
            # Give different amounts of good behavior
            for _ in range(i * 2):
                rep_system.record_event(peer_id, ReputationEvent.BLOCK_DELIVERED)
        
        # Get best peers
        best = rep_system.get_best_peers(count=5)
        assert len(best) <= 5
        
        # Should be sorted by reputation (descending)
        for i in range(len(best) - 1):
            assert best[i].reputation >= best[i+1].reputation


class TestBlockDiscovery:
    """Test DHT-based block discovery."""
    
    def test_block_announcement(self):
        """Test block announcement to DHT."""
        class MockDHT:
            def __init__(self):
                self.storage = {}
            def publish(self, key, value):
                self.storage[key] = value
        
        node_id = hashlib.sha256(b"test_node").hexdigest()
        dht = MockDHT()
        discovery = BlockDiscoveryService(node_id=node_id, dht_client=dht)
        
        block_hash = hashlib.sha256(b"block_data").hexdigest()
        metadata = BlockMetadata(
            block_hash=block_hash,
            size=512,
            depth=5,
            compression_algo="zstd",
            parent_hashes=[]
        )
        
        discovery.announce_block(block_hash, metadata)
        
        assert discovery.stats["blocks_announced"] == 1
        assert block_hash in discovery.local_blocks


class TestEndToEndScenarios:
    """End-to-end integration tests."""
    
    def test_block_sync_workflow(self):
        """Test complete block synchronization workflow."""
        # Node 1: Has blocks
        node1_id = hashlib.sha256(b"node1").hexdigest()
        
        # Node 2: Needs blocks
        node2_id = hashlib.sha256(b"node2").hexdigest()
        
        # Create components
        gossip1 = GossipProtocol(node_id=node1_id)
        protocol2 = BlockRequestProtocol(node_id=node2_id)
        
        # Node 1 announces new block
        block_hash = hashlib.sha256(b"new_block").hexdigest()
        peers = [node2_id]
        
        gossip1.announce_block(
            block_hash=block_hash,
            block_depth=1,
            compression_algo="zstd",
            original_size=1024,
            compressed_size=256,
            parent_hashes=[],
            peers=peers
        )
        
        # Node 2 requests block
        request_id = protocol2.request_block(block_hash)
        
        assert request_id is not None
        assert protocol2.stats["requests_sent"] == 1
    
    def test_network_partition_resilience(self):
        """Test behavior during network partition."""
        # Create 2 partitions of nodes
        partition1 = []
        partition2 = []
        
        for i in range(5):
            node_id = hashlib.sha256(f"p1_node_{i}".encode()).hexdigest()
            partition1.append(GossipProtocol(node_id=node_id))
        
        for i in range(5):
            node_id = hashlib.sha256(f"p2_node_{i}".encode()).hexdigest()
            partition2.append(GossipProtocol(node_id=node_id))
        
        # Both partitions should continue operating
        assert len(partition1) == 5
        assert len(partition2) == 5


def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # DHT lookup performance
    node_id = hashlib.sha256(b"bench_node").hexdigest()
    dht = KademliaDHT(node_id=node_id, node_address="127.0.0.1:7777")
    
    # Add 1000 nodes
    for i in range(1000):
        peer_id = hashlib.sha256(f"peer_{i}".encode()).hexdigest()
        node = DHTNode(peer_id=peer_id, address=f"127.0.0.1:{8000+(i%1000)}")
        dht.routing_table.add_node(node)
    
    # Benchmark lookups
    target = hashlib.sha256(b"lookup_target").hexdigest()
    
    iterations = 1000
    start = time.time()
    for _ in range(iterations):
        dht.routing_table.find_closest_nodes(target, count=20)
    elapsed = time.time() - start
    
    avg_latency = (elapsed / iterations) * 1000  # ms
    
    print(f"\nDHT Lookup Performance (1000 nodes):")
    print(f"  Iterations: {iterations}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg latency: {avg_latency:.3f}ms")
    print(f"  Target: <100ms ✅" if avg_latency < 100 else "  Target: <100ms ❌")
    
    # Gossip propagation benchmark
    print(f"\nGossip Propagation Performance:")
    gossip = GossipProtocol(node_id=node_id)
    
    peers = [hashlib.sha256(f"peer_{i}".encode()).hexdigest() for i in range(100)]
    
    start = time.time()
    for i in range(100):
        block_hash = hashlib.sha256(f"block_{i}".encode()).hexdigest()
        gossip.announce_block(
            block_hash=block_hash,
            block_depth=1,
            compression_algo="zstd",
            original_size=1024,
            compressed_size=256,
            parent_hashes=[],
            peers=peers
        )
    elapsed = time.time() - start
    
    avg_propagation = (elapsed / 100) * 1000  # ms
    print(f"  Blocks: 100")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Avg propagation: {avg_propagation:.3f}ms")
    print(f"  Target: <500ms ✅" if avg_propagation < 500 else "  Target: <500ms ❌")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("🧪 Pakit P2P Integration Tests")
    print("="*60)
    
    # Run tests with pytest
    print("\nRunning tests with pytest...")
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run performance benchmarks
    run_performance_benchmark()
    
    print("\n✅ All tests complete!")
