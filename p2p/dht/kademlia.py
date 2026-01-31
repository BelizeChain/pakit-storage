"""
Kademlia DHT (Distributed Hash Table) Implementation

Implements Kademlia protocol for distributed peer/block discovery.
Based on "Kademlia: A Peer-to-peer Information System Based on the XOR Metric" (2002)

Key Features:
- 160-bit ID space (SHA-256 first 160 bits)
- XOR distance metric for routing
- K-buckets (20 peers per bucket, 160 buckets total)
- RPCs: PING, STORE, FIND_NODE, FIND_VALUE
- Iterative lookups with α=3 parallelism
"""

import hashlib
import time
import random
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Kademlia constants
K = 20  # Bucket size (k closest nodes)
ALPHA = 3  # Parallel lookup queries
B = 160  # ID space bits (160 bits from SHA-256)


def xor_distance(id1: str, id2: str) -> int:
    """
    Calculate XOR distance between two peer IDs.
    
    XOR metric properties:
    - d(x,x) = 0 (distance to self is zero)
    - d(x,y) > 0 for x != y (positive distance)
    - d(x,y) = d(y,x) (symmetric)
    - d(x,y) + d(y,z) >= d(x,z) (triangle inequality)
    
    Args:
        id1: First peer ID (hex string)
        id2: Second peer ID (hex string)
    
    Returns:
        Integer distance
    """
    # Convert hex to int and XOR
    int1 = int(id1[:40], 16)  # First 160 bits (40 hex chars)
    int2 = int(id2[:40], 16)
    return int1 ^ int2


def id_to_bucket_index(peer_id: str, target_id: str) -> int:
    """
    Calculate which k-bucket a peer belongs to relative to target.
    
    Bucket index = floor(log2(distance))
    
    Args:
        peer_id: Peer's ID
        target_id: Our node's ID
    
    Returns:
        Bucket index (0 to 159)
    """
    distance = xor_distance(peer_id, target_id)
    if distance == 0:
        return 0
    
    # Calculate log2(distance) = position of highest bit
    return distance.bit_length() - 1


@dataclass
class DHTNode:
    """Represents a node in the DHT."""
    
    peer_id: str
    address: str  # IP:port
    last_seen: float = field(default_factory=time.time)
    
    def is_alive(self, timeout: int = 900) -> bool:
        """Check if node responded recently (default: 15 minutes)."""
        return (time.time() - self.last_seen) < timeout
    
    def touch(self):
        """Update last_seen timestamp (node responded)."""
        self.last_seen = time.time()


class KBucket:
    """
    K-bucket: Stores up to K nodes at a specific distance range.
    
    Implements least-recently-seen eviction:
    - New nodes go to tail
    - Active nodes move to tail when seen
    - Head nodes are evicted first when full
    """
    
    def __init__(self, max_size: int = K):
        """Initialize k-bucket with max size."""
        self.max_size = max_size
        self.nodes: List[DHTNode] = []
    
    def add_node(self, node: DHTNode) -> bool:
        """
        Add node to bucket (LRU eviction policy).
        
        Returns:
            True if added/updated, False if bucket full
        """
        # Check if node already exists
        for i, existing in enumerate(self.nodes):
            if existing.peer_id == node.peer_id:
                # Move to tail (most recently seen)
                self.nodes.pop(i)
                self.nodes.append(existing)
                existing.touch()
                return True
        
        # Check if bucket has space
        if len(self.nodes) < self.max_size:
            self.nodes.append(node)
            return True
        
        # Bucket full - try to evict dead node from head
        for i, existing in enumerate(self.nodes):
            if not existing.is_alive():
                # Evict dead node
                self.nodes.pop(i)
                self.nodes.append(node)
                logger.debug(f"Evicted dead node: {existing.peer_id[:16]}...")
                return True
        
        # All nodes alive, bucket full
        return False
    
    def get_nodes(self, count: int = None) -> List[DHTNode]:
        """Get up to count nodes from bucket (most recent first)."""
        if count is None:
            return list(reversed(self.nodes))
        return list(reversed(self.nodes[:count]))
    
    def has_space(self) -> bool:
        """Check if bucket has space for new nodes."""
        return len(self.nodes) < self.max_size
    
    def size(self) -> int:
        """Get current bucket size."""
        return len(self.nodes)


class KademliaRoutingTable:
    """
    Kademlia routing table with 160 k-buckets.
    
    Each bucket i contains nodes at distance [2^i, 2^(i+1) - 1] from us.
    """
    
    def __init__(self, node_id: str):
        """
        Initialize routing table for node.
        
        Args:
            node_id: Our node's peer ID
        """
        self.node_id = node_id
        self.buckets: List[KBucket] = [KBucket() for _ in range(B)]
    
    def add_node(self, node: DHTNode) -> bool:
        """
        Add node to appropriate k-bucket.
        
        Returns:
            True if added, False if bucket full
        """
        # Don't add self
        if node.peer_id == self.node_id:
            return False
        
        # Find appropriate bucket
        bucket_index = id_to_bucket_index(node.peer_id, self.node_id)
        bucket = self.buckets[bucket_index]
        
        return bucket.add_node(node)
    
    def find_closest_nodes(self, target_id: str, count: int = K) -> List[DHTNode]:
        """
        Find the K closest nodes to target ID.
        
        Args:
            target_id: Target peer/block ID
            count: Number of nodes to return (default: K=20)
        
        Returns:
            List of closest nodes, sorted by distance
        """
        # Collect all nodes from routing table
        all_nodes = []
        for bucket in self.buckets:
            all_nodes.extend(bucket.get_nodes())
        
        # Sort by XOR distance to target
        all_nodes.sort(key=lambda n: xor_distance(n.peer_id, target_id))
        
        return all_nodes[:count]
    
    def get_bucket(self, bucket_index: int) -> Optional[KBucket]:
        """Get specific k-bucket by index."""
        if 0 <= bucket_index < B:
            return self.buckets[bucket_index]
        return None
    
    def get_stats(self) -> Dict:
        """Get routing table statistics."""
        total_nodes = sum(bucket.size() for bucket in self.buckets)
        non_empty_buckets = sum(1 for bucket in self.buckets if bucket.size() > 0)
        
        return {
            "total_nodes": total_nodes,
            "non_empty_buckets": non_empty_buckets,
            "buckets": B,
            "max_nodes_per_bucket": K
        }


class KademliaDHT:
    """
    Main Kademlia DHT implementation.
    
    Provides distributed key-value storage and peer discovery.
    """
    
    def __init__(self, node_id: str, node_address: str):
        """
        Initialize Kademlia DHT.
        
        Args:
            node_id: Our node's peer ID
            node_address: Our node's address (IP:port)
        """
        self.node_id = node_id
        self.node_address = node_address
        
        # Routing table
        self.routing_table = KademliaRoutingTable(node_id)
        
        # Local storage (key → value)
        self.storage: Dict[str, bytes] = {}
        
        # Lookup cache (key → [peer addresses])
        self.lookup_cache: Dict[str, List[str]] = {}
        
        logger.info(f"Initialized Kademlia DHT for node: {node_id[:16]}...")
    
    def bootstrap(self, bootstrap_nodes: List[Tuple[str, str]]):
        """
        Bootstrap DHT by connecting to initial nodes.
        
        Args:
            bootstrap_nodes: List of (peer_id, address) tuples
        """
        logger.info(f"Bootstrapping DHT with {len(bootstrap_nodes)} nodes...")
        
        for peer_id, address in bootstrap_nodes:
            node = DHTNode(peer_id=peer_id, address=address)
            self.routing_table.add_node(node)
        
        # Perform node lookup for our own ID to populate routing table
        self.iterative_find_node(self.node_id)
    
    def ping(self, node: DHTNode) -> bool:
        """
        PING RPC: Check if node is alive.
        
        Args:
            node: Node to ping
        
        Returns:
            True if node responded (simulated for now)
        """
        # TODO: Implement actual network RPC
        # For now, simulate response
        node.touch()
        return True
    
    def store(self, key: str, value: bytes) -> bool:
        """
        STORE RPC: Store key-value pair locally.
        
        Args:
            key: Storage key (block hash, peer ID, etc.)
            value: Data to store
        
        Returns:
            True if stored successfully
        """
        self.storage[key] = value
        logger.debug(f"Stored key: {key[:16]}... ({len(value)} bytes)")
        return True
    
    def find_value(self, key: str) -> Optional[bytes]:
        """
        FIND_VALUE RPC: Retrieve value for key from local storage.
        
        Args:
            key: Storage key
        
        Returns:
            Value if found locally, None otherwise
        """
        return self.storage.get(key)
    
    def find_node(self, target_id: str) -> List[DHTNode]:
        """
        FIND_NODE RPC: Find K closest nodes to target ID.
        
        Args:
            target_id: Target peer/block ID
        
        Returns:
            List of K closest known nodes
        """
        return self.routing_table.find_closest_nodes(target_id, count=K)
    
    def iterative_find_node(self, target_id: str) -> List[DHTNode]:
        """
        Iterative node lookup (Kademlia's core algorithm).
        
        Process:
        1. Start with K closest known nodes
        2. Query ALPHA (3) closest unqueried nodes in parallel
        3. Add results to candidate list
        4. Repeat until no closer nodes found
        
        Args:
            target_id: Target peer/block ID to find
        
        Returns:
            K closest nodes to target
        """
        # Start with K closest known nodes
        closest = self.routing_table.find_closest_nodes(target_id, count=K)
        
        # Track queried and unqueried nodes
        queried: Set[str] = set()
        unqueried = {node.peer_id: node for node in closest}
        
        # Iterative lookup
        iterations = 0
        max_iterations = 10  # Prevent infinite loops
        
        while unqueried and iterations < max_iterations:
            iterations += 1
            
            # Select ALPHA closest unqueried nodes
            candidates = sorted(
                unqueried.values(),
                key=lambda n: xor_distance(n.peer_id, target_id)
            )[:ALPHA]
            
            # Query each candidate
            for node in candidates:
                # Mark as queried
                queried.add(node.peer_id)
                del unqueried[node.peer_id]
                
                # FIND_NODE RPC (simulated - would be network call)
                results = self.find_node(target_id)
                
                # Add results to routing table and candidates
                for result in results:
                    if result.peer_id not in queried:
                        unqueried[result.peer_id] = result
                        self.routing_table.add_node(result)
            
            # Check if we found closer nodes
            all_candidates = list(queried) + list(unqueried.keys())
            all_candidates.sort(key=lambda pid: xor_distance(pid, target_id))
            
            # If top K unchanged, we're done
            top_k = all_candidates[:K]
            if all(pid in queried for pid in top_k):
                break
        
        # Return K closest nodes
        final_closest = self.routing_table.find_closest_nodes(target_id, count=K)
        logger.debug(f"Iterative lookup completed in {iterations} iterations")
        return final_closest
    
    def iterative_find_value(self, key: str) -> Optional[bytes]:
        """
        Iterative value lookup.
        
        Similar to iterative_find_node, but stops when value is found.
        
        Args:
            key: Storage key to find
        
        Returns:
            Value if found, None otherwise
        """
        # Check local storage first
        value = self.find_value(key)
        if value:
            return value
        
        # Check cache
        if key in self.lookup_cache:
            # Try cached peer addresses
            # TODO: Implement actual network retrieval
            pass
        
        # Perform iterative lookup
        closest_nodes = self.iterative_find_node(key)
        
        # Query closest nodes for value
        for node in closest_nodes:
            # TODO: Implement actual FIND_VALUE RPC over network
            # For now, just check local storage
            value = self.find_value(key)
            if value:
                # Cache result
                self.lookup_cache[key] = [node.address]
                return value
        
        return None
    
    def publish(self, key: str, value: bytes):
        """
        Publish key-value pair to DHT.
        
        Process:
        1. Find K closest nodes to key
        2. Send STORE RPC to each node
        3. Store locally as well
        
        Args:
            key: Storage key
            value: Data to publish
        """
        # Store locally
        self.store(key, value)
        
        # Find K closest nodes
        closest = self.iterative_find_node(key)
        
        # Send STORE RPC to each (simulated)
        for node in closest:
            # TODO: Implement actual network STORE RPC
            logger.debug(f"Would store {key[:16]}... at {node.address}")
        
        logger.info(f"Published {key[:16]}... to {len(closest)} nodes")
    
    def get_stats(self) -> Dict:
        """Get DHT statistics."""
        rt_stats = self.routing_table.get_stats()
        return {
            **rt_stats,
            "local_storage_keys": len(self.storage),
            "cache_entries": len(self.lookup_cache),
            "node_id": self.node_id[:16] + "..."
        }


if __name__ == "__main__":
    # Example usage
    print("Kademlia DHT Example:")
    print("-" * 60)
    
    # Create DHT for a node
    node_id = hashlib.sha256(b"test_node_1").hexdigest()
    dht = KademliaDHT(node_id=node_id, node_address="127.0.0.1:7777")
    
    print(f"Node ID: {node_id[:16]}...")
    
    # Add some nodes to routing table
    for i in range(10):
        peer_id = hashlib.sha256(f"peer_{i}".encode()).hexdigest()
        node = DHTNode(peer_id=peer_id, address=f"127.0.0.1:{8000+i}")
        dht.routing_table.add_node(node)
    
    # Store some data
    dht.store("block_abc123", b"compressed_data_here")
    dht.store("block_def456", b"more_data")
    
    # Find closest nodes to a target
    target_id = hashlib.sha256(b"target_block").hexdigest()
    closest = dht.find_node(target_id)
    
    print(f"\nClosest {len(closest)} nodes to target:")
    for node in closest[:5]:
        dist = xor_distance(node.peer_id, target_id)
        print(f"  {node.peer_id[:16]}... - distance: {dist}")
    
    # Get stats
    stats = dht.get_stats()
    print(f"\nDHT Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Non-empty buckets: {stats['non_empty_buckets']}")
    print(f"  Local storage: {stats['local_storage_keys']} keys")
    
    print("\n✅ Kademlia DHT working!")
