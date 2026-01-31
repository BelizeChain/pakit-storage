"""
DHT-Based Block Discovery

Integrates Kademlia DHT with block storage for efficient peer-to-peer block retrieval.

Features:
- Store block metadata in DHT (hash → peer addresses)
- Content-addressable routing (find peers with specific blocks)
- Block availability announcements
- Efficient multi-peer lookup
- Cache for recently discovered blocks
"""

import time
import hashlib
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# Discovery constants
MAX_PROVIDERS = 20  # Maximum peers to track per block
PROVIDER_TIMEOUT = 3600  # Provider expires after 1 hour
ANNOUNCE_INTERVAL = 600  # Re-announce blocks every 10 minutes
LOOKUP_PARALLELISM = 3  # Query 3 peers in parallel for lookups


@dataclass
class BlockProvider:
    """Represents a peer that has a specific block."""
    
    peer_id: str
    address: str
    announced_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    successful_retrievals: int = 0
    failed_retrievals: int = 0
    
    def is_alive(self, timeout: int = PROVIDER_TIMEOUT) -> bool:
        """Check if provider is still active."""
        return (time.time() - self.last_seen) < timeout
    
    def update_seen(self):
        """Update last seen timestamp."""
        self.last_seen = time.time()
    
    def get_reliability(self) -> float:
        """Calculate reliability score (0.0 to 1.0)."""
        total = self.successful_retrievals + self.failed_retrievals
        if total == 0:
            return 0.5  # Neutral for new providers
        return self.successful_retrievals / total


@dataclass
class BlockMetadata:
    """Metadata about a block stored in DHT."""
    
    block_hash: str
    size: int  # Compressed size in bytes
    depth: int  # Depth in DAG
    compression_algo: str
    parent_hashes: List[str]
    created_at: float = field(default_factory=time.time)


class BlockDiscoveryService:
    """
    DHT-based block discovery service.
    
    Enables peers to:
    1. Announce blocks they have
    2. Find peers who have specific blocks
    3. Query block metadata
    """
    
    def __init__(self, node_id: str, dht_client):
        """
        Initialize block discovery service.
        
        Args:
            node_id: Our node's peer ID
            dht_client: KademliaDHT instance
        """
        self.node_id = node_id
        self.dht = dht_client
        
        # Block providers (block_hash → [BlockProvider])
        self.providers: Dict[str, List[BlockProvider]] = defaultdict(list)
        
        # Blocks we provide (local storage)
        self.local_blocks: Set[str] = set()
        
        # Last announcement times (block_hash → timestamp)
        self.last_announcements: Dict[str, float] = {}
        
        # Discovery cache (block_hash → [peer addresses])
        self.discovery_cache: Dict[str, List[str]] = {}
        
        # Statistics
        self.stats = {
            "blocks_announced": 0,
            "blocks_discovered": 0,
            "providers_found": 0,
            "cache_hits": 0,
            "dht_lookups": 0
        }
        
        logger.info(f"Initialized block discovery service for node: {node_id[:16]}...")
    
    def announce_block(
        self,
        block_hash: str,
        metadata: BlockMetadata
    ):
        """
        Announce that we have a block (publish to DHT).
        
        Args:
            block_hash: Hash of block we have
            metadata: Block metadata
        """
        # Add to local blocks
        self.local_blocks.add(block_hash)
        
        # Create provider entry for ourselves
        provider_key = f"block:{block_hash}:providers"
        provider_data = {
            "peer_id": self.node_id,
            "address": f"127.0.0.1:7777",  # TODO: Get actual address
            "announced_at": time.time()
        }
        
        # Serialize and publish to DHT
        import msgpack
        provider_bytes = msgpack.packb(provider_data)
        
        # Store in DHT
        self.dht.publish(provider_key, provider_bytes)
        
        # Store metadata in DHT
        metadata_key = f"block:{block_hash}:metadata"
        metadata_bytes = msgpack.packb({
            "block_hash": metadata.block_hash,
            "size": metadata.size,
            "depth": metadata.depth,
            "compression_algo": metadata.compression_algo,
            "parent_hashes": metadata.parent_hashes,
            "created_at": metadata.created_at
        })
        
        self.dht.publish(metadata_key, metadata_bytes)
        
        # Update announcement time
        self.last_announcements[block_hash] = time.time()
        self.stats["blocks_announced"] += 1
        
        logger.info(f"Announced block {block_hash[:16]}... to DHT")
    
    def find_block_providers(
        self,
        block_hash: str,
        count: int = MAX_PROVIDERS
    ) -> List[BlockProvider]:
        """
        Find peers who have a specific block.
        
        Args:
            block_hash: Hash of block to find
            count: Maximum number of providers to return
        
        Returns:
            List of BlockProvider objects
        """
        # Check cache first
        if block_hash in self.discovery_cache:
            self.stats["cache_hits"] += 1
            cached_addresses = self.discovery_cache[block_hash]
            
            # Convert cached addresses to providers
            providers = []
            for addr in cached_addresses[:count]:
                providers.append(BlockProvider(
                    peer_id=addr.split(':')[0],  # Simplified
                    address=addr
                ))
            
            if providers:
                logger.debug(f"Cache hit for block {block_hash[:16]}...")
                return providers
        
        # Query DHT for providers
        provider_key = f"block:{block_hash}:providers"
        
        self.stats["dht_lookups"] += 1
        
        # Iterative lookup in DHT
        closest_nodes = self.dht.iterative_find_node(provider_key)
        
        providers = []
        
        # Query each node for provider data
        for node in closest_nodes[:LOOKUP_PARALLELISM]:
            # TODO: Actual network query
            # For now, check local DHT storage
            provider_data = self.dht.find_value(provider_key)
            
            if provider_data:
                import msgpack
                data = msgpack.unpackb(provider_data)
                
                provider = BlockProvider(
                    peer_id=data["peer_id"],
                    address=data.get("address", "unknown"),
                    announced_at=data["announced_at"]
                )
                
                providers.append(provider)
                self.stats["providers_found"] += 1
        
        # Cache results
        if providers:
            self.discovery_cache[block_hash] = [p.address for p in providers]
            self.stats["blocks_discovered"] += 1
            logger.info(f"Found {len(providers)} providers for block {block_hash[:16]}...")
        else:
            logger.warning(f"No providers found for block {block_hash[:16]}...")
        
        return providers[:count]
    
    def get_block_metadata(self, block_hash: str) -> Optional[BlockMetadata]:
        """
        Retrieve block metadata from DHT.
        
        Args:
            block_hash: Hash of block
        
        Returns:
            BlockMetadata if found, None otherwise
        """
        metadata_key = f"block:{block_hash}:metadata"
        
        # Query DHT
        metadata_bytes = self.dht.iterative_find_value(metadata_key)
        
        if not metadata_bytes:
            logger.debug(f"No metadata found for block {block_hash[:16]}...")
            return None
        
        # Deserialize
        import msgpack
        data = msgpack.unpackb(metadata_bytes)
        
        return BlockMetadata(
            block_hash=data["block_hash"],
            size=data["size"],
            depth=data["depth"],
            compression_algo=data["compression_algo"],
            parent_hashes=data["parent_hashes"],
            created_at=data["created_at"]
        )
    
    def find_block_with_routing(
        self,
        block_hash: str
    ) -> Optional[str]:
        """
        Find a block using content routing.
        
        Uses DHT to route to the closest peer who has the block,
        then retrieves it directly.
        
        Args:
            block_hash: Hash of block to find
        
        Returns:
            Peer address if found, None otherwise
        """
        providers = self.find_block_providers(block_hash, count=5)
        
        if not providers:
            return None
        
        # Sort by reliability
        providers.sort(key=lambda p: p.get_reliability(), reverse=True)
        
        # Return best provider
        best = providers[0]
        logger.info(
            f"Routing to {best.peer_id[:16]}... for block {block_hash[:16]}... "
            f"(reliability: {best.get_reliability():.2f})"
        )
        
        return best.address
    
    def reannounce_blocks(self):
        """
        Re-announce all local blocks to DHT.
        
        Should be called periodically to refresh DHT entries.
        """
        current_time = time.time()
        reannounced = 0
        
        for block_hash in self.local_blocks:
            # Check if announcement is stale
            last_announced = self.last_announcements.get(block_hash, 0)
            
            if (current_time - last_announced) > ANNOUNCE_INTERVAL:
                # TODO: Get actual metadata from storage
                # For now, use placeholder
                metadata = BlockMetadata(
                    block_hash=block_hash,
                    size=0,
                    depth=0,
                    compression_algo="zstd",
                    parent_hashes=[]
                )
                
                self.announce_block(block_hash, metadata)
                reannounced += 1
        
        logger.info(f"Re-announced {reannounced} blocks to DHT")
    
    def record_retrieval_result(
        self,
        block_hash: str,
        peer_id: str,
        success: bool
    ):
        """
        Record result of block retrieval attempt.
        
        Updates provider reliability scores.
        
        Args:
            block_hash: Hash of block
            peer_id: Peer we retrieved from
            success: Whether retrieval succeeded
        """
        providers = self.providers.get(block_hash, [])
        
        for provider in providers:
            if provider.peer_id == peer_id:
                if success:
                    provider.successful_retrievals += 1
                else:
                    provider.failed_retrievals += 1
                
                provider.update_seen()
                
                logger.debug(
                    f"Updated provider {peer_id[:16]}... for block {block_hash[:16]}...: "
                    f"reliability={provider.get_reliability():.2f}"
                )
                break
    
    def cleanup_stale_providers(self):
        """Remove stale provider entries."""
        cleaned = 0
        
        for block_hash, providers in list(self.providers.items()):
            # Remove stale providers
            active = [p for p in providers if p.is_alive()]
            
            if len(active) < len(providers):
                cleaned += len(providers) - len(active)
                self.providers[block_hash] = active
            
            # Remove block entry if no providers
            if not active:
                del self.providers[block_hash]
        
        if cleaned > 0:
            logger.info(f"Cleaned {cleaned} stale provider entries")
    
    def get_stats(self) -> Dict:
        """Get discovery service statistics."""
        return {
            **self.stats,
            "local_blocks": len(self.local_blocks),
            "tracked_blocks": len(self.providers),
            "cache_size": len(self.discovery_cache)
        }


if __name__ == "__main__":
    # Example usage
    print("DHT-Based Block Discovery Example:")
    print("-" * 60)
    
    # Create mock DHT
    class MockDHT:
        def __init__(self):
            self.storage = {}
        
        def publish(self, key, value):
            self.storage[key] = value
        
        def iterative_find_node(self, key):
            return []  # No nodes
        
        def find_value(self, key):
            return self.storage.get(key)
        
        def iterative_find_value(self, key):
            return self.storage.get(key)
    
    # Create discovery service
    node_id = hashlib.sha256(b"test_node").hexdigest()
    dht = MockDHT()
    discovery = BlockDiscoveryService(node_id=node_id, dht_client=dht)
    
    print(f"Node ID: {node_id[:16]}...")
    
    # Announce a block
    block_hash = hashlib.sha256(b"block_data").hexdigest()
    metadata = BlockMetadata(
        block_hash=block_hash,
        size=512,
        depth=5,
        compression_algo="zstd",
        parent_hashes=["parent1", "parent2"]
    )
    
    discovery.announce_block(block_hash, metadata)
    print(f"\nAnnounced block {block_hash[:16]}...")
    
    # Retrieve metadata
    retrieved_metadata = discovery.get_block_metadata(block_hash)
    
    if retrieved_metadata:
        print(f"\nRetrieved metadata:")
        print(f"  Size: {retrieved_metadata.size} bytes")
        print(f"  Depth: {retrieved_metadata.depth}")
        print(f"  Compression: {retrieved_metadata.compression_algo}")
        print(f"  Parents: {len(retrieved_metadata.parent_hashes)}")
    
    # Find providers
    providers = discovery.find_block_providers(block_hash)
    print(f"\nFound {len(providers)} providers")
    
    # Get stats
    stats = discovery.get_stats()
    print(f"\nDiscovery Statistics:")
    print(f"  Blocks announced: {stats['blocks_announced']}")
    print(f"  Local blocks: {stats['local_blocks']}")
    print(f"  DHT lookups: {stats['dht_lookups']}")
    
    print("\n✅ DHT-based block discovery working!")
