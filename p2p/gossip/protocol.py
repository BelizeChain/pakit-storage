"""
Block Gossip Protocol

Implements epidemic-style block propagation across the P2P network.
When a node stores a new block, it announces to connected peers who
validate and re-gossip to their peers.

Key Features:
- Fanout=6: Announce to 6 random peers
- TTL=10: Maximum 10 hops to prevent infinite propagation
- Bloom filters: Prevent duplicate announcements
- Validation: Verify block integrity before re-gossip
"""

import time
import hashlib
from typing import Set, List, Dict, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


# Gossip constants
GOSSIP_FANOUT = 6  # Number of peers to gossip to
GOSSIP_TTL = 10  # Maximum hops
BLOOM_FILTER_SIZE = 10000  # Size of bloom filter for seen blocks


class BloomFilter:
    """
    Simple bloom filter for tracking seen block announcements.
    
    Prevents re-gossiping blocks we've already seen.
    """
    
    def __init__(self, size: int = BLOOM_FILTER_SIZE):
        """Initialize bloom filter with given size."""
        self.size = size
        self.bits = [False] * size
        self.num_hashes = 3  # Number of hash functions
    
    def _hash(self, item: str, seed: int) -> int:
        """Hash item with seed to get bit position."""
        h = hashlib.sha256(f"{item}{seed}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.size
    
    def add(self, item: str):
        """Add item to bloom filter."""
        for i in range(self.num_hashes):
            pos = self._hash(item, i)
            self.bits[pos] = True
    
    def might_contain(self, item: str) -> bool:
        """Check if item might be in filter (may have false positives)."""
        for i in range(self.num_hashes):
            pos = self._hash(item, i)
            if not self.bits[pos]:
                return False
        return True
    
    def clear(self):
        """Clear bloom filter."""
        self.bits = [False] * self.size


@dataclass
class BlockAnnouncement:
    """Block announcement message for gossip protocol."""
    
    block_hash: str
    block_depth: int
    compression_algo: str
    original_size: int
    compressed_size: int
    parent_hashes: List[str]
    timestamp: float
    ttl: int = GOSSIP_TTL  # Remaining hops
    origin_peer: str = ""  # Original peer who created block
    
    def to_bytes(self) -> bytes:
        """Serialize announcement to bytes for network transmission."""
        import msgpack
        return msgpack.packb({
            "block_hash": self.block_hash,
            "block_depth": self.block_depth,
            "compression_algo": self.compression_algo,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "parent_hashes": self.parent_hashes,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "origin_peer": self.origin_peer
        })
    
    @staticmethod
    def from_bytes(data: bytes) -> 'BlockAnnouncement':
        """Deserialize announcement from bytes."""
        import msgpack
        d = msgpack.unpackb(data)
        return BlockAnnouncement(
            block_hash=d["block_hash"],
            block_depth=d["block_depth"],
            compression_algo=d["compression_algo"],
            original_size=d["original_size"],
            compressed_size=d["compressed_size"],
            parent_hashes=d["parent_hashes"],
            timestamp=d["timestamp"],
            ttl=d["ttl"],
            origin_peer=d["origin_peer"]
        )


class GossipProtocol:
    """
    Gossip protocol for block announcements.
    
    When a node stores a new block:
    1. Create BlockAnnouncement
    2. Send to FANOUT random peers
    3. Peers validate and re-gossip (TTL-1)
    4. Track seen blocks to prevent duplicates
    """
    
    def __init__(self, node_id: str):
        """
        Initialize gossip protocol.
        
        Args:
            node_id: Our node's peer ID
        """
        self.node_id = node_id
        
        # Track seen blocks (prevent duplicate gossip)
        self.seen_blocks = BloomFilter(size=BLOOM_FILTER_SIZE)
        
        # Recently announced blocks (exact tracking for recent blocks)
        self.recent_announcements: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = {
            "blocks_announced": 0,
            "blocks_received": 0,
            "blocks_re_gossiped": 0,
            "duplicate_blocks": 0
        }
        
        logger.info(f"Initialized gossip protocol for node: {node_id[:16]}...")
    
    def announce_block(
        self,
        block_hash: str,
        block_depth: int,
        compression_algo: str,
        original_size: int,
        compressed_size: int,
        parent_hashes: List[str],
        peers: List[str]
    ) -> int:
        """
        Announce new block to network via gossip.
        
        Args:
            block_hash: Hash of new block
            block_depth: Depth in DAG
            compression_algo: Compression algorithm used
            original_size: Original data size
            compressed_size: Compressed size
            parent_hashes: List of parent block hashes
            peers: List of connected peer IDs
        
        Returns:
            Number of peers announced to
        """
        # Create announcement
        announcement = BlockAnnouncement(
            block_hash=block_hash,
            block_depth=block_depth,
            compression_algo=compression_algo,
            original_size=original_size,
            compressed_size=compressed_size,
            parent_hashes=parent_hashes,
            timestamp=time.time(),
            ttl=GOSSIP_TTL,
            origin_peer=self.node_id
        )
        
        # Mark as seen (don't re-gossip our own blocks)
        self.seen_blocks.add(block_hash)
        self.recent_announcements.append(block_hash)
        
        # Select random FANOUT peers
        import random
        fanout_peers = random.sample(peers, min(GOSSIP_FANOUT, len(peers)))
        
        # Send to each peer (simulated - would be network send)
        for peer_id in fanout_peers:
            self._send_announcement(peer_id, announcement)
        
        self.stats["blocks_announced"] += 1
        logger.info(f"Announced block {block_hash[:16]}... to {len(fanout_peers)} peers")
        
        return len(fanout_peers)
    
    def handle_announcement(
        self,
        announcement: BlockAnnouncement,
        from_peer: str,
        connected_peers: List[str],
        validate_callback=None
    ) -> bool:
        """
        Handle received block announcement.
        
        Process:
        1. Check if already seen (bloom filter + recent list)
        2. Validate block (optional callback)
        3. Re-gossip to FANOUT peers (if TTL > 0)
        
        Args:
            announcement: Received announcement
            from_peer: Peer ID who sent announcement
            connected_peers: List of connected peer IDs
            validate_callback: Optional function to validate block
        
        Returns:
            True if announcement processed, False if duplicate
        """
        self.stats["blocks_received"] += 1
        
        # Check bloom filter first (fast check)
        if self.seen_blocks.might_contain(announcement.block_hash):
            # Check exact list
            if announcement.block_hash in self.recent_announcements:
                self.stats["duplicate_blocks"] += 1
                logger.debug(f"Duplicate block: {announcement.block_hash[:16]}...")
                return False
        
        # Mark as seen
        self.seen_blocks.add(announcement.block_hash)
        self.recent_announcements.append(announcement.block_hash)
        
        # Validate block if callback provided
        if validate_callback:
            if not validate_callback(announcement):
                logger.warning(f"Invalid block: {announcement.block_hash[:16]}...")
                return False
        
        # Re-gossip if TTL > 0
        if announcement.ttl > 0:
            # Decrement TTL
            announcement.ttl -= 1
            
            # Select random FANOUT peers (exclude sender)
            available_peers = [p for p in connected_peers if p != from_peer]
            import random
            fanout_peers = random.sample(
                available_peers,
                min(GOSSIP_FANOUT, len(available_peers))
            )
            
            # Re-gossip to each peer
            for peer_id in fanout_peers:
                self._send_announcement(peer_id, announcement)
            
            self.stats["blocks_re_gossiped"] += 1
            logger.debug(
                f"Re-gossiped block {announcement.block_hash[:16]}... "
                f"to {len(fanout_peers)} peers (TTL={announcement.ttl})"
            )
        
        return True
    
    def _send_announcement(self, peer_id: str, announcement: BlockAnnouncement):
        """
        Send announcement to peer (network transmission).
        
        Args:
            peer_id: Target peer ID
            announcement: Announcement to send
        """
        # TODO: Implement actual network send
        # For now, just log
        logger.debug(f"Sending announcement to {peer_id[:16]}...")
    
    def get_stats(self) -> Dict:
        """Get gossip protocol statistics."""
        return {
            **self.stats,
            "seen_blocks_count": len(self.recent_announcements),
            "bloom_filter_size": self.seen_blocks.size
        }
    
    def clear_seen_blocks(self):
        """Clear seen blocks tracking (for testing or maintenance)."""
        self.seen_blocks.clear()
        self.recent_announcements.clear()
        logger.info("Cleared seen blocks tracking")


if __name__ == "__main__":
    # Example usage
    print("Gossip Protocol Example:")
    print("-" * 60)
    
    # Create gossip protocol for a node
    node_id = hashlib.sha256(b"test_node").hexdigest()
    gossip = GossipProtocol(node_id=node_id)
    
    print(f"Node ID: {node_id[:16]}...")
    
    # Simulate connected peers
    peers = [
        hashlib.sha256(f"peer_{i}".encode()).hexdigest()
        for i in range(10)
    ]
    
    # Announce a block
    block_hash = hashlib.sha256(b"block_data").hexdigest()
    announced_to = gossip.announce_block(
        block_hash=block_hash,
        block_depth=5,
        compression_algo="zstd",
        original_size=2048,
        compressed_size=512,
        parent_hashes=["parent1", "parent2"],
        peers=peers
    )
    
    print(f"\nAnnounced block {block_hash[:16]}... to {announced_to} peers")
    
    # Simulate receiving announcement from peer
    announcement = BlockAnnouncement(
        block_hash=hashlib.sha256(b"new_block").hexdigest(),
        block_depth=6,
        compression_algo="zstd",
        original_size=4096,
        compressed_size=1024,
        parent_hashes=[block_hash],
        timestamp=time.time(),
        ttl=5,
        origin_peer=peers[0]
    )
    
    processed = gossip.handle_announcement(
        announcement=announcement,
        from_peer=peers[0],
        connected_peers=peers[1:]
    )
    
    print(f"Processed announcement: {processed}")
    
    # Get stats
    stats = gossip.get_stats()
    print(f"\nGossip Statistics:")
    print(f"  Blocks announced: {stats['blocks_announced']}")
    print(f"  Blocks received: {stats['blocks_received']}")
    print(f"  Blocks re-gossiped: {stats['blocks_re_gossiped']}")
    print(f"  Duplicate blocks: {stats['duplicate_blocks']}")
    
    print("\n✅ Gossip protocol working!")
