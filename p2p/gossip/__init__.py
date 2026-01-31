"""
Block Gossip Protocol

Epidemic-style block propagation across P2P network.
"""

from .protocol import (
    GossipProtocol,
    BlockAnnouncement,
    BloomFilter,
    GOSSIP_FANOUT,
    GOSSIP_TTL,
    BLOOM_FILTER_SIZE
)

__all__ = [
    "GossipProtocol",
    "BlockAnnouncement",
    "BloomFilter",
    "GOSSIP_FANOUT",
    "GOSSIP_TTL",
    "BLOOM_FILTER_SIZE"
]
