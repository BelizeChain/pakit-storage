"""
Peer Reputation and Trust System

Tracks peer behavior to identify reliable vs malicious nodes.
Uses reputation scores for peer selection and connection management.

Tracked Behaviors:
- Successful block deliveries (+)
- Failed/timeout requests (-)
- Invalid Merkle proofs (---)
- Invalid blocks (---)
- Response time (affects score)

Reputation Range: 0.0 (banned) to 1.0 (excellent)
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Reputation constants
MIN_REPUTATION = 0.0  # Banned threshold
MAX_REPUTATION = 1.0  # Perfect reputation
INITIAL_REPUTATION = 0.5  # New peer starts neutral
BAN_THRESHOLD = 0.1  # Auto-ban below this
DECAY_RATE = 0.01  # Daily reputation decay for inactive peers
DECAY_INTERVAL = 86400  # 24 hours in seconds


class ReputationEvent(Enum):
    """Types of reputation events."""
    
    # Positive events
    BLOCK_DELIVERED = "block_delivered"
    PROOF_VALID = "proof_valid"
    FAST_RESPONSE = "fast_response"
    UPTIME_GOOD = "uptime_good"
    
    # Negative events
    REQUEST_TIMEOUT = "request_timeout"
    REQUEST_FAILED = "request_failed"
    SLOW_RESPONSE = "slow_response"
    
    # Severe violations
    PROOF_INVALID = "proof_invalid"
    BLOCK_INVALID = "block_invalid"
    MALICIOUS_BEHAVIOR = "malicious_behavior"


# Reputation deltas for each event type
REPUTATION_DELTAS = {
    # Positive (small incremental gains)
    ReputationEvent.BLOCK_DELIVERED: +0.01,
    ReputationEvent.PROOF_VALID: +0.02,
    ReputationEvent.FAST_RESPONSE: +0.005,
    ReputationEvent.UPTIME_GOOD: +0.01,
    
    # Negative (moderate penalties)
    ReputationEvent.REQUEST_TIMEOUT: -0.05,
    ReputationEvent.REQUEST_FAILED: -0.03,
    ReputationEvent.SLOW_RESPONSE: -0.01,
    
    # Severe (harsh penalties)
    ReputationEvent.PROOF_INVALID: -0.20,
    ReputationEvent.BLOCK_INVALID: -0.25,
    ReputationEvent.MALICIOUS_BEHAVIOR: -0.50,
}


@dataclass
class PeerReputation:
    """Reputation data for a single peer."""
    
    peer_id: str
    reputation: float = INITIAL_REPUTATION
    
    # Event counters
    blocks_delivered: int = 0
    proofs_valid: int = 0
    requests_timeout: int = 0
    requests_failed: int = 0
    proofs_invalid: int = 0
    blocks_invalid: int = 0
    
    # Response time tracking
    total_response_time: float = 0.0  # Cumulative seconds
    response_count: int = 0
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def avg_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.response_count == 0:
            return 0.0
        return self.total_response_time / self.response_count
    
    def is_banned(self) -> bool:
        """Check if peer is banned (reputation below threshold)."""
        return self.reputation < BAN_THRESHOLD
    
    def update_seen(self):
        """Update last seen timestamp."""
        self.last_seen = time.time()
    
    def get_trust_level(self) -> str:
        """Get human-readable trust level."""
        if self.reputation >= 0.8:
            return "EXCELLENT"
        elif self.reputation >= 0.6:
            return "GOOD"
        elif self.reputation >= 0.4:
            return "FAIR"
        elif self.reputation >= 0.2:
            return "POOR"
        elif self.reputation >= BAN_THRESHOLD:
            return "VERY_POOR"
        else:
            return "BANNED"


class ReputationSystem:
    """
    Manages reputation scores for all peers in the network.
    
    Features:
    - Event-based reputation updates
    - Automatic decay for inactive peers
    - Ban list management
    - Peer ranking and selection
    """
    
    def __init__(self, node_id: str):
        """
        Initialize reputation system.
        
        Args:
            node_id: Our node's peer ID
        """
        self.node_id = node_id
        
        # Peer reputations (peer_id → PeerReputation)
        self.reputations: Dict[str, PeerReputation] = {}
        
        # Ban list (peer_id → ban_timestamp)
        self.banned_peers: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            "events_processed": 0,
            "peers_tracked": 0,
            "peers_banned": 0,
            "decay_runs": 0
        }
        
        logger.info(f"Initialized reputation system for node: {node_id[:16]}...")
    
    def get_reputation(self, peer_id: str) -> PeerReputation:
        """
        Get reputation for peer (creates if doesn't exist).
        
        Args:
            peer_id: Peer ID
        
        Returns:
            PeerReputation object
        """
        if peer_id not in self.reputations:
            self.reputations[peer_id] = PeerReputation(peer_id=peer_id)
            self.stats["peers_tracked"] += 1
        
        return self.reputations[peer_id]
    
    def record_event(
        self,
        peer_id: str,
        event: ReputationEvent,
        response_time: Optional[float] = None
    ):
        """
        Record reputation event for peer.
        
        Args:
            peer_id: Peer ID
            event: Type of event
            response_time: Optional response time in seconds
        """
        rep = self.get_reputation(peer_id)
        
        # Update event counters
        if event == ReputationEvent.BLOCK_DELIVERED:
            rep.blocks_delivered += 1
        elif event == ReputationEvent.PROOF_VALID:
            rep.proofs_valid += 1
        elif event == ReputationEvent.REQUEST_TIMEOUT:
            rep.requests_timeout += 1
        elif event == ReputationEvent.REQUEST_FAILED:
            rep.requests_failed += 1
        elif event == ReputationEvent.PROOF_INVALID:
            rep.proofs_invalid += 1
        elif event == ReputationEvent.BLOCK_INVALID:
            rep.blocks_invalid += 1
        
        # Update response time
        if response_time is not None:
            rep.total_response_time += response_time
            rep.response_count += 1
            
            # Add bonus/penalty for response time
            if response_time < 1.0:  # Fast response (<1s)
                self.record_event(peer_id, ReputationEvent.FAST_RESPONSE)
            elif response_time > 10.0:  # Slow response (>10s)
                self.record_event(peer_id, ReputationEvent.SLOW_RESPONSE)
        
        # Apply reputation delta
        delta = REPUTATION_DELTAS.get(event, 0.0)
        rep.reputation = max(MIN_REPUTATION, min(MAX_REPUTATION, rep.reputation + delta))
        
        # Update timestamps
        rep.last_updated = time.time()
        rep.update_seen()
        
        # Check for ban
        if rep.is_banned() and peer_id not in self.banned_peers:
            self._ban_peer(peer_id)
        
        self.stats["events_processed"] += 1
        
        logger.debug(
            f"Event {event.value} for {peer_id[:16]}...: "
            f"rep={rep.reputation:.3f} (delta={delta:+.3f})"
        )
    
    def _ban_peer(self, peer_id: str):
        """Ban a peer for low reputation."""
        self.banned_peers[peer_id] = time.time()
        self.stats["peers_banned"] += 1
        logger.warning(f"Banned peer {peer_id[:16]}... (reputation below {BAN_THRESHOLD})")
    
    def is_banned(self, peer_id: str) -> bool:
        """Check if peer is banned."""
        return peer_id in self.banned_peers
    
    def unban_peer(self, peer_id: str):
        """Manually unban a peer."""
        if peer_id in self.banned_peers:
            del self.banned_peers[peer_id]
            # Reset reputation to minimum viable
            if peer_id in self.reputations:
                self.reputations[peer_id].reputation = BAN_THRESHOLD + 0.1
            logger.info(f"Unbanned peer {peer_id[:16]}...")
    
    def get_best_peers(self, count: int = 20, min_reputation: float = 0.5) -> List[PeerReputation]:
        """
        Get highest reputation peers.
        
        Args:
            count: Number of peers to return
            min_reputation: Minimum reputation threshold
        
        Returns:
            List of PeerReputation objects sorted by reputation
        """
        # Filter by minimum reputation and not banned
        eligible = [
            rep for rep in self.reputations.values()
            if rep.reputation >= min_reputation and not self.is_banned(rep.peer_id)
        ]
        
        # Sort by reputation (descending)
        eligible.sort(key=lambda r: r.reputation, reverse=True)
        
        return eligible[:count]
    
    def apply_decay(self):
        """
        Apply reputation decay to inactive peers.
        
        Peers lose reputation slowly if they haven't been seen recently.
        Encourages active participation.
        """
        current_time = time.time()
        decayed_count = 0
        
        for rep in self.reputations.values():
            # Check if peer has been inactive
            time_since_seen = current_time - rep.last_seen
            
            if time_since_seen > DECAY_INTERVAL:
                # Apply decay
                days_inactive = time_since_seen / DECAY_INTERVAL
                decay = DECAY_RATE * days_inactive
                
                old_rep = rep.reputation
                rep.reputation = max(MIN_REPUTATION, rep.reputation - decay)
                
                if rep.reputation != old_rep:
                    decayed_count += 1
                    logger.debug(
                        f"Decayed {rep.peer_id[:16]}...: "
                        f"{old_rep:.3f} → {rep.reputation:.3f}"
                    )
                
                # Check for ban after decay
                if rep.is_banned() and rep.peer_id not in self.banned_peers:
                    self._ban_peer(rep.peer_id)
        
        self.stats["decay_runs"] += 1
        logger.info(f"Reputation decay applied to {decayed_count} peers")
    
    def get_peer_summary(self, peer_id: str) -> Dict:
        """Get detailed summary for a peer."""
        if peer_id not in self.reputations:
            return {"error": "Peer not found"}
        
        rep = self.reputations[peer_id]
        
        return {
            "peer_id": peer_id[:16] + "...",
            "reputation": rep.reputation,
            "trust_level": rep.get_trust_level(),
            "banned": self.is_banned(peer_id),
            "blocks_delivered": rep.blocks_delivered,
            "proofs_valid": rep.proofs_valid,
            "requests_timeout": rep.requests_timeout,
            "requests_failed": rep.requests_failed,
            "proofs_invalid": rep.proofs_invalid,
            "blocks_invalid": rep.blocks_invalid,
            "avg_response_time": f"{rep.avg_response_time():.3f}s",
            "age": f"{(time.time() - rep.created_at) / 3600:.1f}h",
            "last_seen": f"{(time.time() - rep.last_seen) / 60:.1f}m ago"
        }
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        if self.reputations:
            avg_reputation = sum(r.reputation for r in self.reputations.values()) / len(self.reputations)
        else:
            avg_reputation = 0.0
        
        return {
            **self.stats,
            "avg_reputation": f"{avg_reputation:.3f}",
            "active_bans": len(self.banned_peers)
        }


if __name__ == "__main__":
    # Example usage
    print("Reputation System Example:")
    print("-" * 60)
    
    # Create reputation system
    import hashlib
    node_id = hashlib.sha256(b"test_node").hexdigest()
    rep_system = ReputationSystem(node_id=node_id)
    
    print(f"Node ID: {node_id[:16]}...")
    
    # Simulate peer behavior
    good_peer = hashlib.sha256(b"good_peer").hexdigest()
    bad_peer = hashlib.sha256(b"bad_peer").hexdigest()
    
    # Good peer: delivers blocks successfully
    print(f"\nSimulating good peer behavior...")
    for i in range(10):
        rep_system.record_event(good_peer, ReputationEvent.BLOCK_DELIVERED, response_time=0.5)
    
    for i in range(5):
        rep_system.record_event(good_peer, ReputationEvent.PROOF_VALID)
    
    # Bad peer: fails requests and sends invalid proofs
    print(f"Simulating bad peer behavior...")
    for i in range(5):
        rep_system.record_event(bad_peer, ReputationEvent.REQUEST_TIMEOUT)
    
    for i in range(3):
        rep_system.record_event(bad_peer, ReputationEvent.PROOF_INVALID)
    
    # Get summaries
    print(f"\nGood Peer Summary:")
    good_summary = rep_system.get_peer_summary(good_peer)
    for key, value in good_summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nBad Peer Summary:")
    bad_summary = rep_system.get_peer_summary(bad_peer)
    for key, value in bad_summary.items():
        print(f"  {key}: {value}")
    
    # Get best peers
    best = rep_system.get_best_peers(count=10)
    print(f"\nBest Peers:")
    for rep in best[:3]:
        print(f"  {rep.peer_id[:16]}... - {rep.reputation:.3f} ({rep.get_trust_level()})")
    
    # Get stats
    stats = rep_system.get_stats()
    print(f"\nSystem Statistics:")
    print(f"  Events processed: {stats['events_processed']}")
    print(f"  Peers tracked: {stats['peers_tracked']}")
    print(f"  Peers banned: {stats['peers_banned']}")
    print(f"  Avg reputation: {stats['avg_reputation']}")
    
    print("\n✅ Reputation system working!")
