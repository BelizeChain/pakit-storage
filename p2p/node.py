"""
P2P Node - Core peer identity and connection management.

Each Pakit node has:
- Unique peer ID (Ed25519 public key hash)
- Network address (IP:port)
- Connection pool to other peers
- Local DAG backend

Nodes participate in:
- DHT for peer/block discovery
- Gossip for block announcements
- Request/response for block retrieval
"""

import hashlib
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
import logging

logger = logging.getLogger(__name__)


@dataclass
class PeerInfo:
    """Information about a remote peer."""
    
    peer_id: str  # Hash of public key
    address: str  # IP:port (e.g., "192.168.1.100:7777")
    public_key: bytes  # Ed25519 public key
    last_seen: float = field(default_factory=time.time)
    reputation: float = 1.0  # 0.0 (malicious) to 1.0 (trusted)
    
    # Statistics
    blocks_received: int = 0
    blocks_sent: int = 0
    failed_requests: int = 0
    invalid_proofs: int = 0
    
    def update_reputation(self):
        """Update reputation based on behavior."""
        # Good behavior increases reputation
        success_rate = (
            self.blocks_received / 
            (self.blocks_received + self.failed_requests + 1)
        )
        
        # Invalid proofs severely damage reputation
        penalty = self.invalid_proofs * 0.1
        
        # Calculate new reputation (0.0 to 1.0)
        self.reputation = max(0.0, min(1.0, success_rate - penalty))
    
    def is_alive(self, timeout: int = 300) -> bool:
        """Check if peer responded recently (default: 5 minutes)."""
        return (time.time() - self.last_seen) < timeout


class PakitNode:
    """
    Main P2P node implementation.
    
    Manages local peer identity, connections to remote peers,
    and participation in distributed protocols (DHT, gossip, etc.).
    """
    
    def __init__(
        self,
        listen_address: str = "0.0.0.0:7777",
        bootstrap_peers: Optional[List[str]] = None,
        max_peers: int = 50,
        data_dir: str = "./pakit_node"
    ):
        """
        Initialize Pakit P2P node.
        
        Args:
            listen_address: Address to listen on (IP:port)
            bootstrap_peers: Initial peers to connect to
            max_peers: Maximum number of peer connections
            data_dir: Directory for node data (keys, DAG, etc.)
        """
        self.listen_address = listen_address
        self.max_peers = max_peers
        self.data_dir = data_dir
        
        # Generate or load peer identity
        self.private_key, self.public_key = self._load_or_generate_keys()
        self.peer_id = self._compute_peer_id(self.public_key)
        
        # Peer management
        self.peers: Dict[str, PeerInfo] = {}  # peer_id -> PeerInfo
        self.bootstrap_peers = bootstrap_peers or []
        
        # Connection tracking
        self.connected_peers: Set[str] = set()  # Currently connected peer IDs
        
        logger.info(f"Initialized Pakit node: {self.peer_id[:16]}...")
        logger.info(f"Listening on: {self.listen_address}")
    
    def _load_or_generate_keys(self) -> tuple:
        """Load existing keys or generate new Ed25519 keypair."""
        import os
        from pathlib import Path
        
        key_path = Path(self.data_dir) / "node_key.pem"
        
        if key_path.exists():
            # Load existing key
            with open(key_path, "rb") as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            public_key = private_key.public_key()
            logger.info("Loaded existing peer identity")
        else:
            # Generate new keypair
            os.makedirs(self.data_dir, exist_ok=True)
            private_key = ed25519.Ed25519PrivateKey.generate()
            public_key = private_key.public_key()
            
            # Save private key
            pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            with open(key_path, "wb") as f:
                f.write(pem)
            
            logger.info("Generated new peer identity")
        
        return private_key, public_key
    
    def _compute_peer_id(self, public_key) -> str:
        """Compute peer ID from public key (SHA-256 hash)."""
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        return hashlib.sha256(pub_bytes).hexdigest()
    
    def add_peer(self, peer_info: PeerInfo) -> bool:
        """
        Add a peer to the peer table.
        
        Args:
            peer_info: Information about the peer
            
        Returns:
            True if added, False if table full or peer exists
        """
        # Don't add self
        if peer_info.peer_id == self.peer_id:
            return False
        
        # Check if already exists
        if peer_info.peer_id in self.peers:
            # Update last_seen
            self.peers[peer_info.peer_id].last_seen = time.time()
            return True
        
        # Check peer limit
        if len(self.peers) >= self.max_peers:
            # Remove worst peer (lowest reputation)
            worst_peer = min(
                self.peers.values(),
                key=lambda p: p.reputation
            )
            del self.peers[worst_peer.peer_id]
            logger.info(f"Evicted low-reputation peer: {worst_peer.peer_id[:16]}...")
        
        # Add new peer
        self.peers[peer_info.peer_id] = peer_info
        logger.info(f"Added peer: {peer_info.peer_id[:16]}... ({peer_info.address})")
        return True
    
    def remove_peer(self, peer_id: str) -> bool:
        """Remove a peer from the peer table."""
        if peer_id in self.peers:
            del self.peers[peer_id]
            self.connected_peers.discard(peer_id)
            logger.info(f"Removed peer: {peer_id[:16]}...")
            return True
        return False
    
    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """Get information about a specific peer."""
        return self.peers.get(peer_id)
    
    def get_random_peers(self, count: int = 10) -> List[PeerInfo]:
        """Get random sample of peers (for gossip, etc.)."""
        import random
        available = list(self.peers.values())
        return random.sample(available, min(count, len(available)))
    
    def get_best_peers(self, count: int = 10) -> List[PeerInfo]:
        """Get peers with highest reputation."""
        sorted_peers = sorted(
            self.peers.values(),
            key=lambda p: p.reputation,
            reverse=True
        )
        return sorted_peers[:count]
    
    def cleanup_dead_peers(self, timeout: int = 300):
        """Remove peers that haven't been seen recently."""
        dead_peers = [
            peer_id for peer_id, peer in self.peers.items()
            if not peer.is_alive(timeout)
        ]
        
        for peer_id in dead_peers:
            self.remove_peer(peer_id)
        
        if dead_peers:
            logger.info(f"Cleaned up {len(dead_peers)} dead peers")
    
    def get_stats(self) -> Dict:
        """Get node statistics."""
        return {
            "peer_id": self.peer_id,
            "address": self.listen_address,
            "total_peers": len(self.peers),
            "connected_peers": len(self.connected_peers),
            "avg_reputation": (
                sum(p.reputation for p in self.peers.values()) / len(self.peers)
                if self.peers else 0.0
            ),
            "bootstrap_peers": len(self.bootstrap_peers)
        }
    
    def sign_message(self, message: bytes) -> bytes:
        """Sign a message with node's private key."""
        return self.private_key.sign(message)
    
    def verify_peer_signature(
        self,
        peer_id: str,
        message: bytes,
        signature: bytes
    ) -> bool:
        """Verify a peer's signature on a message."""
        peer = self.get_peer(peer_id)
        if not peer:
            return False
        
        try:
            # Reconstruct public key from bytes
            public_key = ed25519.Ed25519PublicKey.from_public_bytes(
                peer.public_key
            )
            public_key.verify(signature, message)
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Example usage
    print("Pakit P2P Node Example:")
    print("-" * 60)
    
    # Create a node
    node = PakitNode(
        listen_address="127.0.0.1:7777",
        bootstrap_peers=["127.0.0.1:8888", "127.0.0.1:9999"],
        max_peers=50
    )
    
    print(f"Peer ID: {node.peer_id[:16]}...")
    print(f"Address: {node.listen_address}")
    
    # Simulate adding peers
    for i in range(5):
        # Generate fake peer
        fake_key = ed25519.Ed25519PrivateKey.generate()
        fake_pub = fake_key.public_key()
        
        peer_info = PeerInfo(
            peer_id=node._compute_peer_id(fake_pub),
            address=f"127.0.0.1:{8000 + i}",
            public_key=fake_pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            ),
            reputation=0.8 + (i * 0.04)  # Varying reputations
        )
        node.add_peer(peer_info)
    
    # Get stats
    stats = node.get_stats()
    print(f"\nNode Statistics:")
    print(f"  Total peers: {stats['total_peers']}")
    print(f"  Avg reputation: {stats['avg_reputation']:.2f}")
    
    # Get best peers
    best = node.get_best_peers(count=3)
    print(f"\nTop 3 peers by reputation:")
    for peer in best:
        print(f"  {peer.peer_id[:16]}... - {peer.reputation:.2f}")
    
    print("\n✅ Node initialization complete!")
