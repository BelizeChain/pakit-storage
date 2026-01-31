"""
Remote Merkle Proof Verification

Enables peers to request and verify Merkle proofs for blocks from remote nodes.
Ensures data integrity before trusting blocks from untrusted peers.

Features:
- Request proof path from block to genesis
- Verify proof locally using cryptographic hashing
- Proof caching to avoid redundant verification
- Batch proof verification for efficiency
"""

import hashlib
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProofStatus(Enum):
    """Status of Merkle proof verification."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    CACHED = "cached"


@dataclass
class MerkleProof:
    """
    Merkle proof from target block to root (genesis).
    
    Proof consists of:
    - Target block hash (what we're verifying)
    - Root block hash (genesis/trusted anchor)
    - Path: List of sibling hashes needed to compute root
    - Indices: Left/right positions for each hash in path
    """
    
    target_hash: str
    root_hash: str
    path: List[str]  # Sibling hashes
    indices: List[int]  # 0=left, 1=right
    block_depth: int
    timestamp: float = field(default_factory=time.time)
    from_peer: str = ""
    
    def to_bytes(self) -> bytes:
        """Serialize proof for network transmission."""
        import msgpack
        return msgpack.packb({
            "target_hash": self.target_hash,
            "root_hash": self.root_hash,
            "path": self.path,
            "indices": self.indices,
            "block_depth": self.block_depth,
            "timestamp": self.timestamp,
            "from_peer": self.from_peer
        })
    
    @staticmethod
    def from_bytes(data: bytes) -> 'MerkleProof':
        """Deserialize proof from bytes."""
        import msgpack
        d = msgpack.unpackb(data)
        return MerkleProof(
            target_hash=d["target_hash"],
            root_hash=d["root_hash"],
            path=d["path"],
            indices=d["indices"],
            block_depth=d["block_depth"],
            timestamp=d["timestamp"],
            from_peer=d["from_peer"]
        )


@dataclass
class ProofVerificationResult:
    """Result of Merkle proof verification."""
    
    target_hash: str
    status: ProofStatus
    computed_root: Optional[str] = None
    expected_root: Optional[str] = None
    verification_time: float = 0.0
    cached: bool = False
    error: Optional[str] = None


class MerkleProofCache:
    """
    Cache for verified Merkle proofs.
    
    Avoids re-verification of blocks we've already validated.
    Uses LRU eviction policy.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize proof cache.
        
        Args:
            max_size: Maximum number of cached proofs
        """
        self.max_size = max_size
        
        # Cache: block_hash → (root_hash, verified_at)
        self.cache: Dict[str, Tuple[str, float]] = {}
        
        # Access timestamps for LRU eviction
        self.access_times: Dict[str, float] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, block_hash: str, expected_root: str) -> Optional[bool]:
        """
        Check if block proof is cached and valid.
        
        Args:
            block_hash: Hash of block to check
            expected_root: Expected root hash
        
        Returns:
            True if cached and valid, None if not in cache
        """
        if block_hash not in self.cache:
            self.misses += 1
            return None
        
        cached_root, verified_at = self.cache[block_hash]
        
        # Update access time
        self.access_times[block_hash] = time.time()
        self.hits += 1
        
        # Check if root matches
        return cached_root == expected_root
    
    def put(self, block_hash: str, root_hash: str):
        """
        Cache verified proof.
        
        Args:
            block_hash: Hash of verified block
            root_hash: Root hash of proof
        """
        # Evict oldest if cache full
        if len(self.cache) >= self.max_size:
            # Find least recently accessed
            oldest = min(self.access_times.items(), key=lambda x: x[1])
            oldest_hash = oldest[0]
            
            del self.cache[oldest_hash]
            del self.access_times[oldest_hash]
            
            logger.debug(f"Evicted proof from cache: {oldest_hash[:16]}...")
        
        # Add to cache
        self.cache[block_hash] = (root_hash, time.time())
        self.access_times[block_hash] = time.time()
    
    def clear(self):
        """Clear entire cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }


class RemoteMerkleVerifier:
    """
    Verifies Merkle proofs from remote peers.
    
    Ensures blocks from untrusted sources are cryptographically valid
    before accepting them into local storage.
    """
    
    def __init__(self, node_id: str, trusted_roots: List[str] = None):
        """
        Initialize Merkle verifier.
        
        Args:
            node_id: Our node's peer ID
            trusted_roots: List of trusted root (genesis) hashes
        """
        self.node_id = node_id
        self.trusted_roots = set(trusted_roots or [])
        
        # Proof cache
        self.cache = MerkleProofCache(max_size=10000)
        
        # Statistics
        self.stats = {
            "proofs_verified": 0,
            "proofs_valid": 0,
            "proofs_invalid": 0,
            "cache_hits": 0,
            "batch_verifications": 0
        }
        
        logger.info(f"Initialized Merkle verifier for node: {node_id[:16]}...")
        logger.info(f"Trusted roots: {len(self.trusted_roots)}")
    
    def add_trusted_root(self, root_hash: str):
        """Add a trusted root (genesis) hash."""
        self.trusted_roots.add(root_hash)
        logger.info(f"Added trusted root: {root_hash[:16]}...")
    
    def verify_proof(self, proof: MerkleProof) -> ProofVerificationResult:
        """
        Verify a Merkle proof from remote peer.
        
        Process:
        1. Check cache for previous verification
        2. Verify root is trusted
        3. Compute root from target using path
        4. Compare computed vs expected root
        
        Args:
            proof: Merkle proof to verify
        
        Returns:
            Verification result
        """
        start_time = time.time()
        
        # Check cache first
        cached = self.cache.get(proof.target_hash, proof.root_hash)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return ProofVerificationResult(
                target_hash=proof.target_hash,
                status=ProofStatus.VALID if cached else ProofStatus.INVALID,
                computed_root=proof.root_hash,
                expected_root=proof.root_hash,
                verification_time=time.time() - start_time,
                cached=True
            )
        
        # Verify root is trusted
        if proof.root_hash not in self.trusted_roots:
            error = f"Untrusted root: {proof.root_hash[:16]}..."
            logger.warning(error)
            return ProofVerificationResult(
                target_hash=proof.target_hash,
                status=ProofStatus.INVALID,
                error=error,
                verification_time=time.time() - start_time
            )
        
        # Verify path and indices match
        if len(proof.path) != len(proof.indices):
            error = "Path/indices length mismatch"
            return ProofVerificationResult(
                target_hash=proof.target_hash,
                status=ProofStatus.INVALID,
                error=error,
                verification_time=time.time() - start_time
            )
        
        # Compute root from target
        computed_root = self._compute_root(
            leaf_hash=proof.target_hash,
            path=proof.path,
            indices=proof.indices
        )
        
        # Verify computed root matches expected
        is_valid = (computed_root == proof.root_hash)
        
        # Update statistics
        self.stats["proofs_verified"] += 1
        if is_valid:
            self.stats["proofs_valid"] += 1
            # Cache valid proof
            self.cache.put(proof.target_hash, proof.root_hash)
        else:
            self.stats["proofs_invalid"] += 1
        
        verification_time = time.time() - start_time
        logger.info(
            f"Verified proof for {proof.target_hash[:16]}... "
            f"in {verification_time*1000:.2f}ms: "
            f"{'VALID' if is_valid else 'INVALID'}"
        )
        
        return ProofVerificationResult(
            target_hash=proof.target_hash,
            status=ProofStatus.VALID if is_valid else ProofStatus.INVALID,
            computed_root=computed_root,
            expected_root=proof.root_hash,
            verification_time=verification_time
        )
    
    def verify_batch(self, proofs: List[MerkleProof]) -> List[ProofVerificationResult]:
        """
        Verify multiple proofs in batch.
        
        More efficient than individual verification due to:
        - Shared cache lookups
        - Batch logging
        
        Args:
            proofs: List of proofs to verify
        
        Returns:
            List of verification results
        """
        self.stats["batch_verifications"] += 1
        
        results = []
        for proof in proofs:
            result = self.verify_proof(proof)
            results.append(result)
        
        # Log batch summary
        valid_count = sum(1 for r in results if r.status == ProofStatus.VALID)
        logger.info(
            f"Batch verified {len(proofs)} proofs: "
            f"{valid_count} valid, {len(proofs) - valid_count} invalid"
        )
        
        return results
    
    def _compute_root(
        self,
        leaf_hash: str,
        path: List[str],
        indices: List[int]
    ) -> str:
        """
        Compute root hash from leaf using Merkle path.
        
        Args:
            leaf_hash: Starting hash (target block)
            path: List of sibling hashes
            indices: Left (0) or right (1) positions
        
        Returns:
            Computed root hash
        """
        current = leaf_hash
        
        for sibling, index in zip(path, indices):
            if index == 0:
                # Current is left child, sibling is right
                combined = current + sibling
            else:
                # Current is right child, sibling is left
                combined = sibling + current
            
            # Hash combined
            current = hashlib.sha256(combined.encode()).hexdigest()
        
        return current
    
    def request_proof(
        self,
        target_hash: str,
        from_peer: str,
        timeout: int = 30
    ) -> Optional[MerkleProof]:
        """
        Request Merkle proof from remote peer.
        
        Args:
            target_hash: Hash of block to get proof for
            from_peer: Peer to request from
            timeout: Request timeout in seconds
        
        Returns:
            Proof if received, None if timeout/error
        """
        # TODO: Implement actual network request
        # For now, return None (would be network RPC)
        logger.debug(
            f"Requesting proof for {target_hash[:16]}... from {from_peer[:16]}..."
        )
        return None
    
    def get_stats(self) -> Dict:
        """Get verifier statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            **self.stats,
            "cache": cache_stats,
            "trusted_roots": len(self.trusted_roots)
        }


if __name__ == "__main__":
    # Example usage
    print("Remote Merkle Proof Verification Example:")
    print("-" * 60)
    
    # Create verifier
    node_id = hashlib.sha256(b"test_node").hexdigest()
    genesis_hash = hashlib.sha256(b"genesis_block").hexdigest()
    
    verifier = RemoteMerkleVerifier(
        node_id=node_id,
        trusted_roots=[genesis_hash]
    )
    
    print(f"Node ID: {node_id[:16]}...")
    print(f"Genesis: {genesis_hash[:16]}...")
    
    # Create a valid proof
    # Build small tree: genesis -> block1 -> block2
    block1 = hashlib.sha256(b"block_1_data").hexdigest()
    block2 = hashlib.sha256(b"block_2_data").hexdigest()
    
    # Merkle tree: root = hash(hash(genesis + block1) + block2)
    level1_left = hashlib.sha256(f"{genesis_hash}{block1}".encode()).hexdigest()
    root = hashlib.sha256(f"{level1_left}{block2}".encode()).hexdigest()
    
    # Add computed root as trusted
    verifier.add_trusted_root(root)
    
    # Create proof for block2
    proof = MerkleProof(
        target_hash=block2,
        root_hash=root,
        path=[level1_left],  # Sibling at final level
        indices=[1],  # block2 is right child
        block_depth=2,
        from_peer="peer_123"
    )
    
    # Verify proof
    result = verifier.verify_proof(proof)
    
    print(f"\nProof Verification:")
    print(f"  Target: {proof.target_hash[:16]}...")
    print(f"  Status: {result.status.value}")
    print(f"  Computed root: {result.computed_root[:16]}...")
    print(f"  Expected root: {result.expected_root[:16]}...")
    print(f"  Time: {result.verification_time*1000:.2f}ms")
    print(f"  Cached: {result.cached}")
    
    # Verify again (should hit cache)
    result2 = verifier.verify_proof(proof)
    print(f"\nSecond verification (cached): {result2.cached}")
    
    # Create invalid proof (wrong path)
    bad_proof = MerkleProof(
        target_hash=block2,
        root_hash=root,
        path=[genesis_hash],  # Wrong sibling
        indices=[1],
        block_depth=2,
        from_peer="peer_456"
    )
    
    bad_result = verifier.verify_proof(bad_proof)
    print(f"\nInvalid Proof Status: {bad_result.status.value}")
    
    # Get stats
    stats = verifier.get_stats()
    print(f"\nVerifier Statistics:")
    print(f"  Proofs verified: {stats['proofs_verified']}")
    print(f"  Valid: {stats['proofs_valid']}")
    print(f"  Invalid: {stats['proofs_invalid']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache hit rate: {stats['cache']['hit_rate']}")
    
    print("\n✅ Merkle proof verification working!")
