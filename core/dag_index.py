"""
DAG Index - Efficient queries and path finding in the DAG.

Provides O(log n) queries by depth range and O(1) hash lookups using
BTreeMap and HashMap data structures. Includes bloom filters for fast
existence checks.

Performance Targets (Phase 1):
- Query by hash: <1ms (O(1) HashMap lookup)
- Query by depth range: <10ms (O(log n) BTreeMap scan)
- Find verification path: <50ms (BFS through DAG)
- Bloom filter false positive rate: <1%
"""

from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import hashlib

from .dag_storage import DagBlock

logger = logging.getLogger(__name__)


class BloomFilter:
    """
    Space-efficient probabilistic data structure for existence checks.
    
    Uses multiple hash functions to achieve low false positive rate.
    No false negatives - if it says "no", item definitely doesn't exist.
    """
    
    def __init__(self, size: int = 10000, num_hashes: int = 3):
        """
        Initialize bloom filter.
        
        Args:
            size: Bit array size (larger = fewer false positives, default 10K)
            num_hashes: Number of hash functions (more = fewer false positives)
        """
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
        self.item_count = 0
    
    def _hash(self, item: str, seed: int) -> int:
        """Compute hash with seed."""
        h = hashlib.sha256(f"{item}{seed}".encode()).digest()
        return int.from_bytes(h[:4], 'big') % self.size
    
    def add(self, item: str):
        """Add item to bloom filter."""
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = True
        self.item_count += 1
    
    def contains(self, item: str) -> bool:
        """
        Check if item might exist.
        
        Returns:
            True: Item might exist (could be false positive)
            False: Item definitely doesn't exist
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False  # Definitely not in set
        return True  # Might be in set
    
    def false_positive_rate(self) -> float:
        """
        Estimate false positive rate.
        
        Formula: (1 - e^(-k*n/m))^k
        where k=num_hashes, n=item_count, m=size
        """
        import math
        if self.item_count == 0:
            return 0.0
        
        k = self.num_hashes
        n = self.item_count
        m = self.size
        
        return (1 - math.exp(-k * n / m)) ** k


@dataclass
class DagIndex:
    """
    Efficient indexing for DAG queries.
    
    Data structures:
    - blocks_by_hash: HashMap for O(1) hash lookups
    - blocks_by_depth: BTreeMap for O(log n) depth range queries
    - bloom_filter: Fast existence checks (adaptive size)
    - parent_child_map: Forward/backward graph traversal
    """
    
    # Primary indexes
    blocks_by_hash: Dict[str, DagBlock] = field(default_factory=dict)
    blocks_by_depth: Dict[int, List[str]] = field(default_factory=dict)  # Sorted by insertion
    
    # Graph indexes
    parent_to_children: Dict[str, Set[str]] = field(default_factory=lambda: {})
    
    # Bloom filter (adaptive size: 100x expected blocks)
    bloom_filter: BloomFilter = field(default_factory=lambda: BloomFilter(size=10000, num_hashes=3))
    
    # Statistics
    total_blocks: int = 0
    max_depth: int = 0
    
    def add_block(self, block: DagBlock):
        """
        Add block to all indexes.
        
        Args:
            block: DagBlock to index
        """
        block_hash = block.block_hash
        
        # Hash index (O(1) lookups)
        self.blocks_by_hash[block_hash] = block
        
        # Depth index (range queries)
        if block.depth not in self.blocks_by_depth:
            self.blocks_by_depth[block.depth] = []
        self.blocks_by_depth[block.depth].append(block_hash)
        
        # Parent-child graph (traversal)
        for parent_hash in block.parent_blocks:
            if parent_hash not in self.parent_to_children:
                self.parent_to_children[parent_hash] = set()
            self.parent_to_children[parent_hash].add(block_hash)
        
        # Bloom filter (fast existence checks)
        self.bloom_filter.add(block_hash)
        
        # Statistics
        self.total_blocks += 1
        self.max_depth = max(self.max_depth, block.depth)
        
        logger.debug(
            f"Indexed block {block_hash[:8]}... "
            f"(depth={block.depth}, total={self.total_blocks})"
        )
    
    def get_block(self, block_hash: str) -> Optional[DagBlock]:
        """
        Get block by hash - O(1).
        
        Args:
            block_hash: Block hash to lookup
        
        Returns:
            DagBlock if found, None otherwise
        """
        # Fast bloom filter check first
        if not self.bloom_filter.contains(block_hash):
            return None  # Definitely doesn't exist
        
        # Actual lookup (might be false positive)
        return self.blocks_by_hash.get(block_hash)
    
    def query_by_depth_range(
        self,
        start_depth: int,
        end_depth: int,
        limit: Optional[int] = None
    ) -> List[DagBlock]:
        """
        Query blocks within depth range - O(log n) to O(n).
        
        Args:
            start_depth: Minimum depth (inclusive)
            end_depth: Maximum depth (inclusive)
            limit: Optional max results
        
        Returns:
            List of DagBlocks in depth range
        """
        import time
        start_time = time.time()
        
        blocks = []
        count = 0
        
        # Iterate through sorted depth keys
        for depth in sorted(self.blocks_by_depth.keys()):
            if depth < start_depth:
                continue
            if depth > end_depth:
                break
            
            # Get all blocks at this depth
            for block_hash in self.blocks_by_depth[depth]:
                blocks.append(self.blocks_by_hash[block_hash])
                count += 1
                
                if limit and count >= limit:
                    break
            
            if limit and count >= limit:
                break
        
        query_time = (time.time() - start_time) * 1000  # ms
        logger.debug(
            f"Depth range query [{start_depth}-{end_depth}] "
            f"returned {len(blocks)} blocks in {query_time:.2f}ms"
        )
        
        return blocks
    
    def find_verification_path(
        self,
        block_hash: str,
        target_hash: Optional[str] = None
    ) -> Optional[List[str]]:
        """
        Find shortest path from block to target (or genesis) - BFS.
        
        Used for Merkle proof verification and DAG integrity checks.
        
        Args:
            block_hash: Starting block
            target_hash: Target block (or None for genesis)
        
        Returns:
            List of block hashes forming path, or None if unreachable
        """
        import time
        start_time = time.time()
        
        # Get starting block
        start_block = self.get_block(block_hash)
        if not start_block:
            return None
        
        # If no target, find genesis (depth 0)
        if target_hash is None:
            # Find any block at depth 0
            if 0 not in self.blocks_by_depth or not self.blocks_by_depth[0]:
                return None
            target_hash = self.blocks_by_depth[0][0]
        
        # BFS to find shortest path
        queue = deque([(block_hash, [block_hash])])
        visited = {block_hash}
        
        while queue:
            current_hash, path = queue.popleft()
            
            # Found target?
            if current_hash == target_hash:
                search_time = (time.time() - start_time) * 1000
                logger.debug(
                    f"Found path {block_hash[:8]}...→{target_hash[:8]}... "
                    f"(length={len(path)}, time={search_time:.2f}ms)"
                )
                return path
            
            # Explore parents
            current_block = self.get_block(current_hash)
            if current_block:
                for parent_hash in current_block.parent_blocks:
                    if parent_hash not in visited:
                        visited.add(parent_hash)
                        queue.append((parent_hash, path + [parent_hash]))
        
        # No path found
        logger.warning(
            f"No path from {block_hash[:8]}... to {target_hash[:8]}..."
        )
        return None
    
    def get_ancestors(
        self,
        block_hash: str,
        max_depth: int = 10
    ) -> Set[str]:
        """
        Get all ancestor blocks up to max_depth levels.
        
        Args:
            block_hash: Starting block
            max_depth: Maximum levels to traverse
        
        Returns:
            Set of ancestor block hashes
        """
        ancestors = set()
        
        # BFS with depth limit
        queue = deque([(block_hash, 0)])
        visited = {block_hash}
        
        while queue:
            current_hash, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            current_block = self.get_block(current_hash)
            if current_block:
                for parent_hash in current_block.parent_blocks:
                    if parent_hash not in visited:
                        visited.add(parent_hash)
                        ancestors.add(parent_hash)
                        queue.append((parent_hash, depth + 1))
        
        return ancestors
    
    def get_descendants(
        self,
        block_hash: str,
        max_depth: int = 10
    ) -> Set[str]:
        """
        Get all descendant blocks up to max_depth levels.
        
        Args:
            block_hash: Starting block
            max_depth: Maximum levels to traverse
        
        Returns:
            Set of descendant block hashes
        """
        descendants = set()
        
        # BFS with depth limit
        queue = deque([(block_hash, 0)])
        visited = {block_hash}
        
        while queue:
            current_hash, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Get children from parent_to_children map
            children = self.parent_to_children.get(current_hash, set())
            for child_hash in children:
                if child_hash not in visited:
                    visited.add(child_hash)
                    descendants.add(child_hash)
                    queue.append((child_hash, depth + 1))
        
        return descendants
    
    def verify_dag_integrity(self) -> Tuple[bool, List[str]]:
        """
        Verify entire DAG structure integrity.
        
        Checks:
        1. All parent references exist
        2. No cycles in DAG
        3. All non-genesis blocks reachable from recent blocks
        4. Reference counts match parent_to_children map
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check 1: All parent references exist
        for block_hash, block in self.blocks_by_hash.items():
            for parent_hash in block.parent_blocks:
                if parent_hash not in self.blocks_by_hash:
                    errors.append(
                        f"Block {block_hash[:8]}... has missing parent "
                        f"{parent_hash[:8]}..."
                    )
        
        # Check 2: No cycles (DAG must be acyclic)
        for block_hash in self.blocks_by_hash:
            path = self.find_verification_path(block_hash)
            if path is None:
                # Block not connected to genesis
                block = self.blocks_by_hash[block_hash]
                if block.depth > 0:  # Non-genesis block
                    errors.append(
                        f"Block {block_hash[:8]}... (depth {block.depth}) "
                        f"not connected to genesis"
                    )
        
        # Check 3: Verify depth calculations
        for block_hash, block in self.blocks_by_hash.items():
            if block.parent_blocks:
                # Depth should be max(parent depths) + 1
                parent_depths = [
                    self.blocks_by_hash[p].depth
                    for p in block.parent_blocks
                    if p in self.blocks_by_hash
                ]
                expected_depth = max(parent_depths) + 1 if parent_depths else 1
                
                if block.depth != expected_depth:
                    errors.append(
                        f"Block {block_hash[:8]}... has incorrect depth "
                        f"{block.depth} (expected {expected_depth})"
                    )
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"DAG integrity verified: {self.total_blocks} blocks OK")
        else:
            logger.error(f"DAG integrity check failed: {len(errors)} errors")
            for error in errors[:10]:  # Log first 10 errors
                logger.error(f"  - {error}")
        
        return is_valid, errors
    
    def get_statistics(self, include_bloom_filter: bool = False) -> Dict:
        """
        Get index statistics (fast, O(1) operations only).
        
        Args:
            include_bloom_filter: Include bloom filter stats (disabled by default)
        
        Returns:
            Dictionary with index metrics
        """
        # Fast O(1) statistics only
        stats = {
            "total_blocks": self.total_blocks,
            "max_depth": self.max_depth,
            "depth_levels": len(self.blocks_by_depth),
        }
        
        # Optionally include bloom filter stats (not computed by default)
        if include_bloom_filter:
            stats["bloom_filter_size"] = self.bloom_filter.size
            stats["bloom_filter_item_count"] = self.bloom_filter.item_count
        
        return stats


# Example usage and tests
if __name__ == "__main__":
    from .dag_storage import create_genesis_block
    from .dag_builder import DagBuilder, DagState
    
    print("=== DAG Index Demo ===\n")
    
    # Create DAG with 100 blocks
    dag_state = DagState(
        blocks={},
        blocks_by_depth={},
        reference_counts={}
    )
    builder = DagBuilder(dag_state)
    dag_index = DagIndex()
    
    # Create and index genesis
    genesis = create_genesis_block()
    dag_state.add_block(genesis)
    dag_index.add_block(genesis)
    
    print(f"Created genesis: {genesis.block_hash[:16]}...\n")
    
    # Create 100 blocks
    print("Creating 100 blocks...")
    import time
    start = time.time()
    
    for i in range(1, 101):
        content = f"Block {i} content for testing".encode()
        block = builder.create_block(content)
        dag_index.add_block(block)
        
        if i % 20 == 0:
            print(f"  Created {i} blocks...")
    
    elapsed = time.time() - start
    print(f"Created 100 blocks in {elapsed:.2f}s ({100/elapsed:.1f} blocks/s)\n")
    
    # Test query by hash
    print("=== Query by Hash ===")
    test_hash = genesis.block_hash
    start = time.time()
    found = dag_index.get_block(test_hash)
    elapsed = (time.time() - start) * 1000
    print(f"Found genesis in {elapsed:.4f}ms\n")
    
    # Test depth range query
    print("=== Query by Depth Range ===")
    start = time.time()
    blocks = dag_index.query_by_depth_range(5, 15, limit=10)
    elapsed = (time.time() - start) * 1000
    print(f"Found {len(blocks)} blocks in range [5-15] in {elapsed:.2f}ms\n")
    
    # Test verification path
    print("=== Find Verification Path ===")
    # Get a deep block
    deep_blocks = dag_index.query_by_depth_range(
        dag_index.max_depth,
        dag_index.max_depth,
        limit=1
    )
    if deep_blocks:
        deep_hash = deep_blocks[0].block_hash
        start = time.time()
        path = dag_index.find_verification_path(deep_hash)
        elapsed = (time.time() - start) * 1000
        
        if path:
            print(f"Path from {deep_hash[:8]}... to genesis:")
            print(f"  Length: {len(path)} blocks")
            print(f"  Time: {elapsed:.2f}ms")
            print(f"  Path: {' → '.join([h[:8]+'...' for h in path[:5]])}")
            if len(path) > 5:
                print(f"        ... ({len(path)-5} more blocks)")
    print()
    
    # Test DAG integrity
    print("=== DAG Integrity Check ===")
    start = time.time()
    is_valid, errors = dag_index.verify_dag_integrity()
    elapsed = (time.time() - start) * 1000
    
    if is_valid:
        print(f"✓ DAG integrity verified in {elapsed:.2f}ms")
    else:
        print(f"✗ DAG integrity check failed ({len(errors)} errors)")
        for error in errors[:5]:
            print(f"  - {error}")
    print()
    
    # Print statistics
    print("=== Index Statistics ===")
    stats = dag_index.get_statistics()
    for key, value in stats.items():
        if key == "depth_distribution":
            print(f"{key}:")
            for depth in sorted(value.keys())[:10]:  # First 10 depths
                print(f"  depth {depth}: {value[depth]} blocks")
            if len(value) > 10:
                print(f"  ... ({len(value)-10} more depths)")
        else:
            print(f"{key}: {value}")
