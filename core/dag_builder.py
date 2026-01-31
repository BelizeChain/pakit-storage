"""
DAG Builder - Parent block selection and DAG construction.

Implements the balanced random parent selection strategy that creates
the interconnected web structure of the Pakit DAG network.

Key Strategies:
1. Chain Property: Always link to most recent block (maintains ordering)
2. DAG Property: Link to 2-5 random historical blocks (creates web)
3. Balance: Prefer under-referenced blocks (avoids orphans)
4. Geographic: Prefer Belizean blocks (national sovereignty)
"""

from typing import List, Optional, Dict, Callable
import random
import logging
from dataclasses import dataclass
from collections import defaultdict

from .dag_storage import DagBlock, CompressionAlgorithm, create_genesis_block

logger = logging.getLogger(__name__)


@dataclass
class DagState:
    """
    Current state of the DAG network.
    
    Tracks all blocks, references, and statistics needed for
    intelligent parent selection.
    """
    
    # Block storage
    blocks: Dict[str, DagBlock]  # hash -> block
    blocks_by_depth: Dict[int, List[str]]  # depth -> [hashes]
    
    # Reference tracking
    reference_counts: Dict[str, int]  # hash -> num_children
    
    # Statistics
    total_blocks: int = 0
    max_depth: int = 0
    
    def add_block(self, block: DagBlock):
        """Add block to DAG state."""
        # Store block
        self.blocks[block.block_hash] = block
        
        # Update depth index
        if block.depth not in self.blocks_by_depth:
            self.blocks_by_depth[block.depth] = []
        self.blocks_by_depth[block.depth].append(block.block_hash)
        
        # Update reference counts for parents
        for parent_hash in block.parent_blocks:
            self.reference_counts[parent_hash] = \
                self.reference_counts.get(parent_hash, 0) + 1
        
        # Initialize reference count for this block
        if block.block_hash not in self.reference_counts:
            self.reference_counts[block.block_hash] = 0
        
        # Update statistics
        self.total_blocks += 1
        self.max_depth = max(self.max_depth, block.depth)
    
    def get_latest_block(self) -> Optional[DagBlock]:
        """Get most recent block (highest depth)."""
        if self.max_depth == 0 and self.total_blocks == 1:
            # Only genesis exists
            return self.blocks[self.blocks_by_depth[0][0]]
        
        if self.max_depth not in self.blocks_by_depth:
            return None
        
        # Get last block at max depth
        latest_hash = self.blocks_by_depth[self.max_depth][-1]
        return self.blocks.get(latest_hash)
    
    def get_blocks_in_range(
        self,
        start_depth: int,
        end_depth: int
    ) -> List[DagBlock]:
        """Get all blocks within depth range."""
        blocks = []
        for depth in range(start_depth, end_depth + 1):
            if depth in self.blocks_by_depth:
                for block_hash in self.blocks_by_depth[depth]:
                    blocks.append(self.blocks[block_hash])
        return blocks
    
    def get_under_referenced_blocks(
        self,
        min_references: int = 3,
        max_depth_diff: int = 1000
    ) -> List[str]:
        """
        Get blocks that have fewer than min_references children.
        
        These blocks are at risk of becoming orphaned if not referenced
        by new blocks. Prioritizing them helps balance the DAG.
        
        Args:
            min_references: Minimum number of child references
            max_depth_diff: Only consider blocks within this depth range
        
        Returns:
            List of under-referenced block hashes
        """
        under_referenced = []
        
        # Only look at recent blocks (not ancient history)
        min_depth = max(0, self.max_depth - max_depth_diff)
        
        for block_hash, ref_count in self.reference_counts.items():
            if ref_count < min_references:
                block = self.blocks.get(block_hash)
                if block and block.depth >= min_depth and block.depth < self.max_depth:
                    under_referenced.append(block_hash)
        
        return under_referenced


class DagBuilder:
    """
    DAG block builder with intelligent parent selection.
    
    Creates new blocks and selects parent blocks using the balanced
    random strategy that creates the interconnected web structure.
    """
    
    def __init__(
        self,
        dag_state: DagState,
        compression_engine: Optional[Callable] = None,
        prefer_belizean: bool = True
    ):
        """
        Initialize DAG builder.
        
        Args:
            dag_state: Current DAG state
            compression_engine: Optional compression function
            prefer_belizean: Prefer Belizean storage providers
        """
        self.dag_state = dag_state
        self.compression_engine = compression_engine
        self.prefer_belizean = prefer_belizean
        
        # Parent selection parameters
        self.min_random_parents = 2
        self.max_random_parents = 5
        self.under_reference_weight = 0.7  # 70% chance to help under-referenced
    
    def select_parents(
        self,
        strategy: str = "balanced_random"
    ) -> List[str]:
        """
        Select parent blocks for new block.
        
        Strategies:
        - balanced_random: Mix of recent + random + under-referenced (DEFAULT)
        - chain_only: Only most recent (degenerates to blockchain)
        - full_random: Completely random parents (experimental)
        
        Args:
            strategy: Parent selection strategy
        
        Returns:
            List of parent block hashes
        """
        if strategy == "balanced_random":
            return self._select_balanced_random()
        elif strategy == "chain_only":
            return self._select_chain_only()
        elif strategy == "full_random":
            return self._select_full_random()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _select_balanced_random(self) -> List[str]:
        """
        Balanced random parent selection (DEFAULT).
        
        Selection rules:
        1. Always link to most recent block (chain property)
        2. 70% chance: Link to under-referenced block (balance DAG)
        3. Link to 1-3 random historical blocks (create web)
        
        Returns:
            List of parent block hashes (2-5 typically)
        """
        parents = []
        
        # Rule 1: Most recent block (maintains chain ordering)
        latest_block = self.dag_state.get_latest_block()
        if latest_block:
            parents.append(latest_block.block_hash)
            logger.debug(f"Selected latest block: {latest_block.block_hash[:8]}...")
        
        # Rule 2: Under-referenced block (70% chance)
        if random.random() < self.under_reference_weight:
            under_referenced = self.dag_state.get_under_referenced_blocks(
                min_references=3,
                max_depth_diff=1000
            )
            
            if under_referenced:
                # Exclude already selected parents
                candidates = [h for h in under_referenced if h not in parents]
                if candidates:
                    selected = random.choice(candidates)
                    parents.append(selected)
                    logger.debug(f"Selected under-referenced: {selected[:8]}...")
        
        # Rule 3: Random historical blocks (1-3 blocks)
        num_random = random.randint(1, 3)
        current_depth = self.dag_state.max_depth
        
        if current_depth > 0:
            # Get blocks from last 10K depths (or all if fewer)
            start_depth = max(0, current_depth - 10000)
            historical_blocks = self.dag_state.get_blocks_in_range(
                start_depth=start_depth,
                end_depth=current_depth - 1
            )
            
            # Exclude already selected parents
            candidates = [
                b for b in historical_blocks
                if b.block_hash not in parents
            ]
            
            if candidates:
                # Select random blocks
                num_to_select = min(num_random, len(candidates))
                selected_blocks = random.sample(candidates, num_to_select)
                
                for block in selected_blocks:
                    parents.append(block.block_hash)
                    logger.debug(
                        f"Selected random historical: {block.block_hash[:8]}... "
                        f"(depth {block.depth})"
                    )
        
        # Ensure we have at least 2 parents (unless we're block #1)
        if len(parents) < 2 and current_depth > 0:
            logger.warning(
                f"Only {len(parents)} parent(s) selected, expected 2+. "
                f"DAG may be too small."
            )
        
        return parents
    
    def _select_chain_only(self) -> List[str]:
        """
        Chain-only parent selection (degenerates to blockchain).
        
        Only links to most recent block. This creates a traditional
        blockchain structure, not a DAG. Used for testing/comparison.
        
        Returns:
            List with single parent (most recent block)
        """
        latest_block = self.dag_state.get_latest_block()
        if latest_block:
            return [latest_block.block_hash]
        return []
    
    def _select_full_random(self) -> List[str]:
        """
        Fully random parent selection (experimental).
        
        Selects 2-5 completely random blocks from entire DAG history.
        No preference for recent blocks or under-referenced blocks.
        
        Returns:
            List of random parent block hashes
        """
        num_parents = random.randint(
            self.min_random_parents,
            self.max_random_parents
        )
        
        # Get all blocks
        all_blocks = list(self.dag_state.blocks.values())
        
        if len(all_blocks) <= num_parents:
            # Select all blocks as parents
            return [b.block_hash for b in all_blocks]
        
        # Select random blocks
        selected_blocks = random.sample(all_blocks, num_parents)
        return [b.block_hash for b in selected_blocks]
    
    def create_block(
        self,
        content: bytes,
        compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD,
        metadata: Optional[Dict] = None
    ) -> DagBlock:
        """
        Create new DAG block with content.
        
        Process:
        1. Compress content (if compression_engine provided)
        2. Select parent blocks using balanced random strategy
        3. Create DagBlock with parents and content
        4. Compute block hash
        5. Add to DAG state
        
        Args:
            content: Raw content to store
            compression_algorithm: Compression algorithm to use
            metadata: Optional metadata dict
        
        Returns:
            Created DagBlock
        """
        # Compress content if engine provided
        if self.compression_engine:
            try:
                compressed = self.compression_engine(content, compression_algorithm)
                compressed_size = len(compressed)
                compression_ratio = len(content) / compressed_size
            except Exception as e:
                logger.warning(f"Compression failed: {e}, storing uncompressed")
                compressed = content
                compressed_size = len(content)
                compression_ratio = 1.0
                compression_algorithm = CompressionAlgorithm.NONE
        else:
            # No compression
            compressed = content
            compressed_size = len(content)
            compression_ratio = 1.0
            compression_algorithm = CompressionAlgorithm.NONE
        
        # Select parent blocks
        parent_blocks = self.select_parents(strategy="balanced_random")
        
        # Calculate depth (max parent depth + 1)
        if parent_blocks:
            parent_depths = [
                self.dag_state.blocks[h].depth
                for h in parent_blocks
                if h in self.dag_state.blocks
            ]
            depth = max(parent_depths) + 1 if parent_depths else 1
        else:
            depth = 0  # Genesis block
        
        # Create block
        import time
        block = DagBlock(
            block_hash="",  # Will be computed
            block_size=len(content),
            compressed_size=compressed_size,
            timestamp=time.time(),
            content=compressed,
            compression_algorithm=compression_algorithm,
            compression_ratio=compression_ratio,
            parent_blocks=parent_blocks,
            depth=depth,
            merkle_proof=[],
            metadata=metadata or {}
        )
        
        # Compute hash
        block.block_hash = block.compute_hash()
        
        # Get parent blocks for Merkle proof
        parent_block_objs = [
            self.dag_state.blocks[h]
            for h in parent_blocks
            if h in self.dag_state.blocks
        ]
        
        # Build Merkle proof
        from .dag_storage import MerkleDAG
        block.merkle_proof = MerkleDAG.build_merkle_proof(block, parent_block_objs)
        
        # Add to DAG state
        self.dag_state.add_block(block)
        
        logger.info(
            f"Created block {block.block_hash[:8]}... "
            f"(depth={depth}, parents={len(parent_blocks)}, "
            f"size={len(content)}→{compressed_size} bytes)"
        )
        
        return block
    
    def get_dag_stats(self) -> Dict:
        """
        Get DAG statistics.
        
        Returns:
            Dictionary with DAG metrics
        """
        # Calculate average references per block
        if self.dag_state.total_blocks > 0:
            total_refs = sum(self.dag_state.reference_counts.values())
            avg_refs = total_refs / self.dag_state.total_blocks
        else:
            avg_refs = 0
        
        # Find under-referenced blocks
        under_referenced = self.dag_state.get_under_referenced_blocks()
        
        return {
            "total_blocks": self.dag_state.total_blocks,
            "max_depth": self.dag_state.max_depth,
            "average_references_per_block": avg_refs,
            "under_referenced_blocks": len(under_referenced),
            "genesis_block": (
                self.dag_state.blocks_by_depth.get(0, [None])[0]
                if 0 in self.dag_state.blocks_by_depth else None
            )
        }


# Example usage and tests
if __name__ == "__main__":
    # Initialize DAG state
    dag_state = DagState(
        blocks={},
        blocks_by_depth={},
        reference_counts={}
    )
    
    # Create DAG builder
    builder = DagBuilder(dag_state)
    
    # Create genesis block
    genesis = create_genesis_block()
    dag_state.add_block(genesis)
    
    print("=== DAG Builder Demo ===\n")
    print(f"Genesis: {genesis}\n")
    
    # Create 10 blocks to demonstrate parent selection
    for i in range(1, 11):
        content = f"Block {i} content - testing DAG structure".encode()
        block = builder.create_block(content)
        
        print(f"Block {i}:")
        print(f"  Hash: {block.block_hash[:16]}...")
        print(f"  Depth: {block.depth}")
        print(f"  Parents: {len(block.parent_blocks)}")
        for j, parent in enumerate(block.parent_blocks):
            print(f"    Parent {j+1}: {parent[:16]}...")
        print()
    
    # Print DAG statistics
    stats = builder.get_dag_stats()
    print("\n=== DAG Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Demonstrate under-referenced detection
    under_ref = dag_state.get_under_referenced_blocks()
    print(f"\nUnder-referenced blocks: {len(under_ref)}")
    for block_hash in under_ref[:3]:  # Show first 3
        block = dag_state.blocks[block_hash]
        refs = dag_state.reference_counts[block_hash]
        print(f"  {block_hash[:16]}... (depth {block.depth}, {refs} refs)")
