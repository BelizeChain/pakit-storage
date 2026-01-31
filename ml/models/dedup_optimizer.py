"""
Deduplication Optimizer

Uses SimHash and LSH for intelligent content similarity detection.
Finds duplicate blocks faster and more accurately than fixed chunking.
"""

from typing import Dict, Any, List, Optional, Set
import hashlib
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import simhash
    SIMHASH_AVAILABLE = True
except ImportError:
    SIMHASH_AVAILABLE = False
    logger.warning("SimHash library not available")

from pakit.ml.base_model import PakitMLModel, ModelConfig


class SimHashEngine:
    """
    SimHash implementation for content fingerprinting.
    
    Generates 64-bit fingerprints for similarity detection.
    """
    
    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits
    
    def compute_simhash(self, data: bytes, shingle_size: int = 4) -> int:
        """
        Compute SimHash fingerprint.
        
        Args:
            data: Block data
            shingle_size: Size of shingles (n-grams)
            
        Returns:
            64-bit SimHash value
        """
        if SIMHASH_AVAILABLE:
            # Use simhash library
            return simhash.Simhash(data).value
        else:
            # Simplified implementation
            return self._simple_simhash(data, shingle_size)
    
    def _simple_simhash(self, data: bytes, shingle_size: int) -> int:
        """Simplified SimHash (fallback)."""
        # Create shingles
        shingles = self._create_shingles(data, shingle_size)
        
        # Initialize vector
        v = [0] * self.hash_bits
        
        # Process each shingle
        for shingle in shingles:
            # Hash shingle
            h = int(hashlib.sha256(shingle).hexdigest()[:16], 16)
            
            # Update vector
            for i in range(self.hash_bits):
                if (h >> i) & 1:
                    v[i] += 1
                else:
                    v[i] -= 1
        
        # Generate fingerprint
        fingerprint = 0
        for i in range(self.hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)
        
        return fingerprint
    
    def _create_shingles(self, data: bytes, size: int) -> List[bytes]:
        """Create shingles (n-grams)."""
        shingles = []
        for i in range(len(data) - size + 1):
            shingles.append(data[i:i+size])
        return shingles
    
    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """Compute Hamming distance between two hashes."""
        xor = hash1 ^ hash2
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1
        return distance
    
    def similarity(self, hash1: int, hash2: int) -> float:
        """Compute similarity (0.0 = different, 1.0 = identical)."""
        distance = self.hamming_distance(hash1, hash2)
        return 1.0 - (distance / self.hash_bits)


class LSHIndex:
    """
    Locality-Sensitive Hashing index for fast similarity search.
    
    Uses 16 hash tables with 4 hash functions each.
    """
    
    def __init__(self, num_tables: int = 16, band_size: int = 4):
        self.num_tables = num_tables
        self.band_size = band_size
        self.tables: List[Dict[int, Set[str]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]
    
    def add(self, block_hash: str, simhash_value: int) -> None:
        """
        Add block to LSH index.
        
        Args:
            block_hash: Block hash (identifier)
            simhash_value: SimHash fingerprint
        """
        # Insert into each table
        for table_idx in range(self.num_tables):
            # Extract band
            band = self._extract_band(simhash_value, table_idx)
            
            # Add to table
            self.tables[table_idx][band].add(block_hash)
    
    def query(self, simhash_value: int, max_results: int = 10) -> List[str]:
        """
        Find similar blocks.
        
        Args:
            simhash_value: Query SimHash
            max_results: Maximum results to return
            
        Returns:
            List of candidate block hashes
        """
        candidates = set()
        
        # Query each table
        for table_idx in range(self.num_tables):
            band = self._extract_band(simhash_value, table_idx)
            
            if band in self.tables[table_idx]:
                candidates.update(self.tables[table_idx][band])
            
            if len(candidates) >= max_results:
                break
        
        return list(candidates)[:max_results]
    
    def _extract_band(self, simhash_value: int, table_idx: int) -> int:
        """Extract band for specific table."""
        # Extract bits for this band
        start_bit = table_idx * self.band_size
        mask = (1 << self.band_size) - 1
        band = (simhash_value >> start_bit) & mask
        return band
    
    def size(self) -> int:
        """Get total number of indexed blocks."""
        all_blocks = set()
        for table in self.tables:
            for blocks in table.values():
                all_blocks.update(blocks)
        return len(all_blocks)


class DeduplicationOptimizer(PakitMLModel):
    """
    Intelligent deduplication using SimHash + LSH.
    
    Target: 10-20% deduplication improvement, <5% false positives, <10ms lookup
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.simhash_engine = SimHashEngine(hash_bits=64)
        self.lsh_index = LSHIndex(num_tables=16, band_size=4)
        
        # Cache of computed SimHashes
        self.simhash_cache: Dict[str, int] = {}
        
        # Similarity threshold for deduplication
        self.similarity_threshold = 0.85
    
    def predict(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Find duplicate block.
        
        Args:
            features: Must include 'block_hash' and 'block_data'
            
        Returns:
            Hash of duplicate block, or None if no duplicate
        """
        start = time.time()
        
        block_hash = features.get('block_hash')
        block_data = features.get('block_data')
        
        if not block_data:
            return None
        
        # Compute SimHash
        simhash_value = self.simhash_engine.compute_simhash(block_data)
        self.simhash_cache[block_hash] = simhash_value
        
        # Query LSH index
        candidates = self.lsh_index.query(simhash_value, max_results=10)
        
        # Find best match above threshold
        best_match = None
        best_similarity = 0.0
        
        for candidate_hash in candidates:
            if candidate_hash == block_hash:
                continue
            
            if candidate_hash in self.simhash_cache:
                candidate_simhash = self.simhash_cache[candidate_hash]
                similarity = self.simhash_engine.similarity(
                    simhash_value, candidate_simhash
                )
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match = candidate_hash
        
        # Add to index
        self.lsh_index.add(block_hash, simhash_value)
        
        self._track_inference(time.time() - start)
        
        return best_match
    
    def get_similarity(self, block_hash1: str, block_hash2: str) -> float:
        """Get similarity between two blocks."""
        if block_hash1 not in self.simhash_cache or block_hash2 not in self.simhash_cache:
            return 0.0
        
        simhash1 = self.simhash_cache[block_hash1]
        simhash2 = self.simhash_cache[block_hash2]
        
        return self.simhash_engine.similarity(simhash1, simhash2)
    
    def find_similar_blocks(
        self,
        block_hash: str,
        min_similarity: float = 0.8,
        max_results: int = 5
    ) -> List[tuple]:
        """
        Find blocks similar to given block.
        
        Returns:
            List of (block_hash, similarity) tuples
        """
        if block_hash not in self.simhash_cache:
            return []
        
        simhash_value = self.simhash_cache[block_hash]
        candidates = self.lsh_index.query(simhash_value, max_results=max_results*2)
        
        results = []
        for candidate in candidates:
            if candidate == block_hash:
                continue
            
            similarity = self.get_similarity(block_hash, candidate)
            if similarity >= min_similarity:
                results.append((candidate, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def train(self, dataset: Any, validation_split: float = 0.2) -> Dict[str, float]:
        """
        Build LSH index from dataset.
        
        Args:
            dataset: Dataset of blocks with hashes and data
            validation_split: Unused (unsupervised)
            
        Returns:
            Indexing metrics
        """
        start = time.time()
        indexed = 0
        
        logger.info(f"Building LSH index for {len(dataset)} blocks")
        
        for sample in dataset:
            block_hash = sample.get('block_hash')
            block_data = sample.get('block_data')
            
            if block_data:
                simhash_value = self.simhash_engine.compute_simhash(block_data)
                self.simhash_cache[block_hash] = simhash_value
                self.lsh_index.add(block_hash, simhash_value)
                indexed += 1
        
        elapsed = time.time() - start
        
        logger.info(f"Indexed {indexed} blocks in {elapsed:.2f}s")
        
        return {
            'indexed_blocks': indexed,
            'index_size': self.lsh_index.size(),
            'time_seconds': elapsed,
        }
    
    def save(self, path: str) -> None:
        """Save LSH index to disk."""
        import pickle
        
        data = {
            'simhash_cache': self.simhash_cache,
            'lsh_tables': self.lsh_index.tables,
            'config': self.config.to_dict(),
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {self.config.name} to {path}")
    
    def load(self, path: str) -> None:
        """Load LSH index from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.simhash_cache = data['simhash_cache']
        self.lsh_index.tables = data['lsh_tables']
        
        logger.info(f"Loaded {self.config.name} from {path} ({len(self.simhash_cache)} blocks)")
