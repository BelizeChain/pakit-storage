"""
Content-addressed storage engine.

Every piece of data is identified by its cryptographic hash (content ID).
Enables efficient deduplication and content verification.
"""

import hashlib
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContentID:
    """Content identifier (hash of data)."""
    hash_algorithm: str  # e.g., "sha256"
    hash_value: bytes    # Binary hash
    
    @property
    def hex(self) -> str:
        """Get hex representation of hash."""
        return self.hash_value.hex()
    
    @property
    def base58(self) -> str:
        """Get base58 representation (IPFS-style)."""
        # Simplified base58 encoding
        # In production, use base58 library
        return f"Qm{self.hex[:44]}"  # CID v0 style
    
    def __str__(self) -> str:
        return self.hex
    
    def __repr__(self) -> str:
        return f"ContentID({self.hash_algorithm}:{self.hex[:16]}...)"


class ContentAddressingEngine:
    """
    Content-addressed storage engine.
    
    Uses cryptographic hashes to uniquely identify content.
    Enables:
    - Automatic deduplication (same content = same ID)
    - Content verification (hash check on retrieval)
    - Distributed storage (content ID is location-independent)
    """
    
    def __init__(self, hash_algorithm: str = "sha256"):
        """
        Initialize content addressing engine.
        
        Args:
            hash_algorithm: Hash algorithm to use (sha256, sha3_256, blake2b)
        """
        self.hash_algorithm = hash_algorithm
        
        # Supported hash algorithms
        self.hash_functions = {
            "sha256": hashlib.sha256,
            "sha3_256": hashlib.sha3_256,
            "blake2b": hashlib.blake2b,
            "sha512": hashlib.sha512,
        }
        
        if hash_algorithm not in self.hash_functions:
            raise ValueError(f"Unsupported hash algorithm: {hash_algorithm}")
        
        logger.info(f"Initialized content addressing with {hash_algorithm}")
    
    def compute_content_id(self, data: bytes) -> ContentID:
        """
        Compute content ID for data.
        
        Args:
            data: Raw data bytes
        
        Returns:
            ContentID with cryptographic hash
        """
        hash_func = self.hash_functions[self.hash_algorithm]
        hash_value = hash_func(data).digest()
        
        return ContentID(
            hash_algorithm=self.hash_algorithm,
            hash_value=hash_value
        )
    
    def verify_content(self, data: bytes, content_id: ContentID) -> bool:
        """
        Verify data matches content ID.
        
        Args:
            data: Data to verify
            content_id: Expected content ID
        
        Returns:
            True if data matches content ID
        """
        computed_id = self.compute_content_id(data)
        return computed_id.hash_value == content_id.hash_value
    
    def compute_merkle_root(self, content_ids: list[ContentID]) -> ContentID:
        """
        Compute Merkle root of multiple content IDs.
        
        Useful for:
        - Chunked file storage (prove file integrity from chunk hashes)
        - Batch verification (prove multiple files with single hash)
        - Tree-structured data (efficient partial updates)
        
        Args:
            content_ids: List of content IDs to combine
        
        Returns:
            Merkle root content ID
        """
        if not content_ids:
            raise ValueError("Cannot compute Merkle root of empty list")
        
        if len(content_ids) == 1:
            return content_ids[0]
        
        # Build Merkle tree bottom-up
        level = content_ids
        while len(level) > 1:
            next_level = []
            
            # Pair up hashes and combine
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    # Combine two hashes
                    combined = level[i].hash_value + level[i + 1].hash_value
                else:
                    # Odd one out, duplicate it
                    combined = level[i].hash_value + level[i].hash_value
                
                # Hash the combination
                hash_func = self.hash_functions[self.hash_algorithm]
                parent_hash = hash_func(combined).digest()
                
                next_level.append(ContentID(
                    hash_algorithm=self.hash_algorithm,
                    hash_value=parent_hash
                ))
            
            level = next_level
        
        return level[0]
    
    def chunk_data(
        self,
        data: bytes,
        chunk_size: int = 256 * 1024  # 256KB default
    ) -> list[tuple[bytes, ContentID]]:
        """
        Chunk data into fixed-size blocks and compute content IDs.
        
        Enables:
        - Efficient large file storage
        - Parallel uploads/downloads
        - Partial deduplication (shared chunks)
        
        Args:
            data: Data to chunk
            chunk_size: Size of each chunk in bytes
        
        Returns:
            List of (chunk_data, content_id) tuples
        """
        chunks = []
        offset = 0
        
        while offset < len(data):
            chunk = data[offset:offset + chunk_size]
            content_id = self.compute_content_id(chunk)
            chunks.append((chunk, content_id))
            offset += chunk_size
        
        logger.debug(f"Chunked {len(data)} bytes into {len(chunks)} chunks")
        
        return chunks
    
    def reassemble_chunks(self, chunks: list[bytes]) -> bytes:
        """
        Reassemble data from chunks.
        
        Args:
            chunks: List of chunk data in order
        
        Returns:
            Reassembled original data
        """
        return b"".join(chunks)
