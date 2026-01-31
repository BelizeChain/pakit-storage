"""
Deduplication engine for eliminating redundant storage.

Never store the same content twice - the heart of data farm elimination.
"""

from typing import Dict, Set, Optional
from dataclasses import dataclass
import logging
from threading import RLock  # Use reentrant lock to prevent deadlock

from .content_addressing import ContentID

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationStats:
    """Deduplication statistics."""
    total_stores: int
    unique_content: int
    duplicate_saves: int
    bytes_saved: int
    
    @property
    def deduplication_ratio(self) -> float:
        """Ratio of deduplicated to total stores."""
        return self.duplicate_saves / self.total_stores if self.total_stores > 0 else 0.0
    
    @property
    def efficiency_percent(self) -> float:
        """Deduplication efficiency percentage."""
        return self.deduplication_ratio * 100


class DeduplicationEngine:
    """
    Content deduplication engine.
    
    Tracks which content has been stored and eliminates redundant storage.
    Uses reference counting to know when content can be safely deleted.
    
    This is a key component of data farm elimination:
    - Never store duplicate data
    - Automatic reference counting
    - Safe garbage collection
    """
    
    def __init__(self):
        """Initialize deduplication engine."""
        # Map: content_id -> reference count
        self.reference_counts: Dict[str, int] = {}
        
        # Map: content_id -> size in bytes
        self.content_sizes: Dict[str, int] = {}
        
        # Thread safety (RLock allows same thread to acquire lock multiple times)
        self._lock = RLock()
        
        # Statistics
        self.stats = DeduplicationStats(
            total_stores=0,
            unique_content=0,
            duplicate_saves=0,
            bytes_saved=0,
        )
        
        logger.info("Initialized deduplication engine")
    
    def check_exists(self, content_id: ContentID) -> bool:
        """
        Check if content already exists.
        
        Args:
            content_id: Content ID to check
        
        Returns:
            True if content is already stored
        """
        with self._lock:
            return content_id.hex in self.reference_counts
    
    def add_reference(
        self,
        content_id: ContentID,
        size_bytes: int
    ) -> bool:
        """
        Add reference to content.
        
        Args:
            content_id: Content ID
            size_bytes: Size of content in bytes
        
        Returns:
            True if this is a new (unique) content, False if duplicate
        """
        with self._lock:
            cid_hex = content_id.hex
            
            self.stats.total_stores += 1
            
            if cid_hex in self.reference_counts:
                # Duplicate - increment reference count
                self.reference_counts[cid_hex] += 1
                self.stats.duplicate_saves += 1
                self.stats.bytes_saved += size_bytes
                
                logger.debug(
                    f"Duplicate content {cid_hex[:16]}... "
                    f"(refs: {self.reference_counts[cid_hex]})"
                )
                
                return False  # Not unique
            else:
                # New unique content
                self.reference_counts[cid_hex] = 1
                self.content_sizes[cid_hex] = size_bytes
                self.stats.unique_content += 1
                
                logger.debug(f"New unique content {cid_hex[:16]}... ({size_bytes} bytes)")
                
                return True  # Unique
    
    def remove_reference(self, content_id: ContentID) -> bool:
        """
        Remove reference to content.
        
        Args:
            content_id: Content ID
        
        Returns:
            True if content can be deleted (no more references)
        """
        with self._lock:
            cid_hex = content_id.hex
            
            if cid_hex not in self.reference_counts:
                logger.warning(f"Attempted to remove reference to unknown content {cid_hex[:16]}...")
                return False
            
            self.reference_counts[cid_hex] -= 1
            
            if self.reference_counts[cid_hex] <= 0:
                # No more references - safe to delete
                del self.reference_counts[cid_hex]
                size = self.content_sizes.pop(cid_hex, 0)
                
                logger.debug(f"Content {cid_hex[:16]}... can be deleted (0 refs)")
                
                return True
            else:
                logger.debug(
                    f"Content {cid_hex[:16]}... still has "
                    f"{self.reference_counts[cid_hex]} references"
                )
                return False
    
    def get_reference_count(self, content_id: ContentID) -> int:
        """
        Get reference count for content.
        
        Args:
            content_id: Content ID
        
        Returns:
            Number of references (0 if not found)
        """
        with self._lock:
            return self.reference_counts.get(content_id.hex, 0)
    
    def get_stats(self) -> DeduplicationStats:
        """Get deduplication statistics."""
        with self._lock:
            return DeduplicationStats(
                total_stores=self.stats.total_stores,
                unique_content=self.stats.unique_content,
                duplicate_saves=self.stats.duplicate_saves,
                bytes_saved=self.stats.bytes_saved,
            )
    
    def get_all_content_ids(self) -> Set[str]:
        """
        Get all stored content IDs.
        
        Returns:
            Set of content ID hex strings
        """
        with self._lock:
            return set(self.reference_counts.keys())
    
    def estimate_space_saved(self) -> Dict[str, any]:
        """
        Estimate total space saved by deduplication.
        
        Returns:
            Dictionary with space savings metrics
        """
        with self._lock:
            stats = self.get_stats()
            
            total_unique_bytes = sum(self.content_sizes.values())
            
            # If we stored all duplicates, total would be:
            # unique_bytes * (total_stores / unique_content)
            if stats.unique_content > 0:
                average_duplicates = stats.total_stores / stats.unique_content
                theoretical_total = total_unique_bytes * average_duplicates
            else:
                theoretical_total = 0
            
            return {
                "total_stores": stats.total_stores,
                "unique_content_items": stats.unique_content,
                "duplicate_instances": stats.duplicate_saves,
                "unique_bytes_stored": total_unique_bytes,
                "bytes_saved_deduplication": stats.bytes_saved,
                "theoretical_bytes_without_dedup": theoretical_total,
                "deduplication_ratio": stats.deduplication_ratio,
                "efficiency_percent": stats.efficiency_percent,
                "space_efficiency": (
                    1.0 - (total_unique_bytes / theoretical_total)
                    if theoretical_total > 0 else 0.0
                ),
            }
