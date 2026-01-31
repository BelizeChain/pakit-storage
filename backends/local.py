"""
Local filesystem storage backend.

Stores data on local disk with efficient directory structure.
"""

from typing import Optional, List
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


class LocalBackend:
    """
    Local filesystem storage backend.
    
    Directory structure:
    storage_dir/
        hot/
            AB/
                ABCDEF...123.pak
        warm/
            CD/
                CDEF...456.pak
        cold/
            EF/
                EF0123...789.pak
    
    Uses first 2 characters of content ID as directory prefix
    to avoid having too many files in a single directory.
    """
    
    def __init__(self, storage_dir: Path):
        """
        Initialize local backend.
        
        Args:
            storage_dir: Base directory for storage
        """
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized local backend at {self.storage_dir}")
    
    def store(
        self,
        content_id: str,
        data: bytes,
        tier: str = "warm"
    ) -> bool:
        """
        Store data to local filesystem.
        
        Args:
            content_id: Content ID (hex string)
            data: Data to store
            tier: Storage tier (hot/warm/cold)
        
        Returns:
            True if successful
        """
        try:
            file_path = self._get_file_path(content_id, tier)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(data)
            
            logger.debug(f"Stored {content_id[:16]}... to {file_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to store {content_id[:16]}...: {e}")
            return False
    
    def retrieve(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> Optional[bytes]:
        """
        Retrieve data from local filesystem.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            Data bytes, or None if not found
        """
        try:
            file_path = self._get_file_path(content_id, tier)
            
            if not file_path.exists():
                logger.warning(f"Content {content_id[:16]}... not found at {file_path}")
                return None
            
            with open(file_path, "rb") as f:
                data = f.read()
            
            logger.debug(f"Retrieved {content_id[:16]}... from {file_path}")
            
            return data
        except Exception as e:
            logger.error(f"Failed to retrieve {content_id[:16]}...: {e}")
            return None
    
    def delete(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Delete data from local filesystem.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if successful
        """
        try:
            file_path = self._get_file_path(content_id, tier)
            
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted {content_id[:16]}... from {file_path}")
                
                # Clean up empty directories
                self._cleanup_empty_dirs(file_path.parent)
                
                return True
            else:
                logger.warning(f"Content {content_id[:16]}... not found for deletion")
                return False
        except Exception as e:
            logger.error(f"Failed to delete {content_id[:16]}...: {e}")
            return False
    
    def exists(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Check if content exists.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if exists
        """
        file_path = self._get_file_path(content_id, tier)
        return file_path.exists()
    
    def get_size(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> Optional[int]:
        """
        Get size of stored content.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            Size in bytes, or None if not found
        """
        file_path = self._get_file_path(content_id, tier)
        
        if file_path.exists():
            return file_path.stat().st_size
        else:
            return None
    
    def list_all(self, tier: Optional[str] = None) -> List[str]:
        """
        List all content IDs.
        
        Args:
            tier: Optional tier filter (hot/warm/cold)
        
        Returns:
            List of content ID hex strings
        """
        content_ids = []
        
        tiers = [tier] if tier else ["hot", "warm", "cold"]
        
        for t in tiers:
            tier_dir = self.storage_dir / t
            
            if not tier_dir.exists():
                continue
            
            for prefix_dir in tier_dir.iterdir():
                if not prefix_dir.is_dir():
                    continue
                
                for file_path in prefix_dir.glob("*.pak"):
                    content_id = file_path.stem
                    content_ids.append(content_id)
        
        return content_ids
    
    def get_total_size(self, tier: Optional[str] = None) -> int:
        """
        Get total storage size.
        
        Args:
            tier: Optional tier filter
        
        Returns:
            Total size in bytes
        """
        total_size = 0
        
        tiers = [tier] if tier else ["hot", "warm", "cold"]
        
        for t in tiers:
            tier_dir = self.storage_dir / t
            
            if not tier_dir.exists():
                continue
            
            for prefix_dir in tier_dir.iterdir():
                if not prefix_dir.is_dir():
                    continue
                
                for file_path in prefix_dir.glob("*.pak"):
                    total_size += file_path.stat().st_size
        
        return total_size
    
    def migrate_tier(
        self,
        content_id: str,
        from_tier: str,
        to_tier: str
    ) -> bool:
        """
        Migrate content between tiers.
        
        Args:
            content_id: Content ID (hex string)
            from_tier: Source tier
            to_tier: Destination tier
        
        Returns:
            True if successful
        """
        try:
            # Retrieve from source
            data = self.retrieve(content_id, from_tier)
            
            if data is None:
                return False
            
            # Store to destination
            if not self.store(content_id, data, to_tier):
                return False
            
            # Delete from source
            self.delete(content_id, from_tier)
            
            logger.info(
                f"Migrated {content_id[:16]}... from {from_tier} to {to_tier}"
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to migrate {content_id[:16]}...: {e}")
            return False
    
    def cleanup(self, tier: Optional[str] = None):
        """
        Clean up storage (delete all content).
        
        Args:
            tier: Optional tier to clean (default: all tiers)
        """
        tiers = [tier] if tier else ["hot", "warm", "cold"]
        
        for t in tiers:
            tier_dir = self.storage_dir / t
            
            if tier_dir.exists():
                shutil.rmtree(tier_dir)
                logger.info(f"Cleaned up {t} tier")
    
    # Internal methods
    
    def _get_file_path(self, content_id: str, tier: str) -> Path:
        """Get file path for content ID."""
        tier_dir = self.storage_dir / tier
        prefix_dir = tier_dir / content_id[:2]
        return prefix_dir / f"{content_id}.pak"
    
    def _cleanup_empty_dirs(self, directory: Path):
        """Clean up empty directories."""
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
                logger.debug(f"Cleaned up empty directory {directory}")
                
                # Recursively clean up parent
                self._cleanup_empty_dirs(directory.parent)
        except Exception:
            pass  # Ignore errors
