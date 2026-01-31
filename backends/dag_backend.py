"""
DAG Backend: Persistent storage for DAG blocks using SQLite.

This module provides persistent storage for DAG blocks with efficient querying,
caching, and durability guarantees. Uses SQLite for simplicity and portability.

Key Features:
- Persistent block storage with ACID guarantees
- LRU cache for hot data (configurable size)
- Efficient queries by hash, depth, and timestamp
- Backup/restore functionality
- Fsync for durability
- Thread-safe operations
"""

import sqlite3
import msgpack
import hashlib
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from dataclasses import asdict
from collections import OrderedDict
from threading import RLock
from datetime import datetime

from pakit.core.dag_storage import DagBlock


class LRUCache:
    """Simple LRU cache implementation for hot DAG blocks."""
    
    def __init__(self, capacity: int = 1000):
        """Initialize LRU cache with given capacity."""
        self.capacity = capacity
        self.cache: OrderedDict[str, DagBlock] = OrderedDict()
        self._lock = RLock()
    
    def get(self, key: str) -> Optional[DagBlock]:
        """Get value from cache, moving to end (most recently used)."""
        with self._lock:
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: DagBlock) -> None:
        """Put value in cache, evicting oldest if at capacity."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)  # Remove oldest
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
    
    def stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "usage_pct": int(len(self.cache) / self.capacity * 100)
            }


class DagBackend:
    """
    Persistent storage backend for DAG blocks using SQLite.
    
    Provides efficient storage and retrieval of DAG blocks with caching,
    indexing, and durability guarantees.
    """
    
    def __init__(
        self,
        db_path: str = "pakit_dag.db",
        cache_size: int = 1000,
        enable_wal: bool = True,
        fsync: bool = True
    ):
        """
        Initialize DAG backend.
        
        Args:
            db_path: Path to SQLite database file
            cache_size: Number of blocks to cache in memory
            enable_wal: Enable Write-Ahead Logging for better concurrency
            fsync: Enable fsync for durability (may impact performance)
        """
        self.db_path = db_path
        self.cache = LRUCache(capacity=cache_size)
        self.enable_fsync = fsync
        self._lock = RLock()
        
        # Initialize database
        self._init_db(enable_wal)
    
    def _init_db(self, enable_wal: bool) -> None:
        """Initialize SQLite database with schema."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency
            if enable_wal:
                cursor.execute("PRAGMA journal_mode=WAL")
            
            # Create blocks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dag_blocks (
                    block_hash TEXT PRIMARY KEY,
                    depth INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    compression_algo TEXT NOT NULL,
                    data BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_depth 
                ON dag_blocks(depth)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON dag_blocks(timestamp)
            """)
            
            # Create parent-child relationship table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dag_edges (
                    child_hash TEXT NOT NULL,
                    parent_hash TEXT NOT NULL,
                    PRIMARY KEY (child_hash, parent_hash),
                    FOREIGN KEY (child_hash) REFERENCES dag_blocks(block_hash),
                    FOREIGN KEY (parent_hash) REFERENCES dag_blocks(block_hash)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_child 
                ON dag_edges(child_hash)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_parent 
                ON dag_edges(parent_hash)
            """)
            
            conn.commit()
            conn.close()
    
    def put(self, block: DagBlock) -> bool:
        """
        Store a DAG block to persistent storage.
        
        Args:
            block: DagBlock to store
            
        Returns:
            True if stored successfully, False if already exists
        """
        with self._lock:
            try:
                # Serialize block data
                serialized = msgpack.packb({
                    "block_hash": block.block_hash,
                    "parent_blocks": block.parent_blocks,
                    "depth": block.depth,
                    "timestamp": block.timestamp,
                    "compression_algo": block.compression_algo,
                    "compressed_data": block.compressed_data,
                    "original_size": block.original_size,
                    "compressed_size": block.compressed_size,
                    "metadata": block.metadata
                })
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Insert block
                cursor.execute("""
                    INSERT OR IGNORE INTO dag_blocks 
                    (block_hash, depth, timestamp, compression_algo, data)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    block.block_hash,
                    block.depth,
                    block.timestamp,
                    block.compression_algo,
                    serialized
                ))
                
                # Insert parent relationships
                for parent_hash in block.parent_blocks:
                    cursor.execute("""
                        INSERT OR IGNORE INTO dag_edges 
                        (child_hash, parent_hash)
                        VALUES (?, ?)
                    """, (block.block_hash, parent_hash))
                
                conn.commit()
                
                # Fsync if enabled
                if self.enable_fsync:
                    conn.execute("PRAGMA wal_checkpoint(FULL)")
                
                conn.close()
                
                # Update cache
                self.cache.put(block.block_hash, block)
                
                return cursor.rowcount > 0
                
            except Exception as e:
                print(f"Error storing block: {e}")
                return False
    
    def get(self, block_hash: str) -> Optional[DagBlock]:
        """
        Retrieve a DAG block by hash.
        
        Args:
            block_hash: Hash of block to retrieve
            
        Returns:
            DagBlock if found, None otherwise
        """
        # Check cache first
        cached = self.cache.get(block_hash)
        if cached:
            return cached
        
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT data FROM dag_blocks 
                    WHERE block_hash = ?
                """, (block_hash,))
                
                row = cursor.fetchone()
                conn.close()
                
                if not row:
                    return None
                
                # Deserialize block
                data = msgpack.unpackb(row[0])
                block = DagBlock(
                    block_hash=data["block_hash"],
                    parent_blocks=data["parent_blocks"],
                    depth=data["depth"],
                    timestamp=data["timestamp"],
                    compression_algo=data["compression_algo"],
                    compressed_data=data["compressed_data"],
                    original_size=data["original_size"],
                    compressed_size=data["compressed_size"],
                    metadata=data.get("metadata", {})
                )
                
                # Update cache
                self.cache.put(block_hash, block)
                
                return block
                
            except Exception as e:
                print(f"Error retrieving block: {e}")
                return None
    
    def delete(self, block_hash: str) -> bool:
        """
        Delete a DAG block from storage.
        
        Args:
            block_hash: Hash of block to delete
            
        Returns:
            True if deleted, False otherwise
        """
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Delete edges first (foreign key constraint)
                cursor.execute("""
                    DELETE FROM dag_edges 
                    WHERE child_hash = ? OR parent_hash = ?
                """, (block_hash, block_hash))
                
                # Delete block
                cursor.execute("""
                    DELETE FROM dag_blocks 
                    WHERE block_hash = ?
                """, (block_hash,))
                
                conn.commit()
                conn.close()
                
                # Remove from cache
                if block_hash in self.cache.cache:
                    del self.cache.cache[block_hash]
                
                return cursor.rowcount > 0
                
            except Exception as e:
                print(f"Error deleting block: {e}")
                return False
    
    def query_by_depth(
        self,
        min_depth: int = 0,
        max_depth: int = 1000000
    ) -> List[DagBlock]:
        """
        Query blocks by depth range.
        
        Args:
            min_depth: Minimum depth (inclusive)
            max_depth: Maximum depth (inclusive)
            
        Returns:
            List of blocks in depth range
        """
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT data FROM dag_blocks 
                    WHERE depth BETWEEN ? AND ?
                    ORDER BY depth ASC
                """, (min_depth, max_depth))
                
                rows = cursor.fetchall()
                conn.close()
                
                blocks = []
                for row in rows:
                    data = msgpack.unpackb(row[0])
                    block = DagBlock(
                        block_hash=data["block_hash"],
                        parent_blocks=data["parent_blocks"],
                        depth=data["depth"],
                        timestamp=data["timestamp"],
                        compression_algo=data["compression_algo"],
                        compressed_data=data["compressed_data"],
                        original_size=data["original_size"],
                        compressed_size=data["compressed_size"],
                        metadata=data.get("metadata", {})
                    )
                    blocks.append(block)
                
                return blocks
                
            except Exception as e:
                print(f"Error querying by depth: {e}")
                return []
    
    def get_children(self, parent_hash: str) -> List[str]:
        """
        Get all children of a parent block.
        
        Args:
            parent_hash: Hash of parent block
            
        Returns:
            List of child block hashes
        """
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT child_hash FROM dag_edges 
                    WHERE parent_hash = ?
                """, (parent_hash,))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [row[0] for row in rows]
                
            except Exception as e:
                print(f"Error getting children: {e}")
                return []
    
    def get_parents(self, child_hash: str) -> List[str]:
        """
        Get all parents of a child block.
        
        Args:
            child_hash: Hash of child block
            
        Returns:
            List of parent block hashes
        """
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT parent_hash FROM dag_edges 
                    WHERE child_hash = ?
                """, (child_hash,))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [row[0] for row in rows]
                
            except Exception as e:
                print(f"Error getting parents: {e}")
                return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Total blocks
                cursor.execute("SELECT COUNT(*) FROM dag_blocks")
                total_blocks = cursor.fetchone()[0]
                
                # Total edges
                cursor.execute("SELECT COUNT(*) FROM dag_edges")
                total_edges = cursor.fetchone()[0]
                
                # Max depth
                cursor.execute("SELECT MAX(depth) FROM dag_blocks")
                max_depth = cursor.fetchone()[0] or 0
                
                # Database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                db_size = page_count * page_size
                
                conn.close()
                
                cache_stats = self.cache.stats()
                
                return {
                    "total_blocks": total_blocks,
                    "total_edges": total_edges,
                    "max_depth": max_depth,
                    "db_size_bytes": db_size,
                    "db_size_mb": db_size / 1024 / 1024,
                    "cache": cache_stats
                }
                
            except Exception as e:
                print(f"Error getting statistics: {e}")
                return {}
    
    def backup(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if backup successful
        """
        with self._lock:
            try:
                # Ensure parent directory exists
                Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Copy database file
                shutil.copy2(self.db_path, backup_path)
                
                # Copy WAL files if they exist
                wal_path = f"{self.db_path}-wal"
                if os.path.exists(wal_path):
                    shutil.copy2(wal_path, f"{backup_path}-wal")
                
                shm_path = f"{self.db_path}-shm"
                if os.path.exists(shm_path):
                    shutil.copy2(shm_path, f"{backup_path}-shm")
                
                return True
                
            except Exception as e:
                print(f"Error creating backup: {e}")
                return False
    
    def restore(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful
        """
        with self._lock:
            try:
                # Close any existing connections
                # Copy backup to database path
                shutil.copy2(backup_path, self.db_path)
                
                # Copy WAL files if they exist
                wal_backup = f"{backup_path}-wal"
                if os.path.exists(wal_backup):
                    shutil.copy2(wal_backup, f"{self.db_path}-wal")
                
                shm_backup = f"{backup_path}-shm"
                if os.path.exists(shm_backup):
                    shutil.copy2(shm_backup, f"{self.db_path}-shm")
                
                # Clear cache
                self.cache.clear()
                
                return True
                
            except Exception as e:
                print(f"Error restoring backup: {e}")
                return False
    
    def vacuum(self) -> bool:
        """Vacuum database to reclaim space and optimize."""
        with self._lock:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute("VACUUM")
                conn.close()
                return True
            except Exception as e:
                print(f"Error vacuuming database: {e}")
                return False
    
    def close(self) -> None:
        """Close database connections and clear cache."""
        with self._lock:
            self.cache.clear()


if __name__ == "__main__":
    # Example usage
    print("DAG Backend Example Usage:")
    print("-" * 60)
    
    # Initialize backend
    backend = DagBackend(db_path="test_dag.db", cache_size=100)
    
    # Create a test block
    from pakit.core.dag_storage import create_genesis_block
    genesis = create_genesis_block()
    
    # Store block
    success = backend.put(genesis)
    print(f"Stored genesis block: {success}")
    print(f"Block hash: {genesis.block_hash[:16]}...")
    
    # Retrieve block
    retrieved = backend.get(genesis.block_hash)
    print(f"Retrieved block: {retrieved is not None}")
    print(f"Block depth: {retrieved.depth if retrieved else 'N/A'}")
    
    # Statistics
    stats = backend.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total blocks: {stats['total_blocks']}")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  DB size: {stats['db_size_mb']:.2f} MB")
    print(f"  Cache usage: {stats['cache']['usage_pct']}%")
    
    # Cleanup
    import os
    backend.close()
    if os.path.exists("test_dag.db"):
        os.remove("test_dag.db")
    print("\nCleanup complete!")
