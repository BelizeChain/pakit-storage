"""
Telemetry Collector

Collects block operation events for training data.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of block events."""
    BLOCK_STORED = "stored"
    BLOCK_RETRIEVED = "retrieved"
    BLOCK_COMPRESSED = "compressed"
    BLOCK_DEDUP_CHECK = "dedup_check"
    PEER_REQUEST = "peer_request"
    PREFETCH_HIT = "prefetch_hit"
    PREFETCH_MISS = "prefetch_miss"


@dataclass
class BlockEvent:
    """
    Single block operation event.
    
    Privacy note: block_hash is included but content is NEVER stored.
    """
    event_type: EventType
    block_hash: str
    timestamp: float = field(default_factory=time.time)
    
    # Block characteristics
    block_size: Optional[int] = None
    block_depth: Optional[int] = None
    parent_count: Optional[int] = None
    
    # Compression info
    compression_algo: Optional[str] = None
    compression_ratio: Optional[float] = None
    compression_time_ms: Optional[float] = None
    
    # Deduplication info
    content_similarity: Optional[float] = None
    duplicate_of: Optional[str] = None
    
    # Network info
    peer_id: Optional[str] = None
    latency_ms: Optional[float] = None
    bandwidth_mbps: Optional[float] = None
    
    # Access pattern info
    access_frequency: Optional[int] = None
    last_access_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_type': self.event_type.value,
            'block_hash': self.block_hash,
            'timestamp': self.timestamp,
            'block_size': self.block_size,
            'block_depth': self.block_depth,
            'parent_count': self.parent_count,
            'compression_algo': self.compression_algo,
            'compression_ratio': self.compression_ratio,
            'compression_time_ms': self.compression_time_ms,
            'content_similarity': self.content_similarity,
            'duplicate_of': self.duplicate_of,
            'peer_id': self.peer_id,
            'latency_ms': self.latency_ms,
            'bandwidth_mbps': self.bandwidth_mbps,
            'access_frequency': self.access_frequency,
            'last_access_time': self.last_access_time,
        }


class TelemetryCollector:
    """
    Collects and stores telemetry events.
    
    Uses SQLite for persistent storage of training data.
    """
    
    def __init__(self, db_path: str = "./pakit_telemetry.db", enabled: bool = True):
        self.db_path = Path(db_path)
        self.enabled = enabled
        self._event_count = 0
        
        if self.enabled:
            self._init_database()
            logger.info(f"Telemetry collector initialized: {db_path}")
        else:
            logger.info("Telemetry collector disabled")
    
    def _init_database(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS block_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                block_hash TEXT NOT NULL,
                timestamp REAL NOT NULL,
                block_size INTEGER,
                block_depth INTEGER,
                parent_count INTEGER,
                compression_algo TEXT,
                compression_ratio REAL,
                compression_time_ms REAL,
                content_similarity REAL,
                duplicate_of TEXT,
                peer_id TEXT,
                latency_ms REAL,
                bandwidth_mbps REAL,
                access_frequency INTEGER,
                last_access_time REAL
            )
        """)
        
        # Create indices for common queries
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_type ON block_events(event_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_block_hash ON block_events(block_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON block_events(timestamp)"
        )
        
        conn.commit()
        conn.close()
    
    def record(self, event: BlockEvent) -> None:
        """
        Record a block event.
        
        Args:
            event: Block event to record
        """
        if not self.enabled:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO block_events (
                event_type, block_hash, timestamp,
                block_size, block_depth, parent_count,
                compression_algo, compression_ratio, compression_time_ms,
                content_similarity, duplicate_of,
                peer_id, latency_ms, bandwidth_mbps,
                access_frequency, last_access_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_type.value,
            event.block_hash,
            event.timestamp,
            event.block_size,
            event.block_depth,
            event.parent_count,
            event.compression_algo,
            event.compression_ratio,
            event.compression_time_ms,
            event.content_similarity,
            event.duplicate_of,
            event.peer_id,
            event.latency_ms,
            event.bandwidth_mbps,
            event.access_frequency,
            event.last_access_time,
        ))
        
        conn.commit()
        conn.close()
        
        self._event_count += 1
        
        if self._event_count % 1000 == 0:
            logger.info(f"Recorded {self._event_count} events")
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve events from database.
        
        Args:
            event_type: Filter by event type
            start_time: Filter by start timestamp
            end_time: Filter by end timestamp
            limit: Maximum events to return
            
        Returns:
            List of event dictionaries
        """
        if not self.enabled:
            return []
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM block_events WHERE 1=1"
        params = []
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.value)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total events
        cursor.execute("SELECT COUNT(*) FROM block_events")
        total_events = cursor.fetchone()[0]
        
        # Events by type
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM block_events
            GROUP BY event_type
        """)
        events_by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Time range
        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM block_events")
        min_time, max_time = cursor.fetchone()
        
        conn.close()
        
        return {
            'enabled': True,
            'total_events': total_events,
            'events_by_type': events_by_type,
            'min_timestamp': min_time,
            'max_timestamp': max_time,
            'db_path': str(self.db_path),
        }
    
    def clear(self) -> None:
        """Clear all collected data."""
        if not self.enabled:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM block_events")
        conn.commit()
        conn.close()
        
        self._event_count = 0
        logger.info("Cleared all telemetry data")


# Global collector instance
_collector: Optional[TelemetryCollector] = None


def get_collector() -> TelemetryCollector:
    """Get global telemetry collector."""
    global _collector
    if _collector is None:
        _collector = TelemetryCollector()
    return _collector
