"""
Feature Extractor

Extracts ML features from block events and Pakit operations.
"""

from typing import Dict, Any, Optional, List
import hashlib
import math
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts features from block data for ML models.
    
    Features are privacy-preserving - no block content is exposed.
    """
    
    @staticmethod
    def extract_block_features(
        block_hash: str,
        block_size: int,
        block_depth: int,
        parent_hashes: Optional[List[str]] = None,
        compressed_size: Optional[int] = None,
        compression_algo: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Extract features from block metadata.
        
        Args:
            block_hash: Block hash (SHA-256)
            block_size: Block size in bytes
            block_depth: Depth in DAG
            parent_hashes: List of parent block hashes
            compressed_size: Compressed size (if compressed)
            compression_algo: Compression algorithm used
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Block size features
        features['block_size_bytes'] = float(block_size)
        features['block_size_kb'] = block_size / 1024.0
        features['block_size_log'] = math.log(max(block_size, 1))
        
        # Depth features
        features['block_depth'] = float(block_depth)
        features['block_depth_log'] = math.log(max(block_depth, 1))
        
        # Parent features
        parent_count = len(parent_hashes) if parent_hashes else 0
        features['parent_count'] = float(parent_count)
        features['is_leaf'] = 1.0 if parent_count == 0 else 0.0
        features['is_merge'] = 1.0 if parent_count > 1 else 0.0
        
        # Compression features
        if compressed_size is not None:
            features['compressed_size_bytes'] = float(compressed_size)
            features['compression_ratio'] = (
                block_size / max(compressed_size, 1)
            )
        else:
            features['compressed_size_bytes'] = 0.0
            features['compression_ratio'] = 1.0
        
        # Compression algorithm one-hot encoding
        features['algo_zstd'] = 1.0 if compression_algo == 'zstd' else 0.0
        features['algo_lz4'] = 1.0 if compression_algo == 'lz4' else 0.0
        features['algo_snappy'] = 1.0 if compression_algo == 'snappy' else 0.0
        features['algo_none'] = 1.0 if compression_algo is None else 0.0
        
        # Hash-based features (deterministic pseudo-randomness)
        hash_int = int(block_hash[:16], 16)  # First 64 bits
        features['hash_entropy'] = (hash_int % 256) / 256.0
        
        return features
    
    @staticmethod
    def extract_access_pattern_features(
        access_count: int,
        last_access_time: float,
        current_time: float,
        access_history: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """
        Extract access pattern features.
        
        Args:
            access_count: Total access count
            last_access_time: Last access timestamp
            current_time: Current timestamp
            access_history: List of access timestamps
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Access frequency
        features['access_count'] = float(access_count)
        features['access_count_log'] = math.log(max(access_count, 1))
        
        # Recency
        time_since_access = current_time - last_access_time
        features['time_since_access'] = time_since_access
        features['time_since_access_log'] = math.log(max(time_since_access, 1))
        
        # Access pattern analysis
        if access_history and len(access_history) > 1:
            # Calculate access intervals
            intervals = [
                access_history[i+1] - access_history[i]
                for i in range(len(access_history) - 1)
            ]
            
            features['avg_access_interval'] = sum(intervals) / len(intervals)
            features['access_regularity'] = 1.0 / (
                max(sum((i - features['avg_access_interval'])**2 for i in intervals), 1e-6)
            )
        else:
            features['avg_access_interval'] = 0.0
            features['access_regularity'] = 0.0
        
        return features
    
    @staticmethod
    def extract_network_features(
        peer_id: str,
        latency_ms: float,
        bandwidth_mbps: float,
        reputation_score: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract network/peer features.
        
        Args:
            peer_id: Peer identifier
            latency_ms: Network latency in milliseconds
            bandwidth_mbps: Bandwidth in Mbps
            reputation_score: Peer reputation (0.0-1.0)
            
        Returns:
            Feature dictionary
        """
        features = {}
        
        # Latency features
        features['latency_ms'] = latency_ms
        features['latency_log'] = math.log(max(latency_ms, 0.1))
        features['is_fast'] = 1.0 if latency_ms < 100 else 0.0
        features['is_slow'] = 1.0 if latency_ms > 1000 else 0.0
        
        # Bandwidth features
        features['bandwidth_mbps'] = bandwidth_mbps
        features['bandwidth_log'] = math.log(max(bandwidth_mbps, 0.1))
        
        # Reputation
        if reputation_score is not None:
            features['reputation'] = reputation_score
        else:
            features['reputation'] = 0.5  # Neutral
        
        # Peer ID hash (for clustering)
        peer_hash = int(hashlib.sha256(peer_id.encode()).hexdigest()[:16], 16)
        features['peer_cluster'] = (peer_hash % 10) / 10.0
        
        return features
    
    @staticmethod
    def calculate_content_entropy(data: bytes) -> float:
        """
        Calculate Shannon entropy of data (for compression prediction).
        
        Args:
            data: Block data
            
        Returns:
            Entropy (0.0 = compressible, 8.0 = random)
        """
        if not data:
            return 0.0
        
        # Count byte frequencies
        freq = {}
        for byte in data:
            freq[byte] = freq.get(byte, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in freq.values():
            p = count / data_len
            entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def detect_content_type(data: bytes) -> str:
        """
        Detect content type from magic bytes.
        
        Args:
            data: Block data
            
        Returns:
            Content type string
        """
        if len(data) < 4:
            return "unknown"
        
        # Check magic bytes
        magic = data[:4]
        
        if magic[:2] == b'\x1f\x8b':
            return "gzip"
        elif magic == b'PK\x03\x04':
            return "zip"
        elif magic[:3] == b'\xff\xd8\xff':
            return "jpeg"
        elif magic == b'\x89PNG':
            return "png"
        elif magic[:4] == b'%PDF':
            return "pdf"
        elif magic[:4] == b'{\x0d\x0a' or magic[0:1] == b'{':
            return "json"
        elif data[:5] == b'<?xml':
            return "xml"
        else:
            # Try to detect if text
            try:
                data[:100].decode('utf-8')
                return "text"
            except:
                return "binary"
