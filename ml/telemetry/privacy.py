"""
Privacy Filter

Ensures telemetry data is privacy-preserving.
Never leaks actual block content, only metadata and statistics.
"""

from typing import Dict, Any, Optional
import hashlib
import logging

logger = logging.getLogger(__name__)


class PrivacyFilter:
    """
    Privacy-preserving filter for telemetry data.
    
    Ensures collected data cannot reveal block content.
    """
    
    @staticmethod
    def sanitize_block_data(
        block_data: bytes,
        include_size: bool = True,
        include_entropy: bool = True,
        include_content_type: bool = True
    ) -> Dict[str, Any]:
        """
        Extract privacy-safe metadata from block data.
        
        Args:
            block_data: Raw block data
            include_size: Include size metadata
            include_entropy: Include entropy calculation
            include_content_type: Include content type detection
            
        Returns:
            Safe metadata dictionary (NO CONTENT)
        """
        metadata = {}
        
        # Block hash (always safe)
        metadata['block_hash'] = hashlib.sha256(block_data).hexdigest()
        
        if include_size:
            metadata['size'] = len(block_data)
            metadata['size_kb'] = len(block_data) / 1024.0
        
        if include_entropy:
            # Entropy reveals compressibility, not content
            from pakit.ml.telemetry.features import FeatureExtractor
            metadata['entropy'] = FeatureExtractor.calculate_content_entropy(block_data)
        
        if include_content_type:
            # Content type from magic bytes (safe)
            from pakit.ml.telemetry.features import FeatureExtractor
            metadata['content_type'] = FeatureExtractor.detect_content_type(block_data)
        
        # NO CONTENT in metadata!
        return metadata
    
    @staticmethod
    def anonymize_peer_id(peer_id: str, salt: Optional[str] = None) -> str:
        """
        Anonymize peer ID for privacy.
        
        Args:
            peer_id: Original peer ID
            salt: Optional salt for hashing
            
        Returns:
            Anonymized peer ID
        """
        if salt:
            data = f"{peer_id}:{salt}".encode()
        else:
            data = peer_id.encode()
        
        # Use first 16 chars of hash (64 bits)
        return hashlib.sha256(data).hexdigest()[:16]
    
    @staticmethod
    def validate_telemetry_event(event_data: Dict[str, Any]) -> bool:
        """
        Validate that telemetry event is privacy-safe.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            True if safe, False if contains sensitive data
        """
        # Blacklist of forbidden fields
        forbidden_fields = {
            'block_content',
            'block_data',
            'raw_data',
            'content',
            'payload',
            'decrypted_data',
        }
        
        # Check for forbidden fields
        for field in forbidden_fields:
            if field in event_data:
                logger.error(f"Privacy violation: forbidden field '{field}' in event")
                return False
        
        # Check for suspicious large string values (potential content leak)
        for key, value in event_data.items():
            if isinstance(value, str) and len(value) > 1000:
                logger.warning(f"Suspicious large string in field '{key}' ({len(value)} chars)")
                return False
        
        return True
    
    @staticmethod
    def differential_privacy_noise(
        value: float,
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> float:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value
            epsilon: Privacy parameter (lower = more private)
            sensitivity: Sensitivity of the query
            
        Returns:
            Noisy value
        """
        import numpy as np
        
        # Laplace mechanism
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        
        return value + noise
    
    @staticmethod
    def k_anonymity_check(
        dataset: list,
        quasi_identifiers: list,
        k: int = 5
    ) -> bool:
        """
        Check if dataset satisfies k-anonymity.
        
        Args:
            dataset: List of records
            quasi_identifiers: Fields that could identify individuals
            k: Minimum group size
            
        Returns:
            True if k-anonymous, False otherwise
        """
        from collections import defaultdict
        
        # Group records by quasi-identifier values
        groups = defaultdict(int)
        
        for record in dataset:
            # Create key from quasi-identifiers
            key = tuple(record.get(qi) for qi in quasi_identifiers)
            groups[key] += 1
        
        # Check minimum group size
        for count in groups.values():
            if count < k:
                logger.warning(f"k-anonymity violation: group size {count} < {k}")
                return False
        
        return True
