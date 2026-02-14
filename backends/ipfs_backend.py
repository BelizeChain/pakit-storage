"""
IPFS distributed storage backend.

Stores data on IPFS network with content-addressable storage.
"""

from typing import Optional, List
import logging
import ipfshttpclient
from pathlib import Path
import tempfile
import asyncio
import time

from blockchain.storage_proof_connector import get_storage_proof_connector

logger = logging.getLogger(__name__)


class IPFSBackend:
    """
    IPFS distributed storage backend.
    
    Uses IPFS for decentralized, content-addressable storage.
    Data is stored with its cryptographic hash (CID) and can be
    retrieved from any IPFS node in the network.
    
    Features:
    - Content-addressable storage (automatic deduplication)
    - Distributed retrieval (failover across nodes)
    - Permanent storage via pinning
    - Optional encryption at application layer
    """
    
    def __init__(
        self,
        ipfs_addr: str = "/ip4/127.0.0.1/tcp/5001",
        pin_content: bool = True,
        timeout: int = 60,
        retry_attempts: int = 3,
        retry_backoff: float = 0.5,
    ):
        """
        Initialize IPFS backend.
        
        Args:
            ipfs_addr: IPFS daemon API address (multiaddr format)
            pin_content: Whether to pin content (prevent garbage collection)
            timeout: Timeout for IPFS operations (seconds)
            retry_attempts: Number of retries for transient IPFS failures
            retry_backoff: Base backoff (seconds) between retries
        """
        self.ipfs_addr = ipfs_addr
        self.pin_content = pin_content
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff = max(0.1, retry_backoff)
        self.client = None
        
        self._connect()
        
        logger.info(f"Initialized IPFS backend at {self.ipfs_addr}")
    
    def _connect(self):
        """Connect to IPFS daemon."""
        try:
            self.client = ipfshttpclient.connect(
                self.ipfs_addr,
                timeout=self.timeout
            )
            
            # Test connection
            version = self._execute_with_retries(self.client.version)
            logger.info(f"Connected to IPFS {version['Version']}")
            
        except Exception as e:
            logger.error(f"Failed to connect to IPFS daemon: {e}")
            logger.error(
                "Make sure IPFS daemon is running: ipfs daemon\n"
                "Install IPFS: https://docs.ipfs.tech/install/"
            )
            raise ConnectionError(f"IPFS connection failed: {e}")
    
    def store(
        self,
        content_id: str,
        data: bytes,
        tier: str = "warm"
    ) -> bool:
        """
        Store data to IPFS.
        
        Args:
            content_id: Content ID (used for metadata tagging)
            data: Data to store
            tier: Storage tier (used for pinning strategy)
        
        Returns:
            True if successful
        """
        try:
            # Add data to IPFS
            result = self._execute_with_retries(self.client.add_bytes, data)
            ipfs_cid = result  # CID (Content Identifier)
            
            # Pin content if requested (prevents garbage collection)
            if self.pin_content:
                # Use different pinning strategies based on tier
                if tier == "hot":
                    # Hot tier: pin immediately with high priority
                    self._execute_with_retries(self.client.pin.add, ipfs_cid)
                    logger.debug(f"Pinned (hot) {content_id[:16]}... as {ipfs_cid}")
                elif tier == "warm":
                    # Warm tier: pin normally
                    self._execute_with_retries(self.client.pin.add, ipfs_cid)
                    logger.debug(f"Pinned (warm) {content_id[:16]}... as {ipfs_cid}")
                # Cold tier: don't pin (allow garbage collection)
                # Content stays available via DHT for some time
            
            logger.debug(
                f"Stored {content_id[:16]}... to IPFS as {ipfs_cid} "
                f"(tier: {tier}, size: {len(data)} bytes)"
            )
            
            # Store mapping in local metadata (for reverse lookup)
            # In production, this would be stored on-chain via LandLedger pallet
            self._store_metadata(content_id, ipfs_cid, tier)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store {content_id[:16]}... to IPFS: {e}")
            return False
    
    def retrieve(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> Optional[bytes]:
        """
        Retrieve data from IPFS.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier (for metadata lookup)
        
        Returns:
            Data bytes, or None if not found
        """
        try:
            # Get IPFS CID from metadata
            ipfs_cid = self._get_ipfs_cid(content_id, tier)
            
            if not ipfs_cid:
                logger.warning(
                    f"No IPFS CID found for content {content_id[:16]}..."
                )
                return None
            
            # Retrieve from IPFS
            data = self._execute_with_retries(self.client.cat, ipfs_cid)
            
            logger.debug(
                f"Retrieved {content_id[:16]}... from IPFS "
                f"(CID: {ipfs_cid}, size: {len(data)} bytes)"
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to retrieve {content_id[:16]}... from IPFS: {e}")
            return None
    
    def delete(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Delete data from IPFS (unpin).
        
        Note: Unpinning only removes local pin. Content may still be
        available on IPFS network if pinned by other nodes.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if successful
        """
        try:
            # Get IPFS CID
            ipfs_cid = self._get_ipfs_cid(content_id, tier)
            
            if not ipfs_cid:
                logger.warning(
                    f"No IPFS CID found for content {content_id[:16]}..."
                )
                return False
            
            # Unpin from local node
            if self.pin_content:
                self._execute_with_retries(self.client.pin.rm, ipfs_cid)
                logger.debug(f"Unpinned {content_id[:16]}... (CID: {ipfs_cid})")
            
            # Remove metadata
            self._delete_metadata(content_id, tier)
            
            logger.info(
                f"Deleted {content_id[:16]}... from IPFS "
                f"(unpinned {ipfs_cid})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {content_id[:16]}... from IPFS: {e}")
            return False
    
    def exists(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Check if content exists on IPFS.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if exists
        """
        try:
            # Get IPFS CID
            ipfs_cid = self._get_ipfs_cid(content_id, tier)
            
            if not ipfs_cid:
                return False
            
            # Try to stat the content (lightweight operation)
            self._execute_with_retries(self.client.object.stat, ipfs_cid)
            return True
            
        except Exception:
            return False
    
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
        try:
            ipfs_cid = self._get_ipfs_cid(content_id, tier)
            
            if not ipfs_cid:
                return None
            
            stat = self._execute_with_retries(self.client.object.stat, ipfs_cid)
            return stat['CumulativeSize']
            
        except Exception:
            return None
    
    def list_all(self, tier: Optional[str] = None) -> List[str]:
        """
        List all content IDs stored on IPFS.
        
        Args:
            tier: Optional tier filter (hot/warm/cold)
        
        Returns:
            List of content ID hex strings
        """
        # This would query metadata storage (local DB or on-chain)
        # For now, return empty list as placeholder
        logger.warning("list_all() not fully implemented for IPFS backend")
        return []
    
    def get_total_size(self, tier: Optional[str] = None) -> int:
        """
        Get total storage size on local IPFS node.
        
        Args:
            tier: Optional tier filter
        
        Returns:
            Total size in bytes
        """
        try:
            stat = self._execute_with_retries(self.client.repo.stat)
            return stat['RepoSize']
        except Exception as e:
            logger.error(f"Failed to get IPFS repo size: {e}")
            return 0
    
    def migrate_tier(
        self,
        content_id: str,
        from_tier: str,
        to_tier: str
    ) -> bool:
        """
        Migrate content between tiers (update pinning strategy).
        
        Args:
            content_id: Content ID (hex string)
            from_tier: Source tier
            to_tier: Destination tier
        
        Returns:
            True if successful
        """
        try:
            ipfs_cid = self._get_ipfs_cid(content_id, from_tier)
            
            if not ipfs_cid:
                return False
            
            # Update pinning based on new tier
            if to_tier == "hot" and not self._execute_with_retries(self.client.pin.ls, ipfs_cid):
                self._execute_with_retries(self.client.pin.add, ipfs_cid)
            elif to_tier == "cold" and self._execute_with_retries(self.client.pin.ls, ipfs_cid):
                self._execute_with_retries(self.client.pin.rm, ipfs_cid)
            
            # Update metadata
            self._delete_metadata(content_id, from_tier)
            self._store_metadata(content_id, ipfs_cid, to_tier)
            
            logger.info(
                f"Migrated {content_id[:16]}... from {from_tier} to {to_tier}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate {content_id[:16]}...: {e}")
            return False
    
    def cleanup(self, tier: Optional[str] = None):
        """
        Clean up IPFS storage (unpin all content).
        
        WARNING: This unpins all content from local node.
        Content may still be available on network if pinned elsewhere.
        
        Args:
            tier: Optional tier to clean (default: all tiers)
        """
        logger.warning("cleanup() for IPFS backend unpins all content!")
        
        try:
            # Get all pinned CIDs
            pins = self._execute_with_retries(self.client.pin.ls)
            
            for cid in pins['Keys']:
                self._execute_with_retries(self.client.pin.rm, cid)
                logger.debug(f"Unpinned {cid}")
            
            logger.info("Cleaned up IPFS pins")
            
        except Exception as e:
            logger.error(f"Failed to cleanup IPFS: {e}")
    
    # Internal methods
    
    def _store_metadata(self, content_id: str, ipfs_cid: str, tier: str):
        """
        Store content_id -> IPFS CID mapping on-chain and as fallback.
        
        Tries to store on BelizeChain's LandLedger pallet first,
        falls back to temp file storage if blockchain unavailable.
        """
        # Try on-chain storage first
        try:
            connector = get_storage_proof_connector(mock_mode=True)
            
            # Need to run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(
                connector.store_document_proof(
                    content_id=content_id,
                    ipfs_cid=ipfs_cid,
                    owner="system",
                    metadata={"tier": tier}
                )
            )
            loop.close()
            
            if success:
                logger.debug(f"✅ Stored on-chain metadata for {content_id[:16]}... -> {ipfs_cid[:16]}...")
                return
        except Exception as e:
            logger.warning(f"On-chain storage failed: {e}, using temp file fallback")
        
        # Fallback to temp file storage
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_ipfs_metadata" / tier
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_dir / f"{content_id}.cid"
        metadata_file.write_text(ipfs_cid)
    
    def _get_ipfs_cid(self, content_id: str, tier: str) -> Optional[str]:
        """
        Get IPFS CID for content ID from on-chain storage or fallback.
        
        Queries BelizeChain first, falls back to temp file storage.
        """
        # Try on-chain query first
        try:
            connector = get_storage_proof_connector(mock_mode=True)
            
            # Need to run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            ipfs_cid = loop.run_until_complete(
                connector.get_ipfs_cid(content_id)
            )
            loop.close()
            
            if ipfs_cid:
                logger.debug(f"✅ Retrieved on-chain CID for {content_id[:16]}...")
                return ipfs_cid
        except Exception as e:
            logger.warning(f"On-chain query failed: {e}, trying temp file fallback")
        
        # Fallback to temp file storage
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_ipfs_metadata" / tier
        metadata_file = metadata_dir / f"{content_id}.cid"
        
        if metadata_file.exists():
            return metadata_file.read_text().strip()
        else:
            return None
    
    def _delete_metadata(self, content_id: str, tier: str):
        """Delete metadata for content ID."""
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_ipfs_metadata" / tier
        metadata_file = metadata_dir / f"{content_id}.cid"
        
        if metadata_file.exists():
            metadata_file.unlink()

    def _execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retry and backoff for transient IPFS errors."""
        attempt = 0
        last_exc = None
        while attempt < self.retry_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # Broad on purpose: network client errors vary
                last_exc = exc
                attempt += 1
                if attempt >= self.retry_attempts:
                    break
                delay = self.retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"IPFS operation failed (attempt {attempt}/{self.retry_attempts}): {exc}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        raise last_exc
