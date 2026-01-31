"""
IPFS Pin Garbage Collection
Automated cleanup of expired/unpaid storage pins
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import ipfshttpclient

logger = logging.getLogger(__name__)


class IPFSGarbageCollector:
    """
    Manages IPFS pin lifecycle
    - Tracks pin expiration dates
    - Removes expired pins to free storage
    - Integrates with pallet-economy for payment verification
    """
    
    def __init__(self, ipfs_api: str = "/ip4/127.0.0.1/tcp/5001"):
        self.ipfs_api = ipfs_api
        self.client = None
        self.pin_registry: Dict[str, Dict] = {}  # cid -> {created_at, expires_at, paid}
    
    def connect(self):
        """Connect to IPFS daemon"""
        self.client = ipfshttpclient.connect(self.ipfs_api)
        logger.info(f"✅ Connected to IPFS: {self.ipfs_api}")
    
    async def register_pin(self, cid: str, duration_days: int = 90, paid: bool = False):
        """
        Register a pinned object with expiration
        
        Args:
            cid: IPFS content ID
            duration_days: Pin duration in days
            paid: Whether storage fee is paid
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(days=duration_days)
        
        self.pin_registry[cid] = {
            "created_at": now,
            "expires_at": expires_at,
            "paid": paid,
            "size_bytes": 0  # TODO: Query IPFS for object size
        }
        
        logger.info(f"📌 Registered pin: {cid} (expires: {expires_at.isoformat()})")
    
    async def run_gc(self):
        """
        Run garbage collection cycle
        Unpins expired objects
        """
        if self.client is None:
            self.connect()
        
        now = datetime.utcnow()
        unpinned_count = 0
        
        for cid, metadata in list(self.pin_registry.items()):
            if metadata["expires_at"] < now and not metadata["paid"]:
                # Expired and unpaid - unpin
                try:
                    self.client.pin.rm(cid)
                    del self.pin_registry[cid]
                    unpinned_count += 1
                    logger.info(f"🗑️ Unpinned expired object: {cid}")
                except Exception as e:
                    logger.error(f"Failed to unpin {cid}: {e}")
        
        logger.info(f"GC complete: {unpinned_count} objects unpinned")
        return unpinned_count
    
    async def extend_pin(self, cid: str, additional_days: int):
        """Extend pin expiration (after payment received)"""
        if cid in self.pin_registry:
            metadata = self.pin_registry[cid]
            metadata["expires_at"] += timedelta(days=additional_days)
            metadata["paid"] = True
            logger.info(f"⏰ Extended pin {cid} by {additional_days} days")
    
    async def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        if self.client is None:
            self.connect()
        
        repo_stat = self.client.repo.stat()
        
        return {
            "total_pins": len(self.pin_registry),
            "repo_size_gb": repo_stat["RepoSize"] / (1024**3),
            "num_objects": repo_stat["NumObjects"]
        }
