"""
Pakit Mesh Network Client

Wrapper around nawal-ai's MeshNetworkClient for storage-specific operations.
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import nawal-ai mesh client
try:
    from nawal_ai.blockchain.mesh_network import MeshNetworkClient, MessageType, MeshMessage
    MESH_AVAILABLE = True
except ImportError:
    logger.warning("nawal-ai mesh networking not available. Install with: pip install git+https://github.com/BelizeChain/nawal-ai.git@main#egg=nawal-ai[mesh]")
    MESH_AVAILABLE = False
    MeshNetworkClient = None
    MessageType = None
    MeshMessage = None


@dataclass
class StorageAvailabilityAnnouncement:
    """Announcement of storage block availability."""
    cid: str
    size_bytes: int
    peer_id: str
    timestamp: float
    replication_count: int = 1


@dataclass
class BlockRequest:
    """Request for a specific storage block."""
    cid: str
    requester_peer_id: str
    timestamp: float


class PakitMeshClient:
    """
    Mesh network client for Pakit storage nodes.
    
    Enables direct P2P communication for:
    - File replication coordination
    - Storage availability announcements
    - Block discovery propagation
    - Peer-based content routing
    """
    
    def __init__(
        self,
        peer_id: str,
        listen_port: int = 9091,
        blockchain_rpc: Optional[str] = None,
        enable_mesh: bool = True
    ):
        """
        Initialize Pakit mesh client.
        
        Args:
            peer_id: Unique peer identifier for this storage node
            listen_port: Port to listen on for mesh network
            blockchain_rpc: BelizeChain RPC endpoint (default: ws://localhost:9944)
            enable_mesh: Enable mesh networking (disable for testing)
        """
        self.peer_id = f"pakit_storage_{peer_id}"
        self.listen_port = listen_port
        self.blockchain_rpc = blockchain_rpc or os.getenv("BLOCKCHAIN_RPC", "ws://localhost:9944")
        self.enable_mesh = enable_mesh and MESH_AVAILABLE
        
        # Always initialize storage_announcements (even when mesh disabled)
        self.storage_announcements: Dict[str, StorageAvailabilityAnnouncement] = {}
        self.block_requests: List[BlockRequest] = []
        
        if not self.enable_mesh:
            if not MESH_AVAILABLE:
                logger.info("Mesh networking disabled (nawal-ai not available)")
            else:
                logger.info("Mesh networking disabled by configuration")
            self.mesh = None
            return
        
        # Initialize mesh network client
        self.mesh = MeshNetworkClient(
            peer_id=self.peer_id,
            listen_port=listen_port,
            blockchain_rpc=self.blockchain_rpc,
        )
        
        logger.info(f"Initialized Pakit mesh client: {self.peer_id} on port {listen_port}")
    
    async def start(self):
        """Start mesh network client."""
        if not self.enable_mesh or not self.mesh:
            logger.debug("Mesh networking not enabled, skipping start")
            return
        
        await self.mesh.start()
        
        # Register message handlers
        self.mesh.register_handler(
            MessageType.GOSSIP,
            self._handle_storage_message
        )
        
        logger.info(f"✅ Mesh network started on port {self.listen_port}")
    
    async def stop(self):
        """Stop mesh network client."""
        if not self.enable_mesh or not self.mesh:
            return
        
        await self.mesh.stop()
        logger.info("Mesh network stopped")
    
    async def announce_storage_availability(
        self,
        cid: str,
        size_bytes: int,
        replication_count: int = 1
    ):
        """
        Announce new storage block availability to mesh network.
        
        Args:
            cid: Content identifier (IPFS CID or DAG block hash)
            size_bytes: Size of the block in bytes
            replication_count: Number of replicas available
        """
        if not self.enable_mesh or not self.mesh:
            logger.debug(f"Mesh disabled, skipping announcement for {cid}")
            return
        
        announcement = StorageAvailabilityAnnouncement(
            cid=cid,
            size_bytes=size_bytes,
            peer_id=self.peer_id,
            timestamp=time.time(),
            replication_count=replication_count
        )
        
        # Broadcast via gossip protocol
        await self.mesh._broadcast_message(
            message_type=MessageType.GOSSIP,
            payload={
                "type": "storage_availability",
                "cid": cid,
                "size_bytes": size_bytes,
                "peer_id": self.peer_id,
                "replication_count": replication_count,
                "timestamp": announcement.timestamp
            },
            ttl=5  # Propagate through 5 hops
        )
        
        # Track locally
        self.storage_announcements[cid] = announcement
        
        logger.debug(f"📢 Announced storage availability: {cid} ({size_bytes} bytes)")
    
    async def request_block_from_peers(self, cid: str, timeout: float = 10.0) -> Optional[bytes]:
        """
        Request block from mesh network peers.
        
        Args:
            cid: Content identifier to request
            timeout: Timeout in seconds
            
        Returns:
            Block data if found, None otherwise
        """
        if not self.enable_mesh or not self.mesh:
            logger.debug(f"Mesh disabled, cannot request block {cid}")
            return None
        
        # Broadcast request
        request = BlockRequest(
            cid=cid,
            requester_peer_id=self.peer_id,
            timestamp=time.time()
        )
        
        await self.mesh._broadcast_message(
            message_type=MessageType.GOSSIP,
            payload={
                "type": "block_request",
                "cid": cid,
                "requester": self.peer_id,
                "timestamp": request.timestamp
            },
            ttl=3
        )
        
        self.block_requests.append(request)
        
        # Wait for response (simplified - in production, use response handler)
        # For now, return None (peers would respond via direct messaging)
        logger.debug(f"🔍 Requested block {cid} from mesh peers")
        return None
    
    async def discover_peers(self) -> List[str]:
        """
        Discover peers from mesh network.
        
        Returns:
            List of peer IDs
        """
        if not self.enable_mesh or not self.mesh:
            return []
        
        peers = await self.mesh.discover_peers()
        return [peer.peer_id for peer in peers]
    
    async def get_mesh_health(self) -> Dict[str, Any]:
        """
        Get mesh network health status.
        
        Returns:
            Health status dict
        """
        if not self.enable_mesh or not self.mesh:
            return {
                "enabled": False,
                "status": "disabled",
                "peer_count": 0
            }
        
        peer_count = len([p for p in self.mesh.peers.values() if p.is_alive()])
        
        return {
            "enabled": True,
            "status": "healthy" if peer_count > 0 else "no_peers",
            "peer_id": self.peer_id,
            "peer_count": peer_count,
            "listen_port": self.listen_port,
            "announcements_count": len(self.storage_announcements),
            "requests_count": len(self.block_requests)
        }
    
    async def _handle_storage_message(self, message: Any):
        """
        Handle storage-related gossip messages.
        
        Args:
            message: Mesh message
        """
        if not hasattr(message, 'payload'):
            return
        
        payload = message.payload
        msg_type = payload.get("type")
        
        if msg_type == "storage_availability":
            # Track remote peer's storage announcement
            cid = payload.get("cid")
            if cid and cid not in self.storage_announcements:
                announcement = StorageAvailabilityAnnouncement(
                    cid=cid,
                    size_bytes=payload.get("size_bytes", 0),
                    peer_id=payload.get("peer_id", "unknown"),
                    timestamp=payload.get("timestamp", time.time()),
                    replication_count=payload.get("replication_count", 1)
                )
                self.storage_announcements[cid] = announcement
                logger.debug(f"📥 Received storage announcement: {cid} from {announcement.peer_id}")
        
        elif msg_type == "block_request":
            # Handle block request (in production, check if we have it and respond)
            cid = payload.get("cid")
            requester = payload.get("requester")
            logger.debug(f"📩 Received block request: {cid} from {requester}")
            # TODO: Check if we have the block and respond via direct messaging


class MockMeshClient(PakitMeshClient):
    """Mock mesh client for testing when nawal-ai is not available."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_mesh = False
        self.mesh = None
    
    async def start(self):
        logger.info("Mock mesh client started (no-op)")
    
    async def stop(self):
        logger.info("Mock mesh client stopped (no-op)")
    
    async def announce_storage_availability(self, cid: str, size_bytes: int, replication_count: int = 1):
        logger.debug(f"Mock: Would announce {cid}")
    
    async def request_block_from_peers(self, cid: str, timeout: float = 10.0) -> Optional[bytes]:
        logger.debug(f"Mock: Would request {cid}")
        return None
    
    async def discover_peers(self) -> List[str]:
        return []
