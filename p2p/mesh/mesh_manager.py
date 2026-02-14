"""
Mesh Network Manager for Pakit Storage

High-level manager for coordinating mesh networking across storage operations.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from .mesh_client import PakitMeshClient, MESH_AVAILABLE

logger = logging.getLogger(__name__)


class MeshNetworkManager:
    """
    Manages mesh networking for Pakit storage operations.
    
    Coordinates:
    - Peer discovery via mesh network
    - Storage availability propagation
    - Block request routing
    - Replication coordination
    """
    
    def __init__(
        self,
        peer_id: str,
        listen_port: int = 9091,
        enable_mesh: bool = True
    ):
        """
        Initialize mesh network manager.
        
        Args:
            peer_id: Unique peer identifier
            listen_port: Port for mesh network
            enable_mesh: Enable mesh networking
        """
        self.peer_id = peer_id
        self.listen_port = listen_port
        self.enable_mesh = enable_mesh and MESH_AVAILABLE
        
        # Initialize mesh client
        self.mesh_client = PakitMeshClient(
            peer_id=peer_id,
            listen_port=listen_port,
            enable_mesh=self.enable_mesh
        )
        
        self.running = False
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start mesh network manager."""
        if self.running:
            logger.warning("Mesh network manager already running")
            return
        
        await self.mesh_client.start()
        self.running = True
        
        # Start background tasks
        if self.enable_mesh:
            self._background_tasks.append(
                asyncio.create_task(self._peer_discovery_loop())
            )
        
        logger.info("✅ Mesh network manager started")
    
    async def stop(self):
        """Stop mesh network manager."""
        if not self.running:
            return
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        
        await self.mesh_client.stop()
        self.running = False
        
        logger.info("Mesh network manager stopped")
    
    async def announce_new_block(self, block_hash: str, block_size: int) -> bool:
        """
        Announce new storage block to mesh network.
        
        Args:
            block_hash: Content identifier (block hash)
            block_size: Block size in bytes
            
        Returns:
            True if announcement successful
        """
        try:
            await self.mesh_client.announce_storage_availability(
                cid=block_hash,
                size_bytes=block_size,
                replication_count=1
            )
            return True
        except Exception:
            return False
    
    async def find_block_peers(self, block_hash: str, max_peers: int = 10) -> List[Dict[str, Any]]:
        """
        Find peers that have a specific block.
        
        Args:
            block_hash: Content identifier (block hash)
            max_peers: Maximum number of peers to return
            
        Returns:
            List of peer info dicts
        """
        # Check local announcements
        peers = []
        announcement = self.mesh_client.storage_announcements.get(block_hash)
        if announcement and announcement.peer_id != self.mesh_client.peer_id:
            peers.append({
                "peer_id": announcement.peer_id,
                "block_hash": block_hash
            })
        
        return peers[:max_peers]
    
    async def request_block(self, cid: str, timeout: float = 10.0) -> Optional[bytes]:
        """
        Request block from mesh peers.
        
        Args:
            cid: Content identifier
            timeout: Request timeout
            
        Returns:
            Block data if found
        """
        return await self.mesh_client.request_block_from_peers(cid, timeout)
    
    async def get_peer_count(self) -> int:
        """Get number of connected mesh peers."""
        peers = await self.mesh_client.discover_peers()
        return len(peers)
    
    async def discover_storage_peers(self, max_peers: int = 10) -> List[Dict[str, Any]]:
        """Discover storage provider peers."""
        peers_list = await self.mesh_client.discover_peers()
        # Convert to list of dicts
        result = [{"peer_id": p} for p in peers_list[:max_peers]]
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get mesh network health status."""
        return {
            "running": self.running,
            "mesh_enabled": self.enable_mesh,
            "mesh_client_health": self.mesh_client.get_mesh_health()
        }
    
    async def _peer_discovery_loop(self):
        """Background task for periodic peer discovery."""
        while self.running:
            try:
                peers = await self.mesh_client.discover_peers()
                logger.debug(f"Mesh peer discovery: {len(peers)} peers")
                await asyncio.sleep(60)  # Discover every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in peer discovery loop: {e}")
                await asyncio.sleep(60)
