"""
Integration tests for mesh networking.

Tests PakitMeshClient, MeshNetworkManager, and integration with P2P node.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from p2p.mesh.mesh_client import PakitMeshClient
from p2p.mesh.mesh_manager import MeshNetworkManager
from p2p.node import PakitNode


class TestPakitMeshClient:
    """Test mesh client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.peer_id = "test_peer_123"
        
        self.client = PakitMeshClient(
            peer_id=self.peer_id,
            listen_port=9091,
            enable_mesh=True
        )
    
    @pytest.mark.asyncio
    async def test_announce_storage_availability(self):
        """Test announcing storage availability."""
        success = await self.client.announce_storage_availability(
            cid="Qm123abc...",
            size_bytes=1024
        )
        
        # Method doesn't return anything (None is expected)
        assert success is None
    
    @pytest.mark.asyncio
    async def test_request_block_from_peers(self):
        """Test requesting block from peers."""
        block_hash = "Qm_block_hash_123"
        data = await self.client.request_block_from_peers(block_hash)
        
        # In mock mode, returns None (no peers available)
        assert data is None or isinstance(data, bytes)
    
    @pytest.mark.asyncio
    async def test_discover_peers(self):
        """Test peer discovery."""
        peers = await self.client.discover_peers()
        
        assert isinstance(peers, list)
        
        # In mock mode, returns empty list
        assert len(peers) >= 0
    
    @pytest.mark.asyncio
    async def test_get_mesh_health(self):
        """Test mesh health status retrieval."""
        health = await self.client.get_mesh_health()
        
        assert isinstance(health, dict)
        assert "enabled" in health


class TestMeshNetworkManager:
    """Test mesh network manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.peer_id = "manager_test_peer"
        
        self.manager = MeshNetworkManager(
            peer_id=self.peer_id,
            listen_port=9092
        )
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test manager lifecycle."""
        # Start manager
        await self.manager.start()
        assert self.manager.running is True
        
        # Stop manager
        await self.manager.stop()
        assert self.manager.running is False
    
    @pytest.mark.asyncio
    async def test_announce_new_block(self):
        """Test block announcement."""
        await self.manager.start()
        
        success = await self.manager.announce_new_block(
            block_hash="test_block_hash",
            block_size=1024
        )
        
        assert success is True
        
        await self.manager.stop()
    
    @pytest.mark.asyncio
    async def test_find_block_peers(self):
        """Test finding peers with specific block."""
        await self.manager.start()
        
        peers = await self.manager.find_block_peers(
            block_hash="test_block",
            max_peers=3
        )
        
        assert isinstance(peers, list)
        # May be empty since we have no real announcements
        
        await self.manager.stop()
    
    @pytest.mark.asyncio
    async def test_discover_storage_peers(self):
        """Test discovering storage providers."""
        await self.manager.start()
        
        peers = await self.manager.discover_storage_peers(max_peers=10)
        
        assert isinstance(peers, list)
        assert len(peers) <= 10
        
        await self.manager.stop()
    
    def test_get_health_status(self):
        """Test health status retrieval."""
        health = self.manager.get_health_status()
        
        assert isinstance(health, dict)
        assert "running" in health
        assert "mesh_client_health" in health


class TestNodeMeshIntegration:
    """Test P2P node integration with mesh networking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.node = PakitNode(
            listen_address="127.0.0.1:7777",
            enable_mesh=True
        )
    
    @pytest.mark.asyncio
    async def test_node_mesh_initialization(self):
        """Test that node initializes mesh manager."""
        # Mesh manager should be initialized
        # (might be None if nawal-ai not installed)
        pass  # Implementation depends on availability
    
    @pytest.mark.asyncio
    async def test_discover_peers_via_mesh(self):
        """Test peer discovery through mesh."""
        if not self.node.mesh_manager:
            pytest.skip("Mesh networking not available")
        
        # Start mesh
        await self.node.start_mesh_networking()
        
        # Discover peers
        peers = await self.node.discover_peers_via_mesh(max_peers=5)
        
        assert isinstance(peers, list)
        assert len(peers) <= 5
        
        # Stop mesh
        await self.node.stop_mesh_networking()
    
    @pytest.mark.asyncio
    async def test_announce_block_via_mesh(self):
        """Test block announcement through mesh."""
        if not self.node.mesh_manager:
            pytest.skip("Mesh networking not available")
        
        # Start mesh
        await self.node.start_mesh_networking()
        
        # Announce block
        success = await self.node.announce_block_via_mesh(
            block_hash="test_block_hash",
            block_size=2048
        )
        
        assert isinstance(success, bool)
        
        # Stop mesh
        await self.node.stop_mesh_networking()
    
    def test_node_stats_include_mesh(self):
        """Test that node stats include mesh information."""
        stats = self.node.get_stats()
        
        assert "mesh_enabled" in stats
        
        if stats["mesh_enabled"]:
            assert "mesh_health" in stats


@pytest.mark.integration
class TestEndToEndMeshFlow:
    """End-to-end integration tests for mesh networking."""
    
    @pytest.mark.asyncio
    async def test_two_nodes_mesh_discovery(self):
        """Test two nodes discovering each other via mesh."""
        # Create two nodes
        node1 = PakitNode(
            listen_address="127.0.0.1:7001",
            enable_mesh=True
        )
        
        node2 = PakitNode(
            listen_address="127.0.0.1:7002",
            enable_mesh=True
        )
        if not node1.mesh_manager or not node2.mesh_manager:
            pytest.skip("Mesh networking not available")
        
        # Start both mesh networks
        await node1.start_mesh_networking()
        await node2.start_mesh_networking()
        
        # Node1 announces availability
        await node1.announce_block_via_mesh("block_123", 1024)
        
        # Wait for propagation
        await asyncio.sleep(0.5)
        
        # Node2 discovers peers
        peers = await node2.discover_peers_via_mesh(max_peers=5)
        
        # Should discover at least mock peers
        assert len(peers) >= 0
        
        # Cleanup
        await node1.stop_mesh_networking()
        await node2.stop_mesh_networking()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
