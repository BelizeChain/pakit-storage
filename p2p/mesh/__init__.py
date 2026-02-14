"""
Mesh Networking Module for Pakit Storage

Integrates nawal-ai's MeshNetworkClient for decentralized P2P communication.
Provides:
- Byzantine-resistant peer discovery
- Gossip protocol for message propagation
- Ed25519 cryptographic signing
- Storage availability announcements
"""

from .mesh_client import PakitMeshClient
from .mesh_manager import MeshNetworkManager

__all__ = ["PakitMeshClient", "MeshNetworkManager"]
