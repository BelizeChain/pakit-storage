"""
Pakit P2P (Peer-to-Peer) Network Layer

Implements distributed DAG protocol for sovereign storage across
Belizean nodes. Zero dependency on centralized services.

Components:
- DHT: Kademlia distributed hash table for peer discovery
- Gossip: Block announcement propagation protocol
- Network: TCP/WebSocket transport layer
- Discovery: Bootstrap and peer finding mechanisms
- Reputation: Peer trust and slashing system

Phase 2 Objective: Create national P2P file-sharing network
"""

__version__ = "0.3.0-alpha"
__phase__ = "Phase 2 - Distributed DAG"
