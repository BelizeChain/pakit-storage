# Mesh Networking User Guide

## Overview

Pakit now includes mesh networking capabilities powered by nawal-ai's MeshNetworkClient. This provides:
- **Byzantine-resistant peer discovery**: Filter out malicious nodes automatically
- **Gossip protocol**: Efficient block announcements across the network
- **Ed25519 signing**: Cryptographically signed messages for authenticity
- **LoRa mesh support**: Off-grid access via Meshtastic hardware

## Quick Start

### Basic Mesh Networking

```python
from p2p.node import PakitNode

# Create node with mesh networking enabled
node = PakitNode(
    listen_address="0.0.0.0:7777",
    enable_mesh=True,
    mesh_network_id="pakit-main"
)

# Start mesh networking
await node.start_mesh_networking()

# Discover peers automatically
peers = await node.discover_peers_via_mesh(max_peers=10)
print(f"Discovered {len(peers)} peers via mesh")

# Announce new block to mesh network
await node.announce_block_via_mesh(
    block_hash="abc123...",
    block_size=1024
)

# Check mesh health
stats = node.get_stats()
if stats["mesh_enabled"]:
    print(f"Mesh health: {stats['mesh_health']}")
```

### Advanced Usage

#### Direct Mesh Client Access

```python
from p2p.mesh.mesh_client import PakitMeshClient

# Create mesh client directly
client = PakitMeshClient(
    peer_id="my_peer_123",
    signing_key=b"32_byte_ed25519_private_key_here",
    network_id="pakit-main"
)

# Announce storage availability
await client.announce_storage_availability(
    storage_capacity_gb=1000,
    available_blocks=["block1", "block2", "block3"]
)

# Request block from peers
peer_id = await client.request_block_from_peers("block_abc123")
if peer_id:
    print(f"Block available from peer: {peer_id}")

# Discover peers
peers = await client.discover_peers(max_peers=20)
for peer in peers:
    print(f"Peer: {peer['peer_id']} (reputation: {peer['reputation']:.2f})")

# Check mesh health
health = client.get_mesh_health()
print(f"Connected: {health['connected']}")
print(f"Peers: {health['peer_count']}")
```

## LoRa Mesh Integration (Off-Grid Access)

### Hardware Requirements

- **Meshtastic device**: T-Beam, Heltec LoRa32, RAK WisBlock, etc.
- **Firmware**: Meshtastic 2.3.0+
- **Connection**: USB or Bluetooth to your Pakit node

### Setup

```python
from p2p.mesh.lora_bridge import LoRaMeshBridge

# Initialize LoRa bridge
bridge = LoRaMeshBridge(
    peer_id="my_peer_123",
    lora_channel=1,  # Primary long-range channel
    lora_name="pakit-node-belize"
)

# Connect to Meshtastic device
await bridge.connect()

# Broadcast DAG index update to LoRa mesh
index_data = {
    "version": 5,
    "root_hash": "abc123...",
    "block_count": 1000
}
await bridge.broadcast_index_update(index_data)

# Sync DAG index with other LoRa nodes
synced_index = await bridge.sync_dag_index(local_index)

# Request small file (< 200KB for LoRa constraints)
file_data = await bridge.request_small_file(
    file_hash="file_abc123",
    max_size_kb=50
)

# Emergency broadcast (high priority)
emergency_data = {
    "type": "critical_update",
    "message": "Important system alert"
}
await bridge.broadcast_emergency_data(emergency_data)

# Check stats
stats = bridge.get_stats()
print(f"LoRa connected: {stats['connected']}")
print(f"Messages sent: {stats['messages_sent']}")
```

### LoRa Best Practices

1. **Message Size**: Keep individual messages < 237 bytes (LoRa constraint)
2. **Priority**: Use emergency broadcasts sparingly
3. **Channels**: Use channel 1 for critical coordination
4. **Range**: Expect 5-30km range depending on terrain
5. **Latency**: LoRa is slow (~1 message/5 seconds), use for coordination only

## Mesh Network Manager

For high-level mesh management:

```python
from p2p.mesh.mesh_manager import MeshNetworkManager

# Create manager
manager = MeshNetworkManager(
    peer_id="my_peer_123",
    signing_key=b"32_byte_signing_key",
    network_id="pakit-prod",
    discovery_interval=60  # Discovery every 60 seconds
)

# Start manager (runs background discovery loop)
await manager.start()

# Announce new block
await manager.announce_new_block(
    block_hash="block_xyz",
    block_size=2048
)

# Find peers that have specific block
peers = await manager.find_block_peers(
    block_hash="block_xyz",
    max_peers=5
)

# Discover storage providers
providers = await manager.discover_storage_peers(max_peers=10)

# Get health status
health = manager.get_health_status()
print(f"Manager running: {health['running']}")
print(f"Mesh health: {health['mesh_client_health']}")

# Stop manager
await manager.stop()
```

## Configuration

### Environment Variables

```bash
# Enable mesh networking
MESH_ENABLED=true

# Network ID (isolate from other networks)
MESH_NETWORK_ID=pakit-main

# Discovery interval (seconds)
MESH_DISCOVERY_INTERVAL=60

# Max peers to discover
MESH_MAX_PEERS=50

# Byzantine filter threshold (0.0-1.0, higher = more strict)
MESH_BYZANTINE_THRESHOLD=0.3
```

### Docker Compose

```yaml
services:
  pakit-node:
    environment:
      - MESH_ENABLED=true
      - MESH_NETWORK_ID=pakit-prod
      - MESH_DISCOVERY_INTERVAL=60
    ports:
      - "7777:7777"  # P2P
      - "8080:8080"  # API
```

## Integration with Storage Engine

The P2P node automatically integrates mesh networking with the storage engine:

```python
from core.storage_engine import PakitStorageEngine
from p2p.node import PakitNode

# Create storage engine
engine = PakitStorageEngine(enable_dag=True)

# Create P2P node with mesh
node = PakitNode(enable_mesh=True)
await node.start_mesh_networking()

# Store data
data = b"Important data to store"
content_id = engine.store(data)

# Announce to mesh network
await node.announce_block_via_mesh(
    block_hash=content_id.hex,
    block_size=len(data)
)

# Other nodes can now discover this block
# via mesh networking
```

## Troubleshooting

### Mesh Not Connecting

1. **Check nawal-ai installation**: `pip install nawal-ai[mesh]>=1.1.0`
2. **Verify network ID**: Ensure all nodes use the same `mesh_network_id`
3. **Check logs**: Look for "Mesh networking enabled" message
4. **Firewall**: Ensure mesh gossip ports are open

### LoRa Bridge Issues

1. **Device not found**: Check USB connection (`ls /dev/ttyUSB*`)
2. **Permission denied**: Add user to `dialout` group (`sudo usermod -a -G dialout $USER`)
3. **Firmware version**: Update Meshtastic to 2.3.0+
4. **Channel mismatch**: Ensure all devices on same LoRa channel

### Performance

- **High latency**: Reduce `discovery_interval` (but increases network traffic)
- **Too many peers**: Reduce `max_peers` limit
- **Byzantine nodes**: Increase `byzantine_threshold` to filter more strictly

## Security Considerations

1. **Ed25519 Signing**: All mesh messages are cryptographically signed
2. **Byzantine Resistance**: Malicious nodes automatically filtered
3. **Network Isolation**: Use unique `network_id` per deployment
4. **LoRa Encryption**: Meshtastic provides optional AES-256 encryption
5. **Reputation System**: Low-reputation nodes are deprioritized

## Advanced Topics

### Custom Mesh Protocol

Extend mesh client for custom protocols:

```python
class CustomMeshClient(PakitMeshClient):
    async def broadcast_custom_message(self, data: dict):
        """Send custom message via mesh."""
        return await self.mesh_client.broadcast(
            message_type="custom",
            data=data
        )
```

### Mesh Analytics

Monitor mesh network health:

```python
async def monitor_mesh(node):
    while True:
        stats = node.get_stats()
        health = stats.get("mesh_health", {})
        
        print(f"Mesh peers: {health.get('peer_count', 0)}")
        print(f"Last gossip: {health.get('last_gossip', 'never')}")
        
        await asyncio.sleep(30)
```

## API Reference

See module documentation:
- `p2p.mesh.mesh_client.PakitMeshClient`
- `p2p.mesh.mesh_manager.MeshNetworkManager`
- `p2p.mesh.lora_bridge.LoRaMeshBridge`
- `p2p.node.PakitNode`
