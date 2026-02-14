"""
LoRa Mesh Bridge for Pakit Storage

Enables low-bandwidth storage access over Meshtastic LoRa mesh network.
Provides:
- Off-grid DAG index synchronization
- Small file retrieval (< 1KB) over LoRa
- Emergency data broadcast
- Rural area storage access without internet
"""

import asyncio
import logging
import time
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Try to import Meshtastic
try:
    import meshtastic
    import meshtastic.serial_interface
    MESHTASTIC_AVAILABLE = True
except ImportError:
    logger.warning("Meshtastic not available. Install with: pip install meshtastic")
    MESHTASTIC_AVAILABLE = False
    meshtastic = None


@dataclass
class LoRaMessage:
    """LoRa mesh message."""
    message_type: str  # 'index_update', 'file_request', 'file_response', 'emergency'
    payload: Dict[str, Any]
    timestamp: float
    sender_id: str


class LoRaMeshBridge:
    """
    Bridge between Pakit storage and BelizeChain Mesh pallet via LoRa.
    
    Enables off-grid storage operations:
    - DAG index updates broadcast over LoRa
    - Small file requests/responses
    - Emergency data synchronization
    """
    
    def __init__(
        self,
        blockchain_rpc: str = "ws://localhost:9944",
        device_path: Optional[str] = None,
        enable_lora: bool = False
    ):
        """
        Initialize LoRa mesh bridge.
        
        Args:
            blockchain_rpc: BelizeChain RPC endpoint
            device_path: Path to Meshtastic device (e.g., /dev/ttyUSB0)
            enable_lora: Enable LoRa mesh (requires hardware)
        """
        self.blockchain_rpc = blockchain_rpc
        self.device_path = device_path or os.getenv("LORA_DEVICE", "/dev/ttyUSB0")
        self.enable_lora = enable_lora and MESHTASTIC_AVAILABLE
        
        self.interface = None
        self.running = False
        self._message_queue: List[LoRaMessage] = []
        
        if not self.enable_lora:
            if not MESHTASTIC_AVAILABLE:
                logger.info("LoRa mesh disabled (Meshtastic not available)")
            else:
                logger.info("LoRa mesh disabled by configuration")
    
    async def start(self):
        """Start LoRa mesh bridge."""
        if not self.enable_lora:
            logger.debug("LoRa mesh not enabled, skipping start")
            return
        
        try:
            # Initialize Meshtastic interface
            self.interface = meshtastic.serial_interface.SerialInterface(
                devPath=self.device_path
            )
            self.running = True
            logger.info(f"✅ LoRa mesh bridge started on {self.device_path}")
        except Exception as e:
            logger.error(f"Failed to start LoRa mesh bridge: {e}")
            self.enable_lora = False
    
    async def stop(self):
        """Stop LoRa mesh bridge."""
        if not self.running:
            return
        
        if self.interface:
            self.interface.close()
        
        self.running = False
        logger.info("LoRa mesh bridge stopped")
    
    async def broadcast_index_update(self, dag_index_hash: str):
        """
        Broadcast DAG index update over LoRa mesh.
        
        Args:
            dag_index_hash: Hash of the updated DAG index
        """
        if not self.enable_lora or not self.running:
            logger.debug(f"LoRa disabled, skipping index broadcast: {dag_index_hash}")
            return
        
        message = LoRaMessage(
            message_type="index_update",
            payload={"dag_index_hash": dag_index_hash},
            timestamp=time.time(),
            sender_id="pakit_storage"
        )
        
        # Send via Meshtastic
        try:
            self.interface.sendText(json.dumps({
                "type": message.message_type,
                "hash": dag_index_hash,
                "ts": message.timestamp
            }))
            logger.info(f"📡 Broadcasted DAG index update over LoRa: {dag_index_hash[:16]}...")
        except Exception as e:
            logger.error(f"Failed to broadcast over LoRa: {e}")
    
    async def request_small_file(self, cid: str, timeout: float = 30.0) -> Optional[bytes]:
        """
        Request small file (< 1KB) over LoRa mesh.
        
        For emergency data access during network outages.
        
        Args:
            cid: Content identifier
            timeout: Request timeout
            
        Returns:
            File data if available
        """
        if not self.enable_lora or not self.running:
            logger.debug(f"LoRa disabled, cannot request file {cid}")
            return None
        
        message = LoRaMessage(
            message_type="file_request",
            payload={"cid": cid},
            timestamp=time.time(),
            sender_id="pakit_storage"
        )
        
        try:
            # Send request
            self.interface.sendText(json.dumps({
                "type": "file_request",
                "cid": cid,
                "ts": message.timestamp
            }))
            
            logger.info(f"📡 Requested file over LoRa: {cid}")
            
            # Wait for response (simplified - in production, implement response handler)
            # For now, return None
            return None
        except Exception as e:
            logger.error(f"Failed to request file over LoRa: {e}")
            return None
    
    async def broadcast_emergency_data(self, data: bytes, description: str = ""):
        """
        Broadcast emergency data over LoRa mesh.
        
        For critical data synchronization during network outages.
        
        Args:
            data: Data to broadcast (keep small, < 500 bytes)
            description: Description of the emergency data
        """
        if not self.enable_lora or not self.running:
            logger.debug("LoRa disabled, skipping emergency broadcast")
            return
        
        if len(data) > 500:
            logger.warning(f"Emergency data too large ({len(data)} bytes), may fail over LoRa")
        
        message = LoRaMessage(
            message_type="emergency",
            payload={
                "data": data.hex(),
                "description": description
            },
            timestamp=time.time(),
            sender_id="pakit_storage"
        )
        
        try:
            self.interface.sendText(json.dumps({
                "type": "emergency",
                "desc": description,
                "data": data.hex()[:100],  # Truncate for LoRa
                "ts": message.timestamp
            }))
            logger.warning(f"🚨 Emergency broadcast over LoRa: {description}")
        except Exception as e:
            logger.error(f"Failed to broadcast emergency data: {e}")
    
    async def sync_dag_index(self, local_hash: str) -> Optional[str]:
        """
        Synchronize DAG index with mesh network.
        
        Args:
            local_hash: Local DAG index hash
            
        Returns:
            Latest DAG index hash from network, if different
        """
        if not self.enable_lora or not self.running:
            return None
        
        # Broadcast our hash
        await self.broadcast_index_update(local_hash)
        
        # In production: Listen for other nodes' hashes and determine latest
        # For now, return None
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get LoRa mesh health status.
        
        Returns:
            Health status dict
        """
        if not self.enable_lora:
            return {
                "enabled": False,
                "status": "disabled",
                "device": None
            }
        
        return {
            "enabled": True,
            "status": "healthy" if self.running else "not_started",
            "device": self.device_path,
            "running": self.running,
            "message_queue_size": len(self._message_queue)
        }


class MockLoRaBridge(LoRaMeshBridge):
    """Mock LoRa bridge for testing without hardware."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_lora = False
    
    async def start(self):
        logger.info("Mock LoRa bridge started (no-op)")
        self.running = True
    
    async def stop(self):
        logger.info("Mock LoRa bridge stopped (no-op)")
        self.running = False
    
    async def broadcast_index_update(self, dag_index_hash: str):
        logger.debug(f"Mock: Would broadcast index {dag_index_hash}")
    
    async def request_small_file(self, cid: str, timeout: float = 30.0) -> Optional[bytes]:
        logger.debug(f"Mock: Would request file {cid}")
        return None
    
    async def broadcast_emergency_data(self, data: bytes, description: str = ""):
        logger.debug(f"Mock: Would broadcast emergency data: {description}")
