"""
Pakit Blockchain Storage Proof Connector

Connects Pakit storage backends to BelizeChain's LandLedger pallet
for on-chain metadata storage and verification.

Author: BelizeChain Team
License: Apache-2.0
"""

import asyncio
from typing import Optional, Dict, Any
from loguru import logger

try:
    from substrateinterface import SubstrateInterface, Keypair
    SUBSTRATE_AVAILABLE = True
except ImportError:
    SUBSTRATE_AVAILABLE = False
    logger.warning("substrate-interface not installed. On-chain storage disabled.")


class StorageProofConnector:
    """
    Connector for storing and retrieving storage metadata on BelizeChain.
    
    Integrates with LandLedger pallet for:
    - Document storage proofs
    - IPFS/Arweave CID tracking
    - Content verification
    """
    
    def __init__(
        self,
        node_url: str = "ws://localhost:9944",
        keypair: Optional[Keypair] = None,
        mock_mode: bool = True
    ):
        """
        Initialize blockchain connector.
        
        Args:
            node_url: BelizeChain RPC endpoint
            keypair: Account keypair for signing extrinsics
            mock_mode: If True, operates without blockchain (development)
        """
        self.node_url = node_url
        self.keypair = keypair
        self.mock_mode = mock_mode
        self.substrate: Optional[SubstrateInterface] = None
        self.connected = False
        
        # In-memory cache for mock mode
        self._mock_storage: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self):
        """Connect to BelizeChain node."""
        if self.mock_mode or not SUBSTRATE_AVAILABLE:
            logger.info("📦 Storage proof connector in MOCK mode (no blockchain)")
            self.connected = True
            return
        
        try:
            self.substrate = SubstrateInterface(url=self.node_url)
            self.connected = True
            logger.info("✅ Connected to BelizeChain at {}", self.node_url)
        except Exception as e:
            logger.error("Failed to connect to blockchain: {}", e)
            logger.warning("Falling back to MOCK mode")
            self.mock_mode = True
            self.connected = True
    
    async def disconnect(self):
        """Disconnect from blockchain."""
        if self.substrate:
            self.substrate.close()
        self.connected = False
        logger.info("Disconnected from blockchain")
    
    async def store_document_proof(
        self,
        content_id: str,
        ipfs_cid: Optional[str] = None,
        arweave_tx: Optional[str] = None,
        owner: str = "system",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store document storage proof on-chain.
        
        Args:
            content_id: Pakit content identifier (hash)
            ipfs_cid: IPFS content identifier (if stored on IPFS)
            arweave_tx: Arweave transaction ID (if stored on Arweave)
            owner: Account ID of document owner
            metadata: Additional metadata (size, mime type, etc.)
        
        Returns:
            True if successfully stored
        """
        if not self.connected:
            logger.warning("Not connected to blockchain - storage proof not saved")
            return False
        
        if self.mock_mode:
            # Store in mock storage
            self._mock_storage[content_id] = {
                "ipfs_cid": ipfs_cid,
                "arweave_tx": arweave_tx,
                "owner": owner,
                "metadata": metadata or {},
            }
            logger.debug("📦 [MOCK] Stored proof for {}", content_id[:16])
            return True
        
        try:
            # Submit extrinsic to LandLedger pallet
            if not self.keypair:
                logger.error("No keypair configured - cannot submit extrinsic")
                return False
            
            # Compose call to LandLedger pallet
            call = self.substrate.compose_call(
                call_module='LandLedger',
                call_function='register_document_proof',
                call_params={
                    'content_id': content_id,
                    'ipfs_cid': ipfs_cid,
                    'arweave_tx': arweave_tx or '',
                    'metadata': str(metadata or {}),
                }
            )
            
            # Create and submit signed extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info(f"Successfully registered document proof for {content_id}")
                return True
            else:
                logger.error(f"Extrinsic failed: {receipt.error_message}")
                return False
            
        except Exception as e:
            logger.error("Failed to store document proof: {}", e)
            return False
    
    async def get_document_proof(
        self,
        content_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve document storage proof from blockchain.
        
        Args:
            content_id: Pakit content identifier
        
        Returns:
            Dictionary with ipfs_cid, arweave_tx, owner, metadata
        """
        if not self.connected:
            return None
        
        if self.mock_mode:
            return self._mock_storage.get(content_id)
        
        try:
            # Query LandLedger storage for document proof
            result = self.substrate.query(
                module='LandLedger',
                storage_function='DocumentProofs',
                params=[content_id]
            )
            
            if result.value:
                # Parse storage result
                proof_data = result.value
                return {
                    "ipfs_cid": proof_data.get('ipfs_cid', ''),
                    "arweave_tx": proof_data.get('arweave_tx', ''),
                    "owner": proof_data.get('owner', ''),
                    "metadata": proof_data.get('metadata', {}),
                }
            
            logger.info(f"No proof found for content_id: {content_id}")
            return self._mock_storage.get(content_id)
            
        except Exception as e:
            logger.error("Failed to query document proof: {}", e)
            return None
    
    async def verify_document_exists(self, content_id: str) -> bool:
        """
        Verify that a document proof exists on-chain.
        
        Args:
            content_id: Pakit content identifier
        
        Returns:
            True if proof exists on-chain
        """
        proof = await self.get_document_proof(content_id)
        return proof is not None
    
    async def get_ipfs_cid(self, content_id: str) -> Optional[str]:
        """Get IPFS CID for content ID from blockchain."""
        proof = await self.get_document_proof(content_id)
        if proof:
            return proof.get("ipfs_cid")
        return None
    
    async def get_arweave_tx(self, content_id: str) -> Optional[str]:
        """Get Arweave TX ID for content ID from blockchain."""
        proof = await self.get_document_proof(content_id)
        if proof:
            return proof.get("arweave_tx")
        return None


# Singleton instance for shared use
_global_connector: Optional[StorageProofConnector] = None


def get_storage_proof_connector(
    node_url: str = "ws://localhost:9944",
    mock_mode: bool = True
) -> StorageProofConnector:
    """
    Get or create global storage proof connector.
    
    Args:
        node_url: BelizeChain RPC endpoint
        mock_mode: If True, operates without blockchain
    
    Returns:
        StorageProofConnector instance
    """
    global _global_connector
    
    if _global_connector is None:
        _global_connector = StorageProofConnector(
            node_url=node_url,
            mock_mode=mock_mode
        )
    
    return _global_connector
