"""
Pakit Blockchain Storage Proof Connector

Connects Pakit storage backends to BelizeChain's 16 pallets:
- LandLedger: Document storage proofs
- Mesh: LoRa mesh networking and ZK proofs
- Economy: DALLA/bBZD payment processing
- BNS: .bz domain hosting
- Contracts: Smart contract storage

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

# Import new pallet connectors
try:
    from blockchain.economy_integration import EconomyPalletConnector
    from blockchain.bns_integration import BNSPalletConnector
    from blockchain.contracts_integration import ContractsPalletConnector
    PALLET_CONNECTORS_AVAILABLE = True
except ImportError:
    PALLET_CONNECTORS_AVAILABLE = False
    logger.warning("Pallet connectors not available")


class StorageProofConnector:
    """
    Connector for storing and retrieving storage metadata on BelizeChain.
    
    Integrates with 16 BelizeChain pallets:
    - LandLedger: Document storage proofs, IPFS/Arweave CID tracking
    - Mesh: LoRa mesh networking, ZK storage proofs
    - Economy: DALLA/bBZD payment processing
    - BNS: .bz domain hosting on IPFS
    - Contracts: Smart contract storage backend
    - And 11 more standard pallets
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
        
        # Pallet connectors (initialized in connect())
        self.economy: Optional[EconomyPalletConnector] = None
        self.bns: Optional[BNSPalletConnector] = None
        self.contracts: Optional[ContractsPalletConnector] = None
    
    async def connect(self):
        """Connect to BelizeChain node and initialize pallet connectors."""
        if self.mock_mode or not SUBSTRATE_AVAILABLE:
            logger.info("📦 Storage proof connector in MOCK mode (no blockchain)")
            self.connected = True
            
            # Initialize mock pallet connectors
            if PALLET_CONNECTORS_AVAILABLE:
                self.economy = EconomyPalletConnector(None)
                self.bns = BNSPalletConnector(None)
                self.contracts = ContractsPalletConnector(None)
                logger.info("✅ Initialized pallet connectors in MOCK mode")
            
            return
        
        try:
            self.substrate = SubstrateInterface(url=self.node_url)
            self.connected = True
            logger.info("✅ Connected to BelizeChain at {}", self.node_url)
            
            # Initialize pallet connectors with substrate interface
            if PALLET_CONNECTORS_AVAILABLE:
                self.economy = EconomyPalletConnector(self.substrate)
                self.bns = BNSPalletConnector(self.substrate)
                self.contracts = ContractsPalletConnector(self.substrate)
                logger.info("✅ Initialized all pallet connectors")
            
        except Exception as e:
            logger.error("Failed to connect to blockchain: {}", e)
            logger.warning("Falling back to MOCK mode")
            self.mock_mode = True
            self.connected = True
            
            # Initialize mock pallet connectors
            if PALLET_CONNECTORS_AVAILABLE:
                self.economy = EconomyPalletConnector(None)
                self.bns = BNSPalletConnector(None)
                self.contracts = ContractsPalletConnector(None)
    
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
    
    # ===== New Methods for 16-Pallet Support =====
    
    async def submit_storage_zk_proof(
        self,
        content_id: str,
        zk_proof: Dict[str, Any],
        proof_type: str = "groth16"
    ) -> bool:
        """
        Submit zero-knowledge storage proof to Mesh pallet.
        
        Args:
            content_id: Content identifier
            zk_proof: ZK proof data from StorageProofGenerator
            proof_type: Proof system ('groth16', 'plonk', 'stark')
            
        Returns:
            True if submission successful
        """
        if not self.connected:
            logger.warning("Not connected to blockchain - ZK proof not submitted")
            return False
        
        if self.mock_mode:
            logger.debug("📦 [MOCK] Submitted ZK proof for {}", content_id[:16])
            return True
        
        try:
            if not self.keypair:
                logger.error("No keypair configured - cannot submit ZK proof")
                return False
            
            # Submit to Mesh pallet
            call = self.substrate.compose_call(
                call_module='Mesh',
                call_function='submit_storage_proof',
                call_params={
                    'content_id': content_id,
                    'proof': str(zk_proof),
                    'proof_type': proof_type
                }
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=True
            )
            
            if receipt.is_success:
                logger.info("✅ Submitted ZK storage proof for {}", content_id)
                return True
            else:
                logger.error("ZK proof submission failed: {}", receipt.error_message)
                return False
                
        except Exception as e:
            logger.error("Failed to submit ZK proof: {}", e)
            return False
    
    def get_economy_connector(self) -> Optional[EconomyPalletConnector]:
        """Get Economy pallet connector for payment processing."""
        return self.economy
    
    def get_bns_connector(self) -> Optional[BNSPalletConnector]:
        """Get BNS pallet connector for .bz domain hosting."""
        return self.bns
    
    def get_contracts_connector(self) -> Optional[ContractsPalletConnector]:
        """Get Contracts pallet connector for smart contract storage."""
        return self.contracts
    
    async def get_multi_pallet_status(self) -> Dict[str, Any]:
        """
        Get status from all integrated pallets.
        
        Returns:
            Status dict with all pallet information
        """
        status = {
            "connected": self.connected,
            "mock_mode": self.mock_mode,
            "node_url": self.node_url,
            "pallets": {}
        }
        
        if self.economy:
            status["pallets"]["economy"] = self.economy.get_pricing()
        
        if self.bns:
            status["pallets"]["bns"] = self.bns.get_hosting_stats()
        
        if self.contracts:
            status["pallets"]["contracts"] = self.contracts.get_storage_stats()
        
        return status


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
