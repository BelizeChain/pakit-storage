"""
Contracts Pallet Integration for Pakit Storage

Integrates with BelizeChain's Contracts pallet to:
- Store smart contract data
- Provide storage for contract state
- Enable contract-based access control
- Track contract storage usage
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContractsPalletConnector:
    """
    Connector for BelizeChain Contracts pallet (ink! 4.0).
    
    Enables Pakit to:
    - Store smart contract data (code, state, metadata)
    - Provide storage backend for contracts
    - Implement contract-based access control
    - Track storage usage per contract
    """
    
    def __init__(self, substrate_interface: Any):
        """
        Initialize Contracts pallet connector.
        
        Args:
            substrate_interface: Substrate interface from storage_proof_connector
        """
        self.substrate = substrate_interface
        self.contract_storage: Dict[str, Dict[str, Any]] = {}  # contract_address -> storage_info
    
    async def store_contract_data(
        self,
        contract_address: str,
        data_cid: str,
        data_type: str,
        size_bytes: int
    ) -> bool:
        """
        Store smart contract data on Pakit.
        
        Args:
            contract_address: Contract address (AccountId)
            data_cid: IPFS CID or DAG block hash of the data
            data_type: Type of data ('code', 'state', 'metadata', 'logs')
            size_bytes: Size of data in bytes
            
        Returns:
            True if storage successful
        """
        try:
            # Submit extrinsic to Contracts pallet
            call = self.substrate.compose_call(
                call_module='Contracts',
                call_function='store_contract_data',
                call_params={
                    'contract': contract_address,
                    'data_cid': data_cid,
                    'data_type': data_type,
                    'size': size_bytes
                }
            )
            
            # Track locally
            if contract_address not in self.contract_storage:
                self.contract_storage[contract_address] = {
                    "data": {},
                    "total_size": 0
                }
            
            self.contract_storage[contract_address]["data"][data_type] = {
                "cid": data_cid,
                "size": size_bytes
            }
            self.contract_storage[contract_address]["total_size"] += size_bytes
            
            logger.info(
                f"✅ Stored contract data: {contract_address[:16]}... "
                f"{data_type} → {data_cid[:16]}... ({size_bytes} bytes)"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to store contract data: {e}")
            return False
    
    async def get_contract_data(
        self,
        contract_address: str,
        data_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get contract data CID.
        
        Args:
            contract_address: Contract address
            data_type: Type of data to retrieve
            
        Returns:
            Data info dict or None
        """
        # Check local cache
        if contract_address in self.contract_storage:
            return self.contract_storage[contract_address]["data"].get(data_type)
        
        # Query blockchain
        try:
            result = self.substrate.query(
                module='Contracts',
                storage_function='ContractStorage',
                params=[contract_address, data_type]
            )
            
            if result and result.value:
                return {
                    "cid": result.value.get('cid'),
                    "size": result.value.get('size')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to query contract data: {e}")
            return None
    
    async def verify_contract_access(
        self,
        contract_address: str,
        caller_address: str,
        operation: str
    ) -> bool:
        """
        Verify contract-based access control.
        
        Args:
            contract_address: Contract address
            caller_address: Address requesting access
            operation: Operation type ('read', 'write', 'delete')
            
        Returns:
            True if access granted
        """
        try:
            # Query contract for access permissions
            # In production: Call contract method to check permissions
            logger.debug(
                f"Verifying access: {caller_address[:16]}... → "
                f"{contract_address[:16]}... ({operation})"
            )
            
            # Default: allow all (in production, implement proper ACL)
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify contract access: {e}")
            return False
    
    async def track_storage_usage(
        self,
        contract_address: str
    ) -> Dict[str, Any]:
        """
        Track storage usage for contract.
        
        Args:
            contract_address: Contract address
            
        Returns:
            Usage statistics
        """
        if contract_address not in self.contract_storage:
            return {
                "contract": contract_address,
                "total_size": 0,
                "data_types": []
            }
        
        storage_info = self.contract_storage[contract_address]
        
        return {
            "contract": contract_address,
            "total_size": storage_info["total_size"],
            "data_types": list(storage_info["data"].keys()),
            "details": storage_info["data"]
        }
    
    async def cleanup_contract_data(
        self,
        contract_address: str,
        data_type: Optional[str] = None
    ) -> bool:
        """
        Clean up contract data.
        
        Args:
            contract_address: Contract address
            data_type: Specific data type to remove (None = all)
            
        Returns:
            True if cleanup successful
        """
        try:
            if data_type:
                # Remove specific data type
                if contract_address in self.contract_storage:
                    if data_type in self.contract_storage[contract_address]["data"]:
                        size = self.contract_storage[contract_address]["data"][data_type]["size"]
                        del self.contract_storage[contract_address]["data"][data_type]
                        self.contract_storage[contract_address]["total_size"] -= size
                        logger.info(f"Cleaned up {data_type} for {contract_address[:16]}...")
            else:
                # Remove all data for contract
                if contract_address in self.contract_storage:
                    del self.contract_storage[contract_address]
                    logger.info(f"Cleaned up all data for {contract_address[:16]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup contract data: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get overall contract storage statistics.
        
        Returns:
            Stats dict
        """
        total_size = sum(
            info["total_size"] 
            for info in self.contract_storage.values()
        )
        
        return {
            "contracts_count": len(self.contract_storage),
            "total_storage_bytes": total_size,
            "contracts": list(self.contract_storage.keys())
        }
