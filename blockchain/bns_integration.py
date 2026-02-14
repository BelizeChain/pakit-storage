"""
BNS Pallet Integration for Pakit Storage

Integrates with BelizeChain's BNS (Belize Name Service) pallet to:
- Register .bz domain hosting
- Link domains to IPFS content
- Manage domain marketplace
- Provide domain-based content delivery
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class BNSPalletConnector:
    """
    Connector for BelizeChain BNS pallet.
    
    Enables Pakit to:
    - Host .bz domains on IPFS
    - Link domain names to content CIDs
    - Provide decentralized web hosting
    - Manage domain content updates
    """
    
    def __init__(self, substrate_interface: Any):
        """
        Initialize BNS pallet connector.
        
        Args:
            substrate_interface: Substrate interface from storage_proof_connector
        """
        self.substrate = substrate_interface
        self.hosted_domains: Dict[str, str] = {}  # domain -> CID
    
    async def register_domain_hosting(
        self,
        domain_name: str,
        ipfs_cid: str,
        hosting_account: str
    ) -> bool:
        """
        Register .bz domain hosting on Pakit IPFS.
        
        Args:
            domain_name: Domain name (e.g., 'example.bz')
            ipfs_cid: IPFS CID of the website content
            hosting_account: Blockchain account for hosting fees
            
        Returns:
            True if registration successful
        """
        try:
            # Validate domain
            if not domain_name.endswith('.bz'):
                logger.error(f"Invalid domain: {domain_name} (must end with .bz)")
                return False
            
            # Submit extrinsic to BNS pallet
            call = self.substrate.compose_call(
                call_module='BNS',
                call_function='set_domain_content',
                call_params={
                    'domain': domain_name,
                    'content_cid': ipfs_cid,
                    'hosting_type': 'ipfs'
                }
            )
            
            # Track locally
            self.hosted_domains[domain_name] = ipfs_cid
            
            logger.info(f"✅ Registered domain hosting: {domain_name} → {ipfs_cid}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register domain hosting: {e}")
            return False
    
    async def update_domain_content(
        self,
        domain_name: str,
        new_ipfs_cid: str
    ) -> bool:
        """
        Update content for hosted domain.
        
        Args:
            domain_name: Domain name
            new_ipfs_cid: New IPFS CID
            
        Returns:
            True if update successful
        """
        try:
            call = self.substrate.compose_call(
                call_module='BNS',
                call_function='update_domain_content',
                call_params={
                    'domain': domain_name,
                    'content_cid': new_ipfs_cid
                }
            )
            
            # Update local tracking
            old_cid = self.hosted_domains.get(domain_name)
            self.hosted_domains[domain_name] = new_ipfs_cid
            
            logger.info(
                f"Updated domain content: {domain_name} "
                f"{old_cid[:16] if old_cid else 'none'}... → {new_ipfs_cid[:16]}..."
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update domain content: {e}")
            return False
    
    async def get_domain_content(self, domain_name: str) -> Optional[str]:
        """
        Get IPFS CID for domain.
        
        Args:
            domain_name: Domain name
            
        Returns:
            IPFS CID or None
        """
        # Check local cache first
        if domain_name in self.hosted_domains:
            return self.hosted_domains[domain_name]
        
        # Query blockchain
        try:
            result = self.substrate.query(
                module='BNS',
                storage_function='DomainContent',
                params=[domain_name]
            )
            
            if result and result.value:
                cid = result.value.get('ipfs_cid')
                if cid:
                    self.hosted_domains[domain_name] = cid
                return cid
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to query domain content: {e}")
            return None
    
    async def list_hosted_domains(self) -> List[Dict[str, str]]:
        """
        List all domains hosted on this Pakit node.
        
        Returns:
            List of domain info dicts
        """
        domains = []
        for domain, cid in self.hosted_domains.items():
            domains.append({
                "domain": domain,
                "ipfs_cid": cid,
                "url": f"https://{domain}"
            })
        
        return domains
    
    async def unregister_domain(self, domain_name: str) -> bool:
        """
        Unregister domain hosting.
        
        Args:
            domain_name: Domain name to unregister
            
        Returns:
            True if successful
        """
        try:
            call = self.substrate.compose_call(
                call_module='BNS',
                call_function='remove_domain_content',
                call_params={'domain': domain_name}
            )
            
            # Remove from local tracking
            if domain_name in self.hosted_domains:
                del self.hosted_domains[domain_name]
            
            logger.info(f"Unregistered domain: {domain_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister domain: {e}")
            return False
    
    def get_hosting_stats(self) -> Dict[str, Any]:
        """
        Get domain hosting statistics.
        
        Returns:
            Stats dict
        """
        return {
            "domains_hosted": len(self.hosted_domains),
            "domains": list(self.hosted_domains.keys())
        }
