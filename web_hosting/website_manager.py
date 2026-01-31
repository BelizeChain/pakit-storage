"""
Website Manager for BNS-integrated hosting.

Manages domain-to-content mapping, DAG storage, and blockchain integration.
"""

import logging
import hashlib
import json
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import asyncio
from datetime import datetime, timedelta

from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.exceptions import SubstrateRequestException

logger = logging.getLogger(__name__)


class WebsiteManager:
    """
    Manages website hosting on Pakit with BNS integration.
    
    Responsibilities:
    - Upload website content to DAG storage
    - Register domain-to-content mappings on-chain
    - Verify hosting subscriptions
    - Track bandwidth usage
    - Handle content updates
    
    Integration:
    - BNS Pallet: Domain ownership, hosting activation
    - DAG Backend: Sovereign storage (no external dependencies)
    """
    
    def __init__(
        self,
        dag_backend,
        substrate_url: str = "ws://127.0.0.1:9944",
        keypair: Optional[Keypair] = None
    ):
        """
        Initialize WebsiteManager.
        
        Args:
            dag_backend: DagBackend instance for sovereign storage
            substrate_url: WebSocket URL to BelizeChain node
            keypair: Keypair for signing transactions (optional, for admin operations)
        """
        self.dag = dag_backend
        self.substrate_url = substrate_url
        self.keypair = keypair
        self.substrate = None
        
        # Domain mapping cache: domain -> content_hash
        self.domain_cache: Dict[str, str] = {}
        
        # Hosting stats: domain -> {bandwidth_used, last_updated}
        self.stats: Dict[str, Dict] = {}
        
        self._connect_substrate()
        
        logger.info("WebsiteManager initialized with DAG backend")
    
    def _connect_substrate(self):
        """Connect to BelizeChain substrate node."""
        try:
            self.substrate = SubstrateInterface(
                url=self.substrate_url,
                ss58_format=42,  # BelizeChain uses Substrate default
                type_registry_preset='substrate-node-template'
            )
            
            # Verify connection
            chain = self.substrate.get_chain()
            logger.info(f"Connected to {chain} at {self.substrate_url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to substrate: {e}")
            logger.warning("Blockchain integration disabled - operating in offline mode")
            self.substrate = None
    
    async def upload_website(
        self,
        domain: str,
        content_dir: Path,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Upload website content to DAG storage.
        
        Args:
            domain: Domain name (e.g., "example.bz")
            content_dir: Directory containing website files (must have index.html)
            metadata: Optional metadata (title, description, etc.)
        
        Returns:
            Root content hash (32-byte hex string)
        
        Raises:
            ValueError: If content_dir doesn't have index.html
            ConnectionError: If DAG upload fails
        """
        if not content_dir.exists() or not content_dir.is_dir():
            raise ValueError(f"Content directory not found: {content_dir}")
        
        index_html = content_dir / "index.html"
        if not index_html.exists():
            raise ValueError(f"index.html not found in {content_dir}")
        
        logger.info(f"Uploading website for domain: {domain}")
        
        # Upload to DAG storage
        try:
            root_hash = await self._upload_to_dag(content_dir, metadata)
            logger.info(f"Uploaded to DAG: {root_hash}")
        except Exception as e:
            logger.error(f"DAG upload failed: {e}")
            raise ConnectionError(f"DAG upload failed: {e}")
        
        # Update cache
        self.domain_cache[domain] = root_hash
        self.stats[domain] = {
            'bandwidth_used': 0,
            'last_updated': datetime.now(),
            'content_hash': root_hash,
            'file_count': len(list(content_dir.rglob('*')))
        }
        
        return root_hash
    
    async def _upload_to_dag(self, content_dir: Path, metadata: Optional[Dict] = None) -> str:
        """
        Upload website directory to DAG storage.
        
        Creates a manifest block that references individual file blocks.
        
        Returns:
            Root manifest block hash (32-byte hex)
        """
        # Step 1: Upload each file as a DAG block
        file_manifest = {}
        
        for file_path in content_dir.rglob('*'):
            if file_path.is_file():
                # Get relative path for the manifest
                rel_path = str(file_path.relative_to(content_dir))
                
                # Read file content
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                
                # Determine content type
                content_type = self._get_content_type(file_path.suffix)
                
                # Store file in DAG
                file_metadata = {
                    'filename': rel_path,
                    'content_type': content_type,
                    'size': len(file_data)
                }
                
                # Use storage engine's DAG backend
                block_hash = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda data=file_data, meta=file_metadata: self.dag.store(
                        data=data,
                        metadata=meta
                    )
                )
                
                file_manifest[rel_path] = {
                    'hash': block_hash,
                    'size': len(file_data),
                    'content_type': content_type
                }
                
                logger.debug(f"Uploaded {rel_path}: {block_hash}")
        
        # Step 2: Create manifest block (root)
        manifest_data = {
            'type': 'website_manifest',
            'version': '1.0',
            'domain_metadata': metadata or {},
            'files': file_manifest,
            'index': 'index.html',  # Default entry point
            'created_at': datetime.now().isoformat()
        }
        
        # Store manifest as DAG block
        manifest_json = json.dumps(manifest_data, indent=2).encode('utf-8')
        manifest_hash = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.dag.store(
                data=manifest_json,
                metadata={'type': 'manifest', 'file_count': len(file_manifest)}
            )
        )
        
        logger.info(f"Created manifest block: {manifest_hash} ({len(file_manifest)} files)")
        
        return manifest_hash
    
    def _get_content_type(self, suffix: str) -> str:
        """Determine MIME type from file extension."""
        mime_types = {
            '.html': 'text/html',
            '.css': 'text/css',
            '.js': 'application/javascript',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.ico': 'image/x-icon',
            '.woff': 'font/woff',
            '.woff2': 'font/woff2',
            '.ttf': 'font/ttf',
            '.txt': 'text/plain',
            '.xml': 'application/xml',
        }
        return mime_types.get(suffix.lower(), 'application/octet-stream')
    
    async def register_hosting_onchain(
        self,
        domain: str,
        content_hash: str
    ) -> bool:
        """
        Register hosting content on BelizeChain BNS pallet.
        
        Calls BNS::activate_hosting or BNS::update_hosting_content.
        
        Args:
            domain: Domain name
            content_hash: DAG root block hash (32-byte hex string)
        
        Returns:
            True if transaction succeeded
        """
        if not self.substrate or not self.keypair:
            logger.warning("No blockchain connection or keypair - skipping on-chain registration")
            return False
        
        try:
            # Check if hosting already active
            hosting_info = self.substrate.query(
                module='Bns',
                storage_function='HostingConfigs',
                params=[domain]
            )
            
            # Convert hex string to 32-byte array
            if content_hash.startswith('0x'):
                content_hash = content_hash[2:]
            
            # Ensure exactly 32 bytes (64 hex chars)
            if len(content_hash) != 64:
                raise ValueError(f"Invalid content_hash length: {len(content_hash)} (expected 64 hex chars)")
            
            content_hash_bytes = bytes.fromhex(content_hash)
            
            if hosting_info.value is None:
                # Activate new hosting
                call = self.substrate.compose_call(
                    call_module='Bns',
                    call_function='activate_hosting',
                    call_params={
                        'domain': domain,
                        'tier': 0,  # Basic tier (enum index)
                        'content_hash': list(content_hash_bytes)  # [u8; 32]
                    }
                )
                extrinsic_name = "activate_hosting"
            else:
                # Update existing hosting content
                call = self.substrate.compose_call(
                    call_module='Bns',
                    call_function='update_hosting_content',
                    call_params={
                        'domain': domain,
                        'new_content_hash': list(content_hash_bytes)  # [u8; 32]
                    }
                )
                extrinsic_name = "update_hosting_content"
            
            # Create and submit extrinsic
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            
            if receipt.is_success:
                logger.info(f"Successfully registered hosting for {domain} (tx: {extrinsic_name})")
                return True
            else:
                logger.error(f"On-chain registration failed: {receipt.error_message}")
                return False
                
        except SubstrateRequestException as e:
            logger.error(f"Substrate request failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during on-chain registration: {e}")
            return False
    
    async def get_domain_content(self, domain: str) -> Optional[str]:
        """
        Get content hash for a domain.
        
        Checks cache first, then queries blockchain.
        
        Args:
            domain: Domain name to lookup
        
        Returns:
            DAG root block hash (32-byte hex) or None if not found
        """
        # Check cache
        if domain in self.domain_cache:
            return self.domain_cache[domain]
        
        # Query blockchain
        if self.substrate:
            try:
                hosting_info = self.substrate.query(
                    module='Bns',
                    storage_function='HostingConfigs',
                    params=[domain]
                )
                
                if hosting_info.value:
                    # Extract content_hash (stored as bytes)
                    content_hash = hosting_info.value.get('content_hash')
                    if content_hash:
                        # Convert from hex to string
                        ipfs_cid = bytes.fromhex(content_hash[2:]).decode('utf-8')
                        self.domain_cache[domain] = ipfs_cid
                        return ipfs_cid
                        
            except Exception as e:
                logger.error(f"Failed to query domain {domain}: {e}")
        
        return None
    
    async def verify_hosting_subscription(self, domain: str) -> bool:
        """
        Verify domain has active hosting subscription on-chain.
        
        Args:
            domain: Domain name
        
        Returns:
            True if subscription is active
        """
        if not self.substrate:
            # Offline mode - can't verify
            return True
        
        try:
            hosting_info = self.substrate.query(
                module='Bns',
                storage_function='HostingConfigs',
                params=[domain]
            )
            
            if hosting_info.value is None:
                return False
            
            # Check if expires_at is in the future
            expires_at = hosting_info.value.get('expires_at', 0)
            current_block = self.substrate.query(
                module='System',
                storage_function='Number'
            ).value
            
            return expires_at > current_block
            
        except Exception as e:
            logger.error(f"Failed to verify subscription for {domain}: {e}")
            return False
    
    async def track_bandwidth(self, domain: str, bytes_served: int):
        """
        Track bandwidth usage for a domain.
        
        Args:
            domain: Domain name
            bytes_served: Number of bytes served in this request
        """
        if domain not in self.stats:
            self.stats[domain] = {
                'bandwidth_used': 0,
                'last_updated': datetime.now()
            }
        
        self.stats[domain]['bandwidth_used'] += bytes_served
        self.stats[domain]['last_updated'] = datetime.now()
        
        # Log high usage
        if self.stats[domain]['bandwidth_used'] > 1_000_000_000:  # 1 GB
            logger.info(
                f"High bandwidth usage for {domain}: "
                f"{self.stats[domain]['bandwidth_used'] / 1_000_000_000:.2f} GB"
            )
    
    def get_stats(self, domain: str) -> Optional[Dict]:
        """Get hosting statistics for a domain."""
        return self.stats.get(domain)
    
    def list_hosted_domains(self) -> List[str]:
        """List all domains in cache (recently accessed)."""
        return list(self.domain_cache.keys())
    
    async def update_website(
        self,
        domain: str,
        content_dir: Path,
        use_arweave: bool = False
    ) -> bool:
        """
        Update website content for existing domain.
        
        Args:
            domain: Domain name
            content_dir: New content directory
            use_arweave: Whether to backup to Arweave
        
        Returns:
            True if update succeeded
        """
        # Upload new content
        ipfs_cid, arweave_id = await self.upload_website(
            domain, content_dir, use_arweave
        )
        
        # Register on-chain
        success = await self.register_hosting_onchain(
            domain, ipfs_cid, arweave_id
        )
        
        if success:
            logger.info(f"Successfully updated website for {domain}")
        
        return success
    
    async def delete_website(self, domain: str) -> bool:
        """
        Remove website hosting for domain.
        
        Calls BNS::deactivate_hosting on-chain.
        Does NOT delete IPFS content (immutable).
        
        Args:
            domain: Domain name
        
        Returns:
            True if deactivation succeeded
        """
        if not self.substrate or not self.keypair:
            logger.warning("No blockchain connection - cannot deactivate hosting")
            return False
        
        try:
            call = self.substrate.compose_call(
                call_module='Bns',
                call_function='deactivate_hosting',
                call_params={'domain': domain}
            )
            
            extrinsic = self.substrate.create_signed_extrinsic(
                call=call,
                keypair=self.keypair
            )
            
            receipt = self.substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
            
            if receipt.is_success:
                logger.info(f"Deactivated hosting for {domain}")
                
                # Remove from cache
                self.domain_cache.pop(domain, None)
                self.stats.pop(domain, None)
                
                return True
            else:
                logger.error(f"Deactivation failed: {receipt.error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to deactivate hosting: {e}")
            return False
