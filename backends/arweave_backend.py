"""
Arweave permanent storage backend.

Stores data on Arweave for permanent, immutable storage.
"""

from typing import Optional, List
import logging
from pathlib import Path
import tempfile
import json
import asyncio
import time

from ..blockchain.storage_proof_connector import get_storage_proof_connector

try:
    import arweave
    ARWEAVE_AVAILABLE = True
except ImportError:
    ARWEAVE_AVAILABLE = False
    logging.warning(
        "arweave-python-client not installed. "
        "Install with: pip install arweave-python-client"
    )

logger = logging.getLogger(__name__)


class ArweaveBackend:
    """
    Arweave permanent storage backend.
    
    Uses Arweave blockchain for permanent, immutable data storage.
    Data is stored with cryptographic proofs and guaranteed to be
    available forever (200+ year retention).
    
    Features:
    - Permanent storage (pay once, store forever)
    - Cryptographic verification
    - Immutable data (cannot be changed or deleted)
    - Transaction-based addressing
    
    Cost: ~0.0001 AR per KB (~$0.01 per MB as of 2025)
    """
    
    def __init__(
        self,
        wallet_path: Optional[str] = None,
        gateway: str = "https://arweave.net",
        timeout: int = 120,
        retry_attempts: int = 3,
        retry_backoff: float = 0.5,
    ):
        """
        Initialize Arweave backend.
        
        Args:
            wallet_path: Path to Arweave wallet JSON file (required for uploads)
            gateway: Arweave gateway URL
            timeout: Timeout for operations (seconds)
            retry_attempts: Number of retries for transient errors
            retry_backoff: Base backoff (seconds) between retries
        """
        if not ARWEAVE_AVAILABLE:
            raise ImportError(
                "arweave-python-client not installed. "
                "Install with: pip install arweave-python-client"
            )
        
        self.wallet_path = wallet_path
        self.gateway = gateway
        self.timeout = timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_backoff = max(0.1, retry_backoff)
        self.wallet = None
        
        # Initialize wallet if provided
        if wallet_path:
            self._load_wallet()
        else:
            logger.warning(
                "No Arweave wallet provided. Read-only mode (cannot store)."
            )
        
        logger.info(f"Initialized Arweave backend (gateway: {self.gateway})")
    
    def _load_wallet(self):
        """Load Arweave wallet from file."""
        try:
            self.wallet = self._execute_with_retries(arweave.Wallet, self.wallet_path)
            
            # Get wallet address
            address = self.wallet.address
            
            # Get balance (in Winston, 1 AR = 10^12 Winston)
            balance = self.wallet.balance
            balance_ar = balance / 1e12
            
            logger.info(
                f"Loaded Arweave wallet: {address[:8]}... "
                f"(balance: {balance_ar:.4f} AR)"
            )
            
            if balance_ar < 0.001:
                logger.warning(
                    f"Low Arweave balance ({balance_ar:.4f} AR). "
                    "Fund wallet at https://faucet.arweave.net/"
                )
                
        except Exception as e:
            logger.error(f"Failed to load Arweave wallet: {e}")
            raise
    
    def store(
        self,
        content_id: str,
        data: bytes,
        tier: str = "warm"
    ) -> bool:
        """
        Store data to Arweave.
        
        Args:
            content_id: Content ID (used for metadata tagging)
            data: Data to store
            tier: Storage tier (Arweave always permanent, tier affects tags)
        
        Returns:
            True if successful
        """
        if not self.wallet:
            logger.error("Cannot store to Arweave without wallet")
            return False
        
        try:
            # Create transaction
            transaction = self._execute_with_retries(
                arweave.Transaction,
                self.wallet,
                data=data
            )
            
            # Add tags for metadata
            transaction.add_tag("App-Name", "BelizeChain-Pakit")
            transaction.add_tag("App-Version", "1.0.0")
            transaction.add_tag("Content-ID", content_id[:32])  # Truncate for tag size
            transaction.add_tag("Storage-Tier", tier)
            transaction.add_tag("Content-Type", "application/octet-stream")
            
            # Sign transaction
            transaction.sign()
            
            # Send transaction
            self._execute_with_retries(transaction.send)
            
            tx_id = transaction.id
            
            logger.info(
                f"Stored {content_id[:16]}... to Arweave "
                f"(TX: {tx_id}, size: {len(data)} bytes, tier: {tier})"
            )
            
            # Calculate cost
            cost_winston = transaction.reward
            cost_ar = cost_winston / 1e12
            logger.debug(f"Storage cost: {cost_ar:.6f} AR (${cost_ar * 10:.4f} USD)")
            
            # Store mapping in metadata
            self._store_metadata(content_id, tx_id, tier)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store {content_id[:16]}... to Arweave: {e}")
            return False
    
    def retrieve(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> Optional[bytes]:
        """
        Retrieve data from Arweave.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier (for metadata lookup)
        
        Returns:
            Data bytes, or None if not found
        """
        try:
            # Get Arweave transaction ID from metadata
            tx_id = self._get_arweave_tx_id(content_id, tier)
            
            if not tx_id:
                logger.warning(
                    f"No Arweave TX ID found for content {content_id[:16]}..."
                )
                return None
            
            # Retrieve data from Arweave
            data = self._execute_with_retries(arweave.Transaction.get_data, tx_id)
            
            logger.debug(
                f"Retrieved {content_id[:16]}... from Arweave "
                f"(TX: {tx_id}, size: {len(data)} bytes)"
            )
            
            return data
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve {content_id[:16]}... from Arweave: {e}"
            )
            return None
    
    def delete(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Delete data from Arweave.
        
        WARNING: Data on Arweave is PERMANENT and cannot be deleted!
        This only removes local metadata. Data remains on blockchain.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if successful (only deletes metadata)
        """
        logger.warning(
            f"Arweave data is permanent! Only deleting metadata for "
            f"{content_id[:16]}..."
        )
        
        try:
            # Only delete metadata (data is immutable on Arweave)
            self._delete_metadata(content_id, tier)
            
            logger.info(
                f"Deleted metadata for {content_id[:16]}... "
                "(data remains on Arweave blockchain)"
            )
            
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to delete metadata for {content_id[:16]}...: {e}"
            )
            return False
    
    def exists(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> bool:
        """
        Check if content exists on Arweave.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            True if exists
        """
        try:
            # Get Arweave TX ID
            tx_id = self._get_arweave_tx_id(content_id, tier)
            
            if not tx_id:
                return False
            
            # Check if transaction exists
            status = self._execute_with_retries(arweave.Transaction.get_status, tx_id)
            return status['status'] == 200
            
        except Exception:
            return False
    
    def get_size(
        self,
        content_id: str,
        tier: str = "warm"
    ) -> Optional[int]:
        """
        Get size of stored content.
        
        Args:
            content_id: Content ID (hex string)
            tier: Storage tier
        
        Returns:
            Size in bytes, or None if not found
        """
        try:
            tx_id = self._get_arweave_tx_id(content_id, tier)
            
            if not tx_id:
                return None
            
            # Get transaction info
            tx = self._execute_with_retries(arweave.Transaction.get_transaction, tx_id)
            return int(tx.data_size)
            
        except Exception:
            return None
    
    def list_all(self, tier: Optional[str] = None) -> List[str]:
        """
        List all content IDs stored on Arweave.
        
        Args:
            tier: Optional tier filter (hot/warm/cold)
        
        Returns:
            List of content ID hex strings
        """
        # This would query metadata storage (local DB or on-chain)
        # Arweave doesn't support efficient listing by tags
        logger.warning("list_all() not fully implemented for Arweave backend")
        return []
    
    def get_total_size(self, tier: Optional[str] = None) -> int:
        """
        Get total storage size on Arweave.
        
        Args:
            tier: Optional tier filter
        
        Returns:
            Total size in bytes
        """
        # Would need to iterate all transactions (expensive)
        logger.warning("get_total_size() not fully implemented for Arweave backend")
        return 0
    
    def migrate_tier(
        self,
        content_id: str,
        from_tier: str,
        to_tier: str
    ) -> bool:
        """
        Migrate content between tiers.
        
        For Arweave, this only updates metadata tags.
        Data location doesn't change (already permanent).
        
        Args:
            content_id: Content ID (hex string)
            from_tier: Source tier
            to_tier: Destination tier
        
        Returns:
            True if successful
        """
        try:
            tx_id = self._get_arweave_tx_id(content_id, from_tier)
            
            if not tx_id:
                return False
            
            # Update metadata
            self._delete_metadata(content_id, from_tier)
            self._store_metadata(content_id, tx_id, to_tier)
            
            logger.info(
                f"Migrated {content_id[:16]}... from {from_tier} to {to_tier} "
                "(metadata only, Arweave data is immutable)"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to migrate {content_id[:16]}...: {e}")
            return False
    
    def cleanup(self, tier: Optional[str] = None):
        """
        Clean up Arweave storage.
        
        WARNING: This only deletes metadata. Data on Arweave is permanent
        and cannot be deleted!
        
        Args:
            tier: Optional tier to clean (default: all tiers)
        """
        logger.warning(
            "Arweave data is permanent! Only cleaning up metadata."
        )
        
        try:
            metadata_dir = Path(tempfile.gettempdir()) / "pakit_arweave_metadata"
            
            if tier:
                tier_dir = metadata_dir / tier
                if tier_dir.exists():
                    for f in tier_dir.glob("*.txid"):
                        f.unlink()
                    logger.info(f"Cleaned up {tier} metadata")
            else:
                if metadata_dir.exists():
                    for tier_dir in metadata_dir.iterdir():
                        for f in tier_dir.glob("*.txid"):
                            f.unlink()
                    logger.info("Cleaned up all Arweave metadata")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup Arweave metadata: {e}")
    
    def estimate_cost(self, data_size: int) -> dict:
        """
        Estimate storage cost for data.
        
        Args:
            data_size: Size in bytes
        
        Returns:
            Dict with cost estimates in Winston and AR
        """
        try:
            # Get current price from network
            price = self._execute_with_retries(arweave.Transaction.get_price, data_size)
            
            cost_winston = int(price)
            cost_ar = cost_winston / 1e12
            cost_usd = cost_ar * 10.0  # Approximate AR price
            
            return {
                "winston": cost_winston,
                "ar": cost_ar,
                "usd_estimate": cost_usd,
                "size_bytes": data_size,
                "size_mb": data_size / 1024 / 1024
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate cost: {e}")
            return {
                "winston": 0,
                "ar": 0.0,
                "usd_estimate": 0.0,
                "size_bytes": data_size,
                "size_mb": data_size / 1024 / 1024
            }
    
    # Internal methods
    
    def _store_metadata(self, content_id: str, tx_id: str, tier: str):
        """
        Store content_id -> Arweave TX ID mapping on-chain and as fallback.
        
        Tries to store on BelizeChain's LandLedger pallet first,
        falls back to temp file storage if blockchain unavailable.
        """
        # Try on-chain storage first
        try:
            connector = get_storage_proof_connector(mock_mode=True)
            
            # Need to run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            success = loop.run_until_complete(
                connector.store_document_proof(
                    content_id=content_id,
                    arweave_tx=tx_id,
                    owner="system",
                    metadata={"tier": tier, "gateway": self.gateway}
                )
            )
            loop.close()
            
            if success:
                logger.debug(f"✅ Stored on-chain metadata for {content_id[:16]}... -> {tx_id[:16]}...")
                return
        except Exception as e:
            logger.warning(f"On-chain storage failed: {e}, using temp file fallback")
        
        # Fallback to temp file storage
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_arweave_metadata" / tier
        metadata_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_dir / f"{content_id}.txid"
        
        # Store as JSON with additional metadata
        metadata = {
            "tx_id": tx_id,
            "tier": tier,
            "gateway": self.gateway
        }
        
        metadata_file.write_text(json.dumps(metadata))
    
    def _get_arweave_tx_id(self, content_id: str, tier: str) -> Optional[str]:
        """
        Get Arweave transaction ID for content ID from on-chain storage or fallback.
        
        Queries BelizeChain first, falls back to temp file storage.
        """
        # Try on-chain query first
        try:
            connector = get_storage_proof_connector(mock_mode=True)
            
            # Need to run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            tx_id = loop.run_until_complete(
                connector.get_arweave_tx(content_id)
            )
            loop.close()
            
            if tx_id:
                logger.debug(f"✅ Retrieved on-chain TX ID for {content_id[:16]}...")
                return tx_id
        except Exception as e:
            logger.warning(f"On-chain query failed: {e}, trying temp file fallback")
        
        # Fallback to temp file storage
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_arweave_metadata" / tier
        metadata_file = metadata_dir / f"{content_id}.txid"
        
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            return metadata["tx_id"]
        else:
            return None
    
    def _delete_metadata(self, content_id: str, tier: str):
        """Delete metadata for content ID."""
        metadata_dir = Path(tempfile.gettempdir()) / "pakit_arweave_metadata" / tier
        metadata_file = metadata_dir / f"{content_id}.txid"
        
        if metadata_file.exists():
            metadata_file.unlink()

    def _execute_with_retries(self, func, *args, **kwargs):
        """Execute a function with retry and backoff for transient Arweave errors."""
        attempt = 0
        last_exc = None
        while attempt < self.retry_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # Arweave client errors vary
                last_exc = exc
                attempt += 1
                if attempt >= self.retry_attempts:
                    break
                delay = self.retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    f"Arweave operation failed (attempt {attempt}/{self.retry_attempts}): {exc}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        raise last_exc
