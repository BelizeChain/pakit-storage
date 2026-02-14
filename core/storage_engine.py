"""
Main Pakit storage engine orchestrator.

Coordinates compression, deduplication, content addressing, and backend storage.
This is the primary API for storing and retrieving data with maximum efficiency.
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .compression import CompressionEngine, CompressionAlgorithm, CompressionResult
from .content_addressing import ContentAddressingEngine, ContentID
from .deduplication import DeduplicationEngine
from .dag_storage import DagBlock, CompressionAlgorithm as DagCompressionAlgorithm, create_genesis_block
from .dag_builder import DagBuilder, DagState
from .dag_index import DagIndex
from backends import LocalBackend, IPFSBackend, ArweaveBackend

# Import ZK proof generator (optional)
try:
    from core.zk_storage_proofs import StorageProofGenerator
    ZK_PROOFS_AVAILABLE = True
except ImportError:
    ZK_PROOFS_AVAILABLE = False

logger = logging.getLogger(__name__)


class StorageTier(Enum):
    """Storage tier classification."""
    HOT = "hot"      # RAM + Local SSD (ultra-fast, high cost)
    WARM = "warm"    # Local HDD + DAG (fast, medium cost)
    COLD = "cold"    # DAG permanent storage (sovereign, low cost)
    AUTO = "auto"    # Automatic tier selection


class StorageBackend(Enum):
    """Available storage backends."""
    DAG = "dag"          # Sovereign DAG network (PRIMARY)
    LOCAL = "local"      # Local filesystem
    IPFS = "ipfs"        # IPFS (LEGACY - migration fallback)
    ARWEAVE = "arweave"  # Arweave (LEGACY - migration fallback)


@dataclass
class StorageMetadata:
    """Metadata for stored content."""
    content_id: ContentID
    original_size: int
    compressed_size: int
    compression_algorithm: CompressionAlgorithm
    compression_ratio: float
    storage_tier: StorageTier
    backend: str  # "local", "ipfs", "arweave"
    timestamp: float
    reference_count: int


@dataclass
class StorageStats:
    """Overall storage statistics."""
    total_items: int
    total_original_bytes: int
    total_compressed_bytes: int
    total_deduplication_saves: int
    bytes_saved_compression: int
    bytes_saved_deduplication: int
    
    @property
    def total_bytes_saved(self) -> int:
        """Total bytes saved (compression + dedup)."""
        return self.bytes_saved_compression + self.bytes_saved_deduplication
    
    @property
    def overall_efficiency(self) -> float:
        """Overall storage efficiency ratio."""
        if self.total_original_bytes == 0:
            return 0.0
        return 1.0 - (self.total_compressed_bytes / self.total_original_bytes)
    
    @property
    def efficiency_percent(self) -> float:
        """Overall efficiency as percentage."""
        return self.overall_efficiency * 100


class PakitStorageEngine:
    """
    Main Pakit storage engine.
    
    This orchestrates all storage operations:
    - Content addressing (hash-based IDs)
    - Deduplication (never store duplicates)
    - Compression (multi-algorithm with auto-selection)
    - Backend management (local, IPFS, Arweave)
    - Tier management (hot/warm/cold)
    
    The goal: 1000x storage efficiency compared to traditional data farms.
    """
    
    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.AUTO,
        enable_deduplication: bool = True,
        enable_blockchain_proofs: bool = True,
        enable_dag: bool = True,
        enable_ipfs: bool = False,
        ipfs_addr: str = "/ip4/127.0.0.1/tcp/5001",
        enable_arweave: bool = False,
        arweave_wallet_path: Optional[str] = None,
        migration_mode: bool = False,
    ):
        """
        Initialize Pakit storage engine.
        
        Args:
            storage_dir: Base directory for local storage (default: ./pakit_storage)
            compression_algorithm: Default compression algorithm
            enable_deduplication: Enable content deduplication
            enable_blockchain_proofs: Enable blockchain proof recording
            enable_dag: Enable DAG sovereign storage (PRIMARY - default True)
            enable_ipfs: Enable IPFS distributed storage (LEGACY - migration only)
            ipfs_addr: IPFS daemon API address
            enable_arweave: Enable Arweave permanent storage (LEGACY - migration only)
            arweave_wallet_path: Path to Arweave wallet JSON file
            migration_mode: Enable hybrid DAG+IPFS/Arweave (for migration period)
        """
        # Storage directory
        self.storage_dir = storage_dir or Path("./pakit_storage")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Core engines
        self.compression = CompressionEngine(default_algorithm=compression_algorithm)
        self.content_addressing = ContentAddressingEngine()
        self.deduplication = DeduplicationEngine() if enable_deduplication else None
        
        # DAG storage (PRIMARY - Sovereign Pakit network)
        self.dag_backend = None
        self.dag_state = None
        self.dag_builder = None
        self.dag_index = None
        
        if enable_dag:
            try:
                # Initialize DAG state and index
                self.dag_state = DagState(
                    blocks={},
                    blocks_by_depth={},
                    reference_counts={}
                )
                self.dag_index = DagIndex()
                
                # Create genesis block if this is a new DAG
                genesis = create_genesis_block()
                self.dag_state.add_block(genesis)
                self.dag_index.add_block(genesis)
                
                # Initialize DAG builder
                self.dag_builder = DagBuilder(
                    dag_state=self.dag_state,
                    compression_engine=None  # We handle compression separately
                )
                
                logger.info("DAG backend initialized (sovereign storage)")
            except Exception as e:
                logger.error(f"Failed to initialize DAG backend: {e}")
                enable_dag = False
        
        # Storage backends
        self.local_backend = LocalBackend(self.storage_dir)
        
        # LEGACY backends (migration fallback only)
        self.ipfs_backend = None
        if enable_ipfs:
            try:
                self.ipfs_backend = IPFSBackend(ipfs_addr=ipfs_addr)
                logger.warning("IPFS backend enabled (LEGACY - migration mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize IPFS backend: {e}")
        
        self.arweave_backend = None
        if enable_arweave:
            try:
                self.arweave_backend = ArweaveBackend(wallet_path=arweave_wallet_path)
                logger.warning("Arweave backend enabled (LEGACY - migration mode)")
            except Exception as e:
                logger.warning(f"Failed to initialize Arweave backend: {e}")
        
        # Configuration
        self.enable_dag = enable_dag
        self.enable_deduplication = enable_deduplication
        self.enable_blockchain_proofs = enable_blockchain_proofs
        self.migration_mode = migration_mode
        
        # ZK proof generator (for privacy-preserving storage verification)
        self.zk_proof_generator = None
        if ZK_PROOFS_AVAILABLE and enable_blockchain_proofs:
            try:
                self.zk_proof_generator = StorageProofGenerator()
                logger.info("ZK storage proof generator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ZK proof generator: {e}")
        
        # Metadata storage
        self.metadata: Dict[str, StorageMetadata] = {}
        
        # Statistics
        self.stats = StorageStats(
            total_items=0,
            total_original_bytes=0,
            total_compressed_bytes=0,
            total_deduplication_saves=0,
            bytes_saved_compression=0,
            bytes_saved_deduplication=0,
        )
        
        # Determine primary backend
        if enable_dag:
            primary_backend = "DAG (sovereign)"
        elif enable_ipfs:
            primary_backend = "IPFS (legacy)"
        elif enable_arweave:
            primary_backend = "Arweave (legacy)"
        else:
            primary_backend = "Local only"
        
        logger.info(
            f"Initialized Pakit storage engine "
            f"(primary={primary_backend}, dir={self.storage_dir}, "
            f"compression={compression_algorithm.value}, dedup={enable_deduplication}, "
            f"migration_mode={migration_mode})"
        )
    
    def store(
        self,
        data: bytes,
        tier: StorageTier = StorageTier.AUTO,
        compression: Optional[CompressionAlgorithm] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> ContentID:
        """
        Store data with maximum efficiency.
        
        Process:
        1. Compute content ID (hash-based addressing)
        2. Check for duplicates (deduplication)
        3. Compress data (multi-algorithm selection)
        4. Select storage tier (hot/warm/cold)
        5. Store to backend(s)
        6. Record blockchain proof (optional)
        7. Update statistics
        
        Args:
            data: Raw data to store
            tier: Storage tier (hot/warm/cold/auto)
            compression: Compression algorithm (None = use default)
            tags: Optional metadata tags
        
        Returns:
            ContentID for retrieval
        """
        original_size = len(data)
        
        # Step 1: Compute content ID
        content_id = self.content_addressing.compute_content_id(data)
        cid_hex = content_id.hex
        
        logger.info(f"Storing content {cid_hex[:16]}... ({original_size} bytes)")
        
        # Step 2: Check for duplicates
        if self.enable_deduplication and self.deduplication:
            is_unique = self.deduplication.add_reference(content_id, original_size)
            
            if not is_unique:
                # Duplicate detected - no need to store again!
                self.stats.total_deduplication_saves += 1
                self.stats.bytes_saved_deduplication += original_size
                
                logger.info(
                    f"Duplicate detected for {cid_hex[:16]}... "
                    f"(saved {original_size} bytes, "
                    f"refs: {self.deduplication.get_reference_count(content_id)})"
                )
                
                return content_id
        
        # Step 3: Compress data
        compression_algo = compression or self.compression.default_algorithm
        compression_result = self.compression.compress(data, compression_algo)
        
        compressed_size = compression_result.compressed_size
        compression_ratio = compression_result.compression_ratio
        
        self.stats.bytes_saved_compression += (original_size - compressed_size)
        
        logger.info(
            f"Compressed with {compression_result.algorithm.value}: "
            f"{original_size} -> {compressed_size} bytes "
            f"({compression_ratio:.2%} ratio, "
            f"score={compression_result.efficiency_score:.3f})"
        )
        
        # Step 4: Select storage tier
        selected_tier = self._select_tier(tier, original_size, tags)
        
        # Step 5: Store to backend
        backend = self._store_to_backend(
            content_id,
            compression_result.compressed_data,
            selected_tier
        )
        
        # Step 6: Record metadata
        metadata = StorageMetadata(
            content_id=content_id,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_algorithm=compression_result.algorithm,
            compression_ratio=compression_ratio,
            storage_tier=selected_tier,
            backend=backend,
            timestamp=self._get_timestamp(),
            reference_count=1,
        )
        
        self.metadata[cid_hex] = metadata
        
        # Step 7: Update statistics
        self.stats.total_items += 1
        self.stats.total_original_bytes += original_size
        self.stats.total_compressed_bytes += compressed_size
        
        # Step 8: Generate ZK storage proof (privacy-preserving verification)
        if self.zk_proof_generator and self.enable_blockchain_proofs:
            try:
                # Generate ZK proof for storage verification
                zk_proof = self.zk_proof_generator.generate_storage_proof(
                    content_id=cid_hex,
                    data_size=original_size,
                    storage_location=backend
                )
                
                logger.debug(
                    f"Generated ZK storage proof for {cid_hex[:16]}... "
                    f"(type: {zk_proof.get('type', 'unknown')})"
                )
                
                # Proof can be submitted to blockchain via StorageProofConnector
                # This is handled by the blockchain integration layer
                
            except Exception as e:
                logger.warning(f"Failed to generate ZK proof: {e}")
        
        # Note: Blockchain proofs are now handled directly by backends
        # via StorageProofConnector in ipfs_backend.py and arweave_backend.py
        
        logger.info(
            f"Stored {cid_hex[:16]}... to {backend} ({selected_tier.value} tier)"
        )
        
        return content_id
    
    def retrieve(self, content_id: ContentID) -> Optional[bytes]:
        """
        Retrieve data by content ID.
        
        Process:
        1. Lookup metadata
        2. Retrieve compressed data from backend
        3. Verify content integrity
        4. Decompress data
        5. Update access statistics
        
        Args:
            content_id: Content ID to retrieve
        
        Returns:
            Original uncompressed data, or None if not found
        """
        cid_hex = content_id.hex
        
        logger.info(f"Retrieving content {cid_hex[:16]}...")
        
        # Step 1: Lookup metadata
        if cid_hex not in self.metadata:
            logger.warning(f"Content {cid_hex[:16]}... not found")
            return None
        
        metadata = self.metadata[cid_hex]
        
        # Step 2: Retrieve from backend
        compressed_data = self._retrieve_from_backend(content_id, metadata.backend)
        
        if compressed_data is None:
            logger.error(f"Failed to retrieve {cid_hex[:16]}... from {metadata.backend}")
            return None
        
        # Step 3: Decompress
        original_data = self.compression.decompress(
            compressed_data,
            metadata.compression_algorithm
        )
        
        # Step 4: Verify integrity (verify ORIGINAL data, not compressed)
        if not self.content_addressing.verify_content(original_data, content_id):
            logger.error(f"Content verification failed for {cid_hex[:16]}...")
            return None
        
        logger.info(
            f"Retrieved {cid_hex[:16]}... from {metadata.backend} "
            f"({len(original_data)} bytes)"
        )
        
        return original_data
    
    def delete(self, content_id: ContentID) -> bool:
        """
        Delete content (with reference counting).
        
        Args:
            content_id: Content ID to delete
        
        Returns:
            True if content was deleted, False if still referenced
        """
        cid_hex = content_id.hex
        
        logger.info(f"Deleting content {cid_hex[:16]}...")
        
        if cid_hex not in self.metadata:
            logger.warning(f"Content {cid_hex[:16]}... not found")
            return False
        
        # Check reference count
        if self.enable_deduplication and self.deduplication:
            can_delete = self.deduplication.remove_reference(content_id)
            
            if not can_delete:
                logger.info(
                    f"Content {cid_hex[:16]}... still has references "
                    f"({self.deduplication.get_reference_count(content_id)} remaining)"
                )
                return False
        
        # Safe to delete
        metadata = self.metadata[cid_hex]
        
        self._delete_from_backend(content_id, metadata.backend)
        
        del self.metadata[cid_hex]
        
        self.stats.total_items -= 1
        self.stats.total_original_bytes -= metadata.original_size
        self.stats.total_compressed_bytes -= metadata.compressed_size
        
        logger.info(f"Deleted {cid_hex[:16]}... from {metadata.backend}")
        
        return True
    
    def get_metadata(self, content_id: ContentID) -> Optional[StorageMetadata]:
        """Get metadata for content."""
        return self.metadata.get(content_id.hex)
    
    def get_stats(self) -> StorageStats:
        """Get overall storage statistics."""
        return StorageStats(
            total_items=self.stats.total_items,
            total_original_bytes=self.stats.total_original_bytes,
            total_compressed_bytes=self.stats.total_compressed_bytes,
            total_deduplication_saves=self.stats.total_deduplication_saves,
            bytes_saved_compression=self.stats.bytes_saved_compression,
            bytes_saved_deduplication=self.stats.bytes_saved_deduplication,
        )
    
    def get_efficiency_report(self) -> Dict[str, Any]:
        """
        Get comprehensive efficiency report including DAG metrics.
        
        Returns:
            Dictionary with all efficiency metrics
        """
        stats = self.get_stats()
        
        # Get compression stats (should be fast)
        try:
            compression_stats = self.compression.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get compression stats: {e}")
            compression_stats = {}
        
        # Get deduplication stats (should be fast)
        try:
            dedup_stats = (
                self.deduplication.estimate_space_saved()
                if self.deduplication else {}
            )
        except Exception as e:
            logger.warning(f"Failed to get dedup stats: {e}")
            dedup_stats = {}
        
        # DAG statistics (check if enabled first)
        dag_stats = {}
        if self.enable_dag and self.dag_index:
            try:
                dag_stats = self.dag_index.get_statistics()
            except Exception as e:
                logger.warning(f"Failed to get DAG stats: {e}")
                dag_stats = {"error": str(e)}
        
        return {
            "storage": {
                "total_items": stats.total_items,
                "original_bytes": stats.total_original_bytes,
                "compressed_bytes": stats.total_compressed_bytes,
                "dedup_saves": stats.total_deduplication_saves,
            },
            "savings": {
                "compression_bytes": stats.bytes_saved_compression,
                "deduplication_bytes": stats.bytes_saved_deduplication,
                "total_bytes": stats.total_bytes_saved,
            },
            "efficiency": {
                "overall_percent": stats.efficiency_percent,
                "compression_ratio": (
                    stats.total_compressed_bytes / stats.total_original_bytes
                    if stats.total_original_bytes > 0 else 0.0
                ),
            },
            "compression_stats": compression_stats,
            "deduplication_stats": dedup_stats,
            "dag_stats": dag_stats,
            "backends": {
                "primary": "dag" if self.enable_dag else "local",
                "dag_enabled": self.enable_dag,
                "ipfs_enabled": self.ipfs_backend is not None,
                "arweave_enabled": self.arweave_backend is not None,
                "migration_mode": self.migration_mode,
            },
            "data_farm_comparison": {
                "traditional_storage_bytes": stats.total_original_bytes,
                "pakit_storage_bytes": stats.total_compressed_bytes,
                "efficiency_multiplier": (
                    stats.total_original_bytes / stats.total_compressed_bytes
                    if stats.total_compressed_bytes > 0 else 0.0
                ),
                "data_farm_eliminated": stats.efficiency_percent > 50.0,
            }
        }
    
    # Internal methods
    
    def _select_tier(
        self,
        tier: StorageTier,
        size: int,
        tags: Optional[Dict[str, str]]
    ) -> StorageTier:
        """Select storage tier based on size and access patterns."""
        if tier != StorageTier.AUTO:
            return tier
        
        # Auto-selection logic
        if size < 1024 * 1024:  # < 1MB
            return StorageTier.HOT
        elif size < 100 * 1024 * 1024:  # < 100MB
            return StorageTier.WARM
        else:
            return StorageTier.COLD
    
    def _store_to_backend(
        self,
        content_id: ContentID,
        compressed_data: bytes,
        tier: StorageTier
    ) -> str:
        """
        Store compressed data to appropriate backend(s).
        
        NEW Backend selection (Phase 1+):
        - HOT: DAG + Local cache (sovereign, ultra-fast)
        - WARM: DAG (sovereign, distributed)
        - COLD: DAG permanent (sovereign, low cost)
        
        MIGRATION Mode (hybrid fallback):
        - Try DAG first (primary)
        - Fallback to IPFS/Arweave (legacy) if DAG fails
        - Always cache to local for fast retrieval
        
        Legacy fallback chain: DAG → IPFS → Arweave → Local
        """
        cid_hex = content_id.hex
        tier_str = tier.value
        backend = "local"
        
        # PRIMARY: DAG backend (sovereign Pakit network)
        if self.enable_dag and self.dag_builder and self.dag_index:
            try:
                # Create DAG block with compressed content
                dag_block = self.dag_builder.create_block(
                    content=compressed_data,
                    compression_algorithm=DagCompressionAlgorithm.NONE,  # Already compressed
                    metadata={
                        "content_id": cid_hex,
                        "tier": tier_str,
                        "timestamp": self._get_timestamp()
                    }
                )
                
                # Index block for fast retrieval
                self.dag_index.add_block(dag_block)
                
                # Cache to local for ultra-fast access
                self.local_backend.store(cid_hex, compressed_data, tier_str)
                
                backend = "dag"
                logger.debug(
                    f"Stored to DAG: {cid_hex[:16]}... "
                    f"(block={dag_block.block_hash[:8]}..., depth={dag_block.depth})"
                )
                
                return backend
                
            except Exception as e:
                logger.warning(f"DAG store failed: {e}, falling back to legacy backends")
                
                # In migration mode, try legacy backends
                if self.migration_mode:
                    logger.info("Migration mode: attempting legacy backend storage")
        
        # LEGACY: COLD tier - Prefer Arweave (permanent storage)
        if tier == StorageTier.COLD:
            if self.arweave_backend:
                try:
                    if self.arweave_backend.store(cid_hex, compressed_data, tier_str):
                        backend = "arweave"
                        logger.debug(f"Stored to Arweave (legacy): {cid_hex[:16]}...")
                        # Also store to local for faster retrieval
                        self.local_backend.store(cid_hex, compressed_data, tier_str)
                        return backend
                except Exception as e:
                    logger.warning(f"Arweave store failed: {e}, falling back to IPFS")
            
            # Fallback to IPFS for COLD tier
            if self.ipfs_backend:
                try:
                    if self.ipfs_backend.store(cid_hex, compressed_data, tier_str):
                        backend = "ipfs"
                        logger.debug(f"Stored to IPFS (legacy): {cid_hex[:16]}...")
                        self.local_backend.store(cid_hex, compressed_data, tier_str)
                        return backend
                except Exception as e:
                    logger.warning(f"IPFS store failed: {e}, falling back to local")
        
        # LEGACY: WARM tier - Prefer IPFS (distributed + reasonably fast)
        elif tier == StorageTier.WARM:
            if self.ipfs_backend:
                try:
                    if self.ipfs_backend.store(cid_hex, compressed_data, tier_str):
                        backend = "ipfs"
                        logger.debug(f"Stored to IPFS (legacy): {cid_hex[:16]}...")
                        # Also store to local for faster retrieval
                        self.local_backend.store(cid_hex, compressed_data, tier_str)
                        return backend
                except Exception as e:
                    logger.warning(f"IPFS store failed: {e}, falling back to local")
        
        # HOT tier or final fallback: Always use local
        if self.local_backend.store(cid_hex, compressed_data, tier_str):
            backend = "local"
            logger.debug(f"Stored to local: {cid_hex[:16]}...")
        
        return backend
    
    def _retrieve_from_backend(
        self,
        content_id: ContentID,
        backend: str
    ) -> Optional[bytes]:
        """
        Retrieve compressed data from backend with hybrid fallback.
        
        NEW Retrieval priority (Phase 1+):
        1. DAG sovereign network (if primary backend)
        2. Local cache (ultra-fast)
        3. Legacy IPFS/Arweave (migration fallback)
        
        MIGRATION Mode:
        - Check DAG first
        - Fallback to IPFS/Arweave if not found
        - Cache retrieved data to DAG for future requests
        """
        metadata = self.metadata.get(content_id.hex)
        if not metadata:
            return None
        
        cid_hex = content_id.hex
        tier_str = metadata.storage_tier.value
        
        # PRIMARY: Try DAG backend first
        if backend == "dag" and self.dag_index:
            try:
                # Query DAG index by content_id (stored in metadata)
                # We need to find the DAG block by scanning metadata
                for block_hash, block in self.dag_index.blocks_by_hash.items():
                    if block.metadata.get("content_id") == cid_hex:
                        logger.debug(
                            f"Retrieved from DAG: {cid_hex[:16]}... "
                            f"(block={block.block_hash[:8]}...)"
                        )
                        return block.content
                
                logger.warning(f"Content {cid_hex[:16]}... not found in DAG index")
                
                # In migration mode, fallback to legacy backends
                if self.migration_mode:
                    logger.info("Migration mode: attempting legacy backend retrieval")
                
            except Exception as e:
                logger.warning(f"DAG retrieve failed: {e}, trying legacy backends")
        
        # LEGACY: Try Arweave
        if backend == "arweave" and self.arweave_backend:
            try:
                data = self.arweave_backend.retrieve(cid_hex, tier_str)
                if data:
                    # In migration mode, cache to DAG
                    if self.migration_mode and self.dag_builder:
                        try:
                            dag_block = self.dag_builder.create_block(
                                content=data,
                                compression_algorithm=DagCompressionAlgorithm.NONE,
                                metadata={
                                    "content_id": cid_hex,
                                    "tier": tier_str,
                                    "migrated_from": "arweave"
                                }
                            )
                            self.dag_index.add_block(dag_block)
                            logger.info(f"Migrated {cid_hex[:16]}... from Arweave to DAG")
                        except Exception as e:
                            logger.warning(f"Migration cache failed: {e}")
                    
                    return data
            except Exception as e:
                logger.warning(f"Arweave retrieve failed: {e}")
        
        # LEGACY: Try IPFS
        elif backend == "ipfs" and self.ipfs_backend:
            try:
                data = self.ipfs_backend.retrieve(cid_hex, tier_str)
                if data:
                    # In migration mode, cache to DAG
                    if self.migration_mode and self.dag_builder:
                        try:
                            dag_block = self.dag_builder.create_block(
                                content=data,
                                compression_algorithm=DagCompressionAlgorithm.NONE,
                                metadata={
                                    "content_id": cid_hex,
                                    "tier": tier_str,
                                    "migrated_from": "ipfs"
                                }
                            )
                            self.dag_index.add_block(dag_block)
                            logger.info(f"Migrated {cid_hex[:16]}... from IPFS to DAG")
                        except Exception as e:
                            logger.warning(f"Migration cache failed: {e}")
                    
                    return data
            except Exception as e:
                logger.warning(f"IPFS retrieve failed: {e}")
        
        # Always try local cache/storage as fallback
        if self.local_backend:
            try:
                data = self.local_backend.retrieve(cid_hex, tier_str)
                if data:
                    return data
            except Exception as e:
                logger.warning(f"Local retrieve failed: {e}")
        
        # Last resort: try all backends (migration scenarios)
        if self.migration_mode:
            for fallback_backend in [self.ipfs_backend, self.arweave_backend]:
                if fallback_backend:
                    try:
                        data = fallback_backend.retrieve(cid_hex, tier_str)
                        if data:
                            logger.info(
                                f"Retrieved from fallback backend: {cid_hex[:16]}... "
                                f"(migration mode)"
                            )
                            return data
                    except Exception:
                        pass
        
        return None
    
    def _delete_from_backend(
        self,
        content_id: ContentID,
        backend: str
    ):
        """
        Delete data from backend(s).
        
        Note: Arweave data cannot be deleted (permanent storage).
        """
        metadata = self.metadata.get(content_id.hex)
        if not metadata:
            return
        
        cid_hex = content_id.hex
        tier_str = metadata.storage_tier.value
        
        # Delete from primary backend
        if backend == "arweave" and self.arweave_backend:
            try:
                self.arweave_backend.delete(cid_hex, tier_str)
                logger.debug(f"Deleted metadata for Arweave: {cid_hex[:16]}...")
            except Exception as e:
                logger.warning(f"Arweave delete failed: {e}")
        
        elif backend == "ipfs" and self.ipfs_backend:
            try:
                self.ipfs_backend.delete(cid_hex, tier_str)
                logger.debug(f"Unpinned from IPFS: {cid_hex[:16]}...")
            except Exception as e:
                logger.warning(f"IPFS delete failed: {e}")
        
        # Always delete from local cache/storage
        if self.local_backend:
            try:
                self.local_backend.delete(cid_hex, tier_str)
                logger.debug(f"Deleted from local: {cid_hex[:16]}...")
            except Exception as e:
                logger.warning(f"Local delete failed: {e}")
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


# Convenience alias
Pakit = PakitStorageEngine
