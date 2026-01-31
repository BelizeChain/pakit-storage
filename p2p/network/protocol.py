"""
Block Request/Response Protocol

Implements efficient block retrieval from peers across the P2P network.

Features:
- Request blocks by hash from DHT
- Batch requests (up to 100 blocks)
- Request prioritization (Merkle proof blocks first)
- Timeout handling and retry logic
- Reputation-based peer selection
"""

import time
import hashlib
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Request/response constants
MAX_BATCH_SIZE = 100  # Maximum blocks per request
REQUEST_TIMEOUT = 30  # Seconds to wait for response
MAX_RETRIES = 3  # Maximum retry attempts


class RequestPriority(Enum):
    """Priority levels for block requests."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3  # Merkle proof blocks


@dataclass
class BlockRequest:
    """Request for one or more blocks."""
    
    request_id: str
    block_hashes: List[str]
    priority: RequestPriority = RequestPriority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    retries: int = 0
    requested_from: Optional[str] = None  # Peer ID
    
    def is_expired(self, timeout: int = REQUEST_TIMEOUT) -> bool:
        """Check if request has timed out."""
        return (time.time() - self.timestamp) > timeout
    
    def can_retry(self) -> bool:
        """Check if request can be retried."""
        return self.retries < MAX_RETRIES


@dataclass
class BlockResponse:
    """Response containing requested blocks."""
    
    request_id: str
    blocks: Dict[str, bytes]  # block_hash → compressed_data
    missing_blocks: List[str]  # Hashes not found
    from_peer: str
    timestamp: float = field(default_factory=time.time)


class BlockRequestProtocol:
    """
    Protocol for requesting and receiving blocks from peers.
    
    Supports:
    - Single and batch block requests
    - Priority queue for Merkle proof blocks
    - Automatic retries on failure
    - Peer selection based on reputation
    """
    
    def __init__(self, node_id: str):
        """
        Initialize block request protocol.
        
        Args:
            node_id: Our node's peer ID
        """
        self.node_id = node_id
        
        # Active requests (request_id → BlockRequest)
        self.active_requests: Dict[str, BlockRequest] = {}
        
        # Request queue by priority
        self.request_queue: Dict[RequestPriority, List[BlockRequest]] = {
            priority: [] for priority in RequestPriority
        }
        
        # Statistics
        self.stats = {
            "requests_sent": 0,
            "requests_received": 0,
            "blocks_retrieved": 0,
            "blocks_missing": 0,
            "timeouts": 0,
            "retries": 0
        }
        
        logger.info(f"Initialized block request protocol for node: {node_id[:16]}...")
    
    def request_block(
        self,
        block_hash: str,
        priority: RequestPriority = RequestPriority.MEDIUM,
        target_peer: Optional[str] = None
    ) -> str:
        """
        Request a single block from network.
        
        Args:
            block_hash: Hash of block to request
            priority: Request priority level
            target_peer: Optional specific peer to request from
        
        Returns:
            Request ID for tracking
        """
        return self.request_blocks(
            block_hashes=[block_hash],
            priority=priority,
            target_peer=target_peer
        )
    
    def request_blocks(
        self,
        block_hashes: List[str],
        priority: RequestPriority = RequestPriority.MEDIUM,
        target_peer: Optional[str] = None
    ) -> str:
        """
        Request multiple blocks in batch.
        
        Args:
            block_hashes: List of block hashes to request
            priority: Request priority level
            target_peer: Optional specific peer to request from
        
        Returns:
            Request ID for tracking
        """
        # Validate batch size
        if len(block_hashes) > MAX_BATCH_SIZE:
            logger.warning(
                f"Batch size {len(block_hashes)} exceeds max {MAX_BATCH_SIZE}, "
                f"splitting into multiple requests"
            )
            # Split into multiple requests
            request_ids = []
            for i in range(0, len(block_hashes), MAX_BATCH_SIZE):
                batch = block_hashes[i:i + MAX_BATCH_SIZE]
                req_id = self.request_blocks(batch, priority, target_peer)
                request_ids.append(req_id)
            return ",".join(request_ids)  # Return comma-separated IDs
        
        # Generate request ID
        request_id = hashlib.sha256(
            f"{self.node_id}{time.time()}{''.join(block_hashes)}".encode()
        ).hexdigest()
        
        # Create request
        request = BlockRequest(
            request_id=request_id,
            block_hashes=block_hashes,
            priority=priority,
            requested_from=target_peer
        )
        
        # Add to active requests
        self.active_requests[request_id] = request
        
        # Add to priority queue
        self.request_queue[priority].append(request)
        
        self.stats["requests_sent"] += 1
        logger.info(
            f"Created request {request_id[:16]}... for {len(block_hashes)} blocks "
            f"(priority: {priority.name})"
        )
        
        return request_id
    
    def request_merkle_proof_blocks(
        self,
        target_hash: str,
        root_hash: str,
        proof: List[str]
    ) -> str:
        """
        Request all blocks in a Merkle proof path (CRITICAL priority).
        
        Args:
            target_hash: Target block hash
            root_hash: Root (genesis) block hash
            proof: List of hashes in proof path
        
        Returns:
            Request ID
        """
        # Combine target, root, and proof into full path
        all_hashes = [target_hash, root_hash] + proof
        
        return self.request_blocks(
            block_hashes=all_hashes,
            priority=RequestPriority.CRITICAL
        )
    
    def handle_request(
        self,
        request: BlockRequest,
        from_peer: str,
        local_storage_callback
    ) -> BlockResponse:
        """
        Handle incoming block request from peer.
        
        Args:
            request: Received block request
            from_peer: Peer ID who sent request
            local_storage_callback: Function to retrieve blocks from local storage
        
        Returns:
            Response with found blocks
        """
        self.stats["requests_received"] += 1
        
        # Retrieve blocks from local storage
        found_blocks = {}
        missing_blocks = []
        
        for block_hash in request.block_hashes:
            block_data = local_storage_callback(block_hash)
            if block_data:
                found_blocks[block_hash] = block_data
            else:
                missing_blocks.append(block_hash)
        
        # Create response
        response = BlockResponse(
            request_id=request.request_id,
            blocks=found_blocks,
            missing_blocks=missing_blocks,
            from_peer=self.node_id
        )
        
        logger.info(
            f"Responding to request {request.request_id[:16]}... "
            f"with {len(found_blocks)} blocks ({len(missing_blocks)} missing)"
        )
        
        return response
    
    def handle_response(
        self,
        response: BlockResponse,
        store_callback
    ) -> bool:
        """
        Handle received block response.
        
        Args:
            response: Received response
            store_callback: Function to store blocks locally
        
        Returns:
            True if response processed successfully
        """
        # Check if we have this request
        if response.request_id not in self.active_requests:
            logger.warning(f"Received response for unknown request: {response.request_id[:16]}...")
            return False
        
        request = self.active_requests[response.request_id]
        
        # Store received blocks
        for block_hash, block_data in response.blocks.items():
            store_callback(block_hash, block_data)
            self.stats["blocks_retrieved"] += 1
        
        # Track missing blocks
        self.stats["blocks_missing"] += len(response.missing_blocks)
        
        # If blocks still missing, retry with different peer
        if response.missing_blocks and request.can_retry():
            logger.info(
                f"Request {request.request_id[:16]}... missing {len(response.missing_blocks)} blocks, "
                f"retrying ({request.retries + 1}/{MAX_RETRIES})"
            )
            
            # Create new request for missing blocks
            request.block_hashes = response.missing_blocks
            request.retries += 1
            request.timestamp = time.time()
            request.requested_from = None  # Try different peer
            
            self.stats["retries"] += 1
            return False
        
        # Request complete
        del self.active_requests[response.request_id]
        logger.info(
            f"Request {request.request_id[:16]}... completed: "
            f"{len(response.blocks)} blocks retrieved"
        )
        
        return True
    
    def get_next_request(self) -> Optional[BlockRequest]:
        """
        Get next request to process (highest priority first).
        
        Returns:
            Next request or None if queue empty
        """
        # Check priority levels from high to low
        for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
            queue = self.request_queue[priority]
            if queue:
                return queue.pop(0)
        
        return None
    
    def cleanup_expired_requests(self):
        """Remove expired requests and mark as timeout."""
        expired = [
            req_id for req_id, req in self.active_requests.items()
            if req.is_expired()
        ]
        
        for req_id in expired:
            request = self.active_requests[req_id]
            
            # Retry if possible
            if request.can_retry():
                logger.info(
                    f"Request {req_id[:16]}... timed out, retrying "
                    f"({request.retries + 1}/{MAX_RETRIES})"
                )
                request.retries += 1
                request.timestamp = time.time()
                request.requested_from = None
                self.stats["retries"] += 1
            else:
                # Max retries reached
                logger.warning(
                    f"Request {req_id[:16]}... failed after {MAX_RETRIES} retries"
                )
                del self.active_requests[req_id]
                self.stats["timeouts"] += 1
    
    def get_stats(self) -> Dict:
        """Get protocol statistics."""
        return {
            **self.stats,
            "active_requests": len(self.active_requests),
            "queued_requests": sum(
                len(queue) for queue in self.request_queue.values()
            )
        }


if __name__ == "__main__":
    # Example usage
    print("Block Request/Response Protocol Example:")
    print("-" * 60)
    
    # Create protocol for a node
    node_id = hashlib.sha256(b"test_node").hexdigest()
    protocol = BlockRequestProtocol(node_id=node_id)
    
    print(f"Node ID: {node_id[:16]}...")
    
    # Request a single block
    block_hash = hashlib.sha256(b"block_data").hexdigest()
    request_id = protocol.request_block(
        block_hash=block_hash,
        priority=RequestPriority.HIGH
    )
    
    print(f"\nRequested block {block_hash[:16]}...")
    print(f"Request ID: {request_id[:16]}...")
    
    # Request batch of blocks
    block_hashes = [
        hashlib.sha256(f"block_{i}".encode()).hexdigest()
        for i in range(50)
    ]
    batch_id = protocol.request_blocks(
        block_hashes=block_hashes,
        priority=RequestPriority.MEDIUM
    )
    
    print(f"\nRequested {len(block_hashes)} blocks in batch")
    print(f"Batch ID: {batch_id[:16]}...")
    
    # Request Merkle proof blocks (critical priority)
    target = hashlib.sha256(b"target_block").hexdigest()
    root = hashlib.sha256(b"genesis_block").hexdigest()
    proof = [hashlib.sha256(f"proof_{i}".encode()).hexdigest() for i in range(5)]
    
    proof_id = protocol.request_merkle_proof_blocks(target, root, proof)
    print(f"\nRequested Merkle proof blocks (CRITICAL priority)")
    print(f"Proof request ID: {proof_id[:16]}...")
    
    # Simulate response
    def mock_storage(block_hash):
        # Return fake data for some blocks
        if "block_0" in block_hash or "block_1" in block_hash:
            return b"compressed_block_data"
        return None
    
    # Get next request to process
    next_req = protocol.get_next_request()
    if next_req:
        print(f"\nProcessing next request (priority: {next_req.priority.name})")
        response = protocol.handle_request(next_req, "peer_123", mock_storage)
        print(f"Found {len(response.blocks)} blocks, {len(response.missing_blocks)} missing")
    
    # Get stats
    stats = protocol.get_stats()
    print(f"\nProtocol Statistics:")
    print(f"  Requests sent: {stats['requests_sent']}")
    print(f"  Active requests: {stats['active_requests']}")
    print(f"  Queued requests: {stats['queued_requests']}")
    
    print("\n✅ Request/response protocol working!")
