"""
Pakit Blockchain Integration

Connects Pakit storage backends to BelizeChain for:
- On-chain metadata storage
- Storage proof verification
- Document ownership tracking
"""

from .storage_proof_connector import (
    StorageProofConnector,
    get_storage_proof_connector,
    SUBSTRATE_AVAILABLE
)

__all__ = [
    "StorageProofConnector",
    "get_storage_proof_connector",
    "SUBSTRATE_AVAILABLE"
]
