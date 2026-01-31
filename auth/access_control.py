"""
Pakit Access Control - Two-tier authentication system

PUBLIC TIER (No authentication required):
    - /api/v1/upload
    - /api/v1/download/{cid}
    - /api/v1/list
    - /health
    - /metrics

AUTHENTICATED TIER (Requires BelizeID signature):
    - /api/v1/register_proof
    - /api/v1/verify_ownership
    - /api/v1/admin/*
    
Author: BelizeChain Development Team
License: Apache 2.0
"""

from typing import Optional, List
from enum import Enum
from fastapi import HTTPException, Header, Depends
from pydantic import BaseModel
import os


class AccessTier(str, Enum):
    """Access tier classification for Pakit endpoints"""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"


class EndpointConfig(BaseModel):
    """Configuration for endpoint access control"""
    path: str
    tier: AccessTier
    methods: List[str]
    description: str


# Public endpoints configuration (VIEW-ONLY - NO authentication required)
PUBLIC_ENDPOINTS = [
    EndpointConfig(
        path="/api/v1/list",
        tier=AccessTier.PUBLIC,
        methods=["GET"],
        description="List public files (view-only access)"
    ),
    EndpointConfig(
        path="/api/v1/view/{cid}",
        tier=AccessTier.PUBLIC,
        methods=["GET"],
        description="View file metadata by CID (no download)"
    ),
    EndpointConfig(
        path="/health",
        tier=AccessTier.PUBLIC,
        methods=["GET"],
        description="Health check endpoint"
    ),
    EndpointConfig(
        path="/metrics",
        tier=AccessTier.PUBLIC,
        methods=["GET"],
        description="Prometheus metrics endpoint"
    ),
]

# Authenticated endpoints (requires BelizeID signature)
AUTHENTICATED_ENDPOINTS = [
    EndpointConfig(
        path="/api/v1/upload",
        tier=AccessTier.AUTHENTICATED,
        methods=["POST"],
        description="Upload file to sovereign DAG storage (requires BelizeID)"
    ),
    EndpointConfig(
        path="/api/v1/download/{cid}",
        tier=AccessTier.AUTHENTICATED,
        methods=["GET"],
        description="Download file by CID (requires BelizeID)"
    ),
    EndpointConfig(
        path="/api/v1/register_proof",
        tier=AccessTier.AUTHENTICATED,
        methods=["POST"],
        description="Register storage proof on BelizeChain (requires BelizeID)"
    ),
    EndpointConfig(
        path="/api/v1/verify_ownership",
        tier=AccessTier.AUTHENTICATED,
        methods=["POST"],
        description="Verify file ownership via BelizeID"
    ),
    EndpointConfig(
        path="/api/v1/delete/{cid}",
        tier=AccessTier.AUTHENTICATED,
        methods=["DELETE"],
        description="Delete file (owner verification required)"
    ),
]

# Admin endpoints (requires admin privileges)
ADMIN_ENDPOINTS = [
    EndpointConfig(
        path="/api/v1/admin/stats",
        tier=AccessTier.ADMIN,
        methods=["GET"],
        description="Storage statistics and analytics"
    ),
    EndpointConfig(
        path="/api/v1/admin/purge",
        tier=AccessTier.ADMIN,
        methods=["POST"],
        description="Purge orphaned data"
    ),
]


class AccessControl:
    """
    Pakit access control manager
    
    Implements two-tier authentication:
    1. Public tier: Anyone can upload/download (sovereign storage)
    2. Authenticated tier: Requires BelizeID signature for blockchain proofs
    """
    
    def __init__(self):
        self.public_mode = os.getenv("PAKIT_PUBLIC_MODE", "true").lower() == "true"
        self.blockchain_enabled = os.getenv("BLOCKCHAIN_ENABLED", "false").lower() == "true"
        self.require_belizeid_for_proofs = os.getenv("REQUIRE_BELIZEID_FOR_PROOFS", "true").lower() == "true"
    
    def is_public_endpoint(self, path: str, method: str) -> bool:
        """Check if endpoint allows public access"""
        for endpoint in PUBLIC_ENDPOINTS:
            # Simple path matching (can be enhanced with regex)
            if path.startswith(endpoint.path.replace("{cid}", "")) and method in endpoint.methods:
                return True
        return False
    
    def is_authenticated_endpoint(self, path: str, method: str) -> bool:
        """Check if endpoint requires authentication"""
        for endpoint in AUTHENTICATED_ENDPOINTS:
            if path.startswith(endpoint.path.replace("{cid}", "")) and method in endpoint.methods:
                return True
        return False
    
    def is_admin_endpoint(self, path: str, method: str) -> bool:
        """Check if endpoint requires admin privileges"""
        for endpoint in ADMIN_ENDPOINTS:
            if path.startswith(endpoint.path) and method in endpoint.methods:
                return True
        return False
    
    async def verify_belizeid_signature(
        self,
        signature: Optional[str],
        message: Optional[str],
        public_key: Optional[str]
    ) -> bool:
        """
        Verify BelizeID signature for authenticated operations
        
        Args:
            signature: Hex-encoded signature
            message: Message that was signed
            public_key: BelizeID public key (SS58 format)
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not self.blockchain_enabled:
            # Blockchain disabled - skip verification in dev mode
            return True
        
        if not signature or not message or not public_key:
            return False
        
        # Import here to avoid dependency in public mode
        try:
            from substrateinterface import Keypair
            
            # Create keypair from public key
            keypair = Keypair(ss58_address=public_key)
            
            # Verify signature
            is_valid = keypair.verify(message, signature)
            return is_valid
        except Exception as e:
            print(f"BelizeID signature verification failed: {e}")
            return False


# Dependency injection for FastAPI
async def verify_public_access(
    access_control: AccessControl = Depends(lambda: AccessControl())
) -> bool:
    """Dependency: Always allow public access"""
    return True


async def verify_authenticated_access(
    x_belizeid_signature: Optional[str] = Header(None),
    x_belizeid_message: Optional[str] = Header(None),
    x_belizeid_public_key: Optional[str] = Header(None),
    access_control: AccessControl = Depends(lambda: AccessControl())
) -> bool:
    """
    Dependency: Verify BelizeID signature for authenticated endpoints
    
    Headers required:
        X-BelizeID-Signature: Hex-encoded signature
        X-BelizeID-Message: Message that was signed (timestamp + operation)
        X-BelizeID-Public-Key: SS58-formatted BelizeID public key
    """
    if not access_control.require_belizeid_for_proofs:
        # Skip authentication if not required
        return True
    
    is_valid = await access_control.verify_belizeid_signature(
        x_belizeid_signature,
        x_belizeid_message,
        x_belizeid_public_key
    )
    
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Unauthorized",
                "message": "Valid BelizeID signature required for this operation",
                "required_headers": [
                    "X-BelizeID-Signature",
                    "X-BelizeID-Message",
                    "X-BelizeID-Public-Key"
                ]
            }
        )
    
    return True


async def verify_admin_access(
    x_admin_token: Optional[str] = Header(None),
    access_control: AccessControl = Depends(lambda: AccessControl())
) -> bool:
    """Dependency: Verify admin token for admin endpoints"""
    admin_token = os.getenv("PAKIT_ADMIN_TOKEN")
    
    if not admin_token:
        raise HTTPException(
            status_code=503,
            detail="Admin access not configured"
        )
    
    if x_admin_token != admin_token:
        raise HTTPException(
            status_code=403,
            detail="Invalid admin token"
        )
    
    return True


def get_endpoint_info(path: str, method: str) -> Optional[EndpointConfig]:
    """Get endpoint configuration by path and method"""
    all_endpoints = PUBLIC_ENDPOINTS + AUTHENTICATED_ENDPOINTS + ADMIN_ENDPOINTS
    
    for endpoint in all_endpoints:
        # Simple path matching
        if path.startswith(endpoint.path.replace("{cid}", "")) and method in endpoint.methods:
            return endpoint
    
    return None
