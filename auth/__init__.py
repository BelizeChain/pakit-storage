"""
Pakit Authentication Module

Provides two-tier access control for Pakit sovereign storage:
- Public tier: Open access for uploads/downloads
- Authenticated tier: BelizeID signature required for blockchain proofs
"""

from .access_control import (
    AccessControl,
    AccessTier,
    EndpointConfig,
    PUBLIC_ENDPOINTS,
    AUTHENTICATED_ENDPOINTS,
    ADMIN_ENDPOINTS,
    verify_public_access,
    verify_authenticated_access,
    verify_admin_access,
    get_endpoint_info,
)

__all__ = [
    "AccessControl",
    "AccessTier",
    "EndpointConfig",
    "PUBLIC_ENDPOINTS",
    "AUTHENTICATED_ENDPOINTS",
    "ADMIN_ENDPOINTS",
    "verify_public_access",
    "verify_authenticated_access",
    "verify_admin_access",
    "get_endpoint_info",
]
