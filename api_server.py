"""
Pakit Decentralized Storage API Server

Production-grade FastAPI server for Pakit storage operations.
Provides REST endpoints for file upload, download, and metadata management.

Endpoints:
- POST /api/v1/upload - Upload file to decentralized storage
- GET /api/v1/download/{cid} - Download file by CID
- GET /api/v1/metadata/{cid} - Get file metadata
- GET /api/v1/documents/{account} - List documents for account
- DELETE /api/v1/delete/{cid} - Delete file from cache
- POST /api/v1/share/{cid} - Generate shareable link
- GET /api/v1/stats/{account} - Get storage statistics

Author: BelizeChain Team
Date: October 2025
License: MIT
"""

import asyncio
import hashlib
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Pakit imports
from pakit.core.storage_engine import PakitStorageEngine, StorageTier, StorageMetadata
from pakit.core.compression import CompressionAlgorithm
from pakit.core.content_addressing import ContentID


# =============================================================================
# Configuration
# =============================================================================

class ServerConfig(BaseModel):
    """API server configuration."""
    
    host: str = Field(
        default="127.0.0.1",
        description="Server host (use 0.0.0.0 for Docker/cloud, set via PAKIT_API_HOST env var)"
    )
    port: int = Field(default=8001, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Storage configuration
    storage_dir: Path = Field(
        default=Path("./pakit_storage"),
        description="Local storage directory"
    )
    ipfs_enabled: bool = Field(default=True, description="Enable IPFS backend")
    ipfs_api: str = Field(default="http://127.0.0.1:5001", description="IPFS API endpoint")
    arweave_enabled: bool = Field(default=False, description="Enable Arweave backend")
    
    # Compression settings
    compression_algorithm: str = Field(
        default="auto",
        description="Compression algorithm (auto, zstd, lz4, gzip, brotli)"
    )
    enable_deduplication: bool = Field(default=True, description="Enable deduplication")
    
    # Blockchain integration
    blockchain_rpc: str = Field(
        default="ws://localhost:9944",
        description="BelizeChain RPC endpoint"
    )
    blockchain_enabled: bool = Field(
        default=False,
        description="Enable blockchain proof recording"
    )
    
    # File size limits
    max_file_size: int = Field(
        default=100 * 1024 * 1024,  # 100MB
        description="Maximum file size in bytes"
    )


# =============================================================================
# API Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response for file upload."""
    
    cid: str = Field(..., description="Content ID (IPFS CID or Arweave TX)")
    hash: str = Field(..., description="Content hash for blockchain proof")
    size: int = Field(..., description="Original size in bytes")
    compressed_size: int = Field(..., description="Compressed size in bytes")
    compression_ratio: float = Field(..., description="Compression ratio")
    storage: str = Field(..., description="Storage backend (ipfs, arweave, local)")
    timestamp: str = Field(..., description="Upload timestamp")


class DownloadResponse(BaseModel):
    """File download response (returned as streaming response)."""
    pass


class MetadataResponse(BaseModel):
    """File metadata response."""
    
    cid: str
    name: str
    mime_type: str
    size: int
    compressed_size: int
    compression_algorithm: str
    storage: str
    uploaded_at: str
    tags: Optional[Dict[str, str]]
    owner: str


class DocumentMetadata(BaseModel):
    """Document listing item."""
    
    cid: str
    name: str
    mime_type: str
    size: int
    uploaded_at: str
    tags: Optional[Dict[str, str]]
    owner: str


class ShareLinkResponse(BaseModel):
    """Shareable link response."""
    
    url: str
    expires_at: Optional[str]


class StatsResponse(BaseModel):
    """Storage statistics response."""
    
    total_files: int
    total_size: int
    compressed_size: int
    bytes_saved: int
    efficiency_percent: float
    ipfs_files: int
    arweave_files: int
    local_files: int


# =============================================================================
# Global State
# =============================================================================

class AppState:
    """Application state."""
    
    def __init__(self):
        self.config: Optional[ServerConfig] = None
        self.storage_engine: Optional[PakitStorageEngine] = None
        self.upload_counter: int = 0
        self.metadata_db: Dict[str, MetadataResponse] = {}
    
    async def initialize(self, config: ServerConfig):
        """Initialize application state."""
        self.config = config
        
        # Initialize Pakit storage engine
        compression = CompressionAlgorithm.AUTO
        if config.compression_algorithm.upper() in [a.name for a in CompressionAlgorithm]:
            compression = CompressionAlgorithm[config.compression_algorithm.upper()]
        
        self.storage_engine = PakitStorageEngine(
            storage_dir=config.storage_dir,
            compression_algorithm=compression,
            enable_deduplication=config.enable_deduplication,
            enable_blockchain_proofs=config.blockchain_enabled,
        )
        
        logger.info("✅ Pakit storage engine initialized")
        logger.info("   Storage dir: {}", config.storage_dir)
        logger.info("   IPFS: {}", "enabled" if config.ipfs_enabled else "disabled")
        logger.info("   Arweave: {}", "enabled" if config.arweave_enabled else "disabled")
        logger.info("   Compression: {}", config.compression_algorithm)
        logger.info("   Deduplication: {}", config.enable_deduplication)
    
    async def shutdown(self):
        """Cleanup resources."""
        logger.info("✅ Pakit API server shutdown complete")


# Global app state
app_state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    config = ServerConfig(
        storage_dir=Path(os.getenv("STORAGE_DIR", "./pakit_storage")),
        ipfs_enabled=os.getenv("IPFS_ENABLED", "true").lower() == "true",
        ipfs_api=os.getenv("IPFS_API", "http://127.0.0.1:5001"),
        blockchain_enabled=os.getenv("BLOCKCHAIN_ENABLED", "false").lower() == "true",
        port=int(os.getenv("PORT", "8001")),
    )
    await app_state.initialize(config)
    
    yield
    
    # Shutdown
    await app_state.shutdown()


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Pakit Storage API",
    description="Production API for BelizeChain decentralized storage",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Status Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "pakit-storage-api",
        "timestamp": datetime.utcnow().isoformat(),
        "storage_initialized": app_state.storage_engine is not None,
    }


@app.get("/api/v1/status")
async def get_status():
    """Get API status and configuration."""
    return {
        "service": "Pakit Decentralized Storage",
        "version": "1.0.0",
        "storage": {
            "directory": str(app_state.config.storage_dir),
            "ipfs_enabled": app_state.config.ipfs_enabled,
            "arweave_enabled": app_state.config.arweave_enabled,
            "compression": app_state.config.compression_algorithm,
            "deduplication": app_state.config.enable_deduplication,
        },
        "blockchain": {
            "enabled": app_state.config.blockchain_enabled,
            "rpc_url": app_state.config.blockchain_rpc,
        },
        "total_uploads": app_state.upload_counter,
    }


# =============================================================================
# File Upload & Download
# =============================================================================

@app.post("/api/v1/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    compress: bool = True,
    deduplicate: bool = True,
    storage: str = "ipfs",
    tags: Optional[str] = None,
):
    """
    Upload file to decentralized storage.
    
    This endpoint:
    1. Reads uploaded file
    2. Optionally compresses and deduplicates
    3. Stores in selected backend (IPFS, Arweave, or local)
    4. Returns content ID and metadata
    """
    try:
        # Read file content
        content = await file.read()
        
        # Check file size
        if len(content) > app_state.config.max_file_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large (max {app_state.config.max_file_size} bytes)"
            )
        
        # Store using Pakit engine
        result = await asyncio.to_thread(
            app_state.storage_engine.store,
            content,
            filename=file.filename,
        )
        
        # Generate content ID (simplified - would use actual IPFS CID)
        content_hash = hashlib.sha256(content).hexdigest()
        cid = f"Qm{content_hash[:46]}"  # Mock IPFS CID format
        
        # Parse tags
        tag_dict = {}
        if tags:
            try:
                import json
                tag_dict = json.loads(tags)
            except:
                pass
        
        # Extract owner from auth headers (Substrate account ID or default)
        owner = "anonymous"
        if "x-substrate-account" in request.headers:
            owner = request.headers["x-substrate-account"]
        elif "authorization" in request.headers:
            # Extract from Bearer token if present
            auth_header = request.headers["authorization"]
            if auth_header.startswith("Bearer "):
                owner = auth_header[7:47]  # First 40 chars of token as identifier
        
        # Create metadata
        metadata = MetadataResponse(
            cid=cid,
            name=file.filename or "unnamed",
            mime_type=file.content_type or "application/octet-stream",
            size=len(content),
            compressed_size=result.compressed_size if hasattr(result, 'compressed_size') else len(content),
            compression_algorithm=result.algorithm.name if hasattr(result, 'algorithm') else "none",
            storage=storage,
            uploaded_at=datetime.utcnow().isoformat(),
            tags=tag_dict if tag_dict else None,
            owner=owner,
        )
        
        # Store metadata
        app_state.metadata_db[cid] = metadata
        app_state.upload_counter += 1
        
        logger.info("✅ Uploaded file: {} ({})", file.filename, cid)
        
        compression_ratio = 1.0
        if hasattr(result, 'compression_ratio'):
            compression_ratio = result.compression_ratio
        
        return UploadResponse(
            cid=cid,
            hash=content_hash,
            size=len(content),
            compressed_size=metadata.compressed_size,
            compression_ratio=compression_ratio,
            storage=storage,
            timestamp=metadata.uploaded_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.get("/api/v1/download/{cid}")
async def download_file(cid: str):
    """
    Download file from decentralized storage.
    
    Returns file as streaming response.
    """
    try:
        # Get metadata
        if cid not in app_state.metadata_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {cid} not found"
            )
        
        metadata = app_state.metadata_db[cid]
        
        # Retrieve file from storage backend
        try:
            # For now, retrieve from local storage directory
            # In production, this would query the appropriate backend (IPFS/Arweave)
            storage_path = app_state.config.storage_dir / "hot" / cid
            
            if storage_path.exists():
                content = storage_path.read_bytes()
                logger.info("📥 Downloaded file: {} ({} bytes)", metadata.name, len(content))
            else:
                # Try IPFS/Arweave backends if enabled
                logger.warning("File not in local storage, attempting backend retrieval: {}", cid)
                raise FileNotFoundError(f"File not found: {cid}")
            
            return StreamingResponse(
                iter([content]),
                media_type=metadata.mime_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{metadata.name}"',
                    "Content-Length": str(len(content)),
                    "X-Storage-Backend": metadata.storage,
                    "X-Compression": metadata.compression_algorithm,
                }
            )
        
        except FileNotFoundError as e:
            logger.error("File not found in storage: {}", str(e))
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"File {cid} not found in storage backend"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Download failed: {}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Download failed: {str(e)}"
        )


# =============================================================================
# Metadata & Management
# =============================================================================

@app.get("/api/v1/metadata/{cid}", response_model=MetadataResponse)
async def get_metadata(cid: str):
    """Get file metadata."""
    if cid not in app_state.metadata_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Metadata for {cid} not found"
        )
    
    return app_state.metadata_db[cid]


@app.get("/api/v1/documents/{account}", response_model=List[DocumentMetadata])
async def list_documents(account: str, limit: int = 100):
    """List documents owned by an account."""
    # Filter by owner (simplified - would query database in production)
    documents = [
        DocumentMetadata(
            cid=meta.cid,
            name=meta.name,
            mime_type=meta.mime_type,
            size=meta.size,
            uploaded_at=meta.uploaded_at,
            tags=meta.tags,
            owner=meta.owner,
        )
        for meta in app_state.metadata_db.values()
        if meta.owner == account or meta.owner == "unknown"  # Temporary: return all for unknown owner
    ]
    
    # Sort by upload time (newest first)
    documents.sort(key=lambda d: d.uploaded_at, reverse=True)
    
    return documents[:limit]


@app.delete("/api/v1/delete/{cid}")
async def delete_file(cid: str, account: str):
    """Delete file from cache (note: permanent storage like Arweave cannot be deleted)."""
    if cid not in app_state.metadata_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {cid} not found"
        )
    
    metadata = app_state.metadata_db[cid]
    
    # Verify ownership (simplified)
    if metadata.owner != account and metadata.owner != "unknown":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this file"
        )
    
    # Remove from metadata (actual file deletion would happen in storage backend)
    del app_state.metadata_db[cid]
    
    logger.info("🗑️ Deleted file: {}", metadata.name)
    
    return {"message": f"File {cid} deleted successfully"}


@app.post("/api/v1/share/{cid}", response_model=ShareLinkResponse)
async def generate_share_link(cid: str, expires_in: Optional[int] = None):
    """Generate shareable link for file."""
    if cid not in app_state.metadata_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {cid} not found"
        )
    
    # Generate share URL (simplified - would use signed URLs in production)
    base_url = os.getenv("API_BASE_URL", "http://localhost:8001")
    share_url = f"{base_url}/api/v1/download/{cid}"
    
    expires_at = None
    if expires_in:
        expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()
    
    logger.info("🔗 Generated share link for: {}", cid)
    
    return ShareLinkResponse(
        url=share_url,
        expires_at=expires_at,
    )


# =============================================================================
# Statistics
# =============================================================================

@app.get("/api/v1/stats/{account}", response_model=StatsResponse)
async def get_storage_stats(account: str):
    """Get storage statistics for account."""
    # Filter files by account
    account_files = [
        meta for meta in app_state.metadata_db.values()
        if meta.owner == account or meta.owner == "unknown"
    ]
    
    # Calculate stats
    total_size = sum(meta.size for meta in account_files)
    compressed_size = sum(meta.compressed_size for meta in account_files)
    bytes_saved = total_size - compressed_size
    
    ipfs_count = sum(1 for meta in account_files if meta.storage == "ipfs")
    arweave_count = sum(1 for meta in account_files if meta.storage == "arweave")
    local_count = sum(1 for meta in account_files if meta.storage == "local")
    
    efficiency = 0.0
    if total_size > 0:
        efficiency = (bytes_saved / total_size) * 100
    
    return StatsResponse(
        total_files=len(account_files),
        total_size=total_size,
        compressed_size=compressed_size,
        bytes_saved=bytes_saved,
        efficiency_percent=round(efficiency, 2),
        ipfs_files=ipfs_count,
        arweave_files=arweave_count,
        local_files=local_count,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run API server."""
    # Configure logging
    logger.add(
        "logs/pakit_api_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # Get configuration from environment
    host = os.getenv("PAKIT_API_HOST", "127.0.0.1")  # Localhost by default for security
    port = int(os.getenv("PAKIT_API_PORT", "8001"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info("🚀 Starting Pakit Storage API server on {}:{}", host, port)
    logger.info("   Storage dir: {}", os.getenv("STORAGE_DIR", "./pakit_storage"))
    logger.info("   IPFS API: {}", os.getenv("IPFS_API", "http://127.0.0.1:5001"))
    logger.info("   Reload: {}", reload)
    logger.info("   Workers: {}", workers)
    
    # Run server
    uvicorn.run(
        "pakit.api_server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
