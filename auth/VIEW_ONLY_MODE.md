# Pakit View-Only Public Mode

**Configuration**: Public can VIEW/LIST files, but CANNOT upload, download, or perform any actions without BelizeID authentication.

## Access Model

### ✅ Public Access (No Authentication)
- `/api/v1/list` - List all public files with metadata
- `/api/v1/view/{cid}` - View file metadata (CID, size, owner, uploaded date)
- `/health` - Health check
- `/metrics` - Prometheus metrics

### 🔒 Authenticated Access (Requires BelizeID Signature)
- `/api/v1/upload` - Upload file to sovereign DAG storage
- `/api/v1/download/{cid}` - Download file by CID
- `/api/v1/register_proof` - Register storage proof on blockchain
- `/api/v1/verify_ownership` - Verify file ownership
- `/api/v1/delete/{cid}` - Delete file (owner only)

## Environment Variables

```bash
# View-only public mode
PAKIT_PUBLIC_MODE=view-only
PUBLIC_UPLOAD_ENABLED=false
PUBLIC_DOWNLOAD_ENABLED=false

# Blockchain integration (required for authenticated operations)
BLOCKCHAIN_ENABLED=true
BLOCKCHAIN_RPC=ws://belizechain-node.belizechain.svc.cluster.local:9944
REQUIRE_BELIZEID_FOR_PROOFS=true
```

## Usage Examples

### 1. Public List Files (No Auth)

```bash
# Anyone can browse public files
curl https://storage.belizechain.bz/api/v1/list

# Response:
{
  "files": [
    {
      "cid": "QmHash123...",
      "title": "Public Government Report",
      "size": 1024000,
      "uploaded_at": "2026-01-31T12:00:00Z",
      "owner": "5GrwvaEF...",
      "public": true
    }
  ]
}
```

### 2. Public View File Metadata (No Auth)

```bash
# View file info without downloading
curl https://storage.belizechain.bz/api/v1/view/QmHash123...

# Response:
{
  "cid": "QmHash123...",
  "title": "Public Government Report",
  "description": "Annual transparency report",
  "size": 1024000,
  "compression_ratio": 2.3,
  "uploaded_at": "2026-01-31T12:00:00Z",
  "owner_belizeid": "5GrwvaEF...",
  "blockchain_proof": {
    "block_number": 12345,
    "extrinsic_hash": "0xabc123..."
  },
  "public": true,
  "download_url": "/api/v1/download/QmHash123..."  # Requires auth
}
```

### 3. Upload File (Requires BelizeID)

```bash
# Generate BelizeID signature
MESSAGE="upload:document.pdf:$(date +%s)"
SIGNATURE=$(belizechain-cli sign --message "$MESSAGE" --keyfile ~/.belizechain/keys/alice.json)
PUBLIC_KEY="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

# Upload with BelizeID signature
curl -X POST https://storage.belizechain.bz/api/v1/upload \
  -H "X-BelizeID-Signature: $SIGNATURE" \
  -H "X-BelizeID-Message: $MESSAGE" \
  -H "X-BelizeID-Public-Key: $PUBLIC_KEY" \
  -F "file=@document.pdf" \
  -F "metadata={\"title\": \"My Document\", \"public\": false}"

# Response:
{
  "cid": "QmNewHash...",
  "size": 2048000,
  "owner": "5GrwvaEF...",
  "public": false,
  "uploaded_at": "2026-01-31T13:00:00Z"
}
```

### 4. Download File (Requires BelizeID)

```bash
# Even public files require BelizeID to download
curl https://storage.belizechain.bz/api/v1/download/QmHash123... \
  -H "X-BelizeID-Signature: $SIGNATURE" \
  -H "X-BelizeID-Message: $MESSAGE" \
  -H "X-BelizeID-Public-Key: $PUBLIC_KEY" \
  -o downloaded_file.pdf
```

### 5. Try Download Without Auth (Fails)

```bash
curl https://storage.belizechain.bz/api/v1/download/QmHash123...

# Response: 401 Unauthorized
{
  "error": "Unauthorized",
  "message": "Valid BelizeID signature required for this operation",
  "required_headers": [
    "X-BelizeID-Signature",
    "X-BelizeID-Message",
    "X-BelizeID-Public-Key"
  ]
}
```

## FastAPI Integration

```python
from fastapi import FastAPI, Depends, UploadFile
from pakit.auth import verify_public_access, verify_authenticated_access

app = FastAPI()

# Public view-only endpoint
@app.get("/api/v1/list")
async def list_files():
    # No auth required - anyone can view list
    return await storage_service.list_public_files()

@app.get("/api/v1/view/{cid}")
async def view_file_metadata(cid: str):
    # No auth required - anyone can view metadata
    return await storage_service.get_metadata(cid)

# Authenticated upload endpoint
@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile,
    metadata: dict,
    authenticated: bool = Depends(verify_authenticated_access)  # BelizeID required
):
    # Only BelizeID holders can upload
    return await storage_service.upload(file, metadata)

# Authenticated download endpoint
@app.get("/api/v1/download/{cid}")
async def download_file(
    cid: str,
    authenticated: bool = Depends(verify_authenticated_access)  # BelizeID required
):
    # Only BelizeID holders can download
    return await storage_service.download(cid)
```

## Benefits of View-Only Mode

### 1. **Transparency** ✅
- Public can SEE what files are stored
- Government accountability (public metadata)
- Trust through openness

### 2. **Security** 🔒
- Prevents spam uploads (requires BelizeID)
- Download accountability (who accessed what)
- Owner verification for deletions

### 3. **Sovereignty** 🇧🇿
- Only Belize citizens/residents can store documents
- BelizeID integration ensures national control
- Blockchain proofs for immutability

### 4. **Privacy** 🛡️
- Files marked `public: false` hidden from list
- Download tracking for sensitive documents
- Owner-only access control

## File Visibility Levels

```python
# When uploading, set public flag in metadata
metadata = {
    "title": "Document Title",
    "public": True   # ← Controls visibility in /api/v1/list
}

# Public files:
- Appear in /api/v1/list
- Metadata visible in /api/v1/view/{cid}
- Still require BelizeID to download

# Private files:
- Do NOT appear in /api/v1/list
- Metadata only visible to owner
- Require BelizeID + ownership verification to download
```

## Deployment

### Docker Compose
```bash
# infra/pakit/docker-compose.yml already updated
cd infra/pakit
export POSTGRES_PASSWORD=secure_password
docker compose up -d
```

### Kubernetes
```bash
# Update ConfigMap
kubectl edit configmap belizechain-config -n belizechain

# Set:
data:
  PAKIT_PUBLIC_MODE: "view-only"
  PUBLIC_UPLOAD_ENABLED: "false"
  PUBLIC_DOWNLOAD_ENABLED: "false"
  BLOCKCHAIN_ENABLED: "true"

# Restart Pakit
kubectl rollout restart deployment/pakit -n belizechain

# Apply updated Istio policies
kubectl apply -f infra/k8s/istio/authorization-policies.yaml
kubectl apply -f infra/k8s/istio/virtual-services.yaml
```

## Security Considerations

1. **Rate Limiting**: Implement per-IP limits on /api/v1/list to prevent scraping
2. **Metadata Scrubbing**: Ensure sensitive data not in public metadata
3. **Audit Logging**: Log all download attempts with BelizeID
4. **DDOS Protection**: Cloudflare/WAF in front of public endpoints
5. **File Scanning**: Optional malware scanning before allowing uploads

## Testing

```bash
# Test public list (should work)
curl http://localhost:8001/api/v1/list
# Expected: 200 OK with file list

# Test public view (should work)
curl http://localhost:8001/api/v1/view/QmTestHash
# Expected: 200 OK with metadata

# Test upload without auth (should fail)
curl -X POST http://localhost:8001/api/v1/upload -F "file=@test.pdf"
# Expected: 401 Unauthorized

# Test download without auth (should fail)
curl http://localhost:8001/api/v1/download/QmTestHash
# Expected: 401 Unauthorized

# Test with BelizeID (should work)
curl -X POST http://localhost:8001/api/v1/upload \
  -H "X-BelizeID-Public-Key: 5GrwvaEF..." \
  -F "file=@test.pdf"
# Expected: 200 OK with CID
```
