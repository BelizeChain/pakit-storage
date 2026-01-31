# Pakit Access Control - Usage Examples

## 1. Public Upload (No authentication)

```bash
# Anyone can upload files
curl -X POST https://storage.belizechain.bz/api/v1/upload \
  -F "file=@document.pdf" \
  -F "metadata={\"title\": \"Public Document\"}"

# Response:
{
  "cid": "QmHash123...",
  "size": 1024000,
  "compression_ratio": 2.3,
  "uploaded_at": "2026-01-31T12:00:00Z"
}
```

## 2. Public Download (No authentication)

```bash
# Anyone can download by CID
curl https://storage.belizechain.bz/api/v1/download/QmHash123... \
  -o downloaded_file.pdf
```

## 3. Register Blockchain Proof (Requires BelizeID)

```bash
# Step 1: Generate signature with BelizeID keypair
MESSAGE="register_proof:QmHash123...:$(date +%s)"
SIGNATURE=$(belizechain-cli sign --message "$MESSAGE" --keyfile ~/.belizechain/keys/alice.json)
PUBLIC_KEY="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"

# Step 2: Submit proof registration with signature
curl -X POST https://storage.belizechain.bz/api/v1/register_proof \
  -H "X-BelizeID-Signature: $SIGNATURE" \
  -H "X-BelizeID-Message: $MESSAGE" \
  -H "X-BelizeID-Public-Key: $PUBLIC_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "cid": "QmHash123...",
    "owner": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
    "metadata": {
      "title": "Land Title Certificate",
      "district": "Belize District",
      "parcel_number": "123-456-789"
    }
  }'

# Response (blockchain proof registered):
{
  "cid": "QmHash123...",
  "blockchain_proof": {
    "block_number": 12345,
    "extrinsic_hash": "0xabc123...",
    "pallet": "landledger",
    "call": "register_document_proof"
  },
  "owner": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "registered_at": "2026-01-31T12:05:00Z"
}
```

## 4. Verify Ownership (Requires BelizeID)

```bash
curl -X POST https://storage.belizechain.bz/api/v1/verify_ownership \
  -H "X-BelizeID-Signature: $SIGNATURE" \
  -H "X-BelizeID-Message: $MESSAGE" \
  -H "X-BelizeID-Public-Key: $PUBLIC_KEY" \
  -d '{"cid": "QmHash123..."}'

# Response:
{
  "cid": "QmHash123...",
  "is_owner": true,
  "owner_belizeid": "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
  "proof_exists": true,
  "verified_at": "2026-01-31T12:10:00Z"
}
```

## FastAPI Integration

```python
from fastapi import FastAPI, Depends
from pakit.auth import verify_public_access, verify_authenticated_access

app = FastAPI()

# Public endpoint (no dependency)
@app.post("/api/v1/upload")
async def upload_file(
    file: UploadFile,
    # No auth dependency - public access
):
    # Anyone can upload
    return await storage_service.upload(file)

# Authenticated endpoint (requires BelizeID)
@app.post("/api/v1/register_proof")
async def register_proof(
    cid: str,
    owner: str,
    metadata: dict,
    authenticated: bool = Depends(verify_authenticated_access)  # BelizeID signature required
):
    # Only BelizeID holders can register proofs
    return await blockchain_connector.register_proof(cid, owner, metadata)
```

## Environment Variables

```bash
# Enable public mode (default: true)
PAKIT_PUBLIC_MODE=true

# Enable blockchain integration (default: false)
BLOCKCHAIN_ENABLED=true

# Require BelizeID for proofs (default: true)
REQUIRE_BELIZEID_FOR_PROOFS=true

# Blockchain RPC endpoint
BLOCKCHAIN_RPC=ws://belizechain-node.belizechain.svc.cluster.local:9944

# Admin token (for /api/v1/admin/* endpoints)
PAKIT_ADMIN_TOKEN=your_secure_admin_token
```

## Istio Configuration

Pakit uses dual subsets in Istio:

```yaml
# Public subset (no mTLS)
- name: public
  labels:
    access: public
  trafficPolicy:
    tls:
      mode: DISABLE  # Allow plaintext for public access

# Internal subset (mTLS required)
- name: internal
  labels:
    access: internal
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL  # Require mTLS for blockchain proofs
```

## Security Considerations

1. **Public Upload Limits**
   - Max file size: 100MB (configurable via `MAX_FILE_SIZE`)
   - Rate limiting: Implement per-IP limits in production
   - Content scanning: Optional malware/CSAM detection

2. **BelizeID Signature Verification**
   - Message includes timestamp (5-minute expiry)
   - Replay attack protection via nonce
   - Public key validated against BelizeChain identity registry

3. **Blockchain Proof Integrity**
   - CID verified against on-chain hash
   - Owner field matches signature public key
   - Cannot modify proof after registration (immutable)

## Testing

```bash
# Run access control tests
pytest pakit/tests/test_access_control.py -v

# Test public endpoints
curl http://localhost:8001/api/v1/upload -F "file=@test.pdf"

# Test authenticated endpoints (without signature - should fail)
curl -X POST http://localhost:8001/api/v1/register_proof \
  -d '{"cid": "QmTest", "owner": "5GrwvaEF..."}'
# Expected: 401 Unauthorized

# Test with mock signature (dev mode)
export BLOCKCHAIN_ENABLED=false
curl -X POST http://localhost:8001/api/v1/register_proof \
  -H "X-BelizeID-Public-Key: 5GrwvaEF..." \
  -d '{"cid": "QmTest", "owner": "5GrwvaEF..."}'
# Expected: 200 OK (blockchain disabled)
```
