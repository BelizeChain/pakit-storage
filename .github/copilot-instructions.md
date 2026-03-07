# Pakit Storage — DAG-Based Decentralized Storage

## Project Identity
- **Repo**: `BelizeChain/pakit-storage`
- **Role**: Sovereign DAG-based decentralized storage for BelizeChain
- **Language**: Python
- **Branch**: `main` (default)

## Features
- Content-addressable MerkleDAG storage engine
- P2P networking for distributed storage
- ML-optimized data placement and caching
- Quantum compression support
- CDN layer for content delivery
- Authentication and access control
- REST API server (`api_server.py`)
- Multiple storage backends

## Azure Deployment Target
- **ACR**: `belizechainacr.azurecr.io` → image: `belizechainacr.azurecr.io/pakit`
- **AKS**: `belizechain-aks` (Free tier, 1x Standard_D2s_v3, K8s v1.33.6)
- **Resource Group**: `BelizeChain` in `centralus`
- **Subscription**: `77e6d0a2-78d2-4568-9f5a-34bd62357c40`
- **Tenant**: `belizechain.org`

## Deployment Status: Phase 2 — TODO
### What needs to be done:
1. **Verify Dockerfile** — Ensure Python deps install, API server starts on correct port
2. **Update deploy.yml** — Migrate from VM/SSH to AKS deployment:
   - Use `azure/login@v2` with `${{ secrets.AZURE_CREDENTIALS }}`
   - Use `azure/aks-set-context@v4` with `${{ secrets.AKS_CLUSTER_NAME }}`
   - Push image to `belizechainacr.azurecr.io/pakit`
   - `kubectl apply` Deployment + Service (expose port 8002)
3. **Configure GitHub Secrets**:
   - `ACR_USERNAME` = `belizechainacr`
   - `ACR_PASSWORD` = (from `az acr credential show --name belizechainacr`)
   - `AZURE_CREDENTIALS` = (service principal JSON)
   - `AZURE_RESOURCE_GROUP` = `BelizeChain`
   - `AKS_CLUSTER_NAME` = `belizechain-aks`
4. **K8s namespace**: Deploy into `belizechain` namespace
5. **Persistent storage**: Consider Azure Disk or Azure Files PVC for DAG data persistence
6. **Resource limits**: 100m-250m CPU, 128-512Mi RAM
7. **Connect to blockchain**: `ws://belizechain-node.belizechain.svc.cluster.local:9944`

## Sibling Services (same AKS cluster)
| Service | Image | Ports |
|---------|-------|-------|
| belizechain-node | `belizechainacr.azurecr.io/belizechain-node` | 30333, 9944, 9615 |
| ui | `belizechainacr.azurecr.io/ui` | 80 |
| kinich-quantum | `belizechainacr.azurecr.io/kinich` | 8000 |
| nawal-ai | `belizechainacr.azurecr.io/nawal` | 8001 |

## Dev Commands
```bash
pip install -r requirements.txt          # Install dependencies
python api_server.py                     # Run API server
docker build -t belizechainacr.azurecr.io/pakit .  # Docker image
make test                                # Run tests
```

## Constraints
- **Shared node**: All services share 2 vCPU / 8GB RAM — minimal resource requests
- **Storage persistence**: DAG data must survive pod restarts — needs PVC
- **Cost ceiling**: ~$75/mo total for ALL services
- **No P2P in K8s initially**: P2P networking may need NodePort or HostNetwork config
