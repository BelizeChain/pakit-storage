# Pakit Integration Guide

## Overview

Pakit is BelizeChain's decentralized storage layer, integrating with:
- **Blockchain**: LandLedger pallet (document proofs)
- **Nawal AI**: Model checkpoints, training datasets
- **Kinich Quantum**: Quantum computation results
- **UI Portals**: File uploads/downloads via Maya Wallet

This guide shows how to integrate Pakit into each component.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 BelizeChain Applications                │
│  ┌────────────┬────────────┬────────────┬────────────┐ │
│  │  Nawal AI  │   Kinich   │ LandLedger │ Maya Wallet│ │
│  │ Checkpoints│  Quantum   │  Documents │    Files   │ │
│  └─────┬──────┴─────┬──────┴─────┬──────┴─────┬──────┘ │
└────────┼────────────┼────────────┼────────────┼────────┘
         │            │            │            │
         └────────────┴────────────┴────────────┘
                          │
                          ▼
         ┌─────────────────────────────────────┐
         │           Pakit Storage             │
         │  ┌───────────────────────────────┐  │
         │  │  DAG Backend (Merkle Trees)   │  │
         │  ├───────────────────────────────┤  │
         │  │  P2P Network (Gossip/DHT)     │  │
         │  ├───────────────────────────────┤  │
         │  │  ML Optimization (5 models)   │  │
         │  ├───────────────────────────────┤  │
         │  │  Backends: IPFS + Arweave     │  │
         │  └───────────────────────────────┘  │
         └──────────┬────────────────┬─────────┘
                    │                │
         ┌──────────▼─────┐   ┌─────▼──────────┐
         │  IPFS Network  │   │ Arweave (Perm) │
         └────────────────┘   └────────────────┘
```

## Integration 1: Blockchain (LandLedger Pallet)

### Purpose
Store document proofs on-chain for land registry:
- Property title documents (PDFs, images)
- Survey reports
- Environmental assessments
- Transfer certificates

### Implementation

**Python Side (Pakit):**
```python
from pakit.blockchain import StorageProofConnector

# Initialize connector
connector = StorageProofConnector(
    node_url="ws://localhost:9944",
    keypair=keypair,  # Account for signing
    mock_mode=False  # Set True for development
)

await connector.connect()

# Upload document to Pakit
from pakit.core import PakitStorageEngine

engine = PakitStorageEngine()
content_id = engine.store(
    data=document_bytes,
    metadata={
        'property_id': 12345,
        'document_type': 'title_deed',
        'mime_type': 'application/pdf',
    }
)

# Store proof on blockchain
await connector.store_document_proof(
    content_id=content_id.to_base58(),
    ipfs_cid=ipfs_cid,  # If using IPFS backend
    arweave_tx=arweave_tx,  # If using Arweave
    owner=property_owner_account,
    metadata={
        'property_id': 12345,
        'size_bytes': len(document_bytes),
    }
)
```

**Rust Side (LandLedger Pallet):**
```rust
// Add to LandLedger pallet storage
#[pallet::storage]
pub type DocumentProofs<T: Config> = StorageMap<
    _,
    Blake2_128Concat,
    u32,  // property_id
    DocumentProof,
    OptionQuery,
>;

#[derive(Encode, Decode, Clone, PartialEq, Eq, RuntimeDebug, TypeInfo, MaxEncodedLen)]
pub struct DocumentProof {
    pub content_hash: BoundedVec<u8, ConstU32<64>>,
    pub ipfs_cid: Option<BoundedVec<u8, ConstU32<64>>>,
    pub arweave_tx: Option<BoundedVec<u8, ConstU32<64>>>,
    pub uploaded_at: u64,
}

// Add extrinsic
#[pallet::call]
impl<T: Config> Pallet<T> {
    #[pallet::weight(10_000)]
    pub fn register_document_proof(
        origin: OriginFor<T>,
        property_id: u32,
        content_hash: Vec<u8>,
        ipfs_cid: Option<Vec<u8>>,
        arweave_tx: Option<Vec<u8>>,
    ) -> DispatchResult {
        let who = ensure_signed(origin)?;
        
        // Verify ownership
        let property = Properties::<T>::get(property_id)
            .ok_or(Error::<T>::PropertyNotFound)?;
        ensure!(property.owner == who, Error::<T>::NotOwner);
        
        // Store proof
        let proof = DocumentProof {
            content_hash: BoundedVec::try_from(content_hash)
                .map_err(|_| Error::<T>::InvalidHash)?,
            ipfs_cid: ipfs_cid.map(|c| BoundedVec::try_from(c).unwrap()),
            arweave_tx: arweave_tx.map(|t| BoundedVec::try_from(t).unwrap()),
            uploaded_at: <frame_system::Pallet<T>>::block_number()
                .saturated_into(),
        };
        
        DocumentProofs::<T>::insert(property_id, proof);
        
        Self::deposit_event(Event::DocumentProofRegistered {
            property_id,
            owner: who,
        });
        
        Ok(())
    }
}
```

## Integration 2: Nawal AI (Model Checkpoints)

### Purpose
Store AI model checkpoints and training data:
- Federated learning checkpoints
- Genome evolution snapshots
- Training datasets (privacy-preserved)
- Model distillation results

### Implementation

```python
from nawal.storage import PakitClient, CheckpointManager

# Initialize Pakit client
pakit_client = PakitClient(
    pakit_api_url="http://localhost:8000",
    ipfs_gateway="http://localhost:5001"
)

# Create checkpoint manager
checkpoint_mgr = CheckpointManager(
    checkpoint_dir="./checkpoints",
    pakit_client=pakit_client,
    auto_upload=True  # Auto-upload to Pakit
)

# Save checkpoint (auto-uploads to Pakit)
import torch

model_state = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss,
}

pakit_cid = checkpoint_mgr.save_checkpoint(
    model_state=model_state,
    checkpoint_name=f"checkpoint_epoch_{epoch}.pt",
    metadata={
        'model_type': 'BelizeChainLLM',
        'genome_id': genome_id,
        'training_samples': 50000,
    }
)

print(f"Checkpoint uploaded to Pakit: {pakit_cid}")

# Load checkpoint (from Pakit if local missing)
loaded_state = checkpoint_mgr.load_checkpoint(
    checkpoint_name=f"checkpoint_epoch_{epoch}.pt",
    from_pakit=True  # Download from Pakit
)

model.load_state_dict(loaded_state['model_state_dict'])
```

### Integration with Staking Pallet

After successful training, report to blockchain:

```python
from nawal.blockchain import StakingConnector

staking = StakingConnector(node_url="ws://localhost:9944")
await staking.connect()

# Report training with Pakit checkpoint
await staking.submit_training_proof(
    validator_account=validator_account,
    checkpoint_cid=pakit_cid,  # Pakit CID as proof
    accuracy=0.85,
    samples_trained=50000,
)
```

## Integration 3: Kinich Quantum (Computation Results)

### Purpose
Archive quantum computation results:
- Circuit execution results
- Error mitigation data
- Quantum algorithm benchmarks
- Proof of Quantum Work submissions

### Implementation

```python
from kinich.storage import QuantumResultsStore
from kinich.blockchain import BelizeChainAdapter

# Initialize storage
blockchain = BelizeChainAdapter(node_url="ws://localhost:9944")
quantum_store = QuantumResultsStore(
    pakit_api_url="http://localhost:8000",
    blockchain_connector=blockchain
)

# Run quantum job
from qiskit import QuantumCircuit, execute
from kinich.adapters.azure import AzureQuantumBackend

circuit = QuantumCircuit(3, 3)
circuit.h(0)
circuit.cx(0, 1)
circuit.cx(1, 2)
circuit.measure([0, 1, 2], [0, 1, 2])

backend = AzureQuantumBackend()
job = backend.execute(circuit)
result = job.result()

# Store result in Pakit
content_id = quantum_store.store_quantum_result(
    job_id=job.job_id(),
    circuit_qasm=circuit.qasm(),
    counts=result.get_counts(),
    backend='azure_ionq',
    metadata={
        'shots': 1000,
        'error_mitigation': 'ZNE',
    }
)

print(f"Quantum result stored: {content_id}")

# Result also registered on Consensus pallet for PQW rewards
```

## Integration 4: UI Portals (Maya Wallet)

### Purpose
Allow users to upload/download files via web interface:
- Personal documents
- Photos/videos
- Backups
- Shared files

### Implementation (Maya Wallet - TypeScript/React)

```typescript
// pakit-client.ts
export class PakitClient {
  constructor(private apiUrl: string = 'http://localhost:8000') {}
  
  async uploadFile(file: File, metadata?: Record<string, any>): Promise<string> {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }
    
    const response = await fetch(`${this.apiUrl}/api/v1/upload`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }
    
    const result = await response.json();
    return result.cid || result.content_id;
  }
  
  async downloadFile(contentId: string): Promise<Blob> {
    const response = await fetch(
      `${this.apiUrl}/api/v1/retrieve/${contentId}`
    );
    
    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }
    
    return await response.blob();
  }
  
  async getMetadata(contentId: string): Promise<any> {
    const response = await fetch(
      `${this.apiUrl}/api/v1/metadata/${contentId}`
    );
    
    if (!response.ok) {
      return null;
    }
    
    return await response.json();
  }
}

// In Maya Wallet component
import { PakitClient } from '@/lib/pakit-client';

export default function FileUpload() {
  const pakit = new PakitClient();
  
  const handleUpload = async (file: File) => {
    try {
      const contentId = await pakit.uploadFile(file, {
        uploader: walletAddress,
        timestamp: Date.now(),
      });
      
      console.log('File uploaded:', contentId);
      
      // Optionally register on blockchain
      await registerFileOnChain(contentId);
      
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };
  
  return (
    <input
      type="file"
      onChange={(e) => e.target.files && handleUpload(e.target.files[0])}
    />
  );
}
```

## Starting Pakit Services

### Development Mode (All-in-One)

```bash
cd pakit

# Start Pakit API server + IPFS + Redis
python api_server.py
```

### Production Mode (Separate Services)

```bash
# Terminal 1: Start IPFS daemon
ipfs init
ipfs daemon

# Terminal 2: Start Redis (for caching)
redis-server

# Terminal 3: Start Pakit API
cd pakit
source ../.venv/bin/activate
python api_server.py --host 0.0.0.0 --port 8000

# Terminal 4: Start P2P node (optional)
python -m pakit.node.p2p_node
```

## Testing Integration

### Test Pakit → Blockchain

```bash
# Terminal 1: Start BelizeChain node
./target/release/belizechain-node --dev --tmp

# Terminal 2: Test storage proof
cd pakit
pytest tests/integration/test_blockchain.py -v
```

### Test Nawal → Pakit

```bash
cd nawal
pytest tests/test_storage_integration.py -v
```

### Test Kinich → Pakit

```bash
cd kinich
pytest tests/test_quantum_storage.py -v
```

## API Endpoints

Pakit exposes a REST API:

```
POST   /api/v1/upload           - Upload file
GET    /api/v1/retrieve/:cid    - Download file
GET    /api/v1/metadata/:cid    - Get metadata
POST   /api/v1/pin/:cid         - Pin content
GET    /api/v1/list             - List stored content
DELETE /api/v1/delete/:cid      - Delete content
GET    /api/v1/stats            - Storage statistics
```

## Environment Variables

```bash
# Pakit Configuration
PAKIT_STORAGE_DIR=./pakit_storage
PAKIT_IPFS_GATEWAY=http://localhost:5001
PAKIT_REDIS_URL=redis://localhost:6379
PAKIT_ENABLE_ARWEAVE=false

# Blockchain Connection
BELIZECHAIN_NODE_URL=ws://localhost:9944
BELIZECHAIN_KEYPAIR_URI=//Alice

# ML Optimization
PAKIT_ENABLE_ML=true
PAKIT_ML_DEVICE=cuda  # or 'cpu'
```

## Troubleshooting

### IPFS not connecting
```bash
# Check IPFS daemon
ipfs swarm peers

# Restart daemon
ipfs shutdown && ipfs daemon
```

### Blockchain connection failed
```bash
# Check node is running
curl -H "Content-Type: application/json" \
  -d '{"id":1, "jsonrpc":"2.0", "method": "system_health"}' \
  http://localhost:9944

# Verify keypair
subkey inspect //Alice
```

### Upload fails
```bash
# Check Pakit logs
tail -f pakit/logs/pakit.log

# Verify storage space
df -h

# Check IPFS storage
ipfs repo stat
```

## Next Steps

- **Blockchain**: Add more pallets that need storage (Payroll, Governance docs)
- **Nawal**: Implement dataset versioning with Pakit
- **Kinich**: Add quantum circuit visualization storage
- **UI**: Build file browser in Maya Wallet
- **Monitoring**: Add Prometheus metrics for Pakit storage
