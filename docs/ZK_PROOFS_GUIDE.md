# Zero-Knowledge Storage Proofs Developer Guide

## Overview

Pakit integrates kinich-quantum's ZK proof system for privacy-preserving storage verification. This allows you to prove storage of data without revealing the data itself.

## Supported Proof Systems

- **Groth16**: Fast verification, compact proofs (~200 bytes)
- **PLONK**: Universal trusted setup, medium proof size (~400 bytes)
- **zkSTARK**: No trusted setup, larger proofs (~5KB), quantum-resistant

## Quick Start

### Basic Usage

```python
from core.zk_storage_proofs import StorageProofGenerator

# Create proof generator
generator = StorageProofGenerator()

# Generate storage proof
proof = generator.generate_storage_proof(
    content_id="abc123...",  # Content hash
    data_size=1024,          # Size in bytes
    storage_location="dag",  # Backend: dag/ipfs/arweave
    proof_type="groth16"     # Proof system
)

# Proof structure
print(proof)
# {
#   "proof": "0x1234...",
#   "type": "groth16",
#   "content_id": "abc123...",
#   "data_size": 1024,
#   "storage_location": "dag",
#   "timestamp": 1234567890,
#   "public_inputs": {...}
# }

# Verify proof
is_valid = generator.verify_proof(proof)
print(f"Proof valid: {is_valid}")
```

### Batch Proofs (More Efficient)

```python
# Generate batch proof for multiple content items
content_list = [
    {"content_id": "hash1", "data_size": 1024},
    {"content_id": "hash2", "data_size": 2048},
    {"content_id": "hash3", "data_size": 512},
]

batch_proof = generator.generate_batch_proof(
    content_list=content_list,
    storage_location="dag",
    proof_type="groth16"
)

# Batch proof is more efficient than individual proofs
print(f"Batch size: {batch_proof['batch_size']}")
print(f"Proof type: {batch_proof['type']}")
```

## Integration with Storage Engine

The storage engine automatically generates ZK proofs when storing data:

```python
from core.storage_engine import PakitStorageEngine

# Create engine with ZK proofs enabled
engine = PakitStorageEngine(
    enable_blockchain_proofs=True  # Enables ZK proof generation
)

# Store data - ZK proof generated automatically
data = b"Confidential data"
content_id = engine.store(data)

# ZK proof is generated in background and logged
# Check logs: "Generated ZK storage proof for abc123... (type: groth16)"
```

## Submitting Proofs to Blockchain

Submit ZK proofs to BelizeChain's Mesh pallet:

```python
from blockchain.storage_proof_connector import StorageProofConnector
from core.zk_storage_proofs import StorageProofGenerator

# Initialize blockchain connector
connector = StorageProofConnector(
    node_url="ws://localhost:9944",
    mock_mode=False  # Connect to real blockchain
)
await connector.connect()

# Generate ZK proof
generator = StorageProofGenerator()
proof = generator.generate_storage_proof(
    content_id="content_hash_abc123",
    data_size=2048,
    storage_location="dag",
    proof_type="groth16"
)

# Submit to Mesh pallet
success = await connector.submit_storage_zk_proof(
    content_id="content_hash_abc123",
    zk_proof=proof,
    proof_type="groth16"
)

if success:
    print("✅ ZK proof submitted to blockchain")
else:
    print("❌ Failed to submit ZK proof")

await connector.disconnect()
```

## Choosing a Proof System

### Groth16 (Default: Recommended)

**Pros:**
- Fastest verification (~5ms)
- Smallest proof size (~200 bytes)
- Well-tested and widely used

**Cons:**
- Requires trusted setup per circuit
- Setup ceremony needed for new circuits

**Use When:**
- You need fast verification
- Proof size matters (blockchain storage)
- Standard storage verification (not custom)

```python
proof = generator.generate_storage_proof(
    content_id="hash",
    data_size=1024,
    storage_location="dag",
    proof_type="groth16"  # Default
)
```

### PLONK

**Pros:**
- Universal trusted setup (one-time ceremony)
- More flexible than Groth16
- Medium proof size (~400 bytes)

**Cons:**
- Slower verification (~15ms)
- Slightly larger proofs

**Use When:**
- You want universal setup
- Need circuit flexibility
- Acceptable verification time

```python
proof = generator.generate_storage_proof(
    content_id="hash",
    data_size=1024,
    storage_location="dag",
    proof_type="plonk"
)
```

### zkSTARK

**Pros:**
- No trusted setup required
- Quantum-resistant
- Transparent (no secret randomness)

**Cons:**
- Larger proof size (~5KB)
- Slower verification (~50ms)

**Use When:**
- Quantum resistance required
- No trusted setup possible
- Proof size not critical

```python
proof = generator.generate_storage_proof(
    content_id="hash",
    data_size=1024,
    storage_location="dag",
    proof_type="stark"
)
```

## Configuration

### Environment Variables

```bash
# Enable ZK proofs
BLOCKCHAIN_PROOFS_ENABLED=true

# Default proof system
ZK_PROOF_TYPE=groth16

# Batch proof threshold (items)
ZK_BATCH_THRESHOLD=10

# Auto-submit to blockchain
ZK_AUTO_SUBMIT=true

# Blockchain node URL
BLOCKCHAIN_NODE_URL=ws://localhost:9944
```

### Python Configuration

```python
from core.storage_engine import PakitStorageEngine

engine = PakitStorageEngine(
    enable_blockchain_proofs=True,  # Enable ZK proofs
    enable_dag=True
)

# ZK proof generator automatically initialized
if engine.zk_proof_generator:
    print("ZK proofs enabled")
else:
    print("ZK proofs disabled (kinich-quantum not installed)")
```

## Performance Considerations

### Proof Generation Time

| Proof Type | Generation Time | Verification Time | Proof Size |
|-----------|----------------|------------------|-----------|
| Groth16   | ~100ms        | ~5ms             | ~200 bytes |
| PLONK     | ~150ms        | ~15ms            | ~400 bytes |
| zkSTARK   | ~500ms        | ~50ms            | ~5KB      |

### Batch vs Individual

For 100 proofs:
- **Individual**: 100 × 100ms = 10 seconds
- **Batch**: 1 × 500ms = 0.5 seconds

💡 **Recommendation**: Use batch proofs for > 5 items

### Memory Usage

- **Groth16**: ~50MB per proof
- **PLONK**: ~100MB per proof
- **zkSTARK**: ~200MB per proof

## Advanced Usage

### Custom Circuits (Advanced)

```python
# Define custom verification circuit
circuit_definition = {
    "inputs": ["content_hash", "size", "location"],
    "constraints": [
        # Custom ZK constraints
    ]
}

# Generate proof with custom circuit
proof = generator.generate_storage_proof(
    content_id="hash",
    data_size=1024,
    storage_location="dag",
    proof_type="groth16",
    custom_circuit=circuit_definition
)
```

### Proof Aggregation

```python
# Aggregate multiple proofs into one
proofs = [
    generator.generate_storage_proof("hash1", 1024, "dag"),
    generator.generate_storage_proof("hash2", 2048, "dag"),
    generator.generate_storage_proof("hash3", 512, "dag"),
]

aggregated_proof = generator.aggregate_proofs(proofs)

# Single verification for all proofs
is_valid = generator.verify_proof(aggregated_proof)
```

### Recursive Proofs

```python
# Prove that you have proof of storage
recursive_proof = generator.generate_recursive_proof(
    base_proof=proof,
    recursion_depth=2
)

# Useful for proving long-term storage history
```

## Monitoring and Debugging

### Check Proof Statistics

```python
stats = generator.get_proof_stats()

print(f"Proofs generated: {stats['proofs_generated']}")
print(f"Proofs verified: {stats['proofs_verified']}")
print(f"Batch proofs: {stats['batch_proofs']}")
print(f"Failed verifications: {stats['failed_verifications']}")
```

### Debugging Failed Proofs

```python
proof = generator.generate_storage_proof(
    content_id="hash",
    data_size=1024,
    storage_location="dag"
)

is_valid = generator.verify_proof(proof)

if not is_valid:
    # Check proof structure
    print(f"Proof type: {proof['type']}")
    print(f"Public inputs: {proof['public_inputs']}")
    
    # Verify individual components
    # (implementation-specific debugging)
```

## Security Considerations

1. **Trusted Setup**: Groth16 requires secure setup ceremony
2. **Proof Replay**: Include timestamp to prevent replay attacks
3. **Public Inputs**: Ensure public inputs match actual storage
4. **Verification**: Always verify proofs before trusting
5. **Quantum Resistance**: Use zkSTARK if quantum threats are a concern

## Testing

```bash
# Run ZK proof tests
pytest tests/test_zk_proofs.py -v

# Benchmark proof generation
pytest tests/test_zk_proofs.py::TestZKProofPerformance -v

# Integration tests
pytest tests/test_integration.py -k zk
```

## Troubleshooting

### "kinich-quantum not installed"

```bash
pip install kinich-quantum[zk]>=1.0.0
```

### Slow Proof Generation

- Use batch proofs for multiple items
- Consider Groth16 instead of zkSTARK
- Check system resources (CPU, RAM)

### Verification Failures

- Ensure proof hasn't been tampered with
- Check public inputs match storage
- Verify timestamp is recent
- Ensure same proof system for gen/verify

### Blockchain Submission Fails

- Check blockchain connection
- Verify account has funds for gas
- Ensure Mesh pallet supports proof type
- Check extrinsic parameters

## API Reference

### StorageProofGenerator

```python
class StorageProofGenerator:
    def generate_storage_proof(
        content_id: str,
        data_size: int,
        storage_location: str,
        proof_type: str = "groth16"
    ) -> Dict[str, Any]
    
    def generate_batch_proof(
        content_list: List[Dict],
        storage_location: str,
        proof_type: str = "groth16"
    ) -> Dict[str, Any]
    
    def verify_proof(proof: Dict[str, Any]) -> bool
    
    def get_proof_stats() -> Dict[str, Any]
```

## Further Reading

- [kinich-quantum Documentation](https://github.com/BelizeChain/kinich-quantum)
- [ZK Proofs Explained](https://github.com/BelizeChain/kinich-quantum/blob/main/docs/ZK_INTRODUCTION.md)
- [BelizeChain Mesh Pallet](https://github.com/BelizeChain/belizechain/blob/main/pallets/mesh/README.md)
- [Storage Proof Specification](./STORAGE_PROOF_SPEC.md)
