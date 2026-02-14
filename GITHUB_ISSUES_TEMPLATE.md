# GitHub Issues Template for Pakit Integrations

**Copy these issue templates to create tracking issues for each integration task.**

---

## Issue #1: Mesh Networking Integration (nawal-ai)

**Title**: Integrate nawal-ai Mesh Networking for P2P Communication

**Labels**: `enhancement`, `high-priority`, `integration`, `mesh-networking`

**Assignees**: [TBD]

**Description**:

Integrate nawal-ai's production-ready mesh networking system (`MeshNetworkClient`) to replace/augment DHT-based peer discovery with gossip protocol communication.

### Objective

Enable direct P2P communication between Pakit storage nodes using mesh networking, providing:
- Faster peer discovery (< 5 sec vs ~30 sec DHT)
- Byzantine-resistant peer filtering
- Ed25519 cryptographic signing
- Reduced network overhead

### Tasks

- [ ] Add `nawal-ai` mesh dependency to `requirements.txt`
- [ ] Create `p2p/mesh/` module structure
- [ ] Create `p2p/mesh/mesh_client.py` wrapper for `MeshNetworkClient`
- [ ] Implement storage availability announcements
- [ ] Implement peer-based block discovery
- [ ] Update `p2p/node.py` to use mesh for peer discovery
- [ ] Add mesh health monitoring to API endpoints
- [ ] Write unit tests for mesh integration
- [ ] Write integration tests for P2P mesh communication
- [ ] Update documentation

### Acceptance Criteria

- [ ] Peer discovery completes in < 5 seconds
- [ ] Storage availability announcements propagate via gossip
- [ ] Block requests can be fulfilled via mesh peers
- [ ] Mesh network health endpoint returns status
- [ ] All tests pass with 100% coverage
- [ ] Documentation updated in README and integration guide

### Dependencies

- **Upstream**: nawal-ai v1.1.0+ ([repo](https://github.com/BelizeChain/nawal-ai))
- **Docs**: [Mesh Networking Guide](https://github.com/BelizeChain/nawal-ai/blob/main/docs/guides/mesh-networking.md)

### Estimated Effort

**3 days** (1 developer)

### References

- [INTEGRATION_EVALUATION.md](./INTEGRATION_EVALUATION.md#1-mesh-networking-integration-nawal-ai)
- [INTEGRATION_QUICK_REF.md](./INTEGRATION_QUICK_REF.md#1-mesh-networking-week-1)

---

## Issue #2: Zero-Knowledge Storage Proofs (kinich-quantum)

**Title**: Implement ZK Proofs for Privacy-Preserving Storage Verification

**Labels**: `enhancement`, `high-priority`, `integration`, `zk-proofs`, `privacy`

**Assignees**: [TBD]

**Description**:

Integrate kinich-quantum's zero-knowledge proof system to enable privacy-preserving storage verification using zkSNARK (Groth16/PLONK) and zkSTARK.

### Objective

Generate zero-knowledge proofs for storage operations to:
- Prove data was stored correctly without revealing data
- Prove block availability without revealing locations
- Reduce blockchain storage (proofs ~200 bytes vs ~4KB Merkle proofs)
- Enable privacy-preserving replication verification

### Tasks

- [ ] Add `kinich-quantum` ZK dependency to `requirements.txt`
- [ ] Create `core/zk_storage_proofs.py` module
- [ ] Implement `StorageProofGenerator` class
- [ ] Implement single-block proof generation (zkSNARK Groth16)
- [ ] Implement batch proof generation (zkSTARK)
- [ ] Integrate ZK proofs in `core/storage_engine.py`
- [ ] Update blockchain connector to submit ZK proofs
- [ ] Add proof verification in API endpoints
- [ ] Write unit tests for ZK proof generation
- [ ] Write integration tests for end-to-end storage + proof
- [ ] Benchmark proof generation performance
- [ ] Update documentation

### Acceptance Criteria

- [ ] Single-block ZK proof generated in < 200ms
- [ ] Batch ZK proof for 100 blocks < 1KB
- [ ] Proof verification succeeds for valid storage operations
- [ ] Blockchain connector submits proofs successfully
- [ ] All tests pass with 100% coverage
- [ ] Performance benchmarks meet targets

### Dependencies

- **Upstream**: kinich-quantum v1.0.0+ ([repo](https://github.com/BelizeChain/kinich-quantum))
- **Docs**: [zk_proofs.py](https://github.com/BelizeChain/kinich-quantum/blob/main/security/zk_proofs.py)

### Estimated Effort

**5 days** (1 developer)

### References

- [INTEGRATION_EVALUATION.md](./INTEGRATION_EVALUATION.md#2-zero-knowledge-storage-proofs-kinich-quantum)
- [INTEGRATION_QUICK_REF.md](./INTEGRATION_QUICK_REF.md#2-zk-storage-proofs-week-1-2)

---

## Issue #3: LoRa Mesh Integration (belizechain)

**Title**: Add LoRa Mesh Bridge for Off-Grid Storage Access

**Labels**: `enhancement`, `medium-priority`, `integration`, `lora-mesh`, `resilience`

**Assignees**: [TBD]

**Description**:

Integrate BelizeChain's Mesh pallet to enable LoRa mesh networking for off-grid storage access in rural areas and emergency scenarios.

### Objective

Enable low-bandwidth, off-grid storage operations via Meshtastic LoRa:
- Request small files (< 1KB) over LoRa mesh
- Sync DAG index updates over mesh network
- Emergency data broadcast during network outages
- Rural area storage access without internet

### Tasks

- [ ] Create `p2p/mesh/lora_bridge.py` module
- [ ] Implement Mesh pallet RPC calls in blockchain connector
- [ ] Implement DAG index broadcast over LoRa
- [ ] Implement small file request/response over LoRa
- [ ] Add emergency data sync capability
- [ ] Add mesh network health monitoring
- [ ] Add configuration for Meshtastic device
- [ ] Write unit tests for LoRa bridge
- [ ] Write integration tests for off-grid scenarios
- [ ] Update documentation with off-grid usage guide

### Acceptance Criteria

- [ ] DAG index updates broadcast over LoRa mesh
- [ ] Small files (< 1KB) can be retrieved via LoRa
- [ ] Emergency sync works during network outages
- [ ] Mesh health monitoring shows LoRa status
- [ ] Configuration documented for Meshtastic devices
- [ ] All tests pass with coverage

### Dependencies

- **Upstream**: belizechain Mesh pallet ([repo](https://github.com/BelizeChain/belizechain))
- **Hardware**: Meshtastic LoRa device (optional for testing)

### Estimated Effort

**5 days** (1 developer)

### References

- [INTEGRATION_EVALUATION.md](./INTEGRATION_EVALUATION.md#3-mesh-pallet-integration-belizechain)
- [INTEGRATION_QUICK_REF.md](./INTEGRATION_QUICK_REF.md#4-lora-mesh-integration-week-3)

---

## Issue #4: Enhanced Blockchain Connector (16-Pallet Upgrade)

**Title**: Upgrade Blockchain Connector for 16-Pallet BelizeChain

**Labels**: `enhancement`, `important`, `integration`, `blockchain`, `revenue`

**Assignees**: [TBD]

**Description**:

Upgrade blockchain connector to support BelizeChain's 16 custom pallets, specifically integrating Economy, BNS, Contracts, and Mesh pallets for enhanced functionality and revenue generation.

### Objective

Enable Pakit to leverage full BelizeChain capabilities:
- Accept payments in DALLA/bBZD (Economy pallet)
- Host .bz domains on IPFS (BNS pallet)
- Provide storage for smart contracts (Contracts pallet)
- Coordinate via mesh network (Mesh pallet)
- Submit ZK proofs (Quantum pallet - enhanced)

### Tasks

#### Economy Pallet Integration
- [ ] Create `blockchain/economy_integration.py`
- [ ] Implement storage provider registration
- [ ] Implement payment acceptance (DALLA/bBZD)
- [ ] Add pricing configuration

#### BNS Pallet Integration
- [ ] Create `blockchain/bns_integration.py`
- [ ] Implement .bz domain registration
- [ ] Implement IPFS hosting for domains
- [ ] Add domain marketplace integration

#### Contracts Pallet Integration
- [ ] Create `blockchain/contracts_integration.py`
- [ ] Implement smart contract data storage
- [ ] Add contract-based access control

#### Mesh Pallet Integration
- [ ] Update mesh bridge to use Mesh pallet
- [ ] Implement off-grid coordination

#### General Updates
- [ ] Update `blockchain/storage_proof_connector.py` for multi-pallet
- [ ] Add ZK proof submission enhancement
- [ ] Update configuration for pallet selection
- [ ] Write unit tests for each pallet integration
- [ ] Write integration tests for cross-pallet workflows
- [ ] Update API documentation
- [ ] Update deployment guide

### Acceptance Criteria

- [ ] Storage provider can register and accept DALLA/bBZD payments
- [ ] .bz domains can be hosted on Pakit IPFS
- [ ] Smart contracts can store data on Pakit
- [ ] Mesh pallet coordinates off-grid sync
- [ ] ZK proofs submitted to Quantum pallet
- [ ] All pallet integrations tested
- [ ] Documentation updated

### Dependencies

- **Upstream**: belizechain 16 pallets ([repo](https://github.com/BelizeChain/belizechain))
- **Requires**: Issues #1 (Mesh), #2 (ZK Proofs) completed

### Estimated Effort

**6 days** (1 developer)

### References

- [INTEGRATION_EVALUATION.md](./INTEGRATION_EVALUATION.md#4-enhanced-blockchain-connector-16-pallet-upgrade)
- [INTEGRATION_QUICK_REF.md](./INTEGRATION_QUICK_REF.md#3-blockchain-connector-upgrade-week-2)
- [BelizeChain README](https://github.com/BelizeChain/belizechain/blob/main/README.md)

---

## Issue #5: Integration Testing & Documentation

**Title**: Comprehensive Testing and Documentation for Ecosystem Integrations

**Labels**: `testing`, `documentation`, `high-priority`

**Assignees**: [TBD]

**Description**:

Ensure all integrations (#1-#4) are thoroughly tested, documented, and benchmarked before production release.

### Objective

Validate all integrations work correctly individually and together, with comprehensive documentation for users and developers.

### Tasks

#### Testing
- [ ] Write integration tests for mesh networking
- [ ] Write integration tests for ZK proof generation/verification
- [ ] Write integration tests for LoRa mesh bridge
- [ ] Write integration tests for 16-pallet blockchain
- [ ] Write cross-integration tests (mesh + ZK, etc.)
- [ ] Run performance benchmarks
- [ ] Validate 100% test coverage
- [ ] Test end-to-end scenarios

#### Documentation
- [ ] Update README.md with new features
- [ ] Add Mesh Networking user guide
- [ ] Add ZK Proof developer guide
- [ ] Add LoRa Mesh usage guide
- [ ] Update INTEGRATION_GUIDE.md
- [ ] Update architecture diagrams
- [ ] Add API documentation for new endpoints
- [ ] Update deployment guide with new configs
- [ ] Add troubleshooting section

#### Performance Validation
- [ ] Benchmark peer discovery (target: < 5 sec)
- [ ] Benchmark ZK proof generation (target: < 200ms)
- [ ] Benchmark LoRa mesh sync (target: < 30 sec)
- [ ] Benchmark blockchain integration latency
- [ ] Document performance results

### Acceptance Criteria

- [ ] All integration tests pass
- [ ] 100% code coverage maintained
- [ ] Performance benchmarks meet targets
- [ ] Documentation covers all new features
- [ ] Deployment guide updated
- [ ] Troubleshooting guide comprehensive
- [ ] API docs complete

### Dependencies

- **Requires**: Issues #1, #2, #3, #4 completed

### Estimated Effort

**5 days** (1 developer)

### References

- [INTEGRATION_EVALUATION.md](./INTEGRATION_EVALUATION.md#phase-3-testing--documentation-week-5)

---

## Milestone: Pakit v2.0.0 - Full Ecosystem Integration

**Due Date**: March 21, 2026 (5 weeks from start)

**Description**: Release Pakit v2.0.0 with complete BelizeChain ecosystem integration

**Issues**:
- #1: Mesh Networking Integration
- #2: Zero-Knowledge Storage Proofs
- #3: LoRa Mesh Integration
- #4: Enhanced Blockchain Connector
- #5: Integration Testing & Documentation

**Success Criteria**:
- ✅ All 5 issues completed
- ✅ All tests passing (100% coverage)
- ✅ Documentation complete
- ✅ Performance benchmarks met
- ✅ Production-ready release

**Release Notes**: See [INTEGRATION_SUMMARY.md](./INTEGRATION_SUMMARY.md)

---

## Project Board Columns

Suggested GitHub Project board structure:

1. **Backlog** - All issues initially
2. **In Progress (Week 1-2)** - #1, #2 (Critical Path)
3. **In Progress (Week 3-4)** - #3, #4 (Enhanced Features)
4. **In Progress (Week 5)** - #5 (Testing & Docs)
5. **Code Review** - PRs awaiting review
6. **Testing** - Issues in QA
7. **Done** - Completed and merged

---

**Usage**: Copy each issue template above to create individual GitHub issues in the pakit-storage repository.
