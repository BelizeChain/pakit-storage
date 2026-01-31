# ML Optimization Layer Architecture

## Overview

The ML Optimization Layer sits on top of the deterministic DAG backend, providing intelligent hints for compression, deduplication, prefetching, peer selection, and network tuning. **ML never compromises cryptographic determinism** - it only suggests optimizations that the deterministic core validates and executes.

## Architecture Principle

```
┌────────────────────────────────────────────────────────────┐
│                       User Request                         │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│                  ML Optimization Layer                     │
│  ┌──────────────┬──────────────┬──────────────────────┐   │
│  │ Compression  │ Deduplication│    Prefetch Engine   │   │
│  │  Predictor   │  Optimizer   │    (LSTM)            │   │
│  └──────────────┴──────────────┴──────────────────────┘   │
│  ┌──────────────┬──────────────────────────────────────┐   │
│  │ Peer Selector│    Network Optimizer (Q-Learning)    │   │
│  │  (Bandit)    │                                       │   │
│  └──────────────┴──────────────────────────────────────┘   │
│                                                            │
│  Provides: Hints, Recommendations, Predictions            │
└───────────────────────────┬────────────────────────────────┘
                            │ (suggestions only)
                            ▼
┌────────────────────────────────────────────────────────────┐
│              Deterministic DAG Backend                     │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  • Validates ML suggestions                          │ │
│  │  • Makes final decisions                             │ │
│  │  • Maintains cryptographic guarantees                │ │
│  │  • Falls back gracefully if ML unavailable           │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────┬───────────────────────────────┘
                             │
                             ▼
                        Final Result
```

## ML Models

### 1. Compression Predictor
- **Type**: 3-layer MLP (128→64→32→4 neurons)
- **Input**: 12 features (block_size_log, entropy, content_type, etc.)
- **Output**: Algorithm ('zstd', 'lz4', 'snappy', 'none')
- **Target**: >80% accuracy, <1ms inference
- **Fallback**: Size-based heuristic

### 2. Deduplication Optimizer
- **Type**: SimHash + LSH indexing
- **Input**: Block hash, content fingerprint
- **Output**: List of similar blocks (similarity >0.85)
- **Target**: <10ms lookup, <5% false positives, 10-20% dedup improvement
- **Fallback**: Exact hash matching only

### 3. Prefetch Engine
- **Type**: 2-layer LSTM (64 hidden units)
- **Input**: Sequence of last 10 block accesses
- **Output**: Top-5 likely next blocks
- **Target**: 15-25% hit rate improvement, >60% precision@5
- **Fallback**: No prefetching (conservative)

### 4. Peer Selector
- **Type**: Thompson Sampling / Contextual Bandit
- **Input**: Available peers, block size, network conditions
- **Output**: Best peer ID
- **Target**: 10-20% latency reduction, >95% success rate
- **Fallback**: Random selection

### 5. Network Optimizer
- **Type**: Q-Learning agent
- **Input**: Network size, latency, message loss
- **Output**: Optimal (fanout, TTL) parameters
- **Target**: 15-30% bandwidth reduction, >99.5% coverage
- **Fallback**: Fixed defaults (fanout=6, TTL=10)

## Integration with DAG Backend

### Compression Example
```python
from pakit.ml.integration import MLDAGIntegration

integration = MLDAGIntegration(enable_ml=True)

# Get ML hint
algorithm_hint = integration.get_compression_hint(
    block_hash='abc123',
    block_size=10000,
    block_depth=5,
    content_entropy=7.5
)

# DAG backend validates and executes
if algorithm_hint:
    # Try ML suggestion
    compressed = compress_block(data, algorithm=algorithm_hint)
    
    # Validate compression ratio
    if len(compressed) / len(data) < 0.9:
        use_compression = algorithm_hint
    else:
        use_compression = 'none'  # Override if not beneficial
else:
    # Fallback heuristic
    use_compression = 'zstd' if block_size > 100000 else 'lz4'
```

### Key Principles
1. **ML provides hints, deterministic core decides**
2. **Always validate ML suggestions cryptographically**
3. **Graceful degradation if ML unavailable**
4. **Monitor ML accuracy and revert if degrading**

## Graceful Degradation

### Circuit Breaker Pattern
Each model has a circuit breaker that tracks failures:

- **CLOSED** (normal): Model operates normally
- **OPEN** (failing): Model disabled, fallback used (after 5 failures)
- **HALF_OPEN** (testing): Testing if model recovered

```python
from pakit.ml.integration.fallback import FallbackManager

fallback_mgr = FallbackManager()

# Check if model healthy
if fallback_mgr.is_healthy('compression_predictor'):
    prediction = model.predict(features)
else:
    # Use fallback
    prediction = fallback_heuristic(features)
```

## A/B Testing Framework

Gradual rollout with control vs treatment groups:

```python
from pakit.ml.integration.ab_test import ABTestFramework

ab_test = ABTestFramework()

# Create experiment: 50% control (no ML), 50% treatment (ML)
ab_test.create_experiment(
    'compression_optimization',
    traffic_split=0.5
)

# Get variant for user
variant = ab_test.get_variant('compression_optimization', user_id='user123')

if variant.use_ml:
    # Use ML optimization
    algorithm = ml_model.predict(features)
else:
    # Use baseline
    algorithm = baseline_heuristic(features)

# Record result
ab_test.record_result('compression_optimization', user_id, success=True, latency_ms=50)

# Check if ML is winning
recommendation = ab_test.get_recommendation('compression_optimization')
# Returns: 'rollout', 'rollback', or 'continue'
```

## Model Serving

### Production Deployment
```python
from pakit.ml.serving import ModelServer

server = ModelServer(model_dir='./models')
server.start()

# Get prediction
prediction = server.predict('compression_predictor', features)

# Hot-swap model (zero downtime)
server.hot_swap_model(
    'compression_predictor',
    new_model_path='./models/compression_v2.pt',
    new_version='2.0.0'
)

# Rollback if needed
server.rollback_model('compression_predictor')
```

### Monitoring
```python
from pakit.ml.serving import ModelMonitor

monitor = ModelMonitor()

# Record predictions
monitor.record_prediction('compression_predictor', success=True, latency_ms=0.5)

# Get metrics
metrics = monitor.get_metrics('compression_predictor')
print(f"Success rate: {metrics['success_rate']:.1%}")
print(f"P95 latency: {metrics['latency_p95']:.2f}ms")

# Export Prometheus metrics
prometheus_metrics = monitor.export_prometheus_metrics()
```

## Performance Targets

### Storage Efficiency
- **Baseline**: Deterministic compression only
- **ML Target**: 15-25% reduction
  - Compression: 5-10% improvement
  - Deduplication: 10-20% improvement

### Retrieval Latency
- **Baseline**: No prefetching, random peers
- **ML Target**: 10-20% improvement
  - Peer selection: 10-20% reduction
  - Prefetching: 15-25% hit rate

### Network Bandwidth
- **Baseline**: Fixed fanout=6, TTL=10
- **ML Target**: 15-30% reduction
  - Adaptive parameters based on network conditions

### Inference Speed
- Compression predictor: <1ms
- Deduplication lookup: <10ms
- Prefetch prediction: <10ms
- Peer selection: <1ms
- Network optimization: <1ms

## Privacy & Security

### Telemetry Privacy
All telemetry is **content-free**:
```python
# ✅ ALLOWED: Metadata only
telemetry = {
    'block_hash': sha256(content),  # Hash only
    'block_size': len(content),
    'timestamp': time.time(),
    'compression_ratio': compressed_size / original_size,
}

# ❌ FORBIDDEN: Actual content
telemetry = {
    'content': content,  # NEVER
    'decrypted_data': data,  # NEVER
}
```

### Differential Privacy
Noise added to protect individual contributions:
```python
from pakit.ml.telemetry.privacy import differential_privacy_noise

# Add Laplace noise
noisy_value = value + differential_privacy_noise(epsilon=1.0)
```

### K-Anonymity
Ensure at least k similar records before releasing:
```python
from pakit.ml.telemetry.privacy import k_anonymity_check

if k_anonymity_check(dataset, k=5):
    # Safe to use for training
    model.train(dataset)
```

## Directory Structure

```
pakit/ml/
├── __init__.py                    # Package exports
├── base_model.py                  # PakitMLModel abstract base
├── registry.py                    # ModelRegistry singleton
├── checkpoint.py                  # Model persistence
├── requirements-ml.txt            # ML dependencies
│
├── telemetry/                     # Privacy-preserving data collection
│   ├── collector.py               # Event recording
│   ├── dataset.py                 # Training data management
│   ├── features.py                # Feature extraction
│   └── privacy.py                 # Privacy guarantees
│
├── training/                      # Training infrastructure
│   ├── trainer.py                 # Model training loop
│   ├── evaluator.py               # Model evaluation
│   └── hyperparams.py             # Hyperparameter tuning
│
├── models/                        # ML models
│   ├── compression_predictor.py   # Algorithm selection
│   ├── dedup_optimizer.py         # Similarity detection
│   ├── prefetch_engine.py         # Access prediction
│   ├── peer_selector.py           # Peer ranking
│   └── network_optimizer.py       # Protocol tuning
│
├── integration/                   # DAG integration
│   ├── dag_integration.py         # Main integration
│   ├── fallback.py                # Circuit breaker
│   └── ab_test.py                 # A/B testing
│
└── serving/                       # Production serving
    ├── model_server.py            # Model server
    ├── version_manager.py         # Version control
    └── monitor.py                 # Performance monitoring
```

## Next Steps

1. **Training**: See [ML_TRAINING_GUIDE.md](ML_TRAINING_GUIDE.md)
2. **Deployment**: See [ML_DEPLOYMENT_GUIDE.md](ML_DEPLOYMENT_GUIDE.md)
3. **Testing**: Run `pytest tests/ml/ -v`
