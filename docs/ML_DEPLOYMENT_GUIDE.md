# ML Model Deployment Guide

## Overview

This guide covers deploying ML models to production with the Pakit model serving infrastructure. The deployment process ensures zero downtime, version control, and comprehensive monitoring.

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Model Server                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Version Manager   │  ModelRegistry  │  Monitor   │ │
│  └───────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────┐ │
│  │   Compression  │  Dedup  │  Prefetch  │  Peer    │ │
│  │   Predictor    │  Opt    │  Engine    │  Selector│ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              MLDAGIntegration                           │
│  (Circuit breakers, A/B testing, fallbacks)            │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
                    DAG Backend
```

## Initial Deployment

### 1. Prepare Models

```bash
# Train models (see ML_TRAINING_GUIDE.md)
python scripts/train_compression_predictor.py
python scripts/train_dedup_optimizer.py
python scripts/train_prefetch_engine.py
python scripts/train_peer_selector.py
python scripts/train_network_optimizer.py

# Models saved to ./models/
ls -lh models/
# compression_predictor.pt
# dedup_optimizer.pkl
# prefetch_engine.pt
# peer_selector.pkl
# network_optimizer.pkl
```

### 2. Start Model Server

```python
from pakit.ml.serving import ModelServer

# Initialize server
server = ModelServer(model_dir='./models')

# Start serving
server.start()

# Server loads all models from disk
# Output:
# INFO: Loading models from ./models
# INFO: Loaded compression_predictor v1.0.0
# INFO: Loaded dedup_optimizer v1.0.0
# INFO: Loaded prefetch_engine v1.0.0
# INFO: Loaded peer_selector v1.0.0
# INFO: Loaded network_optimizer v1.0.0
# INFO: Model server started with 5 models
```

### 3. Enable ML Integration

```python
from pakit.ml.integration import MLDAGIntegration

# Enable ML optimization layer
integration = MLDAGIntegration(enable_ml=True)

# ML models now provide hints to DAG backend
print(f"ML enabled: {integration.enable_ml}")
print(f"Models loaded: {len(integration.registry.models)}")
```

## Gradual Rollout (A/B Testing)

### Phase 1: Small Percentage (5%)

```python
from pakit.ml.integration import ABTestFramework

ab_test = ABTestFramework()

# Create experiment with 5% traffic to ML
ab_test.create_experiment(
    experiment_name='ml_compression_rollout',
    traffic_split=0.05  # 5% ML, 95% baseline
)

# In request handler
def handle_request(user_id, block_hash, block_size):
    variant = ab_test.get_variant('ml_compression_rollout', user_id)
    
    if variant.use_ml:
        # ML-optimized path
        hint = integration.get_compression_hint(block_hash, block_size, ...)
    else:
        # Baseline path
        hint = None
    
    # DAG backend processes
    result = dag_backend.process(block_hash, hint=hint)
    
    # Record metrics
    ab_test.record_result(
        'ml_compression_rollout',
        user_id,
        success=result.success,
        latency_ms=result.latency_ms
    )
    
    return result
```

### Phase 2: Monitor Results

```python
# Check results after 1-2 days
results = ab_test.get_results('ml_compression_rollout')

print("Control (baseline):")
print(f"  Success rate: {results['control']['success_rate']:.1%}")
print(f"  Avg latency: {results['control']['avg_latency_ms']:.1f}ms")

print("\nTreatment (ML):")
print(f"  Success rate: {results['treatment']['success_rate']:.1%}")
print(f"  Avg latency: {results['treatment']['avg_latency_ms']:.1f}ms")

# Statistical significance
sig = ab_test.compute_statistical_significance(
    'ml_compression_rollout',
    metric='avg_latency_ms'
)

print(f"\nLatency lift: {sig['lift_percentage']:.1f}%")
print(f"Significant: {sig['is_significant']}")

# Get recommendation
recommendation = ab_test.get_recommendation('ml_compression_rollout')
print(f"Recommendation: {recommendation}")  # 'rollout', 'rollback', or 'continue'
```

### Phase 3: Increase Traffic (if successful)

```python
# Increase to 25%
ab_test.create_experiment(
    experiment_name='ml_compression_rollout_v2',
    traffic_split=0.25
)

# Monitor again...

# Increase to 50%
ab_test.create_experiment(
    experiment_name='ml_compression_rollout_v3',
    traffic_split=0.50
)

# Finally, 100%
integration.enable_ml = True
```

## Model Updates (Hot-Swapping)

### Zero-Downtime Update

```python
from pakit.ml.serving import ModelServer

server = ModelServer(model_dir='./models')
server.start()

# Train new version
# (see ML_TRAINING_GUIDE.md)

# Hot-swap to new version (NO DOWNTIME)
server.hot_swap_model(
    model_name='compression_predictor',
    new_model_path='./models/compression_predictor_v2.pt',
    new_version='2.0.0'
)

# Server continues serving during swap
# Old requests: v1.0.0
# New requests: v2.0.0 (after swap completes)

# Output:
# INFO: Hot-swapping model 'compression_predictor' to version 2.0.0
# INFO: Successfully swapped 'compression_predictor' to v2.0.0
```

### Rollback if Needed

```python
# Monitor metrics after deployment
from pakit.ml.serving import ModelMonitor

monitor = ModelMonitor()

# Check new version performance
metrics = monitor.get_metrics('compression_predictor')

if metrics['success_rate'] < 0.95:  # Below threshold
    print("WARNING: Model performance degraded!")
    
    # Rollback to previous version
    server.rollback_model('compression_predictor')
    
    # Output:
    # INFO: Rolling back model 'compression_predictor'
    # INFO: Successfully swapped 'compression_predictor' to v1.0.0
```

## Monitoring

### Health Checks

```python
# Check server health
health = server.health_check()

print(f"Status: {health['status']}")
print(f"Models loaded: {health['models_loaded']}")
print(f"Models enabled: {health['models_enabled']}")

# Per-model health
for model_name, model_health in health['models'].items():
    print(f"\n{model_name}:")
    print(f"  Enabled: {model_health['enabled']}")
    print(f"  Predictions: {model_health['predictions']:,}")
    print(f"  Success rate: {model_health['success_rate']:.1%}")
    print(f"  Avg latency: {model_health['avg_latency_ms']:.2f}ms")
```

### Prometheus Metrics

```python
# Export metrics for Prometheus
metrics_text = monitor.export_prometheus_metrics()

# Serve on /metrics endpoint
# Example metrics:
# pakit_ml_predictions_total{model="compression_predictor"} 12534
# pakit_ml_success_rate{model="compression_predictor"} 0.95
# pakit_ml_latency_ms{model="compression_predictor",quantile="avg"} 0.5
# pakit_ml_latency_ms{model="compression_predictor",quantile="0.95"} 1.2
```

### Grafana Dashboard

```yaml
# Example Grafana dashboard config
dashboard:
  panels:
    - title: "ML Model Success Rate"
      query: "pakit_ml_success_rate"
      thresholds:
        - value: 0.95
          color: "green"
        - value: 0.90
          color: "yellow"
        - value: 0.0
          color: "red"
    
    - title: "ML Model Latency (P95)"
      query: "pakit_ml_latency_ms{quantile=\"0.95\"}"
      unit: "ms"
    
    - title: "ML Predictions per Second"
      query: "rate(pakit_ml_predictions_total[1m])"
```

### Alerts

```python
# Configure alerts
from pakit.ml.serving import ModelMonitor

monitor = ModelMonitor()

# Set thresholds
monitor.latency_threshold_ms = 5.0  # Alert if >5ms
monitor.error_rate_threshold = 0.05  # Alert if >5% errors

# Get recent alerts
alerts = monitor.get_alerts(limit=10)

for alert in alerts:
    print(f"[{alert['timestamp']}] {alert['model']}: {alert['message']}")

# Example alerts:
# [2026-01-27 10:30:45] compression_predictor: Latency 7.2ms exceeds threshold 5.0ms
# [2026-01-27 10:31:12] dedup_optimizer: Error rate 6.5% exceeds threshold 5.0%
```

## Production Best Practices

### 1. Model Validation

```python
# Before deploying, validate on test set
from pakit.ml.training import ModelEvaluator

evaluator = ModelEvaluator()
test_metrics = evaluator.evaluate(model, test_data)

# Require minimum accuracy
assert test_metrics['accuracy'] >= 0.80, "Model accuracy too low!"
assert test_metrics['f1'] >= 0.75, "Model F1 score too low!"

# Require acceptable latency
import time
latencies = []
for _ in range(100):
    start = time.time()
    model.predict(test_features)
    latencies.append((time.time() - start) * 1000)

p95_latency = sorted(latencies)[95]
assert p95_latency < 10.0, "Model latency too high!"

print("✅ Model validation passed")
```

### 2. Version Control

```python
from pakit.ml.serving import VersionManager

version_mgr = VersionManager(base_dir='./models')

# Register new version
version_mgr.register_version(
    model_name='compression_predictor',
    version='2.0.0',
    model_path='./models/compression_predictor_v2.pt',
    metadata={
        'accuracy': 0.85,
        'training_date': '2026-01-27',
        'training_samples': 50000,
        'git_commit': 'abc123def',
    }
)

# List all versions
versions = version_mgr.list_versions('compression_predictor')
for v in versions:
    print(f"v{v['version']} - deployed {v['deployed_at']}")

# Cleanup old versions (keep latest 3)
deleted = version_mgr.cleanup_old_versions('compression_predictor', keep_latest=3)
print(f"Deleted {deleted} old versions")
```

### 3. Graceful Degradation

```python
from pakit.ml.integration.fallback import FallbackManager

fallback_mgr = FallbackManager()

# Check model health before using
if fallback_mgr.is_healthy('compression_predictor'):
    prediction = model.predict(features)
else:
    # Use fallback heuristic
    prediction = fallback_heuristic(features)
    print("⚠️ Using fallback (model unhealthy)")

# Get health details
health = fallback_mgr.get_model_health('compression_predictor')
print(f"State: {health['state']}")  # CLOSED, OPEN, or HALF_OPEN
print(f"Success rate: {health['success_rate']:.1%}")
print(f"Recent failures: {health['recent_failures']}")
```

### 4. Canary Deployment

```python
# Deploy to canary nodes first (e.g., 1% of fleet)
# Monitor for 24 hours

# If successful, deploy to production
# If failures detected, rollback canary
```

### 5. Load Testing

```python
import time
import concurrent.futures

def load_test_model(server, num_requests=10000):
    """Load test model server."""
    
    start = time.time()
    errors = 0
    
    def make_request(i):
        try:
            result = server.predict('compression_predictor', {
                'block_size_log': 4.0,
                'block_depth_log': 2.0,
                # ... other features
            })
            return 1
        except Exception as e:
            return 0
    
    # Concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        results = list(executor.map(make_request, range(num_requests)))
    
    elapsed = time.time() - start
    success_rate = sum(results) / len(results)
    throughput = num_requests / elapsed
    
    print(f"Load Test Results:")
    print(f"  Requests: {num_requests:,}")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Throughput: {throughput:.0f} req/sec")
    print(f"  Avg latency: {elapsed/num_requests*1000:.2f}ms")

# Run load test
load_test_model(server)
```

## Deployment Checklist

- [ ] Models trained and validated on test set
- [ ] Model artifacts saved and versioned
- [ ] Model server started and health check passing
- [ ] A/B test configured (start with 5% traffic)
- [ ] Monitoring and alerts configured
- [ ] Grafana dashboards set up
- [ ] Rollback procedure tested
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Team notified of deployment

## Troubleshooting

### Model Server Won't Start
```python
# Check logs
server = ModelServer(model_dir='./models')
try:
    server.start()
except Exception as e:
    print(f"Error: {e}")
    # Common issues:
    # - Missing model files
    # - Corrupted model checkpoints
    # - PyTorch version mismatch
    # - Out of memory
```

### High Latency in Production
```python
# Check inference times
metrics = monitor.get_metrics('compression_predictor')
print(f"P95 latency: {metrics['latency_p95']:.2f}ms")

# If too high:
# 1. Use GPU instead of CPU
# 2. Reduce batch size (for batch predictions)
# 3. Optimize model (pruning, quantization)
# 4. Use TorchScript compilation
```

### Low Accuracy in Production
```python
# Check data distribution shift
from pakit.ml.telemetry import TelemetryCollector

collector = TelemetryCollector()
recent_events = collector.get_events(limit=1000)

# Compare with training data
# Look for:
# - Different block size distributions
# - Different content types
# - Different access patterns

# If significant shift detected, retrain model
```

### Circuit Breaker Open
```python
# Check why model is failing
health = fallback_mgr.get_model_health('compression_predictor')
print(f"Failures: {health['failures']}")
print(f"Recent failures: {health['recent_failures']}")

# Manual reset (if issue fixed)
fallback_mgr.reset_circuit_breaker('compression_predictor')
```

## Next Steps

- **Architecture**: See [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md)
- **Training**: See [ML_TRAINING_GUIDE.md](ML_TRAINING_GUIDE.md)
