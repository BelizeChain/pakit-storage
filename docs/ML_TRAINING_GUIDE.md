# ML Model Training Guide

## Overview

This guide covers how to train ML models for the Pakit optimization layer. All models learn from privacy-preserving telemetry collected during normal operations.

## Prerequisites

```bash
# Install ML dependencies
pip install -r pakit/ml/requirements-ml.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check GPU availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Data Collection

### 1. Enable Telemetry
```python
from pakit.ml.telemetry import TelemetryCollector

collector = TelemetryCollector(db_path='./telemetry.db')

# Record block operations (content-free!)
collector.record(
    event_type='BLOCK_STORED',
    block_hash='abc123...',
    block_size=10000,
    block_depth=5,
    compression_algorithm='zstd',
    compression_ratio=0.5,
    # NO CONTENT - privacy preserved
)
```

### 2. Extract Features
```python
from pakit.ml.telemetry import FeatureExtractor

extractor = FeatureExtractor()

# Block features
features = extractor.extract_block_features(
    block_hash='abc123',
    block_size=10000,
    block_depth=5,
    compression_ratio=0.5,
    access_count=10
)

# Access pattern features
access_features = extractor.extract_access_pattern_features(
    block_hash='abc123',
    access_history=access_events
)
```

### 3. Build Training Dataset
```python
from pakit.ml.telemetry import TrainingDataset

dataset = TrainingDataset()

# Add samples
for event in telemetry_events:
    features = extractor.extract_features(event)
    label = event['compression_algorithm']  # or other target
    
    dataset.add_sample(features, label)

# Split into train/val/test
train_data, val_data, test_data = dataset.split(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Save for later
dataset.save('training_dataset.json')
```

## Training Individual Models

### Compression Predictor

```python
from pakit.ml.models import CompressionPredictor
from pakit.ml.training import ModelTrainer, TrainingConfig
from pakit.ml.base_model import ModelConfig

# Create model
model_config = ModelConfig(
    name='compression_predictor',
    model_type='classification',
    device='cuda'  # or 'cpu'
)
model = CompressionPredictor(model_config)

# Training configuration
training_config = TrainingConfig(
    learning_rate=0.001,
    batch_size=32,
    num_epochs=50,
    early_stopping_patience=3,
    checkpoint_dir='./checkpoints/compression'
)

# Create trainer
trainer = ModelTrainer(training_config)

# Train
metrics = trainer.train(
    model=model,
    train_data=train_data,
    val_data=val_data
)

print(f"Final accuracy: {metrics['accuracy']:.1%}")

# Save model
model.save('./models/compression_predictor.pt')
```

### Deduplication Optimizer

```python
from pakit.ml.models import DeduplicationOptimizer

model = DeduplicationOptimizer(model_config)

# Build LSH index from historical blocks
model.train(train_data)

# Test similarity detection
similar_blocks = model.find_similar_blocks('abc123', top_k=5)
print(f"Found {len(similar_blocks)} similar blocks")

# Save
model.save('./models/dedup_optimizer.pkl')
```

### Prefetch Engine

```python
from pakit.ml.models import PrefetchEngine

model = PrefetchEngine(model_config, max_vocab_size=10000)

# Train on access sequences
# Dataset should be list of sequences: [['block1', 'block2', ...], ...]
model.train(access_sequences)

# Test prediction
recent = ['block1', 'block2', 'block3']
next_blocks = model.predict({'recent_accesses': recent})
print(f"Predicted next blocks: {next_blocks}")

# Save
model.save('./models/prefetch_engine.pt')
```

### Peer Selector

```python
from pakit.ml.models import PeerSelector

model = PeerSelector(model_config, use_context=True)

# Train from historical peer interactions
# Dataset: [{'peer_id': 'peer1', 'success': True, 'latency_ms': 50, ...}, ...]
model.train(peer_interactions)

# Test selection
best_peer = model.predict({
    'available_peers': ['peer1', 'peer2', 'peer3'],
    'context': {
        'block_size_log': 4.0,
        'time_of_day': 0.5,
        'network_load': 0.3,
    }
})
print(f"Selected peer: {best_peer}")

# Save
model.save('./models/peer_selector.pkl')
```

### Network Optimizer

```python
from pakit.ml.models import NetworkOptimizer

model = NetworkOptimizer(model_config)

# Train from network episodes
# Dataset: [{'state': (1, 2, 0, 1), 'action': (6, 10), 'reward': 0.8, ...}, ...]
model.train(network_episodes)

# Test optimization
params = model.predict({
    'avg_latency_ms': 100.0,
    'message_loss_rate': 0.05,
})
print(f"Optimized params: fanout={params['fanout']}, TTL={params['ttl']}")

# Save
model.save('./models/network_optimizer.pkl')
```

## Hyperparameter Tuning

### Using Optuna for Bayesian Optimization

```python
from pakit.ml.training import HyperparameterTuner

tuner = HyperparameterTuner(
    model_class=CompressionPredictor,
    train_data=train_data,
    val_data=val_data,
    n_trials=50
)

# Define search space
search_space = {
    'learning_rate': ('float', 1e-4, 1e-2, True),  # log scale
    'batch_size': ('int', 16, 128),
    'hidden_dim': ('categorical', [64, 128, 256]),
}

# Run optimization
best_params = tuner.tune(search_space, metric='accuracy', direction='maximize')

print(f"Best parameters: {best_params}")
print(f"Best accuracy: {tuner.best_value:.1%}")

# Train final model with best params
model = CompressionPredictor(model_config)
trainer = ModelTrainer(TrainingConfig(**best_params))
trainer.train(model, train_data, val_data)
```

## Evaluation

### Model Evaluation

```python
from pakit.ml.training import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate on test set
metrics = evaluator.evaluate(model, test_data)

print(f"Accuracy: {metrics['accuracy']:.1%}")
print(f"Precision: {metrics['precision']:.1%}")
print(f"Recall: {metrics['recall']:.1%}")
print(f"F1 Score: {metrics['f1']:.3f}")

# Confusion matrix
print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")

# Per-class metrics
for class_name, class_metrics in metrics['per_class'].items():
    print(f"{class_name}: P={class_metrics['precision']:.1%}, R={class_metrics['recall']:.1%}")
```

### Cross-Validation

```python
# K-fold cross-validation
cv_metrics = evaluator.cross_validate(
    model_class=CompressionPredictor,
    dataset=full_dataset,
    k_folds=5
)

print(f"CV Accuracy: {cv_metrics['mean_accuracy']:.1%} ± {cv_metrics['std_accuracy']:.1%}")
```

## Continuous Training

### Online Learning Setup

```python
from pakit.ml.training import TrainingScheduler

scheduler = TrainingScheduler()

# Schedule periodic retraining
scheduler.schedule_job(
    model_name='compression_predictor',
    dataset_path='./telemetry.db',
    checkpoint_path='./checkpoints/compression',
    schedule='daily',  # or 'hourly', 'weekly'
    priority=1
)

# Run training jobs
while True:
    job = scheduler.run_next_job()
    if job:
        print(f"Trained {job['model_name']}, accuracy: {job['metrics']['accuracy']:.1%}")
    time.sleep(3600)  # Check every hour
```

## Monitoring Training

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/compression_experiment')

# During training
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch()
    val_loss, val_acc = validate()
    
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

writer.close()

# View in browser
# tensorboard --logdir=./runs
```

## Best Practices

### 1. Data Quality
- ✅ Collect diverse workloads (text, binary, compressed, etc.)
- ✅ Ensure balanced classes (don't oversample one algorithm)
- ✅ Remove outliers and corrupted samples
- ❌ Don't train on synthetic-only data

### 2. Training
- ✅ Use early stopping (patience=3-5 epochs)
- ✅ Monitor validation metrics, not just training
- ✅ Save checkpoints regularly
- ✅ Use learning rate scheduling
- ❌ Don't overtrain (>50 epochs usually not needed)

### 3. Evaluation
- ✅ Always evaluate on held-out test set
- ✅ Use k-fold cross-validation for small datasets
- ✅ Track precision/recall, not just accuracy
- ❌ Don't tune hyperparameters on test set

### 4. Deployment
- ✅ Validate model performance before deployment
- ✅ Use A/B testing for gradual rollout
- ✅ Monitor production metrics
- ✅ Have rollback plan ready

## Troubleshooting

### Low Accuracy (<80%)
- Check data quality (corrupted samples?)
- Try hyperparameter tuning
- Increase model capacity (more layers/neurons)
- Collect more training data

### Overfitting (train acc >> val acc)
- Add dropout (0.1-0.3)
- Reduce model complexity
- Increase training data
- Use data augmentation

### Slow Training
- Use GPU if available (`device='cuda'`)
- Increase batch size (if memory allows)
- Use mixed precision training
- Profile code to find bottlenecks

### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Clear cache: `torch.cuda.empty_cache()`
- Use smaller model

## Example Training Script

```python
#!/usr/bin/env python3
"""
Complete training pipeline for compression predictor.
"""

from pakit.ml.models import CompressionPredictor
from pakit.ml.training import ModelTrainer, TrainingConfig, ModelEvaluator
from pakit.ml.telemetry import TelemetryCollector, FeatureExtractor, TrainingDataset
from pakit.ml.base_model import ModelConfig

def main():
    # 1. Load telemetry
    collector = TelemetryCollector(db_path='./telemetry.db')
    events = collector.get_events(event_type='BLOCK_STORED', limit=10000)
    
    # 2. Extract features
    extractor = FeatureExtractor()
    dataset = TrainingDataset()
    
    for event in events:
        features = extractor.extract_block_features(
            event['block_hash'],
            event['block_size'],
            event['block_depth'],
            event.get('compression_ratio', 1.0),
            event.get('access_count', 0)
        )
        label = event['compression_algorithm']
        dataset.add_sample(features, label)
    
    # 3. Split data
    train_data, val_data, test_data = dataset.split()
    
    # 4. Create model
    model_config = ModelConfig(
        name='compression_predictor',
        model_type='classification',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    model = CompressionPredictor(model_config)
    
    # 5. Train
    training_config = TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        early_stopping_patience=3
    )
    trainer = ModelTrainer(training_config)
    trainer.train(model, train_data, val_data)
    
    # 6. Evaluate
    evaluator = ModelEvaluator()
    test_metrics = evaluator.evaluate(model, test_data)
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"Test F1: {test_metrics['f1']:.3f}")
    
    # 7. Save
    model.save('./models/compression_predictor.pt')
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
```

## Next Steps

- **Deployment**: See [ML_DEPLOYMENT_GUIDE.md](ML_DEPLOYMENT_GUIDE.md)
- **Architecture**: See [ML_ARCHITECTURE.md](ML_ARCHITECTURE.md)
