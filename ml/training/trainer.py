"""
Model Trainer

Handles training of ML models with progress tracking and early stopping.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable, List
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Model config
    model_name: str
    model_type: str
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd, rmsprop
    weight_decay: float = 1e-5
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler: Optional[str] = "step"  # step, cosine, None
    lr_step_size: int = 5
    lr_gamma: float = 0.1
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 3
    min_delta: float = 0.001
    
    # Regularization
    dropout: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    
    # Device
    device: str = "cpu"
    
    # Misc
    verbose: bool = True
    random_seed: int = 42


class ModelTrainer:
    """
    Model trainer with early stopping and checkpointing.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0
        
        logger.info(f"Trainer initialized: {config.model_name}")
    
    def train(
        self,
        model: Any,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        loss_fn: Optional[Callable] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, List[float]]:
        """
        Train model.
        
        Args:
            model: Model to train (PyTorch nn.Module)
            train_data: Training data loader
            val_data: Validation data loader
            loss_fn: Loss function
            callbacks: List of callback functions
            
        Returns:
            Training history (losses, accuracies)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        
        # Default loss function
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Create optimizer
        optimizer = self._create_optimizer(model)
        
        # Create LR scheduler
        scheduler = self._create_scheduler(optimizer)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(
                model, train_data, optimizer, loss_fn
            )
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation
            if val_data:
                val_loss, val_acc = self._validate(model, val_data, loss_fn)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Learning rate schedule step
            if scheduler:
                scheduler.step()
            
            # Logging
            if self.config.verbose:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                )
            
            # Checkpointing
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                self.best_accuracy = val_acc
                self.epochs_without_improvement = 0
                
                if self.config.save_best_only:
                    self._save_checkpoint(model, optimizer, epoch, is_best=True)
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping
            if (self.config.early_stopping and 
                self.epochs_without_improvement >= self.config.patience):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history)
        
        logger.info(f"Training complete: best_loss={self.best_loss:.4f}, best_acc={self.best_accuracy:.4f}")
        
        return history
    
    def _train_epoch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable
    ) -> tuple:
        """Train single epoch."""
        model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        loss_fn: Callable
    ) -> tuple:
        """Validate model."""
        model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "rmsprop":
            return optim.RMSprop(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create LR scheduler."""
        if self.config.lr_scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.epochs
            )
        else:
            return None
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'config': self.config.__dict__,
        }
        
        if is_best:
            path = self.checkpoint_dir / f"{self.config.model_name}_best.pt"
            torch.save(checkpoint, path)
            logger.info(f"Saved best checkpoint: {path}")
