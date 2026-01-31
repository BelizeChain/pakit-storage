"""
Model Evaluator

Evaluates trained models with comprehensive metrics.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Metrics libraries not available")


@dataclass
class EvaluationMetrics:
    """Evaluation metrics container."""
    
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Optional detailed metrics
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'per_class_metrics': self.per_class_metrics,
        }


class ModelEvaluator:
    """
    Model evaluator with cross-validation support.
    """
    
    def __init__(self):
        if not METRICS_AVAILABLE:
            logger.warning("Evaluation libraries not available")
    
    def evaluate(
        self,
        model: Any,
        test_data: Any,
        class_names: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            class_names: Class names for classification
            
        Returns:
            Evaluation metrics
        """
        if not METRICS_AVAILABLE:
            raise RuntimeError("Metrics libraries not available")
        
        # Get predictions
        y_true, y_pred = self._get_predictions(model, test_data)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class = None
        if class_names:
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            per_class = {
                class_name: {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score'],
                }
                for class_name, metrics in report.items()
                if class_name in class_names
            }
        
        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm,
            per_class_metrics=per_class
        )
        
        logger.info(
            f"Evaluation: accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
        )
        
        return metrics
    
    def cross_validate(
        self,
        model_class: Any,
        dataset: Any,
        k_folds: int = 5,
        **train_kwargs
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model_class: Model class to instantiate
            dataset: Full dataset
            k_folds: Number of folds
            **train_kwargs: Additional training arguments
            
        Returns:
            Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        kfold = KFold(n_splits=k_folds, shuffle=True, random_seed=42)
        
        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            logger.info(f"Cross-validation fold {fold+1}/{k_folds}")
            
            # Split data
            train_data = [dataset[i] for i in train_idx]
            val_data = [dataset[i] for i in val_idx]
            
            # Train model
            model = model_class()
            # ... training logic ...
            
            # Evaluate
            metrics = self.evaluate(model, val_data)
            
            results['accuracy'].append(metrics.accuracy)
            results['precision'].append(metrics.precision)
            results['recall'].append(metrics.recall)
            results['f1'].append(metrics.f1)
        
        # Calculate mean and std
        summary = {}
        for metric, values in results.items():
            summary[f'{metric}_mean'] = np.mean(values)
            summary[f'{metric}_std'] = np.std(values)
        
        logger.info(
            f"Cross-validation: "
            f"accuracy={summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}"
        )
        
        return results
    
    def _get_predictions(self, model: Any, test_data: Any) -> tuple:
        """Get predictions from model."""
        # This is model-specific - implement based on model type
        # For PyTorch:
        if torch.is_tensor(test_data):
            model.eval()
            with torch.no_grad():
                outputs = model(test_data)
                _, predictions = outputs.max(1)
            return test_data.cpu().numpy(), predictions.cpu().numpy()
        else:
            # Assume numpy arrays
            return test_data[0], test_data[1]
