"""
Hyperparameter Tuner

Automated hyperparameter tuning with Bayesian optimization.
"""

from typing import Dict, Any, List, Callable, Optional
import logging
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available - hyperparameter tuning disabled")


class HyperparameterTuner:
    """
    Hyperparameter tuner using Bayesian optimization (Optuna).
    """
    
    def __init__(
        self,
        study_name: str = "pakit_ml_study",
        storage: Optional[str] = None
    ):
        self.study_name = study_name
        self.storage = storage or "sqlite:///pakit_optuna.db"
        
        if OPTUNA_AVAILABLE:
            logger.info(f"Tuner initialized: {study_name}")
        else:
            logger.warning("Optuna not available")
    
    def tune(
        self,
        objective_fn: Callable,
        param_space: Dict[str, Dict[str, Any]],
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.
        
        Args:
            objective_fn: Objective function to optimize (returns metric to maximize)
            param_space: Parameter search space
            n_trials: Number of trials
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters
        """
        if not OPTUNA_AVAILABLE:
            logger.error("Cannot tune - Optuna not available")
            return {}
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            load_if_exists=True
        )
        
        # Create wrapped objective
        def wrapped_objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {}
            for param_name, param_config in param_space.items():
                param_type = param_config['type']
                
                if param_type == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_type == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
            
            # Evaluate objective
            return objective_fn(params)
        
        # Run optimization
        study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(
            f"Tuning complete: best_value={best_value:.4f}, "
            f"best_params={best_params}"
        )
        
        return best_params
    
    def get_study_summary(self) -> Dict[str, Any]:
        """Get summary of tuning study."""
        if not OPTUNA_AVAILABLE:
            return {}
        
        study = optuna.load_study(
            study_name=self.study_name,
            storage=self.storage
        )
        
        return {
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial': study.best_trial.number,
        }


class TrainingScheduler:
    """
    Training job scheduler for batch training.
    """
    
    def __init__(self, jobs_dir: str = "./training_jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: List[Dict[str, Any]] = []
        
        logger.info(f"Scheduler initialized: {jobs_dir}")
    
    def schedule_job(
        self,
        model_name: str,
        dataset_path: str,
        config: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """
        Schedule a training job.
        
        Args:
            model_name: Model to train
            dataset_path: Path to dataset
            config: Training configuration
            priority: Job priority (higher = more important)
            
        Returns:
            Job ID
        """
        import uuid
        
        job_id = str(uuid.uuid4())[:8]
        
        job = {
            'job_id': job_id,
            'model_name': model_name,
            'dataset_path': dataset_path,
            'config': config,
            'priority': priority,
            'status': 'pending',
            'created_at': time.time(),
        }
        
        self.jobs.append(job)
        
        # Save job to disk
        job_file = self.jobs_dir / f"{job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job, f, indent=2)
        
        logger.info(f"Scheduled job {job_id}: {model_name}")
        
        return job_id
    
    def run_next_job(self) -> Optional[str]:
        """
        Run the next pending job.
        
        Returns:
            Job ID if executed, None if no jobs
        """
        # Find highest priority pending job
        pending = [j for j in self.jobs if j['status'] == 'pending']
        if not pending:
            return None
        
        next_job = max(pending, key=lambda j: j['priority'])
        
        # Mark as running
        next_job['status'] = 'running'
        next_job['started_at'] = time.time()
        
        logger.info(f"Running job {next_job['job_id']}")
        
        try:
            # Execute training (placeholder - implement actual training)
            # from pakit.ml.training.trainer import ModelTrainer
            # trainer = ModelTrainer(next_job['config'])
            # trainer.train(...)
            
            next_job['status'] = 'completed'
            next_job['completed_at'] = time.time()
            logger.info(f"Job {next_job['job_id']} completed")
        
        except Exception as e:
            next_job['status'] = 'failed'
            next_job['error'] = str(e)
            logger.error(f"Job {next_job['job_id']} failed: {e}")
        
        return next_job['job_id']
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a job."""
        for job in self.jobs:
            if job['job_id'] == job_id:
                return job
        return None
    
    def list_jobs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all jobs, optionally filtered by status."""
        if status:
            return [j for j in self.jobs if j['status'] == status]
        return self.jobs
