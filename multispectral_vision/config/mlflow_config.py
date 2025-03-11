"""MLflow configuration and tracking utilities."""

from dataclasses import dataclass
from typing import Optional, Dict, Any

import mlflow

@dataclass
class MLflowConfig:
    """Configuration for MLflow experiment tracking."""
    
    experiment_name: str
    tracking_uri: str
    run_name: Optional[str] = None
    tags: Optional[Dict[str, str]] = None

class MLflowTracker:
    """MLflow experiment tracking wrapper."""
    
    def __init__(self, config: MLflowConfig):
        """Initialize MLflow tracker.
        
        Args:
            config: MLflow configuration
        """
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
    
    def start_run(self) -> None:
        """Start a new MLflow run."""
        mlflow.start_run(
            run_name=self.config.run_name,
            tags=self.config.tags
        )
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: Any, artifact_path: str) -> None:
        """Log model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path to save model artifacts
        """
        mlflow.pytorch.log_model(model, artifact_path)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()