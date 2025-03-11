"""MLflow configuration and utilities."""

import os
from pathlib import Path
import mlflow
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class MLflowConfig:
    """MLflow experiment configuration."""
    
    experiment_name: str = "multispectral_segmentation"
    tracking_uri: str = "sqlite:///mlflow.db"
    artifact_location: str = "artifacts"
    
    # Tags for experiment organization
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        self.tags = self.tags or {
            "task": "segmentation",
            "model_type": "deeplabv3plus",
            "data_type": "multispectral"
        }
        
        # Ensure artifact directory exists
        Path(self.artifact_location).mkdir(parents=True, exist_ok=True)

def setup_mlflow(config: MLflowConfig) -> str:
    """Setup MLflow experiment tracking.
    
    Args:
        config: MLflow configuration
        
    Returns:
        experiment_id: ID of created or existing experiment
    """
    # Set tracking URI
    mlflow.set_tracking_uri(config.tracking_uri)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=config.experiment_name,
            artifact_location=config.artifact_location,
            tags=config.tags
        )
    else:
        experiment_id = experiment.experiment_id
    
    return experiment_id

class MLflowTracker:
    """MLflow experiment tracker wrapper."""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.experiment_id = setup_mlflow(config)
        self.run_id = None
    
    def start_run(self, run_name: Optional[str] = None, nested: bool = False):
        """Start a new MLflow run."""
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested
        )
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
    
    def log_model(self, model: Any, artifact_path: str):
        """Log model artifacts."""
        mlflow.pytorch.log_model(model, artifact_path)
    
    def log_artifacts(self, local_dir: str):
        """Log artifacts from a local directory."""
        mlflow.log_artifacts(local_dir)
    
    def end_run(self):
        """End the current run."""
        mlflow.end_run()