"""Configuration management for the multispectral vision model.

This module handles loading and validation of configuration files using YAML.
It provides a structured way to configure:
- Model architecture (DeepLabV3+)
- Training parameters
- Dataset paths and processing
- Spectral indices calculation
- Evaluation metrics
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """DeepLabV3+ model configuration."""
    backbone: str = "resnet50"  # Backbone architecture
    output_stride: int = 16  # Output stride for ASPP
    num_classes: int = 21  # Number of output classes
    pretrained: bool = True  # Use pretrained backbone
    freeze_backbone: bool = False  # Freeze backbone weights


@dataclass
class DataConfig:
    """Dataset configuration."""
    data_dir: str  # Root directory for dataset
    train_split: str = "train"  # Training split name
    val_split: str = "val"  # Validation split name
    test_split: str = "test"  # Test split name
    batch_size: int = 16  # Batch size for dataloaders
    num_workers: int = 4  # Number of dataloader workers
    image_size: tuple = (512, 512)  # Input image size (H, W)
    augmentation: bool = True  # Use data augmentation
    bands: list = field(default_factory=lambda: ["red", "green", "blue", "nir"])


@dataclass
class SpectralConfig:
    """Spectral indices configuration."""
    indices: list = field(default_factory=lambda: ["ndvi", "evi", "savi", "ndwi"])
    use_as_features: bool = True  # Use indices as additional input features
    savi_L: float = 0.5  # Soil brightness correction factor for SAVI
    evi_G: float = 2.5  # Gain factor for EVI
    evi_C1: float = 6.0  # Coefficient 1 for EVI
    evi_C2: float = 7.5  # Coefficient 2 for EVI


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100  # Number of training epochs
    learning_rate: float = 0.001  # Initial learning rate
    weight_decay: float = 0.0001  # Weight decay for optimizer
    lr_scheduler: str = "cosine"  # Learning rate scheduler type
    warmup_epochs: int = 5  # Number of warmup epochs
    early_stopping_patience: int = 10  # Patience for early stopping
    grad_clip: float = 1.0  # Gradient clipping value
    mixed_precision: bool = True  # Use mixed precision training


@dataclass
class Config:
    """Main configuration class."""
    # Basic settings
    experiment_name: str  # Name of the experiment
    seed: int = 42  # Random seed
    output_dir: str = "outputs"  # Directory for saving outputs
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _validate_config(self):
        """Validate configuration values."""
        # Validate model config
        valid_backbones = ["resnet50", "resnet101", "mobilenetv2"]
        if self.model.backbone not in valid_backbones:
            raise ValueError(f"Invalid backbone: {self.model.backbone}. "
                           f"Must be one of {valid_backbones}")
        
        if self.model.output_stride not in [8, 16]:
            raise ValueError("Output stride must be 8 or 16")
        
        # Validate data config
        if not os.path.exists(self.data.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data.data_dir}")
        
        if not all(b in ["red", "green", "blue", "nir", "swir1", "swir2"] 
                  for b in self.data.bands):
            raise ValueError("Invalid spectral bands specified")
        
        # Validate training config
        if self.training.lr_scheduler not in ["cosine", "step", "plateau"]:
            raise ValueError("Invalid learning rate scheduler specified")
    
    def save(self, config_path: Union[str, Path]):
        """Save configuration to YAML file."""
        config_dict = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "spectral": self.spectral.__dict__,
            "training": self.training.__dict__
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Config object initialized from YAML file
        """
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        
        # Create component configs
        model_config = ModelConfig(**config_dict.pop("model", {}))
        data_config = DataConfig(**config_dict.pop("data", {}))
        spectral_config = SpectralConfig(**config_dict.pop("spectral", {}))
        training_config = TrainingConfig(**config_dict.pop("training", {}))
        
        # Create main config
        return cls(
            model=model_config,
            data=data_config,
            spectral=spectral_config,
            training=training_config,
            **config_dict
        )