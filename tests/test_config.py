"""Tests for configuration management."""

import os
import pytest
import tempfile
from pathlib import Path
import yaml

from src.config import Config, ModelConfig, DataConfig, SpectralConfig, TrainingConfig


@pytest.fixture
def sample_config_dict():
    """Create a sample configuration dictionary."""
    return {
        "experiment_name": "test_experiment",
        "seed": 42,
        "output_dir": "test_outputs",
        "model": {
            "backbone": "resnet50",
            "output_stride": 16,
            "num_classes": 21,
            "pretrained": True,
            "freeze_backbone": False
        },
        "data": {
            "data_dir": "test_data/",
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
            "batch_size": 16,
            "num_workers": 4,
            "image_size": [512, 512],
            "augmentation": True,
            "bands": ["red", "green", "blue", "nir"]
        },
        "spectral": {
            "indices": ["ndvi", "evi", "savi", "ndwi"],
            "use_as_features": True,
            "savi_L": 0.5,
            "evi_G": 2.5,
            "evi_C1": 6.0,
            "evi_C2": 7.5
        },
        "training": {
            "epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "lr_scheduler": "cosine",
            "warmup_epochs": 5,
            "early_stopping_patience": 10,
            "grad_clip": 1.0,
            "mixed_precision": True
        }
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_model_config():
    """Test ModelConfig initialization and validation."""
    config = ModelConfig()
    assert config.backbone == "resnet50"
    assert config.output_stride == 16
    assert config.num_classes == 21
    assert config.pretrained is True
    assert config.freeze_backbone is False


def test_data_config():
    """Test DataConfig initialization and validation."""
    config = DataConfig(data_dir="test_data/")
    assert config.data_dir == "test_data/"
    assert config.batch_size == 16
    assert config.num_workers == 4
    assert config.image_size == (512, 512)
    assert config.augmentation is True
    assert config.bands == ["red", "green", "blue", "nir"]


def test_spectral_config():
    """Test SpectralConfig initialization and validation."""
    config = SpectralConfig()
    assert config.indices == ["ndvi", "evi", "savi", "ndwi"]
    assert config.use_as_features is True
    assert config.savi_L == 0.5
    assert config.evi_G == 2.5
    assert config.evi_C1 == 6.0
    assert config.evi_C2 == 7.5


def test_training_config():
    """Test TrainingConfig initialization and validation."""
    config = TrainingConfig()
    assert config.epochs == 100
    assert config.learning_rate == 0.001
    assert config.weight_decay == 0.0001
    assert config.lr_scheduler == "cosine"
    assert config.warmup_epochs == 5
    assert config.early_stopping_patience == 10
    assert config.grad_clip == 1.0
    assert config.mixed_precision is True


def test_config_from_yaml(temp_config_file):
    """Test loading configuration from YAML file."""
    config = Config.from_yaml(temp_config_file)
    
    assert config.experiment_name == "test_experiment"
    assert config.seed == 42
    assert config.output_dir == "test_outputs"
    
    # Check model config
    assert config.model.backbone == "resnet50"
    assert config.model.output_stride == 16
    
    # Check data config
    assert config.data.data_dir == "test_data/"
    assert config.data.batch_size == 16
    
    # Check spectral config
    assert "ndvi" in config.spectral.indices
    assert config.spectral.savi_L == 0.5
    
    # Check training config
    assert config.training.epochs == 100
    assert config.training.learning_rate == 0.001


def test_config_save(temp_config_file, tmp_path):
    """Test saving configuration to YAML file."""
    config = Config.from_yaml(temp_config_file)
    save_path = tmp_path / "saved_config.yaml"
    
    config.save(save_path)
    assert os.path.exists(save_path)
    
    # Load saved config and verify
    loaded_config = Config.from_yaml(save_path)
    assert loaded_config.experiment_name == config.experiment_name
    assert loaded_config.model.backbone == config.model.backbone
    assert loaded_config.data.batch_size == config.data.batch_size


def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        ModelConfig(backbone="invalid_backbone")
    
    with pytest.raises(ValueError):
        ModelConfig(output_stride=32)  # Only 8 or 16 allowed
    
    with pytest.raises(ValueError):
        TrainingConfig(lr_scheduler="invalid_scheduler")


def test_default_config():
    """Test loading default configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    assert config_path.exists(), "Default config file not found"
    
    config = Config.from_yaml(config_path)
    assert config.experiment_name == "multispectral_segmentation"
    assert config.model.backbone == "resnet50"
    assert len(config.data.bands) == 4  # red, green, blue, nir
    assert len(config.spectral.indices) == 4  # ndvi, evi, savi, ndwi