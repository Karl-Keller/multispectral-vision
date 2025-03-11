"""Tests for the multi-spectral dataset."""

import os
import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import rasterio

from datasets.multispectral import MultispectralDataset

@pytest.fixture
def sample_data(tmp_path):
    """Create sample data for testing."""
    # Create directories
    ms_dir = tmp_path / "ms"
    rgb_dir = tmp_path / "rgb"
    ms_dir.mkdir()
    rgb_dir.mkdir()
    
    # Create sample MS image
    ms_data = np.random.rand(8, 64, 64).astype(np.float32)
    mask_data = np.random.randint(0, 4, (64, 64), dtype=np.uint8)
    rgb_data = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Save MS image and mask
    profile = {
        'driver': 'GTiff',
        'height': 64,
        'width': 64,
        'count': 8,
        'dtype': 'float32'
    }
    with rasterio.open(ms_dir / "sample.tif", 'w', **profile) as dst:
        dst.write(ms_data)
    
    profile.update(count=1, dtype='uint8')
    with rasterio.open(ms_dir / "sample_mask.tif", 'w', **profile) as dst:
        dst.write(mask_data[None, ...])
    
    # Save RGB image
    Image.fromarray(rgb_data).save(rgb_dir / "sample.jpg")
    
    return {
        'ms_dir': ms_dir,
        'rgb_dir': rgb_dir,
        'ms_data': ms_data,
        'mask_data': mask_data,
        'rgb_data': rgb_data
    }

def test_ms_only_dataset(sample_data):
    """Test dataset with only multi-spectral images."""
    dataset = MultispectralDataset(
        data_dir=str(sample_data['ms_dir']),
        input_size=(64, 64),
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    )
    
    assert len(dataset) == 1
    img, mask = dataset[0]
    
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (8, 64, 64)
    assert mask.shape == (64, 64)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.int64

def test_dual_source_dataset(sample_data):
    """Test dataset with both RGB and multi-spectral images."""
    dataset = MultispectralDataset(
        data_dir=str(sample_data['ms_dir']),
        input_size=(64, 64),
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
        rgb_dir=str(sample_data['rgb_dir'])
    )
    
    assert len(dataset) == 1
    img, mask = dataset[0]
    
    assert isinstance(img, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert img.shape == (11, 64, 64)  # 3 (RGB) + 8 (MS) channels
    assert mask.shape == (64, 64)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.int64

def test_resize(sample_data):
    """Test dataset resizing functionality."""
    dataset = MultispectralDataset(
        data_dir=str(sample_data['ms_dir']),
        input_size=(32, 32),  # Different from input size
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8']
    )
    
    img, mask = dataset[0]
    assert img.shape == (8, 32, 32)
    assert mask.shape == (32, 32)

def test_augmentation(sample_data):
    """Test that training mode enables augmentations."""
    dataset = MultispectralDataset(
        data_dir=str(sample_data['ms_dir']),
        input_size=(64, 64),
        bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8'],
        is_training=True
    )
    
    # Get multiple samples to check they're different due to augmentation
    samples = [dataset[0][0] for _ in range(5)]
    
    # Check that at least some samples are different
    assert not all(torch.allclose(samples[0], sample) for sample in samples[1:])

def test_invalid_inputs(sample_data):
    """Test handling of invalid inputs."""
    # Test invalid directory
    with pytest.raises(ValueError, match="Directory not found"):
        MultispectralDataset(
            data_dir="/nonexistent/dir",
            input_size=(64, 64),
            bands=['B1']
        )
    
    # Test invalid band names
    with pytest.raises(ValueError, match="Invalid band names"):
        MultispectralDataset(
            data_dir=str(sample_data['ms_dir']),
            input_size=(64, 64),
            bands=['invalid_band']
        )
    
    # Test invalid input size
    with pytest.raises(ValueError, match="Input size must be positive"):
        MultispectralDataset(
            data_dir=str(sample_data['ms_dir']),
            input_size=(-1, 64),
            bands=['B1']
        )