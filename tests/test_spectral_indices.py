"""Tests for spectral indices calculation."""

import pytest
import torch
import numpy as np

from src.spectral_indices import (
    check_bands,
    ndvi,
    evi,
    savi,
    ndwi,
    calculate_all_indices
)


@pytest.fixture
def sample_data():
    """Create sample multi-spectral data for testing."""
    # Create a 2x2 image with 12 bands (Sentinel-2 like)
    bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    data = torch.ones((1, len(bands), 2, 2))  # Batch size 1
    
    # Set some realistic values
    # NIR (B8)
    data[0, bands.index('B8')] = 0.8
    # Red (B4)
    data[0, bands.index('B4')] = 0.4
    # Blue (B2)
    data[0, bands.index('B2')] = 0.3
    # Green (B3)
    data[0, bands.index('B3')] = 0.5
    
    return data, bands


def test_check_bands(sample_data):
    """Test band checking functionality."""
    data, bands = sample_data
    
    # Should work with valid bands
    check_bands(data, ['B8', 'B4'], bands)
    
    # Should raise error with invalid bands
    with pytest.raises(ValueError):
        check_bands(data, ['B8', 'B99'], bands)


def test_ndvi(sample_data):
    """Test NDVI calculation."""
    data, bands = sample_data
    result = ndvi(data, bands)
    
    # Manual calculation for verification
    nir = 0.8
    red = 0.4
    expected = (nir - red) / (nir + red)
    
    assert torch.allclose(result, torch.full_like(result, expected))
    assert result.shape == (1, 1, 2, 2)


def test_evi(sample_data):
    """Test EVI calculation."""
    data, bands = sample_data
    result = evi(data, bands)
    
    # Manual calculation for verification
    nir = 0.8
    red = 0.4
    blue = 0.3
    expected = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    
    assert torch.allclose(result, torch.full_like(result, expected))
    assert result.shape == (1, 1, 2, 2)


def test_savi(sample_data):
    """Test SAVI calculation."""
    data, bands = sample_data
    result = savi(data, bands)
    
    # Manual calculation for verification
    nir = 0.8
    red = 0.4
    L = 0.5
    expected = (nir - red) * (1 + L) / (nir + red + L)
    
    assert torch.allclose(result, torch.full_like(result, expected))
    assert result.shape == (1, 1, 2, 2)


def test_ndwi(sample_data):
    """Test NDWI calculation."""
    data, bands = sample_data
    result = ndwi(data, bands)
    
    # Manual calculation for verification
    nir = 0.8
    green = 0.5
    expected = (green - nir) / (green + nir)
    
    assert torch.allclose(result, torch.full_like(result, expected))
    assert result.shape == (1, 1, 2, 2)


def test_calculate_all_indices(sample_data):
    """Test calculation of all indices at once."""
    data, bands = sample_data
    indices = calculate_all_indices(data, bands)
    
    # Should have all indices
    assert set(indices.keys()) == {'ndvi', 'evi', 'savi', 'ndwi'}
    
    # All indices should have correct shape
    for idx in indices.values():
        assert idx.shape == (1, 1, 2, 2)
    
    # Values should match individual calculations
    assert torch.allclose(indices['ndvi'], ndvi(data, bands))
    assert torch.allclose(indices['evi'], evi(data, bands))
    assert torch.allclose(indices['savi'], savi(data, bands))
    assert torch.allclose(indices['ndwi'], ndwi(data, bands))


def test_missing_bands():
    """Test handling of missing bands."""
    # Create data with missing bands
    bands = ['B1', 'B2', 'B3']  # Missing B4 and B8
    data = torch.ones((1, len(bands), 2, 2))
    
    # Should raise error for NDVI (needs B4 and B8)
    with pytest.raises(ValueError):
        ndvi(data, bands)
    
    # calculate_all_indices should return empty dict
    indices = calculate_all_indices(data, bands)
    assert len(indices) == 0