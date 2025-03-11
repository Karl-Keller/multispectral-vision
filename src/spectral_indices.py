"""Spectral indices calculation for multi-spectral imagery.

Common spectral indices include:
- NDVI (Normalized Difference Vegetation Index)
- EVI (Enhanced Vegetation Index)
- SAVI (Soil Adjusted Vegetation Index)
- NDWI (Normalized Difference Water Index)
"""

import torch
import torch.nn.functional as F


def check_bands(tensor: torch.Tensor, required_bands: list[str], band_names: list[str]) -> None:
    """Verify that required bands are present in the input tensor.
    
    Args:
        tensor: Input tensor of shape (B, C, H, W) or (C, H, W)
        required_bands: List of required band names
        band_names: List of all available band names
    
    Raises:
        ValueError: If required bands are not present
    """
    if not all(band in band_names for band in required_bands):
        missing = [b for b in required_bands if b not in band_names]
        raise ValueError(f"Missing required bands: {missing}")


def ndvi(x: torch.Tensor, band_names: list[str]) -> torch.Tensor:
    """Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        band_names: List of band names corresponding to channels
    
    Returns:
        NDVI tensor of shape (B, 1, H, W) or (1, H, W)
    """
    check_bands(x, ['B8', 'B4'], band_names)  # B8=NIR, B4=Red
    
    nir_idx = band_names.index('B8')
    red_idx = band_names.index('B4')
    
    nir = x[..., nir_idx:nir_idx+1, :, :]
    red = x[..., red_idx:red_idx+1, :, :]
    
    return (nir - red) / (nir + red + 1e-8)  # Add epsilon to avoid division by zero


def evi(x: torch.Tensor, band_names: list[str]) -> torch.Tensor:
    """Calculate Enhanced Vegetation Index.
    
    EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
    where G=2.5, C1=6, C2=7.5, L=1
    
    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        band_names: List of band names corresponding to channels
    
    Returns:
        EVI tensor of shape (B, 1, H, W) or (1, H, W)
    """
    check_bands(x, ['B8', 'B4', 'B2'], band_names)  # B8=NIR, B4=Red, B2=Blue
    
    nir_idx = band_names.index('B8')
    red_idx = band_names.index('B4')
    blue_idx = band_names.index('B2')
    
    nir = x[..., nir_idx:nir_idx+1, :, :]
    red = x[..., red_idx:red_idx+1, :, :]
    blue = x[..., blue_idx:blue_idx+1, :, :]
    
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def savi(x: torch.Tensor, band_names: list[str], L: float = 0.5) -> torch.Tensor:
    """Calculate Soil Adjusted Vegetation Index.
    
    SAVI = (NIR - Red) * (1 + L) / (NIR + Red + L)
    where L is a soil brightness correction factor
    
    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        band_names: List of band names corresponding to channels
        L: Soil brightness correction factor (default 0.5)
    
    Returns:
        SAVI tensor of shape (B, 1, H, W) or (1, H, W)
    """
    check_bands(x, ['B8', 'B4'], band_names)  # B8=NIR, B4=Red
    
    nir_idx = band_names.index('B8')
    red_idx = band_names.index('B4')
    
    nir = x[..., nir_idx:nir_idx+1, :, :]
    red = x[..., red_idx:red_idx+1, :, :]
    
    return (nir - red) * (1 + L) / (nir + red + L + 1e-8)


def ndwi(x: torch.Tensor, band_names: list[str]) -> torch.Tensor:
    """Calculate Normalized Difference Water Index.
    
    NDWI = (Green - NIR) / (Green + NIR)
    
    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        band_names: List of band names corresponding to channels
    
    Returns:
        NDWI tensor of shape (B, 1, H, W) or (1, H, W)
    """
    check_bands(x, ['B8', 'B3'], band_names)  # B8=NIR, B3=Green
    
    nir_idx = band_names.index('B8')
    green_idx = band_names.index('B3')
    
    nir = x[..., nir_idx:nir_idx+1, :, :]
    green = x[..., green_idx:green_idx+1, :, :]
    
    return (green - nir) / (green + nir + 1e-8)


def calculate_all_indices(x: torch.Tensor, band_names: list[str]) -> dict[str, torch.Tensor]:
    """Calculate all available spectral indices.
    
    Args:
        x: Input tensor of shape (B, C, H, W) or (C, H, W)
        band_names: List of band names corresponding to channels
    
    Returns:
        Dictionary mapping index names to their tensors
    """
    indices = {}
    
    try:
        indices['ndvi'] = ndvi(x, band_names)
    except ValueError:
        pass
        
    try:
        indices['evi'] = evi(x, band_names)
    except ValueError:
        pass
        
    try:
        indices['savi'] = savi(x, band_names)
    except ValueError:
        pass
        
    try:
        indices['ndwi'] = ndwi(x, band_names)
    except ValueError:
        pass
    
    return indices