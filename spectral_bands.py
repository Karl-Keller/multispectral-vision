import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralBands:
    """Define standard positions for different spectral bands"""
    # Standard band positions
    COASTAL = 0      # ~443nm
    BLUE = 1        # ~490nm
    GREEN = 2       # ~560nm
    RED = 3         # ~665nm
    RED_EDGE1 = 4   # ~705nm
    RED_EDGE2 = 5   # ~740nm
    RED_EDGE3 = 6   # ~783nm
    NIR = 7         # ~842nm
    SWIR1 = 8       # ~1610nm
    SWIR2 = 9       # ~2190nm
    THERMAL = 10    # ~10900nm

    @classmethod
    def get_band_wavelengths(cls):
        """Return approximate central wavelengths for each band in nanometers"""
        return {
            cls.COASTAL: 443,
            cls.BLUE: 490,
            cls.GREEN: 560,
            cls.RED: 665,
            cls.RED_EDGE1: 705,
            cls.RED_EDGE2: 740,
            cls.RED_EDGE3: 783,
            cls.NIR: 842,
            cls.SWIR1: 1610,
            cls.SWIR2: 2190,
            cls.THERMAL: 10900
        }


class SpectralIndices:
    """Calculate various spectral indices from multi-spectral imagery"""
    
    @staticmethod
    def normalize_difference(band1, band2, epsilon=1e-7):
        """Generic normalized difference between two bands"""
        return (band1 - band2) / (band1 + band2 + epsilon)

    @staticmethod
    def extract_band(x, band_idx):
        """Extract a specific band from the input tensor"""
        return x[:, band_idx:band_idx+1]

    def ndvi(self, x):
        """
        Normalized Difference Vegetation Index
        NDVI = (NIR - RED) / (NIR + RED)
        """
        nir = self.extract_band(x, SpectralBands.NIR)
        red = self.extract_band(x, SpectralBands.RED)
        return self.normalize_difference(nir, red)

    def evi(self, x, G=2.5, C1=6, C2=7.5, L=1):
        """
        Enhanced Vegetation Index
        EVI = G * (NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L)
        """
        nir = self.extract_band(x, SpectralBands.NIR)
        red = self.extract_band(x, SpectralBands.RED)
        blue = self.extract_band(x, SpectralBands.BLUE)
        
        return G * (nir - red) / (nir + C1 * red - C2 * blue + L)

    def ndwi(self, x):
        """
        Normalized Difference Water Index
        NDWI = (GREEN - NIR) / (GREEN + NIR)
        """
        green = self.extract_band(x, SpectralBands.GREEN)
        nir = self.extract_band(x, SpectralBands.NIR)
        return self.normalize_difference(green, nir)

    def ndbi(self, x):
        """
        Normalized Difference Built-up Index
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        """
        swir1 = self.extract_band(x, SpectralBands.SWIR1)
        nir = self.extract_band(x, SpectralBands.NIR)
        return self.normalize_difference(swir1, nir)

    def savi(self, x, L=0.5):
        """
        Soil Adjusted Vegetation Index
        SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)
        """
        nir = self.extract_band(x, SpectralBands.NIR)
        red = self.extract_band(x, SpectralBands.RED)
        return ((nir - red) / (nir + red + L)) * (1 + L)

    def ndre(self, x):
        """
        Normalized Difference Red Edge
        NDRE = (NIR - RED_EDGE) / (NIR + RED_EDGE)
        """
        nir = self.extract_band(x, SpectralBands.NIR)
        red_edge = self.extract_band(x, SpectralBands.RED_EDGE2)
        return self.normalize_difference(nir, red_edge)

    def moisture_index(self, x):
        """
        Normalized Difference Moisture Index
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        """
        nir = self.extract_band(x, SpectralBands.NIR)
        swir1 = self.extract_band(x, SpectralBands.SWIR1)
        return self.normalize_difference(nir, swir1)


class SpectralFeatureExtractor(nn.Module):
    """Extract spectral features and indices from multi-spectral imagery"""
    
    def __init__(self, use_indices=True):
        super(SpectralFeatureExtractor, self).__init__()
        self.use_indices = use_indices
        self.indices = SpectralIndices()
        
        # Learnable band weights
        self.band_weights = nn.Parameter(torch.ones(len(SpectralBands.get_band_wavelengths())))
        
    def forward(self, x):
        # Apply band weights
        weighted_bands = x * self.band_weights.view(1, -1, 1, 1)
        
        if not self.use_indices:
            return weighted_bands
            
        # Calculate spectral indices
        ndvi = self.indices.ndvi(weighted_bands)
        evi = self.indices.evi(weighted_bands)
        ndwi = self.indices.ndwi(weighted_bands)
        ndbi = self.indices.ndbi(weighted_bands)
        savi = self.indices.savi(weighted_bands)
        ndre = self.indices.ndre(weighted_bands)
        moisture = self.indices.moisture_index(weighted_bands)
        
        # Concatenate original bands and indices
        indices = torch.cat([
            ndvi, evi, ndwi, ndbi, savi, ndre, moisture
        ], dim=1)
        
        return torch.cat([weighted_bands, indices], dim=1)