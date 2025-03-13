"""
Multi-spectral dataset implementation supporting both single and dual-source inputs.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import albumentations as A
from PIL import Image
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import json
from rasterio.features import rasterize
from rasterio.transform import from_bounds

class MultispectralDataset(Dataset):
    """Dataset for multi-spectral image segmentation.
    
    Supports:
    - Single MS source
    - Dual RGB + MS source
    - Optional spectral indices
    - On-the-fly data augmentation
    """
    
    def __init__(
        self,
        data_dir: str,
        input_size: Tuple[int, int],
        bands: List[str],
        indices: Optional[List[str]] = None,
        is_training: bool = True,
        rgb_dir: Optional[str] = None,
        mask_dir: Optional[str] = None,
        mask_suffix: str = "_mask",
        annotation_format: str = "raster",  # "raster", "geojson", or "shapefile"
        class_map: Optional[Dict[str, int]] = None
    ):
        """Initialize the dataset.
        
        Args:
            data_dir: Directory containing multi-spectral images
            input_size: Model input size (height, width)
            bands: List of band names to use
            indices: Optional list of spectral indices to calculate
            is_training: Whether this is for training (enables augmentation)
            rgb_dir: Optional directory containing RGB images
            mask_dir: Optional directory for masks (defaults to data_dir)
            mask_suffix: Suffix for mask files
            annotation_format: Format of the annotations ("raster", "geojson", or "shapefile")
            class_map: Dictionary mapping feature types to class indices
        """
        self.data_dir = Path(data_dir)
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else self.data_dir
        self.input_size = input_size
        self.bands = bands
        self.indices = indices
        self.is_training = is_training
        self.mask_suffix = mask_suffix
        self.annotation_format = annotation_format
        self.class_map = class_map or {}
        
        # Get list of files
        self.ms_files = sorted([
            f for f in self.data_dir.glob("*.tif")
            if not f.name.endswith(f"{mask_suffix}.tif")
        ])
        
        if not self.ms_files:
            raise ValueError(f"No .tif files found in {data_dir}")
            
        # Validate annotation format
        if annotation_format not in ["raster", "geojson", "shapefile"]:
            raise ValueError(f"Unsupported annotation format: {annotation_format}")
            
        # Setup augmentations
        if is_training:
            self.transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(),
            ])
        else:
            self.transform = A.Compose([
                A.Normalize()
            ])
    
    def _load_vector_annotations(self, ms_path: Path) -> np.ndarray:
        """Load and rasterize vector annotations (GeoJSON or Shapefile).
        
        Args:
            ms_path: Path to the multi-spectral image file
            
        Returns:
            Rasterized mask array [H, W]
        """
        # Get annotation path
        base_name = ms_path.stem
        if self.annotation_format == "geojson":
            anno_path = self.mask_dir / f"{base_name}.geojson"
        else:  # shapefile
            anno_path = self.mask_dir / f"{base_name}.shp"
            
        # Read vector data
        gdf = gpd.read_file(anno_path)
        
        # Get image metadata for rasterization
        with rasterio.open(ms_path) as src:
            height = src.height
            width = src.width
            transform = src.transform
            
        # Prepare shapes for rasterization
        shapes = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, (Polygon, MultiPolygon)):
                class_val = self.class_map.get(row.get('class', 'default'), 1)
                shapes.append((geom, class_val))
                
        # Rasterize the shapes
        mask = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.int32
        )
        
        return mask
        
    def __len__(self) -> int:
        return len(self.ms_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a dataset item.
        
        Returns:
            Tuple of:
                - Image tensor [C, H, W] (C = num_bands or 3+num_bands if RGB)
                - Mask tensor [H, W]
        """
        ms_path = self.ms_files[idx]
        mask_path = self.mask_dir / f"{ms_path.stem}{self.mask_suffix}.tif"
        
        # Load multi-spectral image
        with rasterio.open(ms_path) as src:
            ms_img = src.read()  # [C, H, W]
            
        # Load mask based on annotation format
        if self.annotation_format == "raster":
            with rasterio.open(mask_path) as src:
                mask = src.read(1)  # [H, W]
        else:  # vector format (geojson or shapefile)
            mask = self._load_vector_annotations(ms_path)
        
        # Load RGB if available
        if self.rgb_dir:
            rgb_path = self.rgb_dir / f"{ms_path.stem}.jpg"
            if rgb_path.exists():
                rgb_img = np.array(Image.open(rgb_path))  # [H, W, 3]
                rgb_img = rgb_img.transpose(2, 0, 1)  # [3, H, W]
            else:
                raise ValueError(f"RGB image not found: {rgb_path}")
        
        # Apply augmentations
        if self.is_training:
            # Transpose to [H, W, C] for albumentations
            ms_img = ms_img.transpose(1, 2, 0)
            if self.rgb_dir:
                rgb_img = rgb_img.transpose(1, 2, 0)
            
            # Apply same transform to all images
            if self.rgb_dir:
                transformed = self.transform(
                    image=ms_img,
                    mask=mask,
                    rgb_image=rgb_img
                )
                ms_img = transformed['image']
                mask = transformed['mask']
                rgb_img = transformed['rgb_image']
            else:
                transformed = self.transform(
                    image=ms_img,
                    mask=mask
                )
                ms_img = transformed['image']
                mask = transformed['mask']
            
            # Transpose back to [C, H, W]
            ms_img = ms_img.transpose(2, 0, 1)
            if self.rgb_dir:
                rgb_img = rgb_img.transpose(2, 0, 1)
        
        # Convert to tensor
        ms_img = torch.from_numpy(ms_img).float()
        if self.rgb_dir:
            rgb_img = torch.from_numpy(rgb_img).float()
            # Concatenate RGB and MS
            img = torch.cat([rgb_img, ms_img], dim=0)
        else:
            img = ms_img
            
        mask = torch.from_numpy(mask).long()
        
        # Resize if needed
        if img.shape[1:] != self.input_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),
                size=self.input_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(0)
            
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=self.input_size,
                mode='nearest'
            ).squeeze(0).squeeze(0).long()
        
        return img, mask