"""
Multi-spectral dataset implementation supporting both single and dual-source inputs.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from rasterio.windows import Window
import albumentations as A
from PIL import Image

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
        mask_suffix: str = "_mask"
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
        """
        self.data_dir = Path(data_dir)
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else self.data_dir
        self.input_size = input_size
        self.bands = bands
        self.indices = indices
        self.is_training = is_training
        self.mask_suffix = mask_suffix
        
        # Get list of files
        self.ms_files = sorted([
            f for f in self.data_dir.glob("*.tif")
            if not f.name.endswith(f"{mask_suffix}.tif")
        ])
        
        if not self.ms_files:
            raise ValueError(f"No .tif files found in {data_dir}")
            
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
            
        # Load mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # [H, W]
        
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