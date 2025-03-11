"""
Dual-Source DeepLabV3+ model for multi-spectral segmentation.
Handles both single MS camera and dual RGB+MS camera scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from einops import rearrange
import segmentation_models_pytorch as smp

class AdaptiveFusionModule(nn.Module):
    """Adaptive fusion module for combining RGB and MS features."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.in_channels = in_channels
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels * 2 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2 // reduction, in_channels * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_feat: torch.Tensor, ms_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat: RGB features [B, C, H, W]
            ms_feat: Multi-spectral features [B, C, H, W]
        Returns:
            Fused features [B, C, H, W]
        """
        # Concatenate features for attention
        concat_feat = torch.cat([rgb_feat, ms_feat], dim=1)
        attention = self.attention(concat_feat)
        
        # Split attention weights
        rgb_weights, ms_weights = torch.chunk(attention, 2, dim=1)
        
        # Apply attention and combine
        return rgb_weights * rgb_feat + ms_weights * ms_feat

class DualSourceDeepLabV3Plus(nn.Module):
    """DeepLabV3+ with dual-source input handling."""
    
    def __init__(
        self,
        num_classes: int,
        rgb_encoder: str = "resnet50",
        ms_encoder: str = "resnet50",
        ms_in_channels: int = 8,  # Typical for 8-band MS imagery
        fusion_channels: int = 256,
        pretrained: bool = True
    ):
        super().__init__()
        
        # RGB branch
        self.rgb_model = smp.DeepLabV3Plus(
            encoder_name=rgb_encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=fusion_channels
        )
        
        # MS branch
        self.ms_model = smp.DeepLabV3Plus(
            encoder_name=ms_encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=ms_in_channels,
            classes=fusion_channels
        )
        
        # Fusion module
        self.fusion = AdaptiveFusionModule(fusion_channels)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels, num_classes, 1)
        )
        
    def forward(
        self,
        rgb_img: Optional[torch.Tensor] = None,
        ms_img: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass supporting both single and dual-source inputs.
        
        Args:
            rgb_img: High-quality RGB image [B, 3, H, W] or None
            ms_img: Multi-spectral image [B, C, H, W] or None
            
        Returns:
            Segmentation logits [B, num_classes, H, W]
        """
        if rgb_img is None and ms_img is None:
            raise ValueError("At least one input source must be provided")
        
        # Process available inputs
        rgb_feat = self.rgb_model(rgb_img) if rgb_img is not None else None
        ms_feat = self.ms_model(ms_img) if ms_img is not None else None
        
        # Handle single-source cases
        if rgb_feat is None:
            fused_feat = ms_feat
        elif ms_feat is None:
            fused_feat = rgb_feat
        else:
            # Fusion for dual-source case
            fused_feat = self.fusion(rgb_feat, ms_feat)
        
        return self.classifier(fused_feat)
    
    def predict(
        self,
        rgb_img: Optional[torch.Tensor] = None,
        ms_img: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prediction with softmax activation."""
        logits = self.forward(rgb_img, ms_img)
        return F.softmax(logits, dim=1)