"""Dual-source DeepLabV3+ implementation."""

import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation

class DualSourceDeepLabV3Plus(nn.Module):
    """DeepLabV3+ model modified for dual-source input (RGB + Multi-spectral)."""
    
    def __init__(
        self,
        num_classes: int = 21,
        rgb_pretrained: bool = True,
        ms_pretrained: bool = False,
        fusion_type: str = "concat",
    ):
        """Initialize the dual-source DeepLabV3+ model.
        
        Args:
            num_classes: Number of output classes
            rgb_pretrained: Whether to use pretrained weights for RGB branch
            ms_pretrained: Whether to use pretrained weights for multi-spectral branch
            fusion_type: Type of fusion ("concat", "sum", "attention")
        """
        super().__init__()
        
        # RGB branch
        self.rgb_branch = segmentation.deeplabv3_resnet101(
            pretrained=rgb_pretrained,
            num_classes=num_classes
        )
        
        # Multi-spectral branch
        self.ms_branch = segmentation.deeplabv3_resnet101(
            pretrained=ms_pretrained,
            num_classes=num_classes
        )
        
        # Modify first conv layer of MS branch for different input channels
        self.ms_branch.backbone.conv1 = nn.Conv2d(
            8,  # Assuming 8-channel multi-spectral input
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        self.fusion_type = fusion_type
        if fusion_type == "concat":
            self.fusion = nn.Conv2d(num_classes * 2, num_classes, 1)
        elif fusion_type == "attention":
            self.fusion = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, 1),
                nn.Sigmoid()
            )
    
    def forward(self, rgb_x: torch.Tensor, ms_x: torch.Tensor) -> torch.Tensor:
        """Forward pass with both RGB and multi-spectral inputs.
        
        Args:
            rgb_x: RGB input tensor [B, 3, H, W]
            ms_x: Multi-spectral input tensor [B, 8, H, W]
            
        Returns:
            Segmentation output tensor [B, num_classes, H, W]
        """
        rgb_out = self.rgb_branch(rgb_x)["out"]
        ms_out = self.ms_branch(ms_x)["out"]
        
        if self.fusion_type == "sum":
            return rgb_out + ms_out
        elif self.fusion_type == "concat":
            return self.fusion(torch.cat([rgb_out, ms_out], dim=1))
        elif self.fusion_type == "attention":
            attention = self.fusion(torch.cat([rgb_out, ms_out], dim=1))
            return attention * rgb_out + (1 - attention) * ms_out
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")