import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from spectral_bands import SpectralBands, SpectralIndices, SpectralFeatureExtractor


class BandAttention(nn.Module):
    def __init__(self, num_bands, reduction=16):
        super(BandAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_bands, num_bands // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands // reduction, num_bands, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Atrous convolutions at different rates
        self.aspp1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)


class MultiSpectralDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, use_indices=True):
        super(MultiSpectralDeepLabV3Plus, self).__init__()
        
        # Spectral feature extraction
        self.spectral_extractor = SpectralFeatureExtractor(use_indices=use_indices)
        
        # Calculate total number of input channels
        num_spectral_bands = len(SpectralBands.get_band_wavelengths())
        num_indices = 7 if use_indices else 0  # 7 spectral indices if enabled
        total_channels = num_spectral_bands + num_indices
        
        # Modified ResNet backbone for multi-spectral input
        resnet = resnet50(pretrained=True)
        self.initial_conv = nn.Conv2d(total_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Band attention module
        self.band_attention = BandAttention(total_channels)
        
        # Encoder (modified ResNet)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(256 + 256, 256, 3, padding=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_conv3 = nn.Conv2d(256, num_classes, 1)

        # Low-level features conv
        self.low_level_conv = nn.Conv2d(256, 256, 1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(256)

    def forward(self, x):
        # Extract spectral features and indices
        x = self.spectral_extractor(x)
        
        # Apply band attention
        x = self.band_attention(x)
        
        # Encoder
        x = self.initial_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        low_level_feat = self.layer1(x)
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        
        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)
        x = self.relu(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.relu(x)
        x = self.decoder_conv3(x)
        
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        
        return x


# Example usage
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    height = 512
    width = 512
    num_bands = len(SpectralBands.get_band_wavelengths())
    
    # Initialize model
    model = MultiSpectralDeepLabV3Plus(num_classes=21, use_indices=True)
    
    # Create sample input with all available spectral bands
    x = torch.randn(batch_size, num_bands, height, width)
    
    # Forward pass
    output = model(x)
    
    # Print shapes and band information
    print("\nAvailable Spectral Bands:")
    for name, idx in vars(SpectralBands).items():
        if not name.startswith('_') and isinstance(idx, int):
            wavelength = SpectralBands.get_band_wavelengths()[idx]
            print(f"Band {idx}: {name.lower()} (~{wavelength}nm)")
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate some example indices
    indices = SpectralIndices()
    ndvi = indices.ndvi(x)
    evi = indices.evi(x)
    print(f"\nNDVI shape: {ndvi.shape}")
    print(f"EVI shape: {evi.shape}")