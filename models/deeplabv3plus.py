import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.atrous_1 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.atrous_6 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=6, dilation=6)
        self.atrous_12 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=12, dilation=12)
        self.atrous_18 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channels * 4, out_channels, 1, 1)
        
    def forward(self, x):
        atrous_1 = self.atrous_1(x)
        atrous_6 = self.atrous_6(x)
        atrous_12 = self.atrous_12(x)
        atrous_18 = self.atrous_18(x)
        
        x = torch.cat([atrous_1, atrous_6, atrous_12, atrous_18], dim=1)
        x = self.conv_1x1_output(x)
        return x

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', pretrained=True, in_channels=3):
        super(DeepLabV3Plus, self).__init__()
        
        # Modify the first conv layer to accept arbitrary number of input channels
        if backbone == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
            if in_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
        
        self.aspp = ASPP(2048, 256)
        
        # Low-level features
        self.low_level_conv = nn.Conv2d(256, 48, 1, 1)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(304, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1, 1)
        )
        
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        low_level_feat = self.backbone.layer1(x)
        x = self.backbone.layer2(low_level_feat)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        x = F.interpolate(x, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        # Low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # Combine features
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Upsample to input resolution
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x