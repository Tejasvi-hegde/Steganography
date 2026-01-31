"""
FINAL Decoder v4.0 - Perfect Steganography
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with pre-activation"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = self.bn2(self.conv2(x))
        return F.leaky_relu(x + residual, 0.2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation attention block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class StegoDecoder(nn.Module):
    """
    FINAL Decoder v4.0 - Maximum recovery quality
    
    Features:
    - U-Net architecture with dense skips
    - Multi-scale feature extraction
    - Progressive upsampling
    - Detail enhancement module
    """
    def __init__(self, input_channels=3, hidden_dim=64):
        super().__init__()
        
        # Multi-scale input processing
        self.input_conv = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.input_bn = nn.BatchNorm2d(hidden_dim)
        
        self.detail_branch = nn.Conv2d(input_channels, hidden_dim//4, 1)
        self.color_branch = nn.Conv2d(input_channels, hidden_dim//4, 3, padding=1)
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Bottleneck with residual blocks
        self.bottleneck = nn.Sequential(
            ResidualBlock(hidden_dim*4),
            ResidualBlock(hidden_dim*4),
            ResidualBlock(hidden_dim*4),
            ResidualBlock(hidden_dim*4),
            ResidualBlock(hidden_dim*4),
            ResidualBlock(hidden_dim*4),
            SEBlock(hidden_dim*4),
        )
        
        # Decoder path
        self.up1 = nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(hidden_dim*4, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final refinement with multi-scale inputs
        final_in = hidden_dim + hidden_dim//4 + hidden_dim//4
        self.final = nn.Sequential(
            nn.Conv2d(final_in, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim//2, 3, 3, padding=1),
        )
        
        # Detail enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, padding=1),
        )
        
    def forward(self, stego):
        # Multi-scale input
        detail = self.detail_branch(stego)
        color = self.color_branch(stego)
        
        # Encoder
        x = F.leaky_relu(self.input_bn(self.input_conv(stego)), 0.2)
        e1 = self.enc1(x)
        
        x = self.down1(e1)
        e2 = self.enc2(x)
        
        x = self.down2(e2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skips
        x = self.up1(x)
        x = torch.cat([x, e2], dim=1)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = torch.cat([x, e1], dim=1)
        x = self.dec2(x)
        
        # Final with multi-scale
        x = torch.cat([x, detail, color], dim=1)
        x = self.final(x)
        
        # Detail enhancement
        x = x + self.enhance(x) * 0.2
        
        return torch.tanh(x)