"""
FINAL Encoder v4.0 - Perfect Steganography
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


class StegoEncoder(nn.Module):
    """
    FINAL Encoder v4.0 - Perfect balance of invisibility and capacity
    
    Features:
    - Dual-branch processing (cover + secret)
    - Adaptive residual scaling (learned per-channel)
    - Dense skip connections
    - SE attention for channel importance
    """
    def __init__(self, input_channels=6, hidden_dim=64):
        super().__init__()
        
        # Separate feature extraction
        self.cover_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.secret_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Downsampling
        self.down1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Deep processing with attention
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim*2) for _ in range(8)
        ])
        self.attention = SEBlock(hidden_dim*2)
        
        # Upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Output refinement with skip
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim//2, 3, 3, padding=1),
        )
        
        # Learnable per-channel scaling
        self.scale = nn.Parameter(torch.ones(1, 3, 1, 1) * 0.1)
        
    def forward(self, cover, secret):
        # Extract features separately
        cover_feat = self.cover_encoder(cover)
        secret_feat = self.secret_encoder(secret)
        
        # Fuse features
        x = torch.cat([cover_feat, secret_feat], dim=1)
        x = self.fusion(x)
        skip = x
        
        # Downsample
        x = self.down1(x)
        
        # Deep processing
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.attention(x)
        
        # Upsample
        x = self.up1(x)
        
        # Refine with skip connection
        x = torch.cat([x, skip], dim=1)
        residual = self.refine(x)
        
        # Apply learned scaling (clamped for stability)
        scale = torch.clamp(self.scale, 0.03, 0.12)
        residual = torch.tanh(residual) * scale
        
        # Add to cover
        stego = cover + residual
        return torch.clamp(stego, -1, 1)