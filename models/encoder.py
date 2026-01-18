import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AttentionBlock(nn.Module):
    """Channel attention for better feature selection"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class StegoEncoder(nn.Module):
    def __init__(self, input_channels=6, hidden_dim=64):
        super(StegoEncoder, self).__init__()
        
        # Initial convolution with more channels
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Downsample path
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim*2)
        
        # More residual blocks for better feature extraction
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2),  # Added
            ResidualBlock(hidden_dim*2),  # Added
        )
        
        # Attention for feature refinement
        self.attention = AttentionBlock(hidden_dim*2)
        
        # Upsample to original size
        self.upsample = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Skip connection processing
        self.skip_conv = nn.Conv2d(hidden_dim, hidden_dim, 1)
        
        # Refinement layers
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Final convolution - outputs difference from cover
        self.conv_out = nn.Conv2d(hidden_dim // 2, 3, 3, padding=1)
        
    def forward(self, cover, secret):
        # Concatenate cover and secret images
        x = torch.cat([cover, secret], dim=1)
        
        # Encoder path
        x1 = F.relu(self.bn1(self.conv1(x)))  # Save for skip connection
        x = F.relu(self.bn2(self.conv2(x1)))
        
        # Deep feature extraction
        x = self.res_blocks(x)
        x = self.attention(x)
        
        # Upsample
        x = F.relu(self.bn3(self.upsample(x)))
        
        # Skip connection for preserving spatial details
        skip = self.skip_conv(x1)
        x = torch.cat([x, skip], dim=1)
        
        # Refinement
        x = self.refine(x)
        
        # Output residual (change to apply to cover)
        residual = self.conv_out(x) * 0.1  # Scale down residual
        
        # Add residual to cover for better quality
        stego = cover + residual
        stego = torch.clamp(stego, -1, 1)  # Keep in valid range
        
        return stego