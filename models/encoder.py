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

class StegoEncoder(nn.Module):
    def __init__(self, input_channels=6, hidden_dim=64):
        super(StegoEncoder, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        # Downsample
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim*2)
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2),
            ResidualBlock(hidden_dim*2)
        )
        
        # Upsample to original size
        self.upsample = nn.ConvTranspose2d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        # Final convolution
        self.conv_out = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        
    def forward(self, cover, secret):
        # Concatenate cover and secret images
        x = torch.cat([cover, secret], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.res_blocks(x)
        x = F.relu(self.bn3(self.upsample(x)))
        x = torch.tanh(self.conv_out(x))
        
        return x