import torch
import torch.nn as nn
import torch.nn.functional as F

class StegoEncoderSimple(nn.Module):
    """Encoder network that hides secret image in cover image - Simple version matching Colab training"""
    def __init__(self):
        super(StegoEncoderSimple, self).__init__()
        
        # Cover image encoder
        self.cover_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.cover_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.cover_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Secret image encoder
        self.secret_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.secret_conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.secret_conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        
        # Fusion layers
        self.fusion = nn.Conv2d(512, 256, 3, padding=1)
        
        # Decoder to generate stego image
        self.upsample = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
    def forward(self, cover, secret):
        # Encode cover
        c1 = F.relu(self.bn1(self.cover_conv1(cover)))
        c2 = F.relu(self.bn2(self.cover_conv2(c1)))
        c3 = F.relu(self.cover_conv3(c2))
        
        # Encode secret
        s1 = F.relu(self.secret_conv1(secret))
        s2 = F.relu(self.secret_conv2(s1))
        s3 = F.relu(self.secret_conv3(s2))
        
        # Fuse
        fused = torch.cat([c3, s3], dim=1)
        x = F.relu(self.fusion(fused))
        
        # Decode to stego image
        x = F.relu(self.bn3(self.upsample(x)))
        x = F.relu(self.bn4(self.upsample2(x)))
        stego = torch.tanh(self.final(x))
        
        return stego
