import torch
import torch.nn as nn
import torch.nn.functional as F

class StegoDecoderSimple(nn.Module):
    """Decoder network that recovers secret image from stego image - Simple version matching Colab training"""
    def __init__(self):
        super(StegoDecoderSimple, self).__init__()
        
        # Encoder path
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        
        # Decoder path
        self.up1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        
        # Batch norm
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(64)
        
    def forward(self, stego):
        # Encode
        x1 = F.relu(self.bn1(self.conv1(stego)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.conv4(x3))
        
        # Decode
        x = F.relu(self.bn4(self.up1(x4)))
        x = F.relu(self.bn5(self.up2(x)))
        x = F.relu(self.bn6(self.up3(x)))
        secret = torch.tanh(self.final(x))
        
        return secret
