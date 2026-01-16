import torch
import torch.nn as nn
import torch.nn.functional as F

class StegoDecoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=64):
        super(StegoDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, 3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(hidden_dim*2)
        
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, 3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
        )
        
        self.upsample = nn.ConvTranspose2d(hidden_dim*4, hidden_dim, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_dim)
        
        self.conv_out = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        
    def forward(self, stego_image):
        x = F.relu(self.bn1(self.conv1(stego_image)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv_blocks(x)
        x = F.relu(self.bn3(self.upsample(x)))
        x = torch.tanh(self.conv_out(x))
        
        return x