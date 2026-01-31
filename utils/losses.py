import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss for better visual quality"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        # Normalize to VGG input range
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        y = (y + 1) / 2
        return F.mse_loss(self.vgg(x), self.vgg(y))


class SSIMLoss(nn.Module):
    """Structural Similarity Loss for better perceptual quality"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = self._create_window(window_size, 3)
        
    def _create_window(self, window_size, channel):
        def gaussian(size, sigma=1.5):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - size//2)**2 / (2 * sigma**2)))
                for x in range(size)
            ])
            return gauss / gauss.sum()
        
        _1D = gaussian(window_size).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()
    
    def forward(self, img1, img2):
        # Move to [-1, 1] -> [0, 1]
        img1 = (img1 + 1) / 2
        img2 = (img2 + 1) / 2
        
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=3)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=3) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return 1 - ssim_map.mean()  # 1 - SSIM to make it a loss


class StegoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.7, gamma=0.01, use_perceptual=True, use_ssim=True, device=None):
        super(StegoLoss, self).__init__()
        self.alpha = alpha      # Reconstruction weight
        self.beta = beta        # Stego quality weight
        self.gamma = gamma      # Adversarial weight
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim
        
        # Auto-detect device if not provided
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        if use_perceptual:
            self.perceptual = PerceptualLoss().to(device)
            
        if use_ssim:
            self.ssim = SSIMLoss()
        
    def forward(self, cover, stego, secret, recovered, disc_fake=None):
        # Reconstruction loss (secret recovery) - use L1 + MSE
        recon_loss = self.mse_loss(secret, recovered) + 0.5 * self.l1_loss(secret, recovered)
        
        # Stego quality loss (visual similarity)
        quality_loss = self.mse_loss(cover, stego) + 0.5 * self.l1_loss(cover, stego)
        
        # Add perceptual loss for better visual quality
        if self.use_perceptual:
            quality_loss = quality_loss + 0.1 * self.perceptual(cover, stego)
            recon_loss = recon_loss + 0.1 * self.perceptual(secret, recovered)
            
        # Add SSIM loss for structural similarity
        if self.use_ssim:
            quality_loss = quality_loss + 0.2 * self.ssim(cover, stego)
            recon_loss = recon_loss + 0.2 * self.ssim(secret, recovered)
        
        # Adversarial loss (if using GAN)
        if disc_fake is not None:
            adversarial_loss = -torch.log(disc_fake + 1e-8).mean()
        else:
            adversarial_loss = 0
            
        total_loss = (self.alpha * recon_loss + 
                     self.beta * quality_loss + 
                     self.gamma * adversarial_loss)
        
        return total_loss, recon_loss, quality_loss