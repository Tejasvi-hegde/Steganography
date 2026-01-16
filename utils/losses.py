import torch
import torch.nn as nn

class StegoLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.7, gamma=0.01):
        super(StegoLoss, self).__init__()
        self.alpha = alpha  # Reconstruction weight
        self.beta = beta    # Stego quality weight
        self.gamma = gamma  # Adversarial weight
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, cover, stego, secret, recovered, disc_fake=None):
        # Reconstruction loss (secret recovery)
        recon_loss = self.mse_loss(secret, recovered)
        
        # Stego quality loss (visual similarity)
        quality_loss = self.mse_loss(cover, stego)
        
        # Adversarial loss (if using GAN)
        if disc_fake is not None:
            adversarial_loss = -torch.log(disc_fake).mean()
        else:
            adversarial_loss = 0
            
        total_loss = (self.alpha * recon_loss + 
                     self.beta * quality_loss + 
                     self.gamma * adversarial_loss)
        
        return total_loss, recon_loss, quality_loss