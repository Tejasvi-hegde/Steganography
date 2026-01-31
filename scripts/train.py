import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder
from models.discriminator import Discriminator
from utils.dataset import StegoDataset
from utils.losses import StegoLoss

class StegoTrainer:
    def __init__(self, encoder, decoder, discriminator=None, lr=0.0002):
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        
        self.optimizer_G = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )
        
        if discriminator:
            self.optimizer_D = torch.optim.Adam(
                discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
            )
            
        self.criterion = StegoLoss()
        self.mse = nn.MSELoss()
        
    def train_step(self, cover_imgs, secret_imgs):
        batch_size = cover_imgs.size(0)
        
        # Generate stego images
        stego_imgs = self.encoder(cover_imgs, secret_imgs)
        recovered_imgs = self.decoder(stego_imgs)
        
        # Train Generator (Encoder + Decoder)
        self.optimizer_G.zero_grad()
        
        if self.discriminator:
            disc_fake = self.discriminator(stego_imgs)
        else:
            disc_fake = None
            
        g_loss, recon_loss, quality_loss = self.criterion(
            cover_imgs, stego_imgs, secret_imgs, recovered_imgs, disc_fake
        )
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # Train Discriminator (if using GAN)
        d_loss = 0
        if self.discriminator:
            self.optimizer_D.zero_grad()
            
            # Real images
            pred_real = self.discriminator(cover_imgs)
            loss_real = self.mse(pred_real, torch.ones_like(pred_real))
            
            # Fake images
            pred_fake = self.discriminator(stego_imgs.detach())
            loss_fake = self.mse(pred_fake, torch.zeros_like(pred_fake))
            
            d_loss = (loss_real + loss_fake) * 0.5
            d_loss.backward()
            self.optimizer_D.step()
            
        return {
            'g_loss': g_loss.item(),
            'recon_loss': recon_loss.item(),
            'quality_loss': quality_loss.item(),
            'd_loss': d_loss.item() if self.discriminator else 0,
            'psnr': self.calculate_psnr(secret_imgs, recovered_imgs)
        }
    
    def calculate_psnr(self, original, reconstructed):
        # Denormalize from [-1,1] to [0,1] for proper PSNR calculation
        orig = (original + 1) / 2
        recon = (reconstructed + 1) / 2
        mse = torch.mean((orig - recon) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * torch.log10(1.0 / mse).item()

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 8
    epochs = 100
    image_size = 256
    
    # Create output directories
    os.makedirs('../outputs/checkpoints', exist_ok=True)
    os.makedirs('../outputs/tensorboard', exist_ok=True)
    
    # Dataset and DataLoader
    train_dataset = StegoDataset(
        cover_dir='../data/train/cover',
        secret_dir='../data/train/secret',
        image_size=image_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    discriminator = Discriminator().to(device)
    
    # Trainer
    trainer = StegoTrainer(encoder, decoder, discriminator)
    
    # TensorBoard
    writer = SummaryWriter('../outputs/tensorboard')
    
    # Training loop
    for epoch in range(epochs):
        epoch_losses = {'g_loss': 0, 'recon_loss': 0, 'quality_loss': 0, 'd_loss': 0, 'psnr': 0}
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (cover_imgs, secret_imgs, _) in enumerate(pbar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            losses = trainer.train_step(cover_imgs, secret_imgs)
            
            # Update progress bar
            for k, v in losses.items():
                epoch_losses[k] += v
                
            pbar.set_postfix({
                'G_Loss': f"{losses['g_loss']:.4f}",
                'PSNR': f"{losses['psnr']:.2f}"
            })
            
            # Log to TensorBoard
            if batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                for k, v in losses.items():
                    writer.add_scalar(f'Batch/{k}', v, global_step)
            
        # Print epoch statistics
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Generator Loss: {epoch_losses["g_loss"]:.4f}')
        print(f'  Reconstruction Loss: {epoch_losses["recon_loss"]:.4f}')
        print(f'  Quality Loss: {epoch_losses["quality_loss"]:.4f}')
        print(f'  Discriminator Loss: {epoch_losses["d_loss"]:.4f}')
        print(f'  PSNR: {epoch_losses["psnr"]:.2f} dB')
        
        # Log epoch statistics to TensorBoard
        for k, v in epoch_losses.items():
            writer.add_scalar(f'Epoch/{k}', v, epoch)
        
        # Save models periodically
        if (epoch + 1) % 10 == 0:
            torch.save(encoder.state_dict(), f'../outputs/checkpoints/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'../outputs/checkpoints/decoder_epoch_{epoch+1}.pth')
            if discriminator:
                torch.save(discriminator.state_dict(), f'../outputs/checkpoints/discriminator_epoch_{epoch+1}.pth')
    
    # Save final models
    torch.save(encoder.state_dict(), '../outputs/checkpoints/encoder_final.pth')
    torch.save(decoder.state_dict(), '../outputs/checkpoints/decoder_final.pth')
    if discriminator:
        torch.save(discriminator.state_dict(), '../outputs/checkpoints/discriminator_final.pth')
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    main()