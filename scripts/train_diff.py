import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder
from utils.dataset import StegoDataset
from utils.losses import StegoLoss

def train_different_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("🚀 Training with DIFFERENT cover/secret images")
    
    # Configuration
    batch_size = 4
    epochs = 10  # Train a bit longer for different images
    image_size = 128
    
    # Create output directories
    os.makedirs('../outputs/checkpoints_diff', exist_ok=True)
    
    # Dataset with DIFFERENT images
    train_dataset = StegoDataset(
        cover_dir='../data/train_diff/cover',
        secret_dir='../data/train_diff/secret', 
        image_size=image_size,
        use_different_images=True  # This is the key change!
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    
    # Models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )
    
    criterion = StegoLoss(alpha=1.0, beta=0.7, gamma=0.0)
    
    print(f"Dataset: {len(train_dataset)} DIFFERENT image pairs")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}x{image_size}")
    print("=" * 50)
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_quality_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (cover_imgs, secret_imgs, _) in enumerate(pbar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
            # Forward pass
            stego_imgs = encoder(cover_imgs, secret_imgs)
            recovered_imgs = decoder(stego_imgs)
            
            total_loss, recon_loss, quality_loss = criterion(
                cover_imgs, stego_imgs, secret_imgs, recovered_imgs
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_quality_loss += quality_loss.item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                current_avg = epoch_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f"{total_loss.item():.4f}",
                    'AvgLoss': f"{current_avg:.4f}"
                })
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_quality = epoch_quality_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, Quality: {avg_quality:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f'../outputs/checkpoints_diff/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'../outputs/checkpoints_diff/decoder_epoch_{epoch+1}.pth')
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save final models
    torch.save(encoder.state_dict(), '../outputs/checkpoints_diff/encoder_final.pth')
    torch.save(decoder.state_dict(), '../outputs/checkpoints_diff/decoder_final.pth')
    print("🎉 Training with different images completed!")

if __name__ == "__main__":
    train_different_images()