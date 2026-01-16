import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Set environment variables for optimal CPU utilization
num_cores = os.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(num_cores)
os.environ['MKL_NUM_THREADS'] = str(num_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores)

sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder
from utils.dataset import StegoDataset
from utils.losses import StegoLoss

def train_fast():
    # Configuration for fast training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Available CPU cores: {num_cores}")
    
    # SMALLER settings
    batch_size = 4    
    epochs = 5        
    image_size = 128    
    
    # Create output directories
    os.makedirs('../outputs/checkpoints', exist_ok=True)
    
    # Dataset - use smaller images
    train_dataset = StegoDataset(
        cover_dir='../data/train/cover',
        secret_dir='../data/train/secret', 
        image_size=image_size
    )
    
    # Optimized DataLoader using all CPU cores
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_cores,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    # Models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    # If using CPU for training (not recommended)
    if device.type == 'cpu':
        torch.set_num_threads(num_cores)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001,
        weight_decay=1e-5
    )
    
    criterion = StegoLoss(alpha=1.0, beta=0.7, gamma=0.0)
    
    print("Starting fast training...")
    print(f"Dataset: {len(train_dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Using {num_cores} CPU cores for data loading")
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_quality_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (cover_imgs, secret_imgs, _) in enumerate(pbar):
            cover_imgs = cover_imgs.to(device, non_blocking=True)
            secret_imgs = secret_imgs.to(device, non_blocking=True)
            
            # Forward pass
            stego_imgs = encoder(cover_imgs, secret_imgs)
            recovered_imgs = decoder(stego_imgs)
            
            # Calculate loss
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
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}"
            })
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_quality = epoch_quality_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, Quality: {avg_quality:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(encoder.state_dict(), f'../outputs/checkpoints/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'../outputs/checkpoints/decoder_epoch_{epoch+1}.pth')
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final models
    torch.save(encoder.state_dict(), '../outputs/checkpoints/encoder_final.pth')
    torch.save(decoder.state_dict(), '../outputs/checkpoints/decoder_final.pth')
    print("Training completed! Models saved.")

if __name__ == "__main__":
    train_fast()