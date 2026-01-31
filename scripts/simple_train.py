import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import time

# Set environment variables for optimal CPU utilization
num_cores = os.cpu_count() or 4
# Limit threads to avoid CPU overload
num_threads = min(num_cores, 8)
os.environ['OMP_NUM_THREADS'] = str(num_threads)
os.environ['MKL_NUM_THREADS'] = str(num_threads)
os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder
from utils.dataset import StegoDataset
from utils.losses import StegoLoss


def train_cpu_optimized():
    """
    CPU-optimized training with different cover/secret images
    """
    # Force CPU - optimized settings for laptops without GPU
    device = torch.device('cpu')
    print(f"🖥️  Training on CPU")
    print(f"📊 Available CPU cores: {num_cores}")
    print(f"⚡ Using {num_threads} threads for computation")
    
    # Set PyTorch threads
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(2)  # For parallel operations
    
    # ═══════════════════════════════════════════════════════════
    # CPU-OPTIMIZED SETTINGS
    # ═══════════════════════════════════════════════════════════
    batch_size = 2          # Small batch for CPU memory
    epochs = 20             # More epochs for better convergence
    image_size = 128        # 128x128 is good balance of quality/speed
    learning_rate = 0.0005  # Slightly lower LR for stability
    save_every = 5          # Save checkpoint every N epochs
    
    # Data loading workers (don't use too many on CPU)
    num_workers = min(2, num_cores - 1) if num_cores > 1 else 0
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION (CPU Optimized)")
    print(f"{'='*60}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Epochs:        {epochs}")
    print(f"  Image size:    {image_size}x{image_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Data workers:  {num_workers}")
    print(f"{'='*60}\n")
    # Create output directories
    os.makedirs('../outputs/checkpoints', exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # DATASET - Uses DIFFERENT images for cover and secret
    # ═══════════════════════════════════════════════════════════
    train_dir = '../data/train'
    cover_dir = os.path.join(train_dir, 'cover')
    secret_dir = os.path.join(train_dir, 'secret')
    
    # Check if data exists
    if not os.path.exists(cover_dir) or not os.path.exists(secret_dir):
        print("❌ Training data not found!")
        print("\nPlease run first:")
        print("  python download_dataset.py --dataset picsum --num_images 1000")
        print("\nThis will download ~1000 images and prepare training data.")
        return
    
    train_dataset = StegoDataset(
        cover_dir=cover_dir,
        secret_dir=secret_dir, 
        image_size=image_size,
        use_different_images=True  # IMPORTANT: Use different images!
    )
    
    if len(train_dataset) == 0:
        print("❌ No images found in training directory!")
        return
    
    # Optimized DataLoader for CPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=False,  # No need for pin_memory on CPU
        drop_last=True     # Drop incomplete batches for stability
    )
    
    # Models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"📦 Encoder parameters: {encoder_params:,}")
    print(f"📦 Decoder parameters: {decoder_params:,}")
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    # Loss function (disable perceptual loss for faster CPU training)
    criterion = StegoLoss(
        alpha=1.0,      # Secret reconstruction weight
        beta=0.7,       # Stego quality weight  
        gamma=0.0,      # No adversarial loss (no discriminator)
        use_perceptual=False,  # Disable for CPU (VGG is slow)
        use_ssim=True,  # Keep SSIM for quality
        device=device
    )
    
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    print(f"📁 Dataset: {len(train_dataset)} image pairs")
    print(f"🔄 Batches per epoch: {len(train_loader)}")
    print(f"⏱️  Estimated time per epoch: ~{len(train_loader) * 2}s")
    print(f"{'='*60}\n")
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_quality_loss = 0
        
        encoder.train()
        decoder.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (cover_imgs, secret_imgs, _) in enumerate(pbar):
            cover_imgs = cover_imgs.to(device)
            secret_imgs = secret_imgs.to(device)
            
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
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_quality_loss += quality_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{total_loss.item():.4f}",
                'Recon': f"{recon_loss.item():.4f}",
                'Quality': f"{quality_loss.item():.4f}"
            })
        
        # Step scheduler
        scheduler.step()
        
        # Calculate epoch statistics
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = epoch_recon_loss / len(train_loader)
        avg_quality = epoch_quality_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f'\n📊 Epoch {epoch+1} Summary:')
        print(f'   Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | Quality: {avg_quality:.4f}')
        print(f'   Time: {epoch_time:.1f}s | LR: {current_lr:.6f}')
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), '../outputs/checkpoints/encoder_best.pth')
            torch.save(decoder.state_dict(), '../outputs/checkpoints/decoder_best.pth')
            print(f'   ✓ New best model saved!')
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            torch.save(encoder.state_dict(), f'../outputs/checkpoints/encoder_epoch_{epoch+1}.pth')
            torch.save(decoder.state_dict(), f'../outputs/checkpoints/decoder_epoch_{epoch+1}.pth')
            print(f'   ✓ Checkpoint saved at epoch {epoch+1}')
    
    # Save final models
    torch.save(encoder.state_dict(), '../outputs/checkpoints/encoder_final.pth')
    torch.save(decoder.state_dict(), '../outputs/checkpoints/decoder_final.pth')
    
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"📁 Models saved to: ../outputs/checkpoints/")
    print(f"   - encoder_final.pth")
    print(f"   - decoder_final.pth") 
    print(f"   - encoder_best.pth (lowest loss)")
    print(f"   - decoder_best.pth (lowest loss)")
    print(f"\n📝 Next steps:")
    print(f"   1. Test: python quick_test.py")
    print(f"   2. Hide image: python hide.py --cover img1.png --secret img2.png --output stego.png --encoder ../outputs/checkpoints/encoder_final.pth")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_cpu_optimized()