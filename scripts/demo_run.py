"""
Demo script to test the steganography model with random images.
No external data required!
"""
import torch
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder

def demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models
    print("Creating Encoder and Decoder models...")
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {encoder_params + decoder_params:,}")
    
    # Create random images (batch_size=2, 3 channels, 128x128)
    print("\nCreating random test images...")
    cover = torch.randn(2, 3, 128, 128).to(device)
    secret = torch.randn(2, 3, 128, 128).to(device)
    
    print(f"Cover shape: {cover.shape}")
    print(f"Secret shape: {secret.shape}")
    
    # Forward pass
    print("\nRunning forward pass...")
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode: hide secret inside cover
        stego = encoder(cover, secret)
        print(f"Stego (encoded) shape: {stego.shape}")
        
        # Decode: recover secret from stego
        recovered = decoder(stego)
        print(f"Recovered shape: {recovered.shape}")
    
    # Calculate metrics
    mse = torch.mean((secret - recovered) ** 2).item()
    print(f"\nMSE between secret and recovered: {mse:.4f}")
    print("(Note: Without training, recovery is random - this just tests architecture)")
    
    # Test training mode
    print("\n--- Testing Training Mode ---")
    encoder.train()
    decoder.train()
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=0.001
    )
    
    # Simple training step
    for i in range(3):
        optimizer.zero_grad()
        
        cover_batch = torch.randn(4, 3, 128, 128).to(device)
        secret_batch = torch.randn(4, 3, 128, 128).to(device)
        
        stego = encoder(cover_batch, secret_batch)
        recovered = decoder(stego)
        
        # Simple losses
        cover_loss = torch.mean((cover_batch - stego) ** 2)  # stego should look like cover
        secret_loss = torch.mean((secret_batch - recovered) ** 2)  # recovery should match secret
        loss = cover_loss + secret_loss
        
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}: Loss = {loss.item():.4f}, Cover Loss = {cover_loss.item():.4f}, Secret Loss = {secret_loss.item():.4f}")
    
    print("\n✅ Demo completed successfully!")
    print("The steganography architecture is working correctly.")

if __name__ == "__main__":
    demo()
