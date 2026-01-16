import torch
from torchvision import transforms
from PIL import Image
import argparse
import sys
import os
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder

def hide_secret(cover_path, secret_path, output_path, encoder_path, device='cuda'):
    """Hide secret image in cover image"""
    
    # Load models
    encoder = StegoEncoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    
    # Load and transform images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cover_image = Image.open(cover_path).convert('RGB')
    secret_image = Image.open(secret_path).convert('RGB')
    
    cover_tensor = transform(cover_image).unsqueeze(0).to(device)
    secret_tensor = transform(secret_image).unsqueeze(0).to(device)
    
    # Generate stego image
    with torch.no_grad():
        stego_tensor = encoder(cover_tensor, secret_tensor)
    
    # Convert back to image
    stego_image = transforms.ToPILImage()(stego_tensor.squeeze(0).cpu() * 0.5 + 0.5)
    stego_image.save(output_path)
    print(f"Stego image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hide secret image in cover image')
    parser.add_argument('--cover', type=str, required=True, help='Path to cover image')
    parser.add_argument('--secret', type=str, required=True, help='Path to secret image')
    parser.add_argument('--output', type=str, required=True, help='Path to save stego image')
    parser.add_argument('--encoder', type=str, required=True, help='Path to encoder model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    hide_secret(args.cover, args.secret, args.output, args.encoder, args.device)