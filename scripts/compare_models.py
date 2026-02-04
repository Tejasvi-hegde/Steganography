"""
Compare two different trained steganography models.
Runs the same test images through both models and compares quality metrics.

Usage:
  cd scripts
  python compare_models.py --model1_encoder <path> --model1_decoder <path> --model2_encoder <path> --model2_decoder <path>

Example:
  python compare_models.py \
    --model1_encoder ../outputs/checkpoints_yesterday/encoder_final.pth \
    --model1_decoder ../outputs/checkpoints_yesterday/decoder_final.pth \
    --model2_encoder ../outputs/checkpoints/encoder_final.pth \
    --model2_decoder ../outputs/checkpoints/decoder_final.pth
"""

import sys
sys.path.append('..')

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import math

# Import both model architectures
from models.encoder_simple import StegoEncoderSimple
from models.decoder_simple import StegoDecoderSimple

def calculate_psnr(img1, img2):
    """Calculate PSNR between two tensors (range 0-1)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def calculate_ssim(img1, img2, window_size=11):
    """Simplified SSIM calculation"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = img1.mean()
    mu2 = img2.mean()
    
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim.item()

def load_model(encoder_path, decoder_path, device):
    """Load encoder and decoder from paths"""
    encoder = StegoEncoderSimple()
    decoder = StegoDecoderSimple()
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.to(device).eval()
    decoder.to(device).eval()
    
    return encoder, decoder

def test_model(encoder, decoder, cover, secret, device):
    """Run inference and return metrics"""
    with torch.no_grad():
        cover = cover.to(device)
        secret = secret.to(device)
        
        # Encode
        stego = encoder(cover, secret)
        
        # Decode
        recovered = decoder(stego)
        
        # Denormalize to 0-1 range
        cover_01 = (cover * 0.5 + 0.5).clamp(0, 1)
        secret_01 = (secret * 0.5 + 0.5).clamp(0, 1)
        stego_01 = (stego * 0.5 + 0.5).clamp(0, 1)
        recovered_01 = (recovered * 0.5 + 0.5).clamp(0, 1)
        
        # Calculate metrics
        psnr_stego = calculate_psnr(cover_01, stego_01)
        psnr_recovered = calculate_psnr(secret_01, recovered_01)
        ssim_stego = calculate_ssim(cover_01, stego_01)
        ssim_recovered = calculate_ssim(secret_01, recovered_01)
        
        return {
            'psnr_stego': psnr_stego,
            'psnr_recovered': psnr_recovered,
            'ssim_stego': ssim_stego,
            'ssim_recovered': ssim_recovered,
            'stego': stego_01,
            'recovered': recovered_01
        }

def main():
    parser = argparse.ArgumentParser(description='Compare two steganography models')
    parser.add_argument('--model1_encoder', type=str, required=True, help='Path to model 1 encoder')
    parser.add_argument('--model1_decoder', type=str, required=True, help='Path to model 1 decoder')
    parser.add_argument('--model2_encoder', type=str, required=True, help='Path to model 2 encoder')
    parser.add_argument('--model2_decoder', type=str, required=True, help='Path to model 2 decoder')
    parser.add_argument('--model1_name', type=str, default='Yesterday', help='Name for model 1')
    parser.add_argument('--model2_name', type=str, default='Today', help='Name for model 2')
    parser.add_argument('--cover', type=str, default='../data/train/cover/image_000000.png', help='Cover image path')
    parser.add_argument('--secret', type=str, default='../data/train/secret/image_000010.png', help='Secret image path')
    parser.add_argument('--output_dir', type=str, default='../outputs/comparison', help='Output directory')
    parser.add_argument('--num_tests', type=int, default=5, help='Number of test images')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    print(f"\nLoading {args.model1_name} model...")
    encoder1, decoder1 = load_model(args.model1_encoder, args.model1_decoder, device)
    
    print(f"Loading {args.model2_name} model...")
    encoder2, decoder2 = load_model(args.model2_encoder, args.model2_decoder, device)
    
    # Prepare transform
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    # Get test images
    cover_dir = '../data/train/cover'
    secret_dir = '../data/train/secret'
    
    cover_files = sorted([f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:args.num_tests]
    secret_files = sorted([f for f in os.listdir(secret_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:args.num_tests]
    
    print(f"\n{'='*60}")
    print(f"COMPARISON: {args.model1_name} vs {args.model2_name}")
    print(f"{'='*60}")
    
    model1_psnr_stego = []
    model1_psnr_recovered = []
    model2_psnr_stego = []
    model2_psnr_recovered = []
    
    for i, (cover_file, secret_file) in enumerate(zip(cover_files, secret_files)):
        cover_path = os.path.join(cover_dir, cover_file)
        secret_path = os.path.join(secret_dir, secret_file)
        
        cover_img = Image.open(cover_path).convert('RGB')
        secret_img = Image.open(secret_path).convert('RGB')
        
        cover_t = transform(cover_img).unsqueeze(0)
        secret_t = transform(secret_img).unsqueeze(0)
        
        # Test both models
        results1 = test_model(encoder1, decoder1, cover_t, secret_t, device)
        results2 = test_model(encoder2, decoder2, cover_t, secret_t, device)
        
        model1_psnr_stego.append(results1['psnr_stego'])
        model1_psnr_recovered.append(results1['psnr_recovered'])
        model2_psnr_stego.append(results2['psnr_stego'])
        model2_psnr_recovered.append(results2['psnr_recovered'])
        
        print(f"\nTest {i+1}: {cover_file} + {secret_file}")
        print(f"  {args.model1_name}:")
        print(f"    Stego PSNR: {results1['psnr_stego']:.2f} dB | Recovered PSNR: {results1['psnr_recovered']:.2f} dB")
        print(f"  {args.model2_name}:")
        print(f"    Stego PSNR: {results2['psnr_stego']:.2f} dB | Recovered PSNR: {results2['psnr_recovered']:.2f} dB")
        
        # Save comparison images for first test
        if i == 0:
            # Original images
            cover_01 = (cover_t * 0.5 + 0.5).clamp(0, 1)
            secret_01 = (secret_t * 0.5 + 0.5).clamp(0, 1)
            
            save_image(cover_01, os.path.join(args.output_dir, 'original_cover.png'))
            save_image(secret_01, os.path.join(args.output_dir, 'original_secret.png'))
            save_image(results1['stego'], os.path.join(args.output_dir, f'{args.model1_name.lower()}_stego.png'))
            save_image(results1['recovered'], os.path.join(args.output_dir, f'{args.model1_name.lower()}_recovered.png'))
            save_image(results2['stego'], os.path.join(args.output_dir, f'{args.model2_name.lower()}_stego.png'))
            save_image(results2['recovered'], os.path.join(args.output_dir, f'{args.model2_name.lower()}_recovered.png'))
            
            # Create side-by-side comparison
            comparison = torch.cat([
                cover_01[0], secret_01[0],
                results1['stego'][0], results1['recovered'][0],
                results2['stego'][0], results2['recovered'][0]
            ], dim=2)  # Concatenate horizontally
            save_image(comparison, os.path.join(args.output_dir, 'comparison_grid.png'))
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\n{args.model1_name} Average Metrics:")
    print(f"  Stego PSNR: {np.mean(model1_psnr_stego):.2f} dB")
    print(f"  Recovered PSNR: {np.mean(model1_psnr_recovered):.2f} dB")
    
    print(f"\n{args.model2_name} Average Metrics:")
    print(f"  Stego PSNR: {np.mean(model2_psnr_stego):.2f} dB")
    print(f"  Recovered PSNR: {np.mean(model2_psnr_recovered):.2f} dB")
    
    # Determine winner
    print(f"\n{'='*60}")
    print("WINNER")
    print(f"{'='*60}")
    
    stego_winner = args.model1_name if np.mean(model1_psnr_stego) > np.mean(model2_psnr_stego) else args.model2_name
    recovered_winner = args.model1_name if np.mean(model1_psnr_recovered) > np.mean(model2_psnr_recovered) else args.model2_name
    
    stego_diff = abs(np.mean(model1_psnr_stego) - np.mean(model2_psnr_stego))
    recovered_diff = abs(np.mean(model1_psnr_recovered) - np.mean(model2_psnr_recovered))
    
    print(f"  Better Stego Quality: {stego_winner} (by {stego_diff:.2f} dB)")
    print(f"  Better Secret Recovery: {recovered_winner} (by {recovered_diff:.2f} dB)")
    
    print(f"\nComparison images saved to: {os.path.abspath(args.output_dir)}")

if __name__ == '__main__':
    main()
