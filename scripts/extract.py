import torch
from torchvision import transforms
from PIL import Image
import argparse
import sys
sys.path.append('..')

from models.decoder import StegoDecoder

def extract_secret(stego_path, output_path, decoder_path, device='cuda'):
    """Extract secret image from stego image"""
    
    decoder = StegoDecoder().to(device)
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    stego_image = Image.open(stego_path).convert('RGB')
    stego_tensor = transform(stego_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        secret_tensor = decoder(stego_tensor)
    
    secret_image = transforms.ToPILImage()(secret_tensor.squeeze(0).cpu() * 0.5 + 0.5)
    secret_image.save(output_path)
    print(f"Secret image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract secret image from stego image')
    parser.add_argument('--stego', type=str, required=True, help='Path to stego image')
    parser.add_argument('--output', type=str, required=True, help='Path to save secret image')
    parser.add_argument('--decoder', type=str, required=True, help='Path to decoder model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    extract_secret(args.stego, args.output, args.decoder, args.device)