import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder


def test_trained_model():
    device = torch.device('cpu')  # Force CPU
    print(f"🖥️  Using device: {device}")
    
    # Check for trained models
    checkpoint_dir = '../outputs/checkpoints'
    
    # Try to find the best available model
    model_options = [
        ('encoder_best.pth', 'decoder_best.pth', 'best'),
        ('encoder_final.pth', 'decoder_final.pth', 'final'),
        ('encoder_epoch_20.pth', 'decoder_epoch_20.pth', 'epoch 20'),
        ('encoder_epoch_10.pth', 'decoder_epoch_10.pth', 'epoch 10'),
        ('encoder_epoch_5.pth', 'decoder_epoch_5.pth', 'epoch 5'),
    ]
    
    encoder_path = decoder_path = model_name = None
    for enc, dec, name in model_options:
        enc_full = os.path.join(checkpoint_dir, enc)
        dec_full = os.path.join(checkpoint_dir, dec)
        if os.path.exists(enc_full) and os.path.exists(dec_full):
            encoder_path = enc_full
            decoder_path = dec_full
            model_name = name
            break
    
    if encoder_path is None:
        print("❌ No trained models found!")
        print("Please run training first: python simple_train.py")
        return
    
    print(f"📦 Loading {model_name} model...")
    
    # Load trained models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
    
    encoder.eval()
    decoder.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("🧪 Testing with training images...")
    
    # Find test images
    cover_dir = '../data/train/cover'
    secret_dir = '../data/train/secret'
    
    if not os.path.exists(cover_dir) or not os.path.exists(secret_dir):
        print("❌ Training data not found!")
        return
    
    cover_files = sorted([f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    secret_files = sorted([f for f in os.listdir(secret_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(cover_files) == 0 or len(secret_files) == 0:
        print("❌ No images found in training directories!")
        return
    
    # Test with 3 random pairs (using DIFFERENT images)
    import random
    num_tests = min(3, len(cover_files), len(secret_files))
    test_indices = random.sample(range(min(len(cover_files), len(secret_files))), num_tests)
    
    os.makedirs('../outputs', exist_ok=True)
    
    print(f"\n📊 Testing {num_tests} cover/secret pairs (DIFFERENT images):\n")
    
    for test_num, idx in enumerate(test_indices):
        cover_path = os.path.join(cover_dir, cover_files[idx])
        # Use a different index for secret to ensure different images
        secret_idx = (idx + len(secret_files) // 2) % len(secret_files)
        secret_path = os.path.join(secret_dir, secret_files[secret_idx])
        
        try:
            cover_img = Image.open(cover_path).convert('RGB')
            secret_img = Image.open(secret_path).convert('RGB')
            
            cover_tensor = transform(cover_img).unsqueeze(0).to(device)
            secret_tensor = transform(secret_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                stego_tensor = encoder(cover_tensor, secret_tensor)
                recovered_tensor = decoder(stego_tensor)
            
            # Calculate metrics (denormalize first)
            cover_dn = cover_tensor * 0.5 + 0.5
            stego_dn = stego_tensor * 0.5 + 0.5
            secret_dn = secret_tensor * 0.5 + 0.5
            recovered_dn = recovered_tensor * 0.5 + 0.5
            
            # PSNR for stego quality (cover vs stego)
            mse_stego = torch.mean((cover_dn - stego_dn) ** 2)
            psnr_stego = 10 * torch.log10(1.0 / mse_stego).item() if mse_stego > 0 else float('inf')
            
            # PSNR for recovery quality (secret vs recovered)
            mse_recovery = torch.mean((secret_dn - recovered_dn) ** 2)
            psnr_recovery = 10 * torch.log10(1.0 / mse_recovery).item() if mse_recovery > 0 else float('inf')
            
            # Convert tensors to images for display
            def tensor_to_image(tensor):
                img = tensor.squeeze(0).cpu()
                img = img * 0.5 + 0.5  # Denormalize
                img = torch.clamp(img, 0, 1)
                return transforms.ToPILImage()(img)
            
            stego_img = tensor_to_image(stego_tensor)
            recovered_img = tensor_to_image(recovered_tensor)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            axes[0, 0].imshow(cover_img)
            axes[0, 0].set_title('Cover Image', fontsize=12)
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(secret_img)
            axes[0, 1].set_title('Secret Image (DIFFERENT)', fontsize=12, color='red')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(stego_img)
            axes[1, 0].set_title(f'Stego Image\nPSNR: {psnr_stego:.2f} dB', fontsize=12)
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(recovered_img)
            axes[1, 1].set_title(f'Recovered Secret\nPSNR: {psnr_recovery:.2f} dB', fontsize=12)
            axes[1, 1].axis('off')
            
            plt.suptitle(f'Test {test_num + 1}: Different Cover & Secret Images', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            output_path = f'../outputs/test_result_{test_num + 1}.png'
            plt.savefig(output_path, dpi=120, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Test {test_num + 1}:")
            print(f"  Cover: {cover_files[idx]}")
            print(f"  Secret: {secret_files[secret_idx]}")
            print(f"  Stego PSNR: {psnr_stego:.2f} dB (higher = more invisible)")
            print(f"  Recovery PSNR: {psnr_recovery:.2f} dB (higher = better recovery)")
            print(f"  Saved to: {output_path}\n")
            
        except Exception as e:
            print(f"✗ Error in test {test_num + 1}: {e}")
    
    print(f"{'='*60}")
    print("🎉 Testing completed!")
    print(f"{'='*60}")
    print("\n📊 Quality Guidelines:")
    print("  Stego PSNR > 30 dB: Excellent (nearly invisible)")
    print("  Stego PSNR > 25 dB: Good")
    print("  Stego PSNR < 25 dB: Visible artifacts")
    print("\n  Recovery PSNR > 20 dB: Good recovery")
    print("  Recovery PSNR > 25 dB: Excellent recovery")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_trained_model()

# import torch
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
# import sys
# sys.path.append('..')

# from models.encoder import StegoEncoder
# from models.decoder import StegoDecoder

# def quick_test(epoch=10):
#     """Test with a specific epoch checkpoint"""
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Load models
#     encoder = StegoEncoder().to(device)
#     decoder = StegoDecoder().to(device)
    
#     try:
#         encoder.load_state_dict(torch.load(f'../outputs/checkpoints/encoder_epoch_{epoch}.pth', map_location=device))
#         decoder.load_state_dict(torch.load(f'../outputs/checkpoints/decoder_epoch_{epoch}.pth', map_location=device))
#         print(f"Loaded epoch {epoch} models")
#     except:
#         print(f"Epoch {epoch} not found, using latest")
#         encoder.load_state_dict(torch.load('../outputs/checkpoints/encoder_final.pth', map_location=device))
#         decoder.load_state_dict(torch.load('../outputs/checkpoints/decoder_final.pth', map_location=device))
    
#     encoder.eval()
#     decoder.eval()
    
#     # Transform
#     transform = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
    
#     # Use first training images as test
#     cover_path = '../data/train/cover/image_000000.png'
#     secret_path = '../data/train/secret/image_000000.png'
    
#     cover_img = Image.open(cover_path).convert('RGB')
#     secret_img = Image.open(secret_path).convert('RGB')
    
#     cover_tensor = transform(cover_img).unsqueeze(0).to(device)
#     secret_tensor = transform(secret_img).unsqueeze(0).to(device)
    
#     # Test
#     with torch.no_grad():
#         stego_tensor = encoder(cover_tensor, secret_tensor)
#         recovered_tensor = decoder(stego_tensor)
    
#     # Convert back
#     def tensor_to_image(tensor):
#         img = tensor.squeeze(0).cpu()
#         img = img * 0.5 + 0.5
#         return transforms.ToPILImage()(img)
    
#     stego_img = tensor_to_image(stego_tensor)
#     recovered_img = tensor_to_image(recovered_tensor)
    
#     # Display
#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
#     images = [cover_img, secret_img, stego_img, recovered_img]
#     titles = ['Cover', 'Secret', 'Stego (Hidden)', 'Recovered']
    
#     for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
#         ax.imshow(img)
#         ax.set_title(title)
#         ax.axis('off')
    
#     plt.tight_layout()
#     plt.savefig(f'../outputs/quick_test_epoch_{epoch}.png', dpi=120, bbox_inches='tight')
#     plt.show()
    
#     print(f"Quick test saved to outputs/quick_test_epoch_{epoch}.png")

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epoch', type=int, default=10, help='Epoch to test')
#     args = parser.parse_args()
    
#     quick_test(args.epoch)