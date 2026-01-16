import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder

def test_trained_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    encoder.load_state_dict(torch.load('../outputs/checkpoints/encoder_epoch_5.pth', map_location=device))
    decoder.load_state_dict(torch.load('../outputs/checkpoints/decoder_epoch_5.pth', map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Testing with training images...")
    
    # Test multiple images
    test_images = ['image_000000.png', 'image_000100.png', 'image_000200.png']
    
    for img_name in test_images:
        cover_path = f'../data/train/cover/{img_name}'
        secret_path = f'../data/train/secret/{img_name}'
        
        try:
            cover_img = Image.open(cover_path).convert('RGB')
            secret_img = Image.open(secret_path).convert('RGB')
            
            cover_tensor = transform(cover_img).unsqueeze(0).to(device)
            secret_tensor = transform(secret_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                stego_tensor = encoder(cover_tensor, secret_tensor)
                recovered_tensor = decoder(stego_tensor)
            
            # Calculate metrics
            mse = torch.mean((secret_tensor - recovered_tensor) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
            
            # Convert back to images for display
            def tensor_to_image(tensor):
                img = tensor.squeeze(0).cpu()
                img = img * 0.5 + 0.5  # Denormalize
                return transforms.ToPILImage()(img)
            
            stego_img = tensor_to_image(stego_tensor)
            recovered_img = tensor_to_image(recovered_tensor)
            
            # Display
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            images = [cover_img, secret_img, stego_img, recovered_img]
            titles = ['Cover Image', 'Secret Image', f'Stego Image\n(PSNR: {psnr:.2f} dB)', 'Recovered Secret']
            
            for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
                ax.imshow(img)
                ax.set_title(title, fontsize=12)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'../outputs/test_result_{img_name}.png', dpi=120, bbox_inches='tight')
            plt.show()
            
            print(f"✓ {img_name}: PSNR = {psnr:.2f} dB")
            
        except Exception as e:
            print(f"✗ Error with {img_name}: {e}")
    
    print("\n🎉 Testing completed! Check the output images in outputs/ folder")

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