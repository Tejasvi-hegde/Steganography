import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder

def test_different_images():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("🧪 Testing with DIFFERENT cover/secret images")
    
    # Load models trained on different images
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    
    encoder.load_state_dict(torch.load('../outputs/checkpoints_diff/encoder_final.pth', map_location=device))
    decoder.load_state_dict(torch.load('../outputs/checkpoints_diff/decoder_final.pth', map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test with different image pairs
    test_pairs = [
        ('image_000000.png', 'image_000001.png'),  # Different images!
        ('image_000002.png', 'image_000003.png'),
        ('image_000004.png', 'image_000005.png')
    ]
    
    for cover_name, secret_name in test_pairs:
        cover_path = f'../data/train_diff/cover/{cover_name}'
        secret_path = f'../data/train_diff/secret/{secret_name}'
        
        try:
            cover_img = Image.open(cover_path).convert('RGB')
            secret_img = Image.open(secret_path).convert('RGB')
            
            cover_tensor = transform(cover_img).unsqueeze(0).to(device)
            secret_tensor = transform(secret_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                stego_tensor = encoder(cover_tensor, secret_tensor)
                recovered_tensor = decoder(stego_tensor)
            
            # Calculate PSNR
            mse = torch.mean((secret_tensor - recovered_tensor) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
            
            # Convert back to images
            def tensor_to_image(tensor):
                img = tensor.squeeze(0).cpu()
                img = img * 0.5 + 0.5
                return transforms.ToPILImage()(img)
            
            stego_img = tensor_to_image(stego_tensor)
            recovered_img = tensor_to_image(recovered_tensor)
            
            # Display
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            images = [cover_img, secret_img, stego_img, recovered_img]
            titles = [
                f'Cover Image\n({cover_name})',
                f'Secret Image\n({secret_name})', 
                f'Stego Image\n(PSNR: {psnr:.2f} dB)',
                'Recovered Secret'
            ]
            
            for i, (ax, img, title) in enumerate(zip(axes.flat, images, titles)):
                ax.imshow(img)
                ax.set_title(title, fontsize=12, weight='bold')
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(f'../outputs/different_test_{cover_name}_{secret_name}.png', dpi=120, bbox_inches='tight')
            plt.show()
            
            print(f"✓ {cover_name} + {secret_name}: PSNR = {psnr:.2f} dB")
            
        except Exception as e:
            print(f"✗ Error with {cover_name}/{secret_name}: {e}")

if __name__ == "__main__":
    test_different_images()