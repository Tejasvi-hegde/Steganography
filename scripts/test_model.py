import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from models.encoder import StegoEncoder
from models.decoder import StegoDecoder

def test_steganography(cover_path, secret_path, encoder_path, decoder_path):
    """Test the steganography system"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = StegoEncoder().to(device)
    decoder = StegoDecoder().to(device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()
    
    # Load and transform images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    cover_img = Image.open(cover_path).convert('RGB')
    secret_img = Image.open(secret_path).convert('RGB')
    
    cover_tensor = transform(cover_img).unsqueeze(0).to(device)
    secret_tensor = transform(secret_img).unsqueeze(0).to(device)
    
    # Hide secret
    with torch.no_grad():
        stego_tensor = encoder(cover_tensor, secret_tensor)
        recovered_tensor = decoder(stego_tensor)
    
    # Convert back to images
    def tensor_to_image(tensor):
        img = tensor.squeeze(0).cpu()
        img = img * 0.5 + 0.5  # Denormalize
        return transforms.ToPILImage()(img)
    
    stego_img = tensor_to_image(stego_tensor)
    recovered_img = tensor_to_image(recovered_tensor)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0, 0].imshow(cover_img)
    axes[0, 0].set_title('Cover Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(secret_img)
    axes[0, 1].set_title('Secret Image')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(stego_img)
    axes[1, 0].set_title('Stego Image (Hidden)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(recovered_img)
    axes[1, 1].set_title('Recovered Secret')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('../outputs/test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Test completed! Check outputs/test_result.png")

if __name__ == "__main__":
    # Use any test images
    test_steganography(
        cover_path='../data/train/cover/image_000001.png',
        secret_path='../data/train/secret/image_000001.png', 
        encoder_path='../outputs/checkpoints/encoder_epoch_50.pth',
        decoder_path='../outputs/checkpoints/decoder_epoch_50.pth'
    )