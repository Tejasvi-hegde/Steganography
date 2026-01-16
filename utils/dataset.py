import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class StegoDataset(Dataset):
    def __init__(self, cover_dir, secret_dir, image_size=128, use_different_images=True):
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.use_different_images = use_different_images
        
        self.cover_images = sorted([f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.secret_images = sorted([f for f in os.listdir(secret_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure we have matching pairs
        if use_different_images:
            self.pairs = list(zip(self.cover_images, self.secret_images))
        else:
            # Old behavior - same images
            self.pairs = [(img, img) for img in self.cover_images]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        cover_file, secret_file = self.pairs[idx]
        
        cover_path = os.path.join(self.cover_dir, cover_file)
        secret_path = os.path.join(self.secret_dir, secret_file)
        
        cover_image = Image.open(cover_path).convert('RGB')
        secret_image = Image.open(secret_path).convert('RGB')
        
        cover_tensor = self.transform(cover_image)
        secret_tensor = self.transform(secret_image)
        
        return cover_tensor, secret_tensor, cover_file

# import os
# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as transforms

# class StegoDataset(Dataset):
#     def __init__(self, cover_dir, secret_dir, image_size=256):
#         self.cover_dir = cover_dir
#         self.secret_dir = secret_dir
        
#         self.cover_images = [f for f in os.listdir(cover_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
#         self.secret_images = [f for f in os.listdir(secret_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
        
#     def __len__(self):
#         return min(len(self.cover_images), len(self.secret_images))
    
#     def __getitem__(self, idx):
#         cover_path = os.path.join(self.cover_dir, self.cover_images[idx])
#         secret_path = os.path.join(self.secret_dir, self.secret_images[idx % len(self.secret_images)])
        
#         cover_image = Image.open(cover_path).convert('RGB')
#         secret_image = Image.open(secret_path).convert('RGB')
        
#         cover_tensor = self.transform(cover_image)
#         secret_tensor = self.transform(secret_image)
        
#         return cover_tensor, secret_tensor, self.cover_images[idx]