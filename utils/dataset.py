import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class StegoDataset(Dataset):
    """
    Dataset for steganography training.
    
    Supports two modes:
    1. use_different_images=True: Cover and secret are DIFFERENT images (recommended)
    2. use_different_images=False: Same image used as both cover and secret
    
    For best results, use different images with augmentation enabled.
    """
    
    def __init__(self, cover_dir, secret_dir, image_size=128, 
                 use_different_images=True, augment=True, shuffle_secret=True):
        self.cover_dir = cover_dir
        self.secret_dir = secret_dir
        self.use_different_images = use_different_images
        self.image_size = image_size
        self.augment = augment
        
        # Get image lists
        self.cover_images = sorted([
            f for f in os.listdir(cover_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.secret_images = sorted([
            f for f in os.listdir(secret_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        print(f"📁 Found {len(self.cover_images)} cover images")
        print(f"📁 Found {len(self.secret_images)} secret images")
        
        # Create pairs based on mode
        if use_different_images:
            # Shuffle secret images for random pairing
            if shuffle_secret:
                random.shuffle(self.secret_images)
            
            # Pair cover with different secret images
            min_len = min(len(self.cover_images), len(self.secret_images))
            self.pairs = list(zip(
                self.cover_images[:min_len], 
                self.secret_images[:min_len]
            ))
            print(f"✓ Created {len(self.pairs)} DIFFERENT cover/secret pairs")
        else:
            # Same image as both cover and secret
            self.pairs = [(img, img) for img in self.cover_images]
            print(f"✓ Created {len(self.pairs)} pairs (same image)")
        
        # Base transform (always applied)
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Augmentation transforms (for training variety)
        if augment:
            self.cover_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
            self.secret_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.cover_augment = None
            self.secret_augment = None
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        cover_file, secret_file = self.pairs[idx]
        
        cover_path = os.path.join(self.cover_dir, cover_file)
        secret_path = os.path.join(self.secret_dir, secret_file)
        
        # Load images
        cover_image = Image.open(cover_path).convert('RGB')
        secret_image = Image.open(secret_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment and self.cover_augment:
            cover_image = self.cover_augment(cover_image)
        if self.augment and self.secret_augment:
            secret_image = self.secret_augment(secret_image)
        
        # Apply base transforms
        cover_tensor = self.base_transform(cover_image)
        secret_tensor = self.base_transform(secret_image)
        
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