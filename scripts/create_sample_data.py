"""
Create a small sample dataset for t raining using generated images.
No download required - creates colorful random images for testing.
"""
import os
import sys
import random
from PIL import Image, ImageDraw, ImageFilter

def create_sample_image(size=(256, 256)):
    """Create a colorful sample image with shapes"""
    img = Image.new('RGB', size, color=(
        random.randint(50, 200),
        random.randint(50, 200),
        random.randint(50, 200)
    ))
    draw = ImageDraw.Draw(img)
    
    # Add random shapes
    for _ in range(random.randint(3, 8)):
        shape_type = random.choice(['rectangle', 'ellipse'])
        x1 = random.randint(0, size[0] - 50)
        y1 = random.randint(0, size[1] - 50)
        x2 = x1 + random.randint(30, 100)
        y2 = y1 + random.randint(30, 100)
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        if shape_type == 'rectangle':
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)
    
    # Add some blur for realism
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    return img

def create_dataset(num_images=100):
    """Create cover and secret image directories"""
    
    # Create directories
    cover_dir = '../data/train/cover'
    secret_dir = '../data/train/secret'
    
    os.makedirs(cover_dir, exist_ok=True)
    os.makedirs(secret_dir, exist_ok=True)
    
    print(f"Creating {num_images} sample images...")
    
    for i in range(num_images):
        # Create cover image
        cover_img = create_sample_image()
        cover_path = os.path.join(cover_dir, f'image_{i:06d}.png')
        cover_img.save(cover_path)
        
        # Create secret image (different from cover)
        secret_img = create_sample_image()
        secret_path = os.path.join(secret_dir, f'image_{i:06d}.png')
        secret_img.save(secret_path)
        
        if (i + 1) % 20 == 0:
            print(f"  Created {i + 1}/{num_images} images...")
    
    print(f"\n✅ Dataset created!")
    print(f"   Cover images: {cover_dir} ({num_images} files)")
    print(f"   Secret images: {secret_dir} ({num_images} files)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=100, help='Number of images to create')
    args = parser.parse_args()
    
    create_dataset(args.num)
