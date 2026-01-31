"""
Lightweight Dataset Downloader for Steganography Training

Available datasets (much smaller than COCO's 18GB):
1. STL-10: ~2.5GB - High quality 96x96 images (recommended for good results)
2. CIFAR-10: ~170MB - 32x32 images (too small, not recommended)
3. Tiny ImageNet: ~250MB - 64x64 images
4. Flowers102: ~350MB - Flower images
5. Caltech101: ~130MB - Object images
6. LFW (Faces): ~180MB - Face images

Recommended: STL-10 or download images from web
"""

import os
import sys
import urllib.request
import tarfile
import zipfile
import shutil
import random
from PIL import Image
from tqdm import tqdm

# Disable SSL verification for some downloads
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def download_stl10(data_dir="../data"):
    """
    Download STL-10 dataset (~2.5GB) - 96x96 color images
    Good quality for steganography training
    """
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading STL-10 dataset (~2.5GB)...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete!")
    
    # Extract
    print("Extracting...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    print("STL-10 downloaded and extracted!")
    return os.path.join(data_dir, "stl10_binary")


def download_flowers102(data_dir="../data"):
    """
    Download Flowers102 dataset (~350MB) - Beautiful flower images
    """
    url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
    filename = "102flowers.tgz"
    
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading Flowers102 dataset (~350MB)...")
        urllib.request.urlretrieve(url, filepath)
        print("Download complete!")
    
    # Extract
    print("Extracting...")
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(data_dir)
    
    print("Flowers102 downloaded!")
    return os.path.join(data_dir, "jpg")


def download_sample_images(data_dir="../data", num_images=1000):
    """
    Download sample images from picsum.photos (free, no authentication)
    Creates diverse, high-quality images for training
    
    This is the EASIEST option - just downloads random images
    """
    output_dir = os.path.join(data_dir, "sample_images")
    os.makedirs(output_dir, exist_ok=True)
    
    existing = len([f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))])
    if existing >= num_images:
        print(f"Already have {existing} images in {output_dir}")
        return output_dir
    
    print(f"Downloading {num_images} random images from picsum.photos...")
    print("This may take a few minutes...")
    
    downloaded = existing
    failed = 0
    
    for i in tqdm(range(existing, num_images)):
        try:
            # Random seed ensures different images
            url = f"https://picsum.photos/seed/{i+1000}/256/256"
            img_path = os.path.join(output_dir, f"image_{i:05d}.jpg")
            
            if not os.path.exists(img_path):
                urllib.request.urlretrieve(url, img_path)
                downloaded += 1
        except Exception as e:
            failed += 1
            if failed > 50:  # Stop if too many failures
                print(f"\nToo many failures, stopping at {downloaded} images")
                break
    
    print(f"\nDownloaded {downloaded} images to {output_dir}")
    return output_dir


def download_unsplash_lite(data_dir="../data"):
    """
    Instructions for Unsplash Lite dataset (free, ~1GB)
    High quality professional photos
    """
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║              UNSPLASH LITE DATASET (Recommended)                 ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  High quality, ~25,000 images, ~1GB                              ║
    ║                                                                  ║
    ║  Manual download required:                                       ║
    ║  1. Go to: https://unsplash.com/data                            ║
    ║  2. Download "Lite" dataset                                      ║
    ║  3. Extract to: ../data/unsplash/                               ║
    ║                                                                  ║
    ║  OR use the automatic picsum.photos downloader instead          ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)


def prepare_training_data(source_dir, output_dir="../data/train", max_pairs=500, image_size=128):
    """
    Prepare training data with DIFFERENT images for cover and secret
    
    This is crucial for realistic steganography training!
    """
    cover_dir = os.path.join(output_dir, "cover")
    secret_dir = os.path.join(output_dir, "secret")
    
    os.makedirs(cover_dir, exist_ok=True)
    os.makedirs(secret_dir, exist_ok=True)
    
    # Get all images from source
    all_images = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_images.append(os.path.join(root, f))
    
    print(f"Found {len(all_images)} images in {source_dir}")
    
    if len(all_images) < max_pairs * 2:
        print(f"Warning: Not enough images for {max_pairs} pairs")
        max_pairs = len(all_images) // 2
    
    # Shuffle and split into cover and secret sets
    random.shuffle(all_images)
    cover_images = all_images[:max_pairs]
    secret_images = all_images[max_pairs:max_pairs*2]
    
    print(f"Preparing {max_pairs} cover/secret pairs...")
    print("Cover and secret images will be DIFFERENT!")
    
    # Process cover images
    print("\nProcessing cover images...")
    for i, img_path in enumerate(tqdm(cover_images)):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                # Center crop to square, then resize
                w, h = img.size
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img.save(os.path.join(cover_dir, f"image_{i:05d}.png"))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process secret images
    print("\nProcessing secret images...")
    for i, img_path in enumerate(tqdm(secret_images)):
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                w, h = img.size
                min_dim = min(w, h)
                left = (w - min_dim) // 2
                top = (h - min_dim) // 2
                img = img.crop((left, top, left + min_dim, top + min_dim))
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img.save(os.path.join(secret_dir, f"image_{i:05d}.png"))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\n✓ Training data prepared!")
    print(f"  Cover images: {cover_dir} ({len(os.listdir(cover_dir))} images)")
    print(f"  Secret images: {secret_dir} ({len(os.listdir(secret_dir))} images)")
    print(f"  Image size: {image_size}x{image_size}")
    
    return output_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download dataset for steganography training')
    parser.add_argument('--dataset', type=str, default='picsum',
                       choices=['picsum', 'stl10', 'flowers', 'custom'],
                       help='Dataset to download (picsum is easiest)')
    parser.add_argument('--num_images', type=int, default=1000,
                       help='Number of images to download (for picsum)')
    parser.add_argument('--max_pairs', type=int, default=400,
                       help='Max cover/secret pairs for training')
    parser.add_argument('--image_size', type=int, default=128,
                       help='Image size for training')
    parser.add_argument('--custom_dir', type=str, default=None,
                       help='Path to custom image directory')
    
    args = parser.parse_args()
    
    data_dir = "../data"
    
    if args.dataset == 'picsum':
        print("=" * 60)
        print("EASIEST OPTION: Downloading from picsum.photos")
        print("=" * 60)
        source_dir = download_sample_images(data_dir, args.num_images)
        
    elif args.dataset == 'stl10':
        print("=" * 60)
        print("HIGH QUALITY OPTION: Downloading STL-10 (~2.5GB)")
        print("=" * 60)
        source_dir = download_stl10(data_dir)
        
    elif args.dataset == 'flowers':
        print("=" * 60)
        print("FLOWER IMAGES: Downloading Flowers102 (~350MB)")
        print("=" * 60)
        source_dir = download_flowers102(data_dir)
        
    elif args.dataset == 'custom':
        if not args.custom_dir:
            print("Error: --custom_dir required for custom dataset")
            return
        source_dir = args.custom_dir
    
    # Prepare training data with DIFFERENT cover/secret pairs
    prepare_training_data(
        source_dir, 
        max_pairs=args.max_pairs,
        image_size=args.image_size
    )
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Run training:  python simple_train.py")
    print("2. Test model:    python quick_test.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
