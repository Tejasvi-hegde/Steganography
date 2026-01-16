import os
import urllib.request
import zipfile
import argparse

def download_coco_dataset(download_dir="../data/coco", subset="train2017"):
    """
    Download COCO dataset automatically
    """
    
    # Create directory
    os.makedirs(download_dir, exist_ok=True)
    
    # URLs for COCO 2017 dataset
    urls = {
        'train2017': 'http://images.cocodataset.org/zips/train2017.zip',
        'val2017': 'http://images.cocodataset.org/zips/val2017.zip', 
        'test2017': 'http://images.cocodataset.org/zips/test2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    def download_and_extract(url, filename):
        """Download and extract a zip file"""
        
        filepath = os.path.join(download_dir, filename)
        extract_folder = filename.replace('.zip', '')
        extract_path = os.path.join(download_dir, extract_folder)
        
        # CHECK IF ALREADY EXISTS - ADDED THIS CHECK
        if os.path.exists(extract_path):
            # Count files to verify it's not empty
            if os.path.isdir(extract_path):
                files = os.listdir(extract_path)
                if files:
                    print(f"✓ {extract_folder} already exists ({len(files)} files), skipping download")
                    return
            # If folder exists but is empty, continue with download
        
        print(f"Downloading {filename}...")
        
        # Download
        urllib.request.urlretrieve(url, filepath)
        
        # Extract
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        
        # Remove zip file to save space
        os.remove(filepath)
        print(f"Completed {filename}")
    
    # Download based on subset
    if subset == "all":
        for key, url in urls.items():
            filename = f"{key}.zip"
            download_and_extract(url, filename)
    else:
        if subset in urls:
            filename = f"{subset}.zip"
            download_and_extract(urls[subset], filename)
        
        # Always download annotations if downloading images
        if subset in ['train2017', 'val2017']:
            # Check if annotations already exist
            annotations_path = os.path.join(download_dir, "annotations")
            if not os.path.exists(annotations_path) or not os.listdir(annotations_path):
                download_and_extract(urls['annotations'], "annotations.zip")
            else:
                print("✓ Annotations already exist, skipping download")

def prepare_for_steganography(coco_dir="../data/coco", output_dir="../data/train", max_images=10000):
    """
    Prepare COCO images for steganography training
    """
    
    import shutil
    from tqdm import tqdm
    import random
    
    # Create output directories
    os.makedirs(f"{output_dir}/cover", exist_ok=True)
    os.makedirs(f"{output_dir}/secret", exist_ok=True)
    
    # Get all image paths
    train_images = []
    train_dir = f"{coco_dir}/train2017"
    if os.path.exists(train_dir):
        for file in os.listdir(train_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                train_images.append(os.path.join(train_dir, file))
    
    val_images = []
    val_dir = f"{coco_dir}/val2017"
    if os.path.exists(val_dir):
        for file in os.listdir(val_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                val_images.append(os.path.join(val_dir, file))
    
    all_images = train_images + val_images
    print(f"Found {len(all_images)} total images")
    
    if len(all_images) == 0:
        print("❌ No images found! Please check your COCO dataset installation.")
        return
    
    # Randomly select images
    selected_images = random.sample(all_images, min(max_images, len(all_images)))
    
    # Copy to cover and secret directories
    print("Copying images for steganography training...")
    for i, img_path in enumerate(tqdm(selected_images)):
        filename = f"image_{i:06d}.jpg"
        
        # Copy to cover
        shutil.copy2(img_path, f"{output_dir}/cover/{filename}")
        
        # Copy to secret (using same images)
        shutil.copy2(img_path, f"{output_dir}/secret/{filename}")
    
    print(f"Prepared {len(selected_images)} images for training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download COCO dataset')
    parser.add_argument('--subset', type=str, default='train2017', 
                       choices=['train2017', 'val2017', 'test2017', 'all'],
                       help='Which subset to download')
    parser.add_argument('--max_images', type=int, default=10000,
                       help='Maximum number of images to use for training')
    parser.add_argument('--download_dir', type=str, default='../data/coco',
                       help='Directory to download COCO dataset')
    
    args = parser.parse_args()
    
    # Download COCO
    download_coco_dataset(args.download_dir, args.subset)
    
    # Prepare for steganography
    prepare_for_steganography(args.download_dir, max_images=args.max_images)