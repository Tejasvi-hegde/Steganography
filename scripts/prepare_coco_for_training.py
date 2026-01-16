import os
import shutil
import random
from tqdm import tqdm
from PIL import Image

def prepare_different_images(max_images=500, image_size=128):
    """
    Prepare COCO dataset with DIFFERENT images for cover and secret
    """
    
    coco_dir = "../data/coco"
    output_dir = "../data/train_diff"
    
    # Create output directories
    os.makedirs(f"{output_dir}/cover", exist_ok=True)
    os.makedirs(f"{output_dir}/secret", exist_ok=True)
    
    # Get all image paths from COCO
    image_paths = []
    
    # Train images
    train_dir = f"{coco_dir}/train2017"
    if os.path.exists(train_dir):
        for img_file in os.listdir(train_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(train_dir, img_file))
    
    # Val images
    val_dir = f"{coco_dir}/val2017"
    if os.path.exists(val_dir):
        for img_file in os.listdir(val_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(val_dir, img_file))
    
    print(f"Found {len(image_paths)} total COCO images")
    
    # Shuffle images
    random.shuffle(image_paths)
    
    # Ensure we have enough images for pairs
    if len(image_paths) < max_images * 2:
        print(f"Warning: Not enough images for {max_images} pairs")
        max_images = len(image_paths) // 2
    
    # Split into cover and secret images
    cover_images = image_paths[:max_images]
    secret_images = image_paths[max_images:max_images*2]
    
    print(f"Preparing {len(cover_images)} DIFFERENT cover/secret pairs...")
    print(f"Cover images: {len(cover_images)}")
    print(f"Secret images: {len(secret_images)}")
    
    # Process cover images
    print("Processing cover images...")
    for i, img_path in enumerate(tqdm(cover_images)):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img_resized.save(f"{output_dir}/cover/image_{i:06d}.png")
        except Exception as e:
            print(f"Error processing cover {img_path}: {e}")
    
    # Process secret images
    print("Processing secret images...")
    for i, img_path in enumerate(tqdm(secret_images)):
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                img_resized.save(f"{output_dir}/secret/image_{i:06d}.png")
        except Exception as e:
            print(f"Error processing secret {img_path}: {e}")
    
    print(f"✓ Prepared {len(cover_images)} DIFFERENT cover/secret pairs")
    print(f"Cover images: {output_dir}/cover/")
    print(f"Secret images: {output_dir}/secret/")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_images', type=int, default=500)
    parser.add_argument('--image_size', type=int, default=128)
    
    args = parser.parse_args()
    prepare_different_images(args.max_images, args.image_size)

# import os
# import shutil
# import random
# from tqdm import tqdm
# from PIL import Image

# def prepare_coco_for_training(max_images=10000, image_size=256):
#     """
#     Prepare COCO dataset for steganography training
#     """
    
#     coco_dir = "../data/coco"
#     output_dir = "../data/train"
    
#     # Create output directories
#     os.makedirs(f"{output_dir}/cover", exist_ok=True)
#     os.makedirs(f"{output_dir}/secret", exist_ok=True)
    
#     # Get all image paths
#     image_paths = []
    
#     # Train images
#     train_dir = f"{coco_dir}/train2017"
#     if os.path.exists(train_dir):
#         for img_file in os.listdir(train_dir):
#             if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_paths.append(os.path.join(train_dir, img_file))
    
#     # Val images  
#     val_dir = f"{coco_dir}/val2017"
#     if os.path.exists(val_dir):
#         for img_file in os.listdir(val_dir):
#             if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
#                 image_paths.append(os.path.join(val_dir, img_file))
    
#     print(f"Found {len(image_paths)} total COCO images")
    
#     # Randomly select images
#     random.shuffle(image_paths)
#     selected_images = image_paths[:max_images]
    
#     print(f"Preparing {len(selected_images)} images for training...")
    
#     # Process and copy images
#     for i, img_path in enumerate(tqdm(selected_images)):
#         try:
#             with Image.open(img_path) as img:
#                 if img.mode != 'RGB':
#                     img = img.convert('RGB')
                
#                 img_resized = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
#                 # Save as cover and secret
#                 cover_filename = f"image_{i:06d}.png"
#                 img_resized.save(f"{output_dir}/cover/{cover_filename}")
                
#                 secret_filename = f"image_{i:06d}.png" 
#                 img_resized.save(f"{output_dir}/secret/{secret_filename}")
                
#         except Exception as e:
#             print(f"Error processing {img_path}: {e}")
    
#     print(f"✓ Prepared {len(selected_images)} images for training")

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--max_images', type=int, default=10000)
#     parser.add_argument('--image_size', type=int, default=256)
    
#     args = parser.parse_args()
#     prepare_coco_for_training(args.max_images, args.image_size)