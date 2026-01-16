import os
import json

def verify_coco_setup():
    base_path = "../data/coco"
    
    required_files = {
        'annotations/instances_train2017.json': 'Train annotations',
        'annotations/instances_val2017.json': 'Val annotations', 
        'train2017/': 'Train images directory',
        'val2017/': 'Val images directory'
    }
    
    print("Verifying COCO dataset setup...")
    print("=" * 50)
    
    all_good = True
    
    for file_path, description in required_files.items():
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            if file_path.endswith('.json'):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                    print(f"✓ {description}: {len(data.get('images', []))} images")
                except Exception as e:
                    print(f"✗ {description}: Invalid JSON - {e}")
                    all_good = False
            else:
                image_count = len([f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print(f"✓ {description}: {image_count} images")
        else:
            print(f"✗ {description}: Missing")
            all_good = False
    
    print("=" * 50)
    if all_good:
        print("✓ COCO dataset is properly set up!")
        return True
    else:
        print("✗ Some files are missing or invalid")
        return False

if __name__ == "__main__":
    verify_coco_setup()