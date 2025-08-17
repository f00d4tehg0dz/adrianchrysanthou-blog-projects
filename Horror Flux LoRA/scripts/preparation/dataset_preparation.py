import os
import shutil
import json
from PIL import Image
import random
from pathlib import Path
import warnings

# Suppress warnings for better output
warnings.filterwarnings("ignore", category=UserWarning)

def check_cuda():
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("CUDA not available, using CPU for image processing")
            return False
    except ImportError:
        print("PyTorch not installed, using CPU for image processing")
        return False

def prepare_dataset(thumbnails_dir="dataset", output_dir="../dataset"):
    """
    Prepare the scraped horror content for LoRA training.
    Organizes images and creates metadata files.
    CUDA 12.8 Compatible
    """
    
    print("Preparing dataset for LoRA training...")
    print("CUDA 12.8 Compatible")
    print("=" * 50)
    
    # Check CUDA availability
    cuda_available = check_cuda()
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"Created directories: {train_dir}, {val_dir}")
    
    # Get all thumbnail files
    thumbnail_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        thumbnail_files.extend(Path(thumbnails_dir).glob(ext))
    
    print(f"Found {len(thumbnail_files)} horror content files")
    
    if len(thumbnail_files) == 0:
        print("No horror content files found!")
        print(f"   Expected location: {thumbnails_dir}")
        print("   Supported formats: .jpg, .jpeg, .png, .webp")
        return
    
    # Split into train/val (90/10)
    random.shuffle(thumbnail_files)
    split_idx = int(len(thumbnail_files) * 0.9)
    train_files = thumbnail_files[:split_idx]
    val_files = thumbnail_files[split_idx:]
    
    print(f"Split: {len(train_files)} training, {len(val_files)} validation")
    
    # Process training files
    train_metadata = []
    print("\n Processing training files...")
    
    for i, file_path in enumerate(train_files):
        try:
            # Open and validate image
            with Image.open(file_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard dimensions (1024x1024)
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                # Save processed image
                new_filename = f"train_{i:06d}.jpg"
                new_path = os.path.join(train_dir, new_filename)
                img.save(new_path, 'JPEG', quality=95)
                
                # Create metadata entry with horror themes
                metadata_entry = {
                    "file_name": new_filename,
                    "text": "horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style, unsettling atmosphere, horror aesthetic, low light, shadowy figures, eerie lighting, disturbing imagery, horror movie quality",
                    "caption": "horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style, unsettling atmosphere, horror aesthetic, low light, shadowy figures, eerie lighting, disturbing imagery, horror movie quality"
                }
                train_metadata.append(metadata_entry)
                
        except Exception as e:
            print(f" Error processing {file_path}: {e}")
            continue
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(train_files)} training files")
    
    # Process validation files
    val_metadata = []
    print("\n Processing validation files...")
    
    for i, file_path in enumerate(val_files):
        try:
            with Image.open(file_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
                
                new_filename = f"val_{i:06d}.jpg"
                new_path = os.path.join(val_dir, new_filename)
                img.save(new_path, 'JPEG', quality=95)
                
                metadata_entry = {
                    "file_name": new_filename,
                    "text": "horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style, unsettling atmosphere, horror aesthetic, low light, shadowy figures, eerie lighting, disturbing imagery, horror movie quality",
                    "caption": "horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style, unsettling atmosphere, horror aesthetic, low light, shadowy figures, eerie lighting, disturbing imagery, horror movie quality"
                }
                val_metadata.append(metadata_entry)
                
        except Exception as e:
            print(f" Error processing {file_path}: {e}")
            continue
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(val_files)} validation files")
    
    # Save metadata files
    print("\n Saving metadata files...")
    
    with open(os.path.join(train_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
        for entry in train_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    with open(os.path.join(val_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
        for entry in val_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    
    # Check file sizes
    train_size = sum(os.path.getsize(os.path.join(train_dir, f)) for f in os.listdir(train_dir) if f.endswith('.jpg'))
    val_size = sum(os.path.getsize(os.path.join(val_dir, f)) for f in os.listdir(val_dir) if f.endswith('.jpg'))
  
    # Verify files exist
    train_metadata_file = os.path.join(train_dir, "metadata.jsonl")
    val_metadata_file = os.path.join(val_dir, "metadata.jsonl")
    
    if os.path.exists(train_metadata_file):
        print(f" Training metadata: {train_metadata_file}")
    else:
        print(f" Training metadata missing: {train_metadata_file}")
    
    if os.path.exists(val_metadata_file):
        print(f" Validation metadata: {val_metadata_file}")
    else:
        print(f" Validation metadata missing: {val_metadata_file}")
    
    print("\n Horror dataset is ready for training!")
    if cuda_available:
        print(" CUDA is available - training should be fast!")
    else:
        print(" CUDA not available - training will be slow on CPU")

if __name__ == "__main__":
    prepare_dataset()