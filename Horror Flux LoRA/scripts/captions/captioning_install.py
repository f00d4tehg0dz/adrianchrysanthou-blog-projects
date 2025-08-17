#!/usr/bin/env python3
"""
Run Enhanced Captioning on YouTube Thumbnails Dataset
CUDA 12.8 Compatible
"""

import os
import sys
import subprocess
from pathlib import Path
import warnings

# Suppress warnings for better output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*xFormers.*")

def check_cuda():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available, will use CPU (will be slow)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def install_dependencies():
    """Install captioning dependencies"""
    print("📦 Installing captioning dependencies...")
    
    # Check if requirements file exists
    if os.path.exists("captioning_requirements.txt"):
        print("Installing from captioning_requirements.txt...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "captioning_requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            return False
        else:
            print("✅ Dependencies installed successfully")
            return True
    else:
        print("⚠️  captioning_requirements.txt not found, installing core dependencies...")
        
        # Install core dependencies manually
        core_deps = [
            "torch>=2.1.0",
            "torchvision>=0.16.0", 
            "transformers>=4.35.0",
            "easyocr>=1.7.0",
            "Pillow>=9.0.0",
            "tqdm>=4.64.0"
        ]
        
        for dep in core_deps:
            print(f"Installing {dep}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"⚠️  Failed to install {dep}: {result.stderr}")
        
        return True

def run_captioning_script(dataset_dir, output_file, script_name="enhanced_captioning.py"):
    """Run the captioning script on a dataset"""
    if not os.path.exists(script_name):
        print(f"❌ {script_name} not found")
        return False
    
    print(f"🔄 Running {script_name} on {dataset_dir}...")
    
    cmd = [
        sys.executable, script_name,
        "--dataset_dir", dataset_dir,
        "--output_file", output_file
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {script_name} completed successfully")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Last 500 chars
        return True
    else:
        print(f"❌ {script_name} failed")
        print("Error:", result.stderr)
        return False

def main():
    print("🎨 Enhanced Captioning for YouTube Thumbnails")
    print("CUDA 12.8 Compatible")
    print("=" * 50)
    
    # Check if we're in the right directory
    script_dir = Path(__file__).resolve().parent.parent.parent  # points to lora_training
    dataset_dir = script_dir / "dataset"
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found at {dataset_dir}")
        return
    os.chdir(script_dir)  # Change working directory to lora_training
    
    # Check CUDA availability
    print("🔍 Checking CUDA availability...")
    cuda_available = check_cuda()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Process training dataset
    print("\n🔄 Processing training dataset...")
    if os.path.exists("dataset/train"):
        success = run_captioning_script(
            "dataset/train", 
            "dataset/train/metadata.jsonl"
        )
        if not success:
            print("⚠️  Training dataset processing failed, but continuing...")
    else:
        print("❌ Training dataset not found at dataset/train")
    
    # Process validation dataset
    print("\n🔄 Processing validation dataset...")
    if os.path.exists("dataset/val"):
        success = run_captioning_script(
            "dataset/val", 
            "dataset/val/metadata.jsonl"
        )
        if not success:
            print("⚠️  Validation dataset processing failed")
    else:
        print("❌ Validation dataset not found at dataset/val")
    
    # Check results
    print("\n📊 Checking results...")
    train_metadata = "dataset/train/metadata.jsonl"
    val_metadata = "dataset/val/metadata.jsonl"
    
    if os.path.exists(train_metadata):
        with open(train_metadata, 'r') as f:
            train_count = sum(1 for line in f)
        print(f"✅ Training metadata: {train_count} entries")
    else:
        print("❌ Training metadata not found")
    
    if os.path.exists(val_metadata):
        with open(val_metadata, 'r') as f:
            val_count = sum(1 for line in f)
        print(f"✅ Validation metadata: {val_count} entries")
    else:
        print("❌ Validation metadata not found")
    
    print("\n✅ Captioning process completed!")
    print("You can now run training with enhanced captions.")
    
    if cuda_available:
        print("🚀 CUDA is available - training should be fast!")
    else:
        print("⚠️  CUDA not available - training will be slow on CPU")

if __name__ == "__main__":
    main() 