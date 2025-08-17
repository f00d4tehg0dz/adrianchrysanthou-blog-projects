#!/usr/bin/env python3
"""
Convert JSONL metadata files to per-image .txt files for ai-toolkit compatibility
Optimized for horror and creepy content datasets
"""

import json
import os
import sys
from pathlib import Path

def convert_jsonl_to_txt(jsonl_path, output_dir=None):
    """
    Convert a JSONL file to individual .txt files for each image
    
    Args:
        jsonl_path: Path to the JSONL file
        output_dir: Directory to save .txt files (defaults to same directory as JSONL)
    """
    jsonl_path = Path(jsonl_path)
    
    if not jsonl_path.exists():
        print(f"❌ Error: JSONL file not found: {jsonl_path}")
        return False
    
    if output_dir is None:
        output_dir = jsonl_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 Converting {jsonl_path} to .txt files...")
    print(f"📁 Output directory: {output_dir}")
    
    converted_count = 0
    error_count = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Extract filename and caption
                    # Try different possible field names
                    if 'file_name' in data:
                        img_name = data['file_name']
                    elif 'image' in data:
                        img_name = data['image']
                    else:
                        print(f"⚠️  Line {line_num}: No file_name or image field found")
                        error_count += 1
                        continue
                    
                    # Try different possible caption field names
                    if 'text' in data:
                        caption = data['text']
                    elif 'caption' in data:
                        caption = data['caption']
                    else:
                        print(f"⚠️  Line {line_num}: No text or caption field found")
                        error_count += 1
                        continue
                    
                    # Create .txt filename (replace extension)
                    txt_name = img_name.rsplit('.', 1)[0] + '.txt'
                    txt_path = output_dir / txt_name
                    
                    # Write caption to .txt file
                    with open(txt_path, 'w', encoding='utf-8') as out:
                        out.write(caption.strip())
                    
                    converted_count += 1
                    
                    # Progress indicator
                    if converted_count % 100 == 0:
                        print(f"✅ Converted {converted_count} files...")
                
                except json.JSONDecodeError as e:
                    print(f"❌ Line {line_num}: Invalid JSON - {e}")
                    error_count += 1
                except Exception as e:
                    print(f"❌ Line {line_num}: Error - {e}")
                    error_count += 1
    
    except Exception as e:
        print(f"❌ Error reading JSONL file: {e}")
        return False
    
    print(f"\n🎉 Conversion complete!")
    print(f"✅ Successfully converted: {converted_count} files")
    if error_count > 0:
        print(f"⚠️  Errors: {error_count} lines")
    
    return True

def main():
    """Main function to convert train and val datasets"""
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset"
    
    print("🚀 JSONL to TXT Converter for Horror Content Dataset")
    print("=" * 60)
    
    # Convert train dataset
    train_jsonl = dataset_dir / "train" / "metadata.jsonl"
    if train_jsonl.exists():
        print(f"\n📁 Processing train dataset...")
        success = convert_jsonl_to_txt(train_jsonl)
        if not success:
            print("❌ Failed to convert train dataset")
            return
    else:
        print(f"⚠️  Train metadata.jsonl not found: {train_jsonl}")
    
    # Convert val dataset
    val_jsonl = dataset_dir / "val" / "metadata.jsonl"
    if val_jsonl.exists():
        print(f"\n📁 Processing val dataset...")
        success = convert_jsonl_to_txt(val_jsonl)
        if not success:
            print("❌ Failed to convert val dataset")
            return
    else:
        print(f"⚠️  Val metadata.jsonl not found: {val_jsonl}")
    
    print(f"\n🎉 All conversions complete!")
    print(f"📝 Your horror content dataset is now ready for ai-toolkit training!")
    print(f"💡 Make sure to update your config file to point to the dataset folder")
    print(f"🎭 The LoRA will now learn to generate realistic horror and creepy content!")

if __name__ == "__main__":
    main() 