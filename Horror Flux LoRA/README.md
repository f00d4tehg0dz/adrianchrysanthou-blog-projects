# Horror LoRA Training & Dataset Preparation

Scripts and logic for preparing a horror and creepy content dataset and fine-tuning a LoRA on the FLUX.1-dev model.

---

## Directory Structure
```
lora_training/
├── dataset/
│   ├── train/
│   └── val/
├── scripts/
│   ├── captions/
│   │   ├── enhanced_captioning.py
│   │   └── captioning_requirements.txt
│   └── preparation/
│       ├── dataset_preparation.py
├── train_lora_flux_24gb.yaml
└── convert_jsonl_to_txt.py
```

---

## Quick Start

1. **Generate enhanced captions for horror content:**
   ```bash
   python scripts/captions/enhanced_captioning.py --dataset_dir dataset/train --output_file dataset/train/metadata.jsonl
   ```

2. **Convert JSONL metadata to TXT files:**
   ```bash
   python scripts/captions/convert_jsonl_to_txt.py
   ```

3. **Start training:**
   ```bash
   ai-toolkit train train_lora_flux_24gb.yaml
   ```

---

## Dataset Preparation

### 1. **Enhanced Captioning for Horror Content**

Generate rich captions using AI vision models with automatic text image filtering:

```bash
python scripts/captions/enhanced_captioning.py \
  --dataset_dir dataset/train \
  --output_file dataset/train/metadata.jsonl
```

**Features:**
- **AI Vision Captioning**: Uses Florence-2 for intelligent image descriptions
- **OCR Text Extraction**: Extracts text from images using EasyOCR
- **Text Image Filtering**: Automatically skips images with text overlays
- **Horror Theme Enhancement**: Adds consistent horror and creepy content descriptors
- **Rich Training Data**: Creates detailed captions for better LoRA training

**Text Filtering Options:**
- **Enabled by default**: Images with text are automatically skipped
- **Disable filtering**: Use `--no_filter_text` to process all images
- **Florence-2 Detection**: Uses Microsoft's Florence-2 model for accurate text identification

**Sample Output:**
```
🔍 Text filtering is ENABLED - images with text will be skipped
🗑️  Skipped 12 images with text:
  - train_000045.jpg
  - train_000056.jpg
  ... and 10 more

📊 Statistics:
  Total images processed: 78
  Images skipped (with text): 12
  Text filtering rate: 15.4%
```

### 2. **JSONL to TXT Conversion**

Convert the generated metadata to ai-toolkit's required TXT format:

```bash
python scripts/captions/convert_jsonl_to_txt.py
```

This script:
- Reads `dataset/train/metadata.jsonl` and `dataset/val/metadata.jsonl`
- Creates `.txt` caption files for each image
- Uses enhanced horror-themed captions as training prompts
- Organizes files in `dataset/train/` and `dataset/val/`

---

## Training Configuration

The `train_lora_flux_24gb.yaml` configuration is optimized for horror content generation:

**Key Features:**
- **Model**: FLUX.1-dev with flow matching scheduler
- **LoRA Rank**: 32 (optimal for horror content learning)
- **Resolutions**: Multiple resolutions (512x768, 1024x1280) for diverse training
- **Horror Prompts**: 10 representative horror and creepy content prompts

**Sample Training Prompts:**
- "dark abandoned hallway with flickering lights, horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style"
- "shadowy figure lurking in doorway, horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style"
- "low light scene with eerie shadows and broken windows, horror content, creepy atmosphere, grainy footage, dark and unsettling, horror movie style"

---

## Hardware Requirements

- **GPU**: RTX 4090 (24GB VRAM) or equivalent
- **Memory**: Minimum 20GB VRAM for FLUX.1-dev training
- **Storage**: 100GB+ for model weights and datasets
- **CUDA**: Version 12.8+ recommended

---

## Notes

- All scripts are designed to be run locally
- Training requires ~24GB VRAM (RTX 4090 recommended)
- FLUX models require specific optimizations (see AI TOOLKIT README)
- Text filtering is recommended for clean horror content training

---

## License

MIT