#!/usr/bin/env python3
"""
Enhanced Captioning Script for Horror and Creepy Content
Combines OCR text extraction with vision-language captioning using Florence-2
Includes text image filtering to remove images with text overlays
CUDA 12.8 Compatible

Features:
- Florence-2 model for superior image understanding and captioning
- OCR text extraction for horror content text recognition
- Text image filtering to remove images with text overlays
- Fallback to BLIP2 if Florence-2 fails to load
- Optimized for horror, creepy, and grainy footage content
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
import easyocr
from tqdm import tqdm
import logging
import re
import warnings

# Suppress warnings for better output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*triton.*")
warnings.filterwarnings("ignore", message=".*xFormers.*")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedCaptioner:
    def __init__(self, filter_text_images=True):
        """
        Initialize captioning models with CUDA 12.8 compatibility
        """
        logger.info("Loading captioning models...")
        
        # Store text filtering preference
        self.filter_text_images = filter_text_images
        
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name()}")
            logger.info(f"✅ CUDA version: {torch.version.cuda}")
            logger.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = "cpu"
            logger.warning("⚠️  CUDA not available, using CPU (will be slow)")
        
        # Initialize OCR for text extraction
        try:
            # Use GPU for EasyOCR if available
            gpu_available = torch.cuda.is_available()
            self.ocr_reader = easyocr.Reader(['en'], gpu=gpu_available)
            logger.info("✅ EasyOCR loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load EasyOCR: {e}")
            logger.info("Continuing without OCR text extraction...")
            self.ocr_reader = None
        
        # Initialize vision-language model for captioning
        try:
            # Use Florence-2, a more recent and powerful vision-language model
            # Florence-2 is better at understanding complex visual content like horror content
            model_name = "microsoft/Florence-2-base"
            logger.info(f"Loading Florence-2 captioning model: {model_name}")
            
            # Set device for pipeline
            device_id = 0 if torch.cuda.is_available() else -1
            
            # Florence-2 uses a different pipeline approach
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            # Determine the dtype to use consistently
            model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.caption_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=model_dtype,
                trust_remote_code=True,
                device_map=None  # Don't use device_map initially
            )
            
            # Manually move model to device if CUDA is available
            if torch.cuda.is_available():
                self.caption_model = self.caption_model.to(self.device)
                logger.info(f"✅ Model moved to {self.device}")
            
            # Store the dtype for consistent use
            self.model_dtype = model_dtype
            
            logger.info("✅ Florence-2 captioning model loaded successfully")
            logger.info("🎯 Florence-2 provides superior understanding of complex visual content")
            logger.info(f"📊 Model parameters: ~230M (efficient for GPU memory) using {model_dtype}")
        except Exception as e:
            logger.error(f"❌ Failed to load Florence-2 model: {e}")
            logger.info("Trying BLIP2 as fallback...")
            
            try:
                # Fallback to BLIP2
                model_name = "Salesforce/blip2-opt-2.7b"
                logger.info(f"Loading fallback BLIP2 model: {model_name}")
                
                self.caption_model = pipeline(
                    "image-to-text",
                    model=model_name,
                    device=device_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self.processor = None  # BLIP2 doesn't need separate processor
                self.model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                logger.info("✅ BLIP2 fallback model loaded successfully")
                logger.info("📊 Model parameters: ~2.7B (larger but well-established)")
            except Exception as e2:
                logger.error(f"❌ Failed to load BLIP2 fallback: {e2}")
                logger.info("Trying smaller alternative model...")
                
                try:
                    # Final fallback to a smaller model
                    model_name = "nlpconnect/vit-gpt2-image-captioning"
                    self.caption_model = pipeline(
                        "image-to-text",
                        model=model_name,
                        device=device_id,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                    self.processor = None
                    self.model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                    logger.info("✅ Alternative captioning model loaded successfully")
                    logger.info("📊 Model parameters: ~150M (lightweight fallback)")
                except Exception as e3:
                    logger.error(f"❌ Failed to load alternative model: {e3}")
                    self.caption_model = None
                    self.processor = None
                    self.model_dtype = None
    
    def detect_text_in_image(self, image_path):
        """
        Use Florence-2 to detect if an image contains text
        Returns True if text is detected, False otherwise
        """
        if not self.filter_text_images or not self.processor or not self.caption_model:
            return False, "Text filtering disabled or model not available"
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Use Florence-2's text detection prompt
            prompt = "<TEXT_DETECTION>"
            
            inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            )
            
            # Ensure consistent data types
            if hasattr(self, 'model_dtype') and self.model_dtype:
                # Only convert pixel_values to model dtype, keep input_ids as integers
                inputs = {
                    "input_ids": inputs["input_ids"],  # Keep as Long/Int
                    "pixel_values": inputs["pixel_values"].to(dtype=self.model_dtype)  # Convert to model dtype
                }
            
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1
                )
            
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            # Parse the generated text to extract text detection result
            try:
                parsed_answer = self.processor.post_process_generation(
                    generated_text, 
                    task=prompt, 
                    image_size=(image.width, image.height)
                )
                
                # Extract the text detection result
                if isinstance(parsed_answer, dict) and prompt in parsed_answer:
                    text_detection_result = parsed_answer[prompt]
                else:
                    # Fallback: use the raw generated text
                    text_detection_result = generated_text.replace(prompt, "").strip()
            except Exception as parse_error:
                logger.warning(f"Failed to parse Florence-2 output: {parse_error}")
                # Fallback: use the raw generated text
                text_detection_result = generated_text.replace(prompt, "").strip()
            
            # Check if text was detected
            # Florence-2 typically returns "yes" or "no" for text detection
            has_text = "yes" in text_detection_result.lower() or "text" in text_detection_result.lower()
            
            return has_text, text_detection_result
            
        except Exception as e:
            logger.error(f"Error detecting text in {image_path}: {e}")
            return False, f"Error: {str(e)}"
    
    def extract_text_from_image(self, image_path):
        """
        Extract text from image using EasyOCR
        """
        if not self.ocr_reader:
            return ""
        
        try:
            # Read text from image
            results = self.ocr_reader.readtext(str(image_path))
            
            # Extract and clean text
            extracted_texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter by confidence
                    cleaned_text = re.sub(r'[^\w\s\-!?.,]', '', text).strip()
                    if cleaned_text and len(cleaned_text) > 2:
                        extracted_texts.append(cleaned_text)
            
            return " | ".join(extracted_texts) if extracted_texts else ""
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def generate_caption(self, image_path, extracted_text=""):
        """
        Generate rich caption for the image using Florence-2/BLIP2
        """
        if not self.caption_model:
            return self._generate_fallback_caption(extracted_text)
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Generate base caption based on model type
            if self.processor:  # Florence-2
                # Florence-2 uses processor + model approach with specific prompts
                # Use the detailed caption prompt for better results
                prompt = "<DETAILED_CAPTION>"
                
                inputs = self.processor(
                    text=prompt, 
                    images=image, 
                    return_tensors="pt"
                )
                
                # Ensure consistent data types
                if hasattr(self, 'model_dtype') and self.model_dtype:
                    # Only convert pixel_values to model dtype, keep input_ids as integers
                    inputs = {
                        "input_ids": inputs["input_ids"],  # Keep as Long/Int
                        "pixel_values": inputs["pixel_values"].to(dtype=self.model_dtype)  # Convert to model dtype
                    }
                
                if torch.cuda.is_available():
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.caption_model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=3
                    )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                
                # Parse the generated text to extract the caption
                try:
                    parsed_answer = self.processor.post_process_generation(
                        generated_text, 
                        task=prompt, 
                        image_size=(image.width, image.height)
                    )
                    
                    # Extract the caption from the parsed answer
                    if isinstance(parsed_answer, dict) and prompt in parsed_answer:
                        base_caption = parsed_answer[prompt]
                    else:
                        # Fallback: use the raw generated text
                        base_caption = generated_text.replace(prompt, "").strip()
                except Exception as parse_error:
                    logger.warning(f"Failed to parse Florence-2 caption output: {parse_error}")
                    # Fallback: use the raw generated text
                    base_caption = generated_text.replace(prompt, "").strip()
                
            else:  # BLIP2 or other pipeline models
                # Use pipeline approach
                caption_result = self.caption_model(str(image_path))
                base_caption = caption_result[0]['generated_text'] if caption_result else ""
            
            # Enhance caption for horror content
            enhanced_caption = self._enhance_caption_for_horror(base_caption, extracted_text)
            
            return enhanced_caption
            
        except Exception as e:
            logger.error(f"Error generating caption for {image_path}: {e}")
            return self._generate_fallback_caption(extracted_text)
    
    def _enhance_caption_for_horror(self, base_caption, extracted_text):
        """
        Enhance caption specifically for horror and creepy content
        """
        # Start with base caption
        enhanced_parts = []
        
        if base_caption:
            enhanced_parts.append(base_caption)
        
        # Add horror and creepy content specific elements
        enhanced_parts.append("horror content")
        enhanced_parts.append("creepy atmosphere")
        enhanced_parts.append("grainy footage")
        enhanced_parts.append("dark and unsettling")
        enhanced_parts.append("horror movie style")
        
        # Add extracted text if available
        if extracted_text:
            enhanced_parts.append(f"text: {extracted_text}")
        
        # Add horror visual style descriptors
        enhanced_parts.extend([
            "dark and moody",
            "grainy texture",
            "creepy people",
            "unsettling atmosphere",
            "horror aesthetic",
            "low light",
            "shadowy figures",
            "eerie lighting",
            "disturbing imagery",
            "horror movie quality"
        ])
        
        return ", ".join(enhanced_parts)
    
    def _generate_fallback_caption(self, extracted_text=""):
        """
        Generate fallback caption when models fail
        """
        enhanced_parts = ["horror content", "creepy atmosphere", "grainy footage", "dark and unsettling", "horror movie style"]
        
        # Add extracted text if available
        if extracted_text:
            enhanced_parts.append(f"text: {extracted_text}")
        
        return ", ".join(enhanced_parts)
    
    def process_image(self, image_path):
        """
        Process a single image: detect text, extract text, and generate caption
        """
        filename = os.path.basename(image_path)
        logger.info(f"Processing: {filename}")
        
        # First, check if image has text (if filtering is enabled)
        if self.filter_text_images:
            has_text, text_detection_result = self.detect_text_in_image(image_path)
            if has_text:
                logger.info(f"Text detected in {filename}: {text_detection_result}")
                logger.info(f"Skipping {filename} due to text detection")
                return None  # Return None to indicate this image should be skipped
            else:
                logger.info(f"No text detected in {filename}")
        
        # Extract text using OCR
        extracted_text = self.extract_text_from_image(image_path)
        if extracted_text:
            logger.info(f"Extracted text: {extracted_text[:100]}...")
        else:
            logger.info("No text extracted from image")
        
        # Generate caption
        caption = self.generate_caption(image_path, extracted_text)
        logger.info(f"Generated caption: {caption[:100]}...")
        
        return {
            "file_name": filename,
            "text": caption,
            "caption": caption,
            "extracted_text": extracted_text
        }

def process_dataset(dataset_dir, output_file, filter_text_images=True):
    """
    Process entire dataset with enhanced captioning and text filtering
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return
    
    # Initialize captioner with text filtering
    try:
        captioner = EnhancedCaptioner(filter_text_images)
    except Exception as e:
        logger.error(f"Failed to initialize captioner: {e}")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [
        f for f in dataset_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} images to process")
    if filter_text_images:
        logger.info("🔍 Text filtering is ENABLED - images with text will be skipped")
    else:
        logger.info("🔍 Text filtering is DISABLED - all images will be processed")
    
    # Process each image
    metadata = []
    skipped_images = []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            result = captioner.process_image(str(image_file))
            if result is None:
                # Image was skipped due to text detection
                skipped_images.append(image_file.name)
                logger.info(f"Skipped {image_file.name} (contains text)")
            else:
                metadata.append(result)
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            # Add fallback entry
            fallback_caption = captioner._generate_fallback_caption("")
            metadata.append({
                "file_name": image_file.name,
                "text": fallback_caption,
                "caption": fallback_caption,
                "extracted_text": ""
            })
    
    # Save metadata
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"✅ Processed {len(metadata)} images. Metadata saved to: {output_file}")
    
    if filter_text_images and skipped_images:
        logger.info(f"🗑️  Skipped {len(skipped_images)} images with text:")
        for img in skipped_images[:10]:  # Show first 10
            logger.info(f"  - {img}")
        if len(skipped_images) > 10:
            logger.info(f"  ... and {len(skipped_images) - 10} more")
    
    # Print some examples
    logger.info("\n📝 Sample captions:")
    for i, item in enumerate(metadata[:5]):
        logger.info(f"  {item['file_name']}: {item['text'][:100]}...")
    
    # Print statistics
    total_with_text = sum(1 for item in metadata if item['extracted_text'])
    logger.info(f"\n📊 Statistics:")
    logger.info(f"  Total images processed: {len(metadata)}")
    logger.info(f"  Images skipped (with text): {len(skipped_images)}")
    logger.info(f"  Images with extracted text: {total_with_text}")
    if len(metadata) > 0:
        logger.info(f"  Text extraction rate: {total_with_text/len(metadata)*100:.1f}%")
    if len(image_files) > 0:
        logger.info(f"  Text filtering rate: {len(skipped_images)/len(image_files)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Captioning for Horror and Creepy Content with Text Filtering")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing images")
    parser.add_argument("--output_file", required=True, help="Output metadata.jsonl file")
    parser.add_argument("--sample", action="store_true", help="Process only first 10 images as a sample")
    parser.add_argument("--no_filter_text", action="store_true", help="Disable text image filtering")
    
    args = parser.parse_args()
    
    if args.sample:
        logger.info("🧪 Running in sample mode (first 10 images only)")
        # For sample mode, we'll process all but you can limit the dataset directory
    
    # Enable text filtering by default unless --no_filter_text is specified
    filter_text_images = not args.no_filter_text
    
    process_dataset(args.dataset_dir, args.output_file, filter_text_images)

if __name__ == "__main__":
    main() 