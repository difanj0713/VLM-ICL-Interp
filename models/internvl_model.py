import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
import math
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from .base_model import BaseVLLM
import logging
from typing import List

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLModel(BaseVLLM):
    def __init__(self, model_name="OpenGVLab/InternVL3-8B-Instruct", **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def build_transform(self, input_size=448):
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        
        return processed_images
    
    def load_image(self, image, input_size=448, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(
            image, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            use_fast=False
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        
        logger.info(f"Model loaded: {self.model_name}")
        logger.info(f"Model device: {next(self.model.parameters()).device}")
    
    def generate_text(self, images: List[Image.Image], prompt: str, max_new_tokens=32, debug=False, **kwargs) -> str:
        if debug:
            logger.info(f"Input prompt: {prompt}")
            logger.info(f"Number of images: {len(images)}")
        
        # Process images according to InternVL3 documentation
        if not images:
            pixel_values = None
            num_patches_list = None
        elif len(images) == 1:
            # Single image case
            pixel_values = self.load_image(images[0], max_num=12).to(torch.bfloat16).cuda()
            num_patches_list = None
            if debug:
                logger.info(f"Single image - pixel_values shape: {pixel_values.shape}")
        else:
            # Multiple images case - following InternVL3 documentation exactly
            pixel_values_list = []
            num_patches_list = []
            
            for i, img in enumerate(images):
                # Process each image individually
                img_pixel_values = self.load_image(img, max_num=6)  # Reduce max_num for multiple images
                pixel_values_list.append(img_pixel_values)
                num_patches_list.append(img_pixel_values.size(0))
                
                if debug:
                    logger.info(f"Image {i+1} - pixel_values shape: {img_pixel_values.shape}, patches: {img_pixel_values.size(0)}")
            
            # Concatenate following InternVL3 pattern
            pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
            
            if debug:
                logger.info(f"Multiple images - pixel_values shape: {pixel_values.shape}")
                logger.info(f"num_patches_list: {num_patches_list}")
        
        # Generation configuration
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate response using InternVL3's chat method
        try:
            if pixel_values is not None and num_patches_list is not None:
                # Multiple images - following InternVL3 documentation
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config,
                    num_patches_list=num_patches_list
                )
            else:
                # Single image or no image
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    prompt,
                    generation_config
                )
            
            response = response.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            if debug:
                import traceback
                logger.error(traceback.format_exc())
            response = ""
        
        if debug:
            logger.info(f"Model response: '{response}'")
        
        return response
    
    def generate_image(self, prompt: str, context_images: List[Image.Image] = None, **kwargs) -> Image.Image:
        raise NotImplementedError("InternVL doesn't support image generation")