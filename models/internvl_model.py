from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
from vllm import LLM, SamplingParams
import math
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from .base_model import BaseVLLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVLModel(BaseVLLM):
    def __init__(self, model_name="OpenGVLab/InternVL3-8B", use_vllm=True, **kwargs):
        super().__init__(model_name, **kwargs)
        self.use_vllm = use_vllm
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
    
    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) 
            for i in range(1, n + 1) for j in range(1, n + 1) 
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = orig_width * orig_height
        
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        
        target_width = image_size * best_ratio[0]
        target_height = image_size * best_ratio[1]
        blocks = best_ratio[0] * best_ratio[1]
        
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
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    def load_model(self):
        if self.use_vllm:
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=8192,
                tensor_parallel_size=1,
                dtype=torch.bfloat16
            )
            self.tokenizer = self.model.get_tokenizer()
        else:
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
    
    def format_prompt_with_images(self, images: List[Image.Image], prompt: str):
        if not images:
            return prompt, None
        
        image_tokens = ""
        pixel_values_list = []
        
        for img in images:
            pixel_values = self.load_image(img)
            pixel_values_list.append(pixel_values)
            image_tokens += "<image>\n"
        
        formatted_prompt = image_tokens + prompt
        
        if pixel_values_list:
            all_pixel_values = torch.cat(pixel_values_list, dim=0)
            return formatted_prompt, all_pixel_values
        
        return formatted_prompt, None
    
    def generate_text(self, images: List[Image.Image], prompt: str, max_new_tokens=512, debug=False, **kwargs) -> str:
        if debug:
            logger.info(f"Input prompt: {prompt}")
            logger.info(f"Number of images: {len(images)}")
        
        if self.use_vllm:
            formatted_prompt, pixel_values = self.format_prompt_with_images(images, prompt)
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_new_tokens,
                stop_token_ids=[self.tokenizer.eos_token_id] if hasattr(self.tokenizer, 'eos_token_id') else None
            )
            
            outputs = self.model.generate([formatted_prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
        else:
            formatted_prompt, pixel_values = self.format_prompt_with_images(images, prompt)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
            
            # Fixed generation config to avoid warnings
            generation_config = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.model.chat(
                self.tokenizer, 
                pixel_values, 
                formatted_prompt, 
                generation_config
            )
            response = response.strip()
        
        if debug:
            logger.info(f"Model response: {response}")
        
        return response
    
    def generate_image(self, prompt: str, context_images: List[Image.Image] = None, **kwargs) -> Image.Image:
        raise NotImplementedError("InternVL doesn't support image generation")