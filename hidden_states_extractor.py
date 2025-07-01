import torch
import numpy as np
import json
import os
from PIL import Image
from transformers import AutoTokenizer
import copy
import random
from typing import List, Dict
import pickle
from tqdm import tqdm
import types

from models.model_factory import create_model
from tasks.i2t_tasks import OperatorInductionTask

class VLICLHiddenStatesExtractor:
    def __init__(self, model_name: str, data_dir: str):
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.tokenizer = None
        self.task = None
        self.base_instruction = ("The image contains two digit numbers and a ? "
                               "representing the mathematical operator. "
                               "Induce the mathematical operator (addition, multiplication, minus) according to the "
                               "results of the in-context examples and calculate the result.")
        self.debug_sample_count = {}
        
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = create_model('internvl', self.model_name)
        self.tokenizer = self.model.tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.task = OperatorInductionTask(self.data_dir)
        
    def create_vlicl_prompt_and_images(self, query: Dict, n_shot: int = 4):
        demonstrations = self.task.select_demonstrations(query, n_shot)
        
        prompt_parts = [self.base_instruction]
        images = []
        
        for demo in demonstrations:
            demo_text = self.task.format_demonstration(demo, include_image_token=True, mode="constrained")
            prompt_parts.append(demo_text)
            if 'image' in demo:
                for img_path in demo['image']:
                    images.append(self.task.load_image(img_path))
        
        query_text = self.task.format_query(query, include_image_token=True, mode="constrained") + " Answer: "
        prompt_parts.append(query_text)
        
        if 'image' in query:
            for img_path in query['image']:
                images.append(self.task.load_image(img_path))
        
        full_prompt = "\n\n".join(prompt_parts)
        return full_prompt, images

    def find_real_token_positions(self, prompt: str, images: List, n_shot: int, sample_idx: int = 0):
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        
        actual_input_ids = None
        original_generate = self.model.model.generate
        
        def instrumented_generate(self, pixel_values=None, input_ids=None, attention_mask=None, **kwargs):
            nonlocal actual_input_ids
            if input_ids is not None:
                actual_input_ids = input_ids.clone()
            return original_generate(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        self.model.model.generate = types.MethodType(instrumented_generate, self.model.model)
        
        try:
            with torch.no_grad():
                generation_config = dict(max_new_tokens=1, do_sample=False)
                
                if images:
                    if len(images) == 1:
                        pixel_values = self.model.load_image(images[0], max_num=12).to(torch.bfloat16).cuda()
                        num_patches_list = None
                    else:
                        pixel_values_list = []
                        num_patches_list = []
                        for img in images:
                            img_pixel_values = self.model.load_image(img, max_num=6)
                            pixel_values_list.append(img_pixel_values)
                            num_patches_list.append(img_pixel_values.size(0))
                        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
                    
                    if num_patches_list is not None:
                        response = self.model.model.chat(
                            self.tokenizer, pixel_values, prompt, generation_config, 
                            num_patches_list=num_patches_list
                        )
                    else:
                        response = self.model.model.chat(
                            self.tokenizer, pixel_values, prompt, generation_config
                        )
                else:
                    response = self.model.model.chat(
                        self.tokenizer, None, prompt, generation_config
                    )
        finally:
            self.model.model.generate = original_generate
        
        if actual_input_ids is None:
            raise ValueError("Failed to capture actual input_ids")
            
        actual_token_ids = actual_input_ids[0].tolist()
        actual_token_texts = [self.tokenizer.decode([tid]) for tid in actual_token_ids]
        
        answer_positions = []
        colon_positions = []
        img_token_positions = []
        
        for i, token_text in enumerate(actual_token_texts):
            if 'Answer' in token_text or 'answer' in token_text:
                answer_positions.append(i)
            if ':' in token_text:
                colon_positions.append(i)
            if actual_token_ids[i] == img_context_token_id:
                img_token_positions.append(i)
        
        if not answer_positions:
            raise ValueError("No answer positions found in actual token sequence")
            
        last_answer_pos = answer_positions[-1]
        
        query_forerunner_pos = None
        for colon_pos in colon_positions:
            if colon_pos > last_answer_pos:
                query_forerunner_pos = colon_pos
                break
        
        if query_forerunner_pos is None:
            query_forerunner_pos = last_answer_pos + 1
        
        target_positions = {
            'query_forerunner': query_forerunner_pos,
            'last_input_text': last_answer_pos - 1 if last_answer_pos > 0 else 0,
            'query_label': query_forerunner_pos + 1 if query_forerunner_pos + 1 < len(actual_token_ids) else query_forerunner_pos
        }
        
        if n_shot == 0:
            query_start_pos = None
            for i, token_id in enumerate(actual_token_ids):
                if token_id == img_context_token_id:
                    query_start_pos = i
                    break
            if query_start_pos is None:
                raise ValueError("No image tokens found for 0-shot query")
        else:
            img_token_groups = []
            current_group_start = None
            
            for i, token_id in enumerate(actual_token_ids):
                if token_id == img_context_token_id:
                    if current_group_start is None:
                        current_group_start = i
                else:
                    if current_group_start is not None:
                        img_token_groups.append((current_group_start, i - 1))
                        current_group_start = None
            
            if current_group_start is not None:
                img_token_groups.append((current_group_start, len(actual_token_ids) - 1))
            
            if len(img_token_groups) < n_shot + 1:
                raise ValueError(f"Expected {n_shot + 1} image groups, found {len(img_token_groups)}")
            
            query_img_start, _ = img_token_groups[-1]
            query_start_pos = query_img_start
        
        query_end_pos = last_answer_pos - 1
        while query_end_pos > query_start_pos and 'Answer' in actual_token_texts[query_end_pos]:
            query_end_pos -= 1
        
        query_positions = list(range(query_start_pos - 1, query_end_pos + 1))
        
        safety_check_passed = True
        token_verification = {}
        
        for pos_name, pos in target_positions.items():
            if pos >= len(actual_token_texts):
                safety_check_passed = False
                break
            token_text = actual_token_texts[pos]
            token_id = actual_token_ids[pos]
            is_img_context = token_id == img_context_token_id
            
            token_verification[pos_name] = {
                'position': pos,
                'token_text': token_text,
                'token_id': token_id,
                'is_img_context': is_img_context
            }
            
            if is_img_context and pos_name in ['query_forerunner', 'last_input_text', 'query_label']:
                safety_check_passed = False
                break
        
        expected_tokens = {
            'query_forerunner': ':',
            'last_input_text': '?',
            'query_label': ' '
        }
        
        for pos_name, expected in expected_tokens.items():
            if pos_name in token_verification:
                actual_token = token_verification[pos_name]['token_text']
                if expected not in actual_token:
                    print(f"Warning: {pos_name} expected '{expected}' but got '{actual_token}'")

        if n_shot not in self.debug_sample_count:
            self.debug_sample_count[n_shot] = 0
        
        should_debug = self.debug_sample_count[n_shot] < 3
        
        if should_debug:
            self.debug_sample_count[n_shot] += 1
            
            print(f"\n{'='*60}")
            print(f"TOKEN POSITION VERIFICATION (k={n_shot}, sample={sample_idx})")
            print(f"{'='*60}")

            print("Critical Position Tokens:")
            for pos_name, info in token_verification.items():
                pos = info['position']
                token_text = info['token_text']
                is_img = info['is_img_context']
                print(f"  {pos_name:20} [pos {pos:4d}]: '{token_text}' (IMG_CONTEXT: {is_img})")

            print(f"\nQuery Positions Range: [{query_positions[0]} - {query_positions[-1]}] (total: {len(query_positions)} tokens)")
            
            query_tokens = []
            img_token_count = 0
            for pos in query_positions:
                if pos < len(actual_token_texts):
                    if actual_token_ids[pos] == img_context_token_id:
                        query_tokens.append("<IMG_CONTEXT>")
                    else:
                        query_tokens.append(actual_token_texts[pos])
            
            query_sentence = "".join(query_tokens)
            print(f"Decoded Query Sentence: '{query_sentence}'")
            print(f"Query contains {img_token_count} image tokens")
            
            print(f"\nContext Around Critical Positions:")
            for pos_name, info in token_verification.items():
                pos = info['position']
                start_ctx = max(0, pos - 2)
                end_ctx = min(len(actual_token_texts), pos + 3)
                context_tokens = []
                
                for i in range(start_ctx, end_ctx):
                    if actual_token_ids[i] == img_context_token_id:
                        context_tokens.append("<IMG>")
                    else:
                        if i == pos:
                            context_tokens.append(f"**{actual_token_texts[i]}**")
                        else:
                            context_tokens.append(actual_token_texts[i])
                
                context_str = "".join(context_tokens)
                print(f"  {pos_name:20}: ...{context_str}...")
            
            print(f"{'='*60}\n")
        
        if not safety_check_passed:
            error_msg = "Safety check failed:\n"
            for pos_name, info in token_verification.items():
                error_msg += f"  {pos_name}: pos {info['position']} -> '{info['token_text']}' (IMG_CONTEXT: {info['is_img_context']})\n"
            raise ValueError(error_msg)
        
        return target_positions, query_positions, len(actual_token_ids)
    
    def extract_representations(self, prompt: str, images: List, target_positions: Dict, query_positions: List):
        layer_outputs = {}
        hooks = []
        
        def create_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                if hidden_states.dtype == torch.bfloat16:
                    hidden_states = hidden_states.float()
                layer_outputs[layer_name] = hidden_states.detach().cpu()
            return hook_fn
        
        language_model = self.model.model.language_model
        num_layers = len(language_model.model.layers)
        
        for layer_idx in range(num_layers):
            layer = language_model.model.layers[layer_idx]
            hook = layer.register_forward_hook(create_hook(f"layer_{layer_idx}"))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                generation_config = dict(max_new_tokens=1, do_sample=False)
                
                if images:
                    if len(images) == 1:
                        pixel_values = self.model.load_image(images[0], max_num=12).to(torch.bfloat16).cuda()
                        num_patches_list = None
                    else:
                        pixel_values_list = []
                        num_patches_list = []
                        for img in images:
                            img_pixel_values = self.model.load_image(img, max_num=6)
                            pixel_values_list.append(img_pixel_values)
                            num_patches_list.append(img_pixel_values.size(0))
                        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
                    
                    if num_patches_list is not None:
                        response = self.model.model.chat(
                            self.tokenizer, pixel_values, prompt, generation_config, 
                            num_patches_list=num_patches_list
                        )
                    else:
                        response = self.model.model.chat(
                            self.tokenizer, pixel_values, prompt, generation_config
                        )
                else:
                    response = self.model.model.chat(
                        self.tokenizer, None, prompt, generation_config
                    )
        
        finally:
            for hook in hooks:
                hook.remove()
        
        extracted_reps = {}
        query_mean_pooled = []
        
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx}"
            if layer_name in layer_outputs:
                hidden_states = layer_outputs[layer_name]
                
                if hidden_states.dim() == 3:
                    seq_len = hidden_states.shape[1]
                    
                    valid_query_positions = [pos for pos in query_positions if 0 <= pos < seq_len]
                    
                    if valid_query_positions:
                        query_states = hidden_states[0, valid_query_positions, :].numpy()
                        query_mean = np.mean(query_states, axis=0)
                        query_mean_pooled.append(query_mean)
                    else:
                        query_mean_pooled.append(np.zeros(3584))
                else:
                    query_mean_pooled.append(np.zeros(3584))
            else:
                query_mean_pooled.append(np.zeros(3584))
        
        for pos_name, pos in target_positions.items():
            layer_reps = []
            for layer_idx in range(num_layers):
                layer_name = f"layer_{layer_idx}"
                if layer_name in layer_outputs:
                    hidden_states = layer_outputs[layer_name]
                    if hidden_states.dim() == 3 and 0 <= pos < hidden_states.shape[1]:
                        rep = hidden_states[0, pos, :].numpy()
                        layer_reps.append(rep)
                    else:
                        layer_reps.append(np.zeros(3584))
                else:
                    layer_reps.append(np.zeros(3584))
            extracted_reps[pos_name] = layer_reps
        
        extracted_reps['query_mean_pooled'] = query_mean_pooled
        
        return extracted_reps
    
    def process_single_sample(self, query: Dict, n_shot: int = 4, sample_idx: int = 0):
        prompt, images = self.create_vlicl_prompt_and_images(query, n_shot)
        
        target_positions, query_positions, actual_seq_len = self.find_real_token_positions(prompt, images, n_shot, sample_idx)
        
        extracted_reps = self.extract_representations(prompt, images, target_positions, query_positions)
        
        results = {
            'query_forerunner': extracted_reps['query_forerunner'],
            'last_input_text': extracted_reps['last_input_text'],
            'query_label': extracted_reps['query_label'],
            'query_mean_pooled': extracted_reps['query_mean_pooled'],
            'sample_info': {
                'operator': query.get('operator', 'unknown'),
                'n_shot': n_shot,
                'num_images': len(images),
                'sample_idx': sample_idx,
                'target_positions': target_positions,
                'query_positions': query_positions,
                'actual_seq_len': actual_seq_len
            }
        }
        
        return results
    
    def validate_final_results(self, complete_results: Dict):
        print(f"\n{'='*60}")
        print("FINAL VALIDATION")
        print(f"{'='*60}")
        
        test_k_values = [k for k in [0, 4] if k in complete_results['data']]
        
        for k in test_k_values:
            results = complete_results['data'][k]
            
            if not results['query_forerunner'] or len(results['query_forerunner']) < 2:
                continue
                
            print(f"\nk={k}:")
            
            final_layer_idx = len(results['query_forerunner'][0]) - 1
            sample_0_forerunner = results['query_forerunner'][0][final_layer_idx]
            sample_0_last_input = results['last_input_text'][0][final_layer_idx]   
            sample_0_label = results['query_label'][0][final_layer_idx]
            
            within_sim_1 = np.dot(sample_0_forerunner, sample_0_last_input) / (np.linalg.norm(sample_0_forerunner) * np.linalg.norm(sample_0_last_input))
            within_sim_2 = np.dot(sample_0_forerunner, sample_0_label) / (np.linalg.norm(sample_0_forerunner) * np.linalg.norm(sample_0_label))
            
            print(f"  Within-sample diversity (final layer):")
            print(f"    forerunner vs last_input: {within_sim_1:.4f}")
            print(f"    forerunner vs label: {within_sim_2:.4f}")
            
            num_test = min(5, len(results['query_forerunner']))
            cross_sims = []
            
            for i in range(num_test):
                for j in range(i+1, num_test):
                    rep_i = results['query_forerunner'][i][final_layer_idx]
                    rep_j = results['query_forerunner'][j][final_layer_idx]
                    cos_sim = np.dot(rep_i, rep_j) / (np.linalg.norm(rep_i) * np.linalg.norm(rep_j))
                    cross_sims.append(cos_sim)
            
            if cross_sims:
                avg_cross_sim = np.mean(cross_sims)
                max_cross_sim = np.max(cross_sims)
                print(f"  Cross-sample diversity (final layer):")
                print(f"    average similarity: {avg_cross_sim:.4f}")
                print(f"    max similarity: {max_cross_sim:.4f}")
                
                if k == 0:
                    print(f"    (k=0: high similarity expected for same image files)")
                elif avg_cross_sim < 0.95:
                    print(f"    Good diversity")
                else:
                    print(f"    High similarity")
    
    def extract_complete_dataset(self, num_samples: int = 100, k_values: List[int] = [0, 1, 2, 4, 8], save_path: str = "vlicl_hidden_states.pkl"):
        if self.model is None:
            self.load_model()
        
        query_samples = self.task.query_data[:num_samples]
        print(f"Extracting hidden states for {len(query_samples)} samples with k_values: {k_values}")
        
        complete_results = {
            'extraction_info': {
                'model_name': self.model_name,
                'data_dir': self.data_dir,
                'num_samples': len(query_samples),
                'k_values': k_values,
                'task': 'operator_induction',
                'base_instruction': self.base_instruction,
                'extraction_method': 'mean_pooled_query_reference'
            },
            'data': {}
        }
        
        for k in k_values:
            print(f"\nProcessing k={k}...")
            
            k_results = {
                'query_forerunner': [],
                'last_input_text': [],
                'query_label': [],
                'query_mean_pooled': [],
                'sample_info': []
            }
            
            for i, query in enumerate(tqdm(query_samples, desc=f"k={k}")):
                try:
                    sample_results = self.process_single_sample(query, k, sample_idx=i)
                    
                    for key in ['query_forerunner', 'last_input_text', 'query_label', 'query_mean_pooled', 'sample_info']:
                        k_results[key].append(sample_results[key])
                    
                except Exception as e:
                    print(f"Error processing sample {i} for k={k}: {e}")
                    continue
            
            complete_results['data'][k] = k_results
            print(f"k={k} completed: {len(k_results['query_forerunner'])} samples")
        
        self.validate_final_results(complete_results)
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving results to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print("Extraction complete!")
        
        print(f"\nFinal data structure:")
        for k, k_data in complete_results['data'].items():
            if k_data['query_forerunner']:
                n_samples = len(k_data['query_forerunner'])
                n_layers = len(k_data['query_forerunner'][0])
                hidden_dim = k_data['query_forerunner'][0][0].shape[0]
                print(f"  k={k}: {n_samples} samples × {n_layers} layers × {hidden_dim} dims")
        
        return complete_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./VL-ICL", help="VL-ICL data directory")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-38B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=60, help="Number of samples to process")
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 2, 4, 6], help="K values to process")
    parser.add_argument("--save_path", type=str, default="./data/InternVL3-38B-Instruct/vlicl_hidden_states_final.pkl", help="Save path for results")
    
    args = parser.parse_args()
    
    extractor = VLICLHiddenStatesExtractor(args.model_name, args.data_dir)
    results = extractor.extract_complete_dataset(
        num_samples=args.num_samples,
        k_values=args.k_values,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()