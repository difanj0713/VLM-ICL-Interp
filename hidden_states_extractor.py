import os
import sys
import torch
import numpy as np
import pickle
import gc
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import create_model
from tasks.i2t_tasks import OperatorInductionTask

class VLICLHiddenStatesExtractor:
    def __init__(self, model_name="OpenGVLab/InternVL3-8B-Instruct", data_dir="./VL-ICL"):
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.tokenizer = None
        self.task = None
        self.bge_model = None
        self.bge_tokenizer = None
        
        self.base_instruction = ("The image contains two digit numbers and a ? representing the mathematical operator. "
                               "Induce the mathematical operator (addition, multiplication, minus) according to the "
                               "results of the in-context examples and calculate the result.")
        
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = create_model('internvl', self.model_name)
        self.tokenizer = self.model.tokenizer
        self.task = OperatorInductionTask(self.data_dir)
        
    def load_bge_model(self):
        self.bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.bge_model = AutoModel.from_pretrained('BAAI/bge-m3')
        self.bge_model.cuda()
        self.bge_model.eval()
        
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
    
    def find_answer_positions(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze()
        tokens = input_ids.tolist()
        token_texts = [self.tokenizer.decode([tid]) for tid in tokens]
        
        answer_positions = []
        colon_positions = []
        
        for i, text in enumerate(token_texts):
            if 'Answer' in text or 'answer' in text:
                answer_positions.append(i)
            if ':' in text and i > 0:
                colon_positions.append(i)
        
        return answer_positions, colon_positions, len(tokens)
    
    def extract_representations(self, prompt: str, images: List, target_positions: Dict):
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
                # MEMORY OPTIMIZATION: Only keep what we need, move to CPU immediately
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
                
                # MEMORY OPTIMIZATION: Clear GPU cache after forward pass
                torch.cuda.empty_cache()
        
        finally:
            for hook in hooks:
                hook.remove()
        
        # Extract specific positions and convert to numpy
        extracted_reps = {}
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
                        # Get hidden dimension from first valid layer
                        if layer_idx == 0 and hidden_states.dim() == 3:
                            hidden_dim = hidden_states.shape[-1]
                        else:
                            hidden_dim = 3584  # fallback
                        layer_reps.append(np.zeros(hidden_dim))
                else:
                    layer_reps.append(np.zeros(3584))
            extracted_reps[pos_name] = layer_reps
        
        # Add mean-pooling across tokens for each layer
        mean_pooling_reps = []
        for layer_idx in range(num_layers):
            layer_name = f"layer_{layer_idx}"
            if layer_name in layer_outputs:
                hidden_states = layer_outputs[layer_name]
                if hidden_states.dim() == 3:
                    mean_pooled = torch.mean(hidden_states[0], dim=0).numpy()
                    mean_pooling_reps.append(mean_pooled)
                else:
                    if layer_idx == 0 and hidden_states.dim() == 3:
                        hidden_dim = hidden_states.shape[-1]
                    else:
                        hidden_dim = 3584  # fallback
                    mean_pooling_reps.append(np.zeros(hidden_dim))
            else:
                mean_pooling_reps.append(np.zeros(3584))
        
        extracted_reps['mean_pooling'] = mean_pooling_reps
        
        layer_outputs.clear()
        del layer_outputs
        gc.collect()
        
        return extracted_reps
    
    def get_bge_reference_embedding(self, query_identifier: str):
        if self.bge_model is None:
            self.load_bge_model()
        
        inputs = self.bge_tokenizer(query_identifier, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
            if embedding_tensor.dtype == torch.bfloat16:
                embedding_tensor = embedding_tensor.float()
            embedding = embedding_tensor.numpy()
        
        # MEMORY OPTIMIZATION: Clear BGE inputs
        del inputs, outputs, embedding_tensor
        torch.cuda.empty_cache()
        
        return embedding
    
    def get_bge_token_embeddings(self):
        """Get BGE embeddings for the three critical token types"""
        if self.bge_model is None:
            self.load_bge_model()
        
        token_texts = {
            'forerunner': "mathematical operator forerunner token",
            'last_input': "last input text token", 
            'label': "result label token"
        }
        
        token_embeddings = {}
        for token_name, token_text in token_texts.items():
            inputs = self.bge_tokenizer(token_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.bge_model(**inputs)
                embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
                if embedding_tensor.dtype == torch.bfloat16:
                    embedding_tensor = embedding_tensor.float()
                token_embeddings[token_name] = embedding_tensor.numpy()
            
            # MEMORY OPTIMIZATION: Clear intermediate tensors
            del inputs, outputs, embedding_tensor
        
        torch.cuda.empty_cache()
        return token_embeddings
    
    def create_unique_query_identifier(self, query: Dict, sample_idx: int):
        operator = query.get('operator', 'unknown')
        base_text = f"Mathematical expression with {operator} operator"
        
        if 'image' in query and query['image']:
            img_info = f"image_{os.path.basename(query['image'][0])}"
            unique_text = f"{base_text} {img_info} sample_{sample_idx}"
        else:
            unique_text = f"{base_text} sample_{sample_idx}"
        
        return unique_text
    
    def process_single_sample(self, query: Dict, n_shot: int = 4, sample_idx: int = 0):
        prompt, images = self.create_vlicl_prompt_and_images(query, n_shot)
        
        answer_positions, colon_positions, text_len = self.find_answer_positions(prompt)
        
        if not answer_positions:
            raise ValueError(f"No answer positions found in sample {sample_idx}")
        
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
            'query_label': query_forerunner_pos + 1 if query_forerunner_pos + 1 < text_len else query_forerunner_pos
        }
        
        extracted_reps = self.extract_representations(prompt, images, target_positions)
        
        query_identifier = self.create_unique_query_identifier(query, sample_idx)
        bge_reference = self.get_bge_reference_embedding(query_identifier)
        bge_token_embeddings = self.get_bge_token_embeddings()
        
        results = {
            'query_forerunner': extracted_reps['query_forerunner'],
            'last_input_text': extracted_reps['last_input_text'],
            'query_label': extracted_reps['query_label'],
            'mean_pooling': extracted_reps['mean_pooling'],
            'bge_reference': bge_reference,
            'bge_token_embeddings': bge_token_embeddings,
            'sample_info': {
                'operator': query.get('operator', 'unknown'),
                'n_shot': n_shot,
                'num_images': len(images),
                'sample_idx': sample_idx,
                'query_identifier': query_identifier,
                'target_positions': target_positions
            }
        }
        
        del extracted_reps
        gc.collect()
        
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
            
            sample_0_forerunner = results['query_forerunner'][0][0]
            sample_0_last_input = results['last_input_text'][0][0]   
            sample_0_label = results['query_label'][0][0]
            sample_0_mean_pool = results['mean_pooling'][0][0]
            
            within_sim_1 = np.dot(sample_0_forerunner, sample_0_last_input) / (np.linalg.norm(sample_0_forerunner) * np.linalg.norm(sample_0_last_input))
            within_sim_2 = np.dot(sample_0_forerunner, sample_0_label) / (np.linalg.norm(sample_0_forerunner) * np.linalg.norm(sample_0_label))
            within_sim_3 = np.dot(sample_0_forerunner, sample_0_mean_pool) / (np.linalg.norm(sample_0_forerunner) * np.linalg.norm(sample_0_mean_pool))
            
            print(f"  Within-sample diversity (layer 0):")
            print(f"    forerunner vs last_input: {within_sim_1:.4f}")
            print(f"    forerunner vs label: {within_sim_2:.4f}")
            print(f"    forerunner vs mean_pooling: {within_sim_3:.4f}")
            
            print(f"  Dimensions:")
            print(f"    forerunner: {sample_0_forerunner.shape}")
            print(f"    mean_pooling: {sample_0_mean_pool.shape}")
            print(f"    bge_reference: {results['bge_reference'][0].shape}")
            print(f"    bge_token_embeddings: {results['bge_token_embeddings'][0]['forerunner'].shape}")
            
            num_test = min(5, len(results['query_forerunner']))
            cross_sims = []
            
            for i in range(num_test):
                for j in range(i+1, num_test):
                    rep_i = results['query_forerunner'][i][0]
                    rep_j = results['query_forerunner'][j][0]
                    cos_sim = np.dot(rep_i, rep_j) / (np.linalg.norm(rep_i) * np.linalg.norm(rep_j))
                    cross_sims.append(cos_sim)
            
            if cross_sims:
                avg_cross_sim = np.mean(cross_sims)
                max_cross_sim = np.max(cross_sims)
                print(f"  Cross-sample diversity (layer 0):")
                print(f"    average similarity: {avg_cross_sim:.4f}")
                print(f"    max similarity: {max_cross_sim:.4f}")
                
                if k == 0:
                    print(f"    (k=0: high similarity expected for same image files)")
                elif avg_cross_sim < 0.95:
                    print(f"    ✅ Good diversity")
                else:
                    print(f"    ⚠️  High similarity")
    
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
                'extraction_method': 'memory_optimized_version'
            },
            'data': {}
        }
        
        for k in k_values:
            print(f"\nProcessing k={k}...")
            print(f"GPU Memory before k={k}: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            k_results = {
                'query_forerunner': [],
                'last_input_text': [],
                'query_label': [],
                'mean_pooling': [],
                'bge_reference': [],
                'bge_token_embeddings': [],
                'sample_info': []
            }
            
            for i, query in enumerate(tqdm(query_samples, desc=f"k={k}")):
                try:
                    sample_results = self.process_single_sample(query, k, sample_idx=i)
                    
                    for key in ['query_forerunner', 'last_input_text', 'query_label', 'mean_pooling', 'bge_reference', 'bge_token_embeddings', 'sample_info']:
                        k_results[key].append(sample_results[key])
                    
                    del sample_results
                    
                    if (i + 1) % 10 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"  Sample {i+1}/{len(query_samples)}: GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                    
                except Exception as e:
                    print(f"Error processing sample {i} for k={k}: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            complete_results['data'][k] = k_results
            print(f"k={k} completed: {len(k_results['query_forerunner'])} samples")
            
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU Memory after k={k}: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        self.validate_final_results(complete_results)
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving results to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print("✅ Extraction complete!")
        print(f"\nFinal data structure:")
        for k, k_data in complete_results['data'].items():
            if k_data['query_forerunner']:
                n_samples = len(k_data['query_forerunner'])
                n_layers = len(k_data['query_forerunner'][0])
                hidden_dim = k_data['query_forerunner'][0][0].shape[0]
                print(f"  k={k}: {n_samples} samples × {n_layers} layers × {hidden_dim} dims")
                print(f"    + mean_pooling: {n_samples} samples × {n_layers} layers × {hidden_dim} dims")
                print(f"    + bge_token_embeddings: 3 token types × 1024 dims")
        
        return complete_results

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./VL-ICL", help="VL-ICL data directory")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-38B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=60, help="Number of samples to process")
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 2, 4], help="K values to process")
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