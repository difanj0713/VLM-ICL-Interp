import os
import sys
import torch
import numpy as np
import json
import pickle
from PIL import Image
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
        self.hidden_states = []
        self.hooks = []
        
    def load_model(self):
        print(f"Loading InternVL model: {self.model_name}")
        self.model = create_model('internvl', self.model_name)
        self.tokenizer = self.model.tokenizer
        self.task = OperatorInductionTask(self.data_dir)
        print("InternVL model and task loaded successfully")
        
    def load_bge_model(self):
        print("Loading BGE M3 reference encoder...")
        self.bge_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
        self.bge_model = AutoModel.from_pretrained('BAAI/bge-m3')
        self.bge_model.cuda()
        self.bge_model.eval()
        print("BGE M3 model loaded successfully")
        
    def create_vlicl_prompt_and_images(self, query: Dict, n_shot: int = 4) -> Tuple[str, List[Image.Image]]:
        demonstrations = self.task.select_demonstrations(query, n_shot)
        
        prompt_parts = []
        images = []
        
        for demo in demonstrations:
            demo_text = self.task.format_demonstration(demo, include_image_token=True, mode="constrained")
            prompt_parts.append(demo_text)
            
            if 'image' in demo:
                for img_path in demo['image']:
                    images.append(self.task.load_image(img_path))
        
        query_text = self.task.format_query(query, include_image_token=True, mode="constrained") + " Answer:"
        prompt_parts.append(query_text)
        
        if 'image' in query:
            for img_path in query['image']:
                images.append(self.task.load_image(img_path))
        
        full_prompt = "\n\n".join(prompt_parts)
        return full_prompt, images
    
    def find_query_token_positions(self, input_ids: torch.Tensor) -> Dict[str, int]:
        if input_ids.dim() > 1:
            tokens = input_ids.squeeze().tolist()
        else:
            tokens = input_ids.tolist()
        
        token_texts = [self.tokenizer.decode([tid]) for tid in tokens]
        
        positions = {
            'query_forerunner': None,
            'last_input_text': None, 
            'query_label': None,
            'all_answer_positions': []
        }
        
        answer_patterns = ['Answer:', 'answer:', 'Answer', 'answer']
        
        for i, token_text in enumerate(token_texts):
            for pattern in answer_patterns:
                if pattern in token_text:
                    positions['all_answer_positions'].append(i)
                    if i + 1 < len(token_texts) and ':' in token_texts[i + 1]:
                        positions['all_answer_positions'].append(i + 1)
                    break
        
        if positions['all_answer_positions']:
            last_answer_pos = positions['all_answer_positions'][-1]
            
            for i in range(last_answer_pos, min(last_answer_pos + 3, len(token_texts))):
                if ':' in token_texts[i]:
                    positions['query_forerunner'] = i
                    break
            
            if positions['query_forerunner'] is not None:
                positions['query_label'] = positions['query_forerunner'] + 1
                
                answer_start = None
                for pos in reversed(positions['all_answer_positions']):
                    if 'Answer' in token_texts[pos]:
                        answer_start = pos
                        break
                
                if answer_start is not None and answer_start > 0:
                    positions['last_input_text'] = answer_start - 1
        
        return positions
    
    def register_hooks(self):
        self.hidden_states = []
        self.hooks = []
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            if hidden_state.dtype == torch.bfloat16:
                hidden_state = hidden_state.float()
            self.hidden_states.append(hidden_state.detach().cpu())
        
        if hasattr(self.model.model, 'language_model'):
            layers = self.model.model.language_model.model.layers
        elif hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'layers'):
            layers = self.model.model.model.layers
        else:
            raise ValueError("Cannot find language model layers")
        
        for layer in layers:
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def extract_hidden_states_at_position(self, position: int) -> List[np.ndarray]:
        layer_representations = []
        
        for layer_hidden in self.hidden_states:
            if layer_hidden.dim() == 3:
                batch_size, seq_len, hidden_dim = layer_hidden.shape
                if position is not None and 0 <= position < seq_len:
                    extracted_tensor = layer_hidden[0, position, :]
                else:
                    extracted_tensor = layer_hidden[0, -1, :]
            else:
                extracted_tensor = layer_hidden.flatten()[:3584]
            
            if extracted_tensor.dtype == torch.bfloat16:
                extracted_tensor = extracted_tensor.float()
            
            extracted = extracted_tensor.numpy()
            layer_representations.append(extracted)
        
        return layer_representations
    
    def get_bge_reference_embedding(self, query_identifier: str) -> np.ndarray:
        """
        Create a unique BGE embedding for each sample
        For OperatorInduction, use the operator and image info to create unique identifiers
        """
        if self.bge_model is None:
            self.load_bge_model()
        
        # Create a unique text for each sample instead of using the same question
        inputs = self.bge_tokenizer(query_identifier, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bge_model(**inputs)
            embedding_tensor = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
            
            if embedding_tensor.dtype == torch.bfloat16:
                embedding_tensor = embedding_tensor.float()
            
            embedding = embedding_tensor.numpy()
        
        return embedding
    
    def create_unique_query_identifier(self, query: Dict, sample_idx: int) -> str:
        """
        Create a unique identifier for each sample to get diverse BGE embeddings
        """
        operator = query.get('operator', 'unknown')
        
        # Create a unique string that captures the essence of this specific sample
        base_text = f"Mathematical expression with {operator} operator"
        
        # Add sample-specific information
        if 'image' in query and query['image']:
            # Use image path as additional identifier
            img_info = f"image_{os.path.basename(query['image'][0])}"
            unique_text = f"{base_text} {img_info} sample_{sample_idx}"
        else:
            unique_text = f"{base_text} sample_{sample_idx}"
        
        return unique_text
    
    def process_single_sample(self, query: Dict, n_shot: int = 4, sample_idx: int = 0) -> Dict:
        prompt, images = self.create_vlicl_prompt_and_images(query, n_shot)
        
        input_ids = self.tokenizer(prompt, return_tensors='pt')['input_ids']
        token_positions = self.find_query_token_positions(input_ids)
        
        try:
            self.register_hooks()
            
            with torch.no_grad():
                if images:
                    pixel_values = None
                    num_patches_list = None
                    
                    if len(images) == 1:
                        pixel_values = self.model.load_image(images[0], max_num=12).to(torch.bfloat16).cuda()
                    else:
                        pixel_values_list = []
                        num_patches_list = []
                        for img in images:
                            img_pixel_values = self.model.load_image(img, max_num=6)
                            pixel_values_list.append(img_pixel_values)
                            num_patches_list.append(img_pixel_values.size(0))
                        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.bfloat16).cuda()
                    
                    generation_config = dict(max_new_tokens=1, do_sample=False)
                    
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
                    inputs = self.tokenizer(prompt, return_tensors='pt')
                    inputs = {k: v.to(next(self.model.model.parameters()).device) for k, v in inputs.items()}
                    outputs = self.model.model(**inputs)
            
            results = {}
            
            if token_positions['query_forerunner'] is not None:
                results['query_forerunner'] = self.extract_hidden_states_at_position(token_positions['query_forerunner'])
            
            if token_positions['last_input_text'] is not None:
                results['last_input_text'] = self.extract_hidden_states_at_position(token_positions['last_input_text'])
            
            if token_positions['query_label'] is not None:
                results['query_label'] = self.extract_hidden_states_at_position(token_positions['query_label'])
            
            # Create unique BGE reference embedding for this specific sample
            query_identifier = self.create_unique_query_identifier(query, sample_idx)
            results['bge_reference'] = self.get_bge_reference_embedding(query_identifier)
            
            results['sample_info'] = {
                'operator': query.get('operator', 'unknown'),
                'n_shot': n_shot,
                'num_images': len(images),
                'sample_idx': sample_idx,
                'query_identifier': query_identifier
            }
            
            return results
            
        finally:
            self.remove_hooks()
            self.hidden_states.clear()
            torch.cuda.empty_cache()
    
    def debug_sample_diversity(self, results: Dict, k: int, num_debug_samples: int = 5):
        """Debug function to check if samples are actually diverse"""
        print(f"\n=== DEBUGGING SAMPLE DIVERSITY FOR k={k} ===")
        
        if 'query_forerunner' not in results or not results['query_forerunner']:
            print("No query_forerunner data found")
            return
        
        # Check BGE reference diversity
        bge_refs = results['bge_reference'][:num_debug_samples]
        print(f"\nBGE Reference Embeddings diversity:")
        for i in range(len(bge_refs)):
            for j in range(i+1, len(bge_refs)):
                cos_sim = np.dot(bge_refs[i], bge_refs[j]) / (np.linalg.norm(bge_refs[i]) * np.linalg.norm(bge_refs[j]))
                print(f"  BGE sample {i} vs {j}: cosine_sim = {cos_sim:.4f}")
        
        # Check hidden states diversity (just layer 0 and layer 15)
        forerunner_states = results['query_forerunner'][:num_debug_samples]
        for layer_idx in [0, 15]:
            if layer_idx < len(forerunner_states[0]):
                print(f"\nHidden states diversity at layer {layer_idx}:")
                layer_states = [sample[layer_idx] for sample in forerunner_states]
                for i in range(len(layer_states)):
                    for j in range(i+1, len(layer_states)):
                        cos_sim = np.dot(layer_states[i], layer_states[j]) / (np.linalg.norm(layer_states[i]) * np.linalg.norm(layer_states[j]))
                        print(f"  Hidden sample {i} vs {j}: cosine_sim = {cos_sim:.4f}")
        
        # Check sample info diversity
        sample_infos = results['sample_info'][:num_debug_samples]
        print(f"\nSample info:")
        for i, info in enumerate(sample_infos):
            print(f"  Sample {i}: operator={info.get('operator')}, identifier='{info.get('query_identifier', '')[:50]}...'")
    
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
                'task': 'operator_induction'
            },
            'data': {}
        }
        
        for k in k_values:
            print(f"\nProcessing k={k} demonstrations...")
            
            k_results = {
                'query_forerunner': [],
                'last_input_text': [],
                'query_label': [],
                'bge_reference': [],
                'sample_info': []
            }
            
            for i, query in enumerate(tqdm(query_samples, desc=f"k={k}")):
                try:
                    sample_results = self.process_single_sample(query, k, sample_idx=i)
                    
                    for key in ['query_forerunner', 'last_input_text', 'query_label', 'bge_reference', 'sample_info']:
                        if key in sample_results:
                            k_results[key].append(sample_results[key])
                        else:
                            print(f"Warning: {key} not found in sample {i} for k={k}")
                    
                except Exception as e:
                    print(f"Error processing sample {i} for k={k}: {e}")
                    continue
            
            # Debug sample diversity for first k value
            if k == k_values[0]:
                self.debug_sample_diversity(k_results, k)
            
            complete_results['data'][k] = k_results
            print(f"k={k} completed: {len(k_results['query_forerunner'])} samples")
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving complete results to {save_path}...")
        with open(save_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print(f"Extraction complete! Saved to {save_path}")
        self.print_extraction_summary(complete_results)
        
        return complete_results
    
    def print_extraction_summary(self, results: Dict):
        print(f"\n{'='*60}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*60}")
        
        info = results['extraction_info']
        print(f"Model: {info['model_name']}")
        print(f"Task: {info['task']}")
        print(f"Total samples: {info['num_samples']}")
        print(f"K values: {info['k_values']}")
        
        print(f"\nExtracted data structure:")
        for k, k_data in results['data'].items():
            if k_data['query_forerunner']:
                n_samples = len(k_data['query_forerunner'])
                n_layers = len(k_data['query_forerunner'][0])
                hidden_dim = k_data['query_forerunner'][0][0].shape[0]
                print(f"  k={k}: {n_samples} samples × {n_layers} layers × {hidden_dim} dims")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./VL-ICL", help="VL-ICL data directory")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-8B-Instruct", help="Model name")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to process")
    parser.add_argument("--k_values", nargs="+", type=int, default=[0, 1, 2, 4, 8], help="K values to process")
    parser.add_argument("--save_path", type=str, default="./data/vlicl_hidden_states.pkl", help="Save path for results")
    parser.add_argument("--debug_only", action="store_true", help="Only run token position debugging")
    
    args = parser.parse_args()
    
    extractor = VLICLHiddenStatesExtractor(args.model_name, args.data_dir)
    
    if args.debug_only:
        extractor.load_model()
        print("Running token position debugging only...")
        
        query = extractor.task.query_data[0]
        prompt, images = extractor.create_vlicl_prompt_and_images(query, 4)
        
        print(f"Sample operator: {query.get('operator', 'unknown')}")
        print(f"Images: {len(images)}")
        print(f"Prompt (last 300 chars): ...{prompt[-300:]}")
        
        input_ids = extractor.tokenizer(prompt, return_tensors='pt')['input_ids']
        positions = extractor.find_query_token_positions(input_ids)
        print(f"Token positions: {positions}")
    else:
        print("Running complete extraction...")
        results = extractor.extract_complete_dataset(
            num_samples=args.num_samples,
            k_values=args.k_values,
            save_path=args.save_path
        )

if __name__ == "__main__":
    main()