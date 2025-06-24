import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from typing import Dict, List, Tuple
import os

class VLICLKernelAlignment:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded data from {self.data_path}")
        print(f"Available k values: {list(self.data['data'].keys())}")
        
        for k, k_data in self.data['data'].items():
            if k_data['query_forerunner']:
                n_samples = len(k_data['query_forerunner'])
                n_layers = len(k_data['query_forerunner'][0])
                print(f"k={k}: {n_samples} samples, {n_layers} layers")
    
    def sim_graph(self, representations: List[np.ndarray]) -> np.ndarray:
        n = len(representations)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    norm_i = np.linalg.norm(representations[i])
                    norm_j = np.linalg.norm(representations[j])
                    if norm_i > 0 and norm_j > 0:
                        similarity_matrix[i, j] = np.dot(representations[i], representations[j]) / (norm_i * norm_j)
                    else:
                        similarity_matrix[i, j] = 0.0
        
        return similarity_matrix
    
    def overlap(self, list1: List, list2: List) -> int:
        return len(set(list1).intersection(set(list2)))
    
    def kernel_alignment(self, simGraph_1: np.ndarray, simGraph_2: np.ndarray, k: int = 8) -> Tuple[float, float, List[float]]:
        n = len(simGraph_1)
        k = min(k, n-1)
        
        aligns = []
        for i in range(n):
            top_k_1 = np.argsort(simGraph_1[i])[::-1][:k]
            top_k_2 = np.argsort(simGraph_2[i])[::-1][:k]
            
            overlap_count = self.overlap(top_k_1.tolist(), top_k_2.tolist())
            aligns.append(overlap_count / k)
        
        return np.mean(aligns), np.std(aligns), aligns
    
    def cosine_similarity_analysis(self, representations_1: List[np.ndarray], representations_2: List[np.ndarray]) -> Tuple[float, float, List[float]]:
        similarities = []
        
        for rep1, rep2 in zip(representations_1, representations_2):
            norm1 = np.linalg.norm(rep1)
            norm2 = np.linalg.norm(rep2)
            
            if norm1 > 0 and norm2 > 0:
                cosine_sim = np.dot(rep1, rep2) / (norm1 * norm2)
                similarities.append(cosine_sim)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities), np.std(similarities), similarities
    
    def analyze_token_type_comparison(self, k: int = 4) -> Dict:
        print(f"Analyzing token type comparison (k={k})...")
        
        if k not in self.data['data']:
            raise ValueError(f"k={k} not found in data")
        
        k_data = self.data['data'][k]
        
        query_forerunner_states = k_data['query_forerunner']
        last_input_text_states = k_data['last_input_text'] 
        query_label_states = k_data['query_label']
        query_mean_pooled_states = k_data['query_mean_pooled']
        
        if not query_forerunner_states:
            print("No data available for analysis")
            return {}
        
        n_layers = len(query_forerunner_states[0])
        n_samples = len(query_forerunner_states)
        print(f"Analyzing {n_layers} layers, {n_samples} samples")
        
        comparison_results = {
            'query_forerunner': {'kernel_alignments': [], 'cosine_similarities': []},
            'last_input_text': {'kernel_alignments': [], 'cosine_similarities': []},
            'query_label': {'kernel_alignments': [], 'cosine_similarities': []}
        }
        
        for token_type in ['query_forerunner', 'last_input_text', 'query_label']:
            print(f"Processing {token_type}...")
            token_states = k_data[token_type]
            
            for layer_idx in tqdm(range(n_layers), desc=f"{token_type} layers"):
                layer_states = [sample_states[layer_idx] for sample_states in token_states]
                query_layer_states = [sample_states[layer_idx] for sample_states in query_mean_pooled_states]
                
                layer_sim_graph = self.sim_graph(layer_states)
                query_sim_graph = self.sim_graph(query_layer_states)
                
                mean_align, std_align, individual_aligns = self.kernel_alignment(layer_sim_graph, query_sim_graph)
                comparison_results[token_type]['kernel_alignments'].append((mean_align, std_align, individual_aligns))
                
                mean_cosine, std_cosine, individual_cosines = self.cosine_similarity_analysis(layer_states, query_layer_states)
                comparison_results[token_type]['cosine_similarities'].append((mean_cosine, std_cosine, individual_cosines))
        
        return comparison_results
    
    def analyze_different_k_values(self) -> Dict:
        print("Analyzing different k values (query_forerunner only)...")
        
        k_analysis_results = {}
        
        for k, k_data in self.data['data'].items():
            print(f"Processing k={k}...")
            
            query_forerunner_states = k_data['query_forerunner']
            query_mean_pooled_states = k_data['query_mean_pooled']
            
            if not query_forerunner_states:
                print(f"No data for k={k}")
                continue
            
            n_layers = len(query_forerunner_states[0])
            n_samples = len(query_forerunner_states)
            
            print(f"k={k}: {n_layers} layers, {n_samples} samples")
            
            k_results = {'kernel_alignments': [], 'cosine_similarities': []}
            
            for layer_idx in tqdm(range(n_layers), desc=f"k={k} layers"):
                layer_states = [sample_states[layer_idx] for sample_states in query_forerunner_states]
                query_layer_states = [sample_states[layer_idx] for sample_states in query_mean_pooled_states]
                
                layer_sim_graph = self.sim_graph(layer_states)
                query_sim_graph = self.sim_graph(query_layer_states)
                
                mean_align, std_align, individual_aligns = self.kernel_alignment(layer_sim_graph, query_sim_graph)
                k_results['kernel_alignments'].append((mean_align, std_align, individual_aligns))
                
                mean_cosine, std_cosine, individual_cosines = self.cosine_similarity_analysis(layer_states, query_layer_states)
                k_results['cosine_similarities'].append((mean_cosine, std_cosine, individual_cosines))
            
            k_analysis_results[k] = k_results
        
        return k_analysis_results
    
    def get_model_name(self) -> str:
        if 'extraction_info' in self.data and 'model_name' in self.data['extraction_info']:
            model_name = self.data['extraction_info']['model_name']
            return model_name.split('/')[-1]
        else:
            return os.path.basename(self.data_path).split('_')[0]
    
    def plot_token_type_comparison(self, comparison_results: Dict):
        model_name = self.get_model_name()
        os.makedirs("./figs", exist_ok=True)
        
        colors = {
            'query_forerunner': '#1f77b4',
            'last_input_text': '#ff7f0e', 
            'query_label': '#2ca02c'
        }
        
        labels = {
            'query_forerunner': 'Forerunner Token of Label',
            'last_input_text': 'Last Token of Input Text',
            'query_label': 'Label Token'
        }
        
        n_layers = len(comparison_results['query_forerunner']['kernel_alignments'])
        layer_indices = list(range(1, n_layers + 1))
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for token_type, results in comparison_results.items():
            kernel_means = [result[0] for result in results['kernel_alignments']]
            ax.plot(layer_indices, kernel_means, color=colors[token_type], 
                   label=labels[token_type], linewidth=2)
        
        random_baseline = 8 / len(comparison_results['query_forerunner']['kernel_alignments'][0][2])
        ax.axhline(y=random_baseline, color='black', linestyle='--', linewidth=1, label='Random Baseline')
        
        ax.set_xlabel('Transformer Block Number')
        ax.set_ylabel('Kernel Alignment')
        ax.set_title(f'Token Type Comparison - Kernel Alignment ({model_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        kernel_path = f"./figs/{model_name}_token_type_kernel_alignment.png"
        plt.savefig(kernel_path, dpi=300, bbox_inches='tight')
        print(f"Saved kernel alignment plot to {kernel_path}")
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for token_type, results in comparison_results.items():
            cosine_means = [result[0] for result in results['cosine_similarities']]
            ax.plot(layer_indices, cosine_means, color=colors[token_type], 
                   label=labels[token_type], linewidth=2)
        
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1, label='Zero Baseline')
        
        ax.set_xlabel('Transformer Block Number')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Token Type Comparison - Cosine Similarity ({model_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cosine_path = f"./figs/{model_name}_token_type_cosine_similarity.png"
        plt.savefig(cosine_path, dpi=300, bbox_inches='tight')
        print(f"Saved cosine similarity plot to {cosine_path}")
        plt.close()
    
    def plot_k_value_comparison(self, k_analysis_results: Dict):
        model_name = self.get_model_name()
        os.makedirs("./figs", exist_ok=True)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(k_analysis_results)))
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for i, (k, results) in enumerate(k_analysis_results.items()):
            kernel_means = [result[0] for result in results['kernel_alignments']]
            n_layers = len(kernel_means)
            layer_indices = list(range(1, n_layers + 1))
            
            ax.plot(layer_indices, kernel_means, color=colors[i], 
                   label=f'k={k}', linewidth=2)
        
        if k_analysis_results:
            first_result = list(k_analysis_results.values())[0]
            random_baseline = 8 / len(first_result['kernel_alignments'][0][2])
            ax.axhline(y=random_baseline, color='black', linestyle='--', linewidth=1, label='Random Baseline')
        
        ax.set_xlabel('Transformer Block Number')
        ax.set_ylabel('Kernel Alignment')
        ax.set_title(f'K Value Comparison - Kernel Alignment ({model_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        kernel_path = f"./figs/{model_name}_k_value_kernel_alignment.png"
        plt.savefig(kernel_path, dpi=300, bbox_inches='tight')
        print(f"Saved kernel alignment plot to {kernel_path}")
        plt.close()
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        for i, (k, results) in enumerate(k_analysis_results.items()):
            cosine_means = [result[0] for result in results['cosine_similarities']]
            n_layers = len(cosine_means)
            layer_indices = list(range(1, n_layers + 1))
            
            ax.plot(layer_indices, cosine_means, color=colors[i], 
                   label=f'k={k}', linewidth=2)
        
        ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1, label='Zero Baseline')
        
        ax.set_xlabel('Transformer Block Number')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'K Value Comparison - Cosine Similarity ({model_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        cosine_path = f"./figs/{model_name}_k_value_cosine_similarity.png"
        plt.savefig(cosine_path, dpi=300, bbox_inches='tight')
        print(f"Saved cosine similarity plot to {cosine_path}")
        plt.close()
    
    def run_complete_analysis(self, output_dir: str = "./analysis_results"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("VL-ICL KERNEL ALIGNMENT ANALYSIS")
        print("="*60)
        
        print("\n1. Token Type Comparison Analysis")
        token_comparison = self.analyze_token_type_comparison(k=4)
        self.plot_token_type_comparison(token_comparison)
        
        print("\n2. K Value Comparison Analysis")
        k_comparison = self.analyze_different_k_values()
        self.plot_k_value_comparison(k_comparison)
        
        results = {
            'token_type_comparison': token_comparison,
            'k_value_comparison': k_comparison
        }
        
        model_name = self.get_model_name()
        results_path = os.path.join(output_dir, f"{model_name}_kernel_alignment_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved analysis results to {results_path}")
        
        print("\n3. Summary Statistics")
        self.print_summary_statistics(token_comparison, k_comparison)
        
        return results
    
    def print_summary_statistics(self, token_comparison: Dict, k_comparison: Dict):
        print("\nToken Type Peak Performance (Kernel Alignment):")
        for token_type, results in token_comparison.items():
            kernel_means = [result[0] for result in results['kernel_alignments']]
            max_val = max(kernel_means)
            max_layer = kernel_means.index(max_val) + 1
            print(f"  {token_type}: {max_val:.4f} at layer {max_layer}")
        
        print("\nK Value Peak Performance (Kernel Alignment):")
        for k, results in k_comparison.items():
            kernel_means = [result[0] for result in results['kernel_alignments']]
            max_val = max(kernel_means)
            max_layer = kernel_means.index(max_val) + 1
            print(f"  k={k}: {max_val:.4f} at layer {max_layer}")
        
        print("\nToken Type Peak Performance (Cosine Similarity):")
        for token_type, results in token_comparison.items():
            cosine_means = [result[0] for result in results['cosine_similarities']]
            max_val = max(cosine_means)
            max_layer = cosine_means.index(max_val) + 1
            print(f"  {token_type}: {max_val:.4f} at layer {max_layer}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/InternVL3-38B-Instruct/vlicl_hidden_states_final.pkl", help="Path to VL-ICL hidden states pickle file")
    parser.add_argument("--output_dir", type=str, default="./figs", help="Output directory for results")
    
    args = parser.parse_args()
    
    analyzer = VLICLKernelAlignment(args.data_path)
    results = analyzer.run_complete_analysis(args.output_dir)

if __name__ == "__main__":
    main()