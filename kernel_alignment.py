import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from tqdm import tqdm

class KernelAlignmentAnalyzer:
    def __init__(self):
        pass
    
    def sim_graph(self, features: List[np.ndarray]) -> List[List[float]]:
        """Calculate similarity graph using cosine similarity (exact replication from ICL_Circuit repo)"""
        simGraph = []
        for i in tqdm(range(len(features)), desc="Computing similarity graph"):
            line = []
            for j in range(len(features)):
                if i == j:
                    line.append(0)  # Set diagonal to 0 as per paper
                else:
                    dot_product = np.dot(features[i], features[j])
                    norm_i = np.linalg.norm(features[i])
                    norm_j = np.linalg.norm(features[j])
                    if norm_i > 0 and norm_j > 0:
                        cos_sim = dot_product / (norm_i * norm_j)
                    else:
                        cos_sim = 0.0
                    line.append(cos_sim)
            simGraph.append(line)
        return simGraph
    
    def overlap(self, a: List[int], b: List[int]) -> int:
        """Count overlap between two lists (exact replication from ICL_Circuit repo)"""
        return len(set(a) & set(b))
    
    def kernel_alignment(self, simGraph_1: List[List[float]], simGraph_2: List[List[float]], k: int = 64) -> Tuple[float, float, List[float]]:
        """
        Calculate mutual nearest-neighbor kernel alignment (exact replication from ICL_Circuit repo)
        
        Args:
            simGraph_1: First similarity graph
            simGraph_2: Second similarity graph
            k: Number of nearest neighbors (default 64 as per paper)
        Returns:
            (mean_alignment, std_alignment, individual_alignments)
        """
        n = len(simGraph_1)
        k = min(k, n-1)  # Ensure k doesn't exceed available neighbors
        
        aligns = []
        for i in range(n):
            # Get top-k indices for both graphs (sorted descending)
            top_k_1 = np.argsort(simGraph_1[i])[::-1][:k]
            top_k_2 = np.argsort(simGraph_2[i])[::-1][:k]
            
            # Calculate overlap ratio
            overlap_count = self.overlap(top_k_1.tolist(), top_k_2.tolist())
            aligns.append(overlap_count / k)
        
        return np.mean(aligns), np.std(aligns), aligns
    
    def analyze_token_type_comparison(self, k4_data: Dict) -> Dict:
        """Analyze token type comparison for Figure 2 Left (using k=4 data)"""
        print("Analyzing token type comparison (Figure 2 Left)...")
        
        query_forerunner_states = k4_data['query_forerunner']
        last_input_text_states = k4_data['last_input_text'] 
        query_label_states = k4_data['query_label']
        bge_references = k4_data['bge_reference']
        
        if not query_forerunner_states:
            print("No data available for analysis")
            return {}
        
        n_layers = len(query_forerunner_states[0])
        n_samples = len(query_forerunner_states)
        print(f"Analyzing {n_layers} layers, {n_samples} samples")
        
        # Calculate BGE reference similarity graph
        print("Computing BGE reference similarity graph...")
        bge_sim_graph = self.sim_graph(bge_references)
        
        comparison_results = {
            'query_forerunner': {'kernel_alignments': []},
            'last_input_text': {'kernel_alignments': []},
            'query_label': {'kernel_alignments': []}
        }
        
        # Process each token type
        for token_type in ['query_forerunner', 'last_input_text', 'query_label']:
            print(f"Processing {token_type}...")
            token_states = k4_data[token_type]
            
            for layer_idx in tqdm(range(n_layers), desc=f"{token_type} layers"):
                # Extract layer-specific states for all samples
                layer_states = [sample_states[layer_idx] for sample_states in token_states]
                
                # Calculate similarity graph for this layer
                layer_sim_graph = self.sim_graph(layer_states)
                
                # Calculate kernel alignment with BGE reference
                mean_align, std_align, individual_aligns = self.kernel_alignment(layer_sim_graph, bge_sim_graph)
                comparison_results[token_type]['kernel_alignments'].append((mean_align, std_align, individual_aligns))
        
        return comparison_results
    
    def analyze_different_k_values(self, all_data: Dict) -> Dict:
        """Analyze different k values for Figure 2 Middle (using query_forerunner only)"""
        print("Analyzing different k values (Figure 2 Middle)...")
        
        k_analysis_results = {}
        
        for k, k_data in all_data.items():
            print(f"Processing k={k}...")
            
            query_forerunner_states = k_data['query_forerunner']
            bge_references = k_data['bge_reference']
            
            if not query_forerunner_states:
                print(f"No data for k={k}")
                continue
            
            n_layers = len(query_forerunner_states[0])
            n_samples = len(query_forerunner_states)
            print(f"  {n_samples} samples, {n_layers} layers")
            
            # Calculate BGE reference similarity graph for this k
            bge_sim_graph = self.sim_graph(bge_references)
            
            k_alignments = []
            for layer_idx in tqdm(range(n_layers), desc=f"k={k} layers"):
                # Extract layer-specific states for all samples
                layer_states = [sample_states[layer_idx] for sample_states in query_forerunner_states]
                
                # Calculate similarity graph for this layer
                layer_sim_graph = self.sim_graph(layer_states)
                
                # Calculate kernel alignment with BGE reference
                mean_align, std_align, individual_aligns = self.kernel_alignment(layer_sim_graph, bge_sim_graph)
                k_alignments.append((mean_align, std_align, individual_aligns))
            
            k_analysis_results[k] = {'kernel_alignments': k_alignments}
        
        return k_analysis_results
    
    def load_and_analyze(self, data_path: str) -> Tuple[Dict, Dict]:
        """Load extracted data and run both analyses"""
        print(f"Loading data from {data_path}...")
        with open(data_path, 'rb') as f:
            results = pickle.load(f)
        
        print("Data loaded successfully!")
        print(f"Model: {results['extraction_info']['model_name']}")
        print(f"Task: {results['extraction_info']['task']}")
        print(f"K values: {results['extraction_info']['k_values']}")
        
        all_data = results['data']
        
        # Analysis 1: Token type comparison (Figure 2 Left) using k=4
        if 4 in all_data:
            comparison_results = self.analyze_token_type_comparison(all_data[4])
        else:
            print("Warning: k=4 data not found for token comparison")
            comparison_results = {}
        
        # Analysis 2: Different k values (Figure 2 Middle) 
        k_analysis_results = self.analyze_different_k_values(all_data)
        
        return comparison_results, k_analysis_results

def plot_figure2_left(comparison_results: Dict, save_path: str = "figs/figure2_left_vlicl.png"):
    """Plot Figure 2 Left: Token type comparison"""
    plt.figure(figsize=(10, 6))
    
    # Colors and labels matching the paper
    colors = {
        'query_forerunner': 'blue', 
        'last_input_text': 'green', 
        'query_label': 'red'
    }
    labels = {
        'query_forerunner': 'Forerunner Token of Label', 
        'last_input_text': 'Last Token of Input Text', 
        'query_label': 'Label Token'
    }
    
    for token_type, results in comparison_results.items():
        if 'kernel_alignments' in results and results['kernel_alignments']:
            alignments = [ka[0] for ka in results['kernel_alignments']]
            layers = list(range(len(alignments)))
            plt.plot(layers, alignments, color=colors[token_type], 
                    label=labels[token_type], linewidth=2)
    
    # Random baseline as per paper
    plt.axhline(y=0.125, color='black', linestyle='--', linewidth=1, label='Random Baseline')
    
    plt.xlabel('Transformer Block Number', fontsize=12)
    plt.ylabel('Kernel Alignment', fontsize=12)
    plt.title('VL-ICL: Input Text Encoding in Hidden States (Figure 2 Left)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure 2 Left saved to {save_path}")

def plot_figure2_middle(k_analysis_results: Dict, save_path: str = "figs/figure2_middle_vlicl.png"):
    """Plot Figure 2 Middle: Different k values"""
    plt.figure(figsize=(10, 6))
    
    colors = ['purple', 'orange', 'green', 'blue', 'red']
    
    # Sort k values for consistent plotting
    sorted_k_values = sorted(k_analysis_results.keys())
    
    for i, k in enumerate(sorted_k_values):
        results = k_analysis_results[k]
        if 'kernel_alignments' in results and results['kernel_alignments']:
            alignments = [ka[0] for ka in results['kernel_alignments']]
            layers = list(range(len(alignments)))
            plt.plot(layers, alignments, color=colors[i % len(colors)], 
                    label=f'k={k}', linewidth=2)
    
    # Random baseline as per paper
    plt.axhline(y=0.125, color='black', linestyle='--', linewidth=1, label='Random Baseline')
    
    plt.xlabel('Transformer Block Number', fontsize=12)
    plt.ylabel('Kernel Alignment', fontsize=12)
    plt.title('VL-ICL: Enhancement by Demonstrations (Figure 2 Middle)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure 2 Middle saved to {save_path}")

def print_analysis_summary(comparison_results: Dict, k_analysis_results: Dict):
    """Print summary of analysis results"""
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Figure 2 Left summary
    if comparison_results:
        print("\nFigure 2 Left - Token Type Comparison:")
        for token_type, results in comparison_results.items():
            if 'kernel_alignments' in results and results['kernel_alignments']:
                alignments = [ka[0] for ka in results['kernel_alignments']]
                max_alignment = max(alignments)
                max_layer = np.argmax(alignments)
                print(f"  {token_type}: max={max_alignment:.4f} at layer {max_layer}")
    
    # Figure 2 Middle summary
    if k_analysis_results:
        print("\nFigure 2 Middle - Different K Values:")
        for k in sorted(k_analysis_results.keys()):
            results = k_analysis_results[k]
            if 'kernel_alignments' in results and results['kernel_alignments']:
                alignments = [ka[0] for ka in results['kernel_alignments']]
                max_alignment = max(alignments)
                max_layer = np.argmax(alignments)
                print(f"  k={k}: max={max_alignment:.4f} at layer {max_layer}")

def save_analysis_results(comparison_results: Dict, k_analysis_results: Dict, save_path: str = "vlicl_analysis_results.pkl"):
    """Save analysis results for future use"""
    analysis_data = {
        'comparison_results': comparison_results,
        'k_analysis_results': k_analysis_results,
        'analysis_info': {
            'analyzer': 'KernelAlignmentAnalyzer',
            'method': 'mutual_nearest_neighbor_kernel_alignment',
            'k_neighbors': 64,
            'random_baseline': 0.125
        }
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(analysis_data, f)
    
    print(f"Analysis results saved to {save_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/vlicl_hidden_states.pkl", 
                       help="Path to extracted hidden states")
    parser.add_argument("--save_analysis", type=str, default="./data/vlicl_analysis_results.pkl",
                       help="Path to save analysis results")
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting")
    
    args = parser.parse_args()
    
    print("VL-ICL Kernel Alignment Analysis")
    print("="*50)
    
    # Initialize analyzer
    analyzer = KernelAlignmentAnalyzer()
    
    # Load data and run analysis
    comparison_results, k_analysis_results = analyzer.load_and_analyze(args.data_path)
    
    # Print summary
    print_analysis_summary(comparison_results, k_analysis_results)
    
    # Save analysis results
    save_analysis_results(comparison_results, k_analysis_results, args.save_analysis)
    
    # Create plots
    if not args.no_plots:
        print("\nCreating plots...")
        
        if comparison_results:
            plot_figure2_left(comparison_results)
        else:
            print("Skipping Figure 2 Left - no comparison data")
        
        if k_analysis_results:
            plot_figure2_middle(k_analysis_results)
        else:
            print("Skipping Figure 2 Middle - no k analysis data")
    
    print("\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()