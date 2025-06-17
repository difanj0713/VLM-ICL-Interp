#!/usr/bin/env python3
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple
from tqdm import tqdm

class KernelAlignmentAnalyzer:
    def __init__(self, k_neighbors: int = 16):
        self.k_neighbors = k_neighbors
        # Random baseline = k/n
    
    def sim_graph(self, features: List[np.ndarray]) -> List[List[float]]:
        """Calculate similarity graph using cosine similarity"""
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
    
    def kernel_alignment(self, simGraph_1: List[List[float]], simGraph_2: List[List[float]], k: int = None) -> Tuple[float, float, List[float]]:
        """Calculate mutual nearest-neighbor kernel alignment"""
        if k is None:
            k = self.k_neighbors
        
        n = len(simGraph_1)
        k = min(k, n - 1)  # Ensure k doesn't exceed available neighbors
        
        # Similar to their repo
        aligns = []
        for i in range(n):
            top_k_1 = np.argsort(simGraph_1[i])[::-1][:k]
            top_k_2 = np.argsort(simGraph_2[i])[::-1][:k]
            
            overlap = len(set(top_k_1.tolist()) & set(top_k_2.tolist()))
            aligns.append(overlap / k)
        
        return np.mean(aligns), np.std(aligns), aligns
    
    def average_cosine_similarity(self, features1: List[np.ndarray], features2: List[np.ndarray]) -> float:
        similarities = []
        for f1, f2 in zip(features1, features2):
            cos_sim = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            similarities.append(cos_sim)
        return np.mean(similarities)
    
    def analyze_token_types(self, k4_data: Dict) -> Dict:
        print("Analyzing token type comparison...")
        
        query_forerunner_states = k4_data['query_forerunner']
        last_input_text_states = k4_data['last_input_text'] 
        query_label_states = k4_data['query_label']
        mean_pooling_states = k4_data['mean_pooling']
        bge_references = k4_data['bge_reference']
        
        n_layers = len(query_forerunner_states[0])
        n_samples = len(query_forerunner_states)
        
        print(f"Processing {n_samples} samples, {n_layers} layers")
        print(f"Using k_neighbors = {self.k_neighbors}")
        print(f"Random baseline = {self.k_neighbors}/{n_samples} = {self.k_neighbors/n_samples:.4f}")
        
        # Calculate BGE reference similarity graph (FIXED - same for all layers)
        print("Computing BGE reference similarity graph...")
        bge_sim_graph = self.sim_graph(bge_references)
        
        results = {
            'forerunner': {'cosine_vs_mean_pool': [], 'kernel_align_vs_bge': []},
            'last_input_text': {'cosine_vs_mean_pool': [], 'kernel_align_vs_bge': []},
            'label': {'cosine_vs_mean_pool': [], 'kernel_align_vs_bge': []}
        }
        
        for layer_idx in tqdm(range(n_layers), desc="Processing layers"):
            layer_features = {
                'forerunner': [query_forerunner_states[i][layer_idx] for i in range(n_samples)],
                'last_input_text': [last_input_text_states[i][layer_idx] for i in range(n_samples)],
                'label': [query_label_states[i][layer_idx] for i in range(n_samples)]
            }
            
            mean_pool_layer = [mean_pooling_states[i][layer_idx] for i in range(n_samples)]
            
            for token_type, features in layer_features.items():
                # Same-dimension cosine similarity
                cos_vs_mean = self.average_cosine_similarity(features, mean_pool_layer)
                results[token_type]['cosine_vs_mean_pool'].append(cos_vs_mean)
                
                token_sim_graph = self.sim_graph(features)
                ka_vs_bge, _, _ = self.kernel_alignment(token_sim_graph, bge_sim_graph, k=self.k_neighbors)
                results[token_type]['kernel_align_vs_bge'].append(ka_vs_bge)
                
                if layer_idx == 0:
                    print(f"Layer 0 - {token_type}: kernel_align={ka_vs_bge:.4f}")
        
        return results
    
    def analyze_k_values(self, complete_data: Dict) -> Dict:
        print("Analyzing k value comparison...")
        
        results = {}
        
        for k in complete_data['data'].keys():
            k_data = complete_data['data'][k]
            if not k_data['query_forerunner'] or not k_data['bge_reference']:
                continue
                
            print(f"Processing k={k}")
            
            query_forerunner_states = k_data['query_forerunner']
            mean_pooling_states = k_data['mean_pooling']
            bge_references = k_data['bge_reference']
            
            n_layers = len(query_forerunner_states[0])
            n_samples = len(query_forerunner_states)
            
            # DEBUG: Check if BGE references are actually different
            if k == 0:
                bge_sims = []
                for i in range(min(5, n_samples)):
                    for j in range(i+1, min(5, n_samples)):
                        cos_sim = np.dot(bge_references[i], bge_references[j]) / (np.linalg.norm(bge_references[i]) * np.linalg.norm(bge_references[j]))
                        bge_sims.append(cos_sim)
                print(f"  BGE diversity check (k={k}): mean_sim={np.mean(bge_sims):.4f}, std={np.std(bge_sims):.4f}")
            
            # Calculate BGE similarity graph (FIXED - same for all layers of this k)
            bge_sim_graph = self.sim_graph(bge_references)
            
            cosine_vs_mean_pool = []
            kernel_align_vs_bge = []
            
            for layer_idx in range(n_layers):
                forerunner_features = [query_forerunner_states[i][layer_idx] for i in range(n_samples)]
                mean_pool_layer = [mean_pooling_states[i][layer_idx] for i in range(n_samples)]
                
                # Cosine similarity
                cos_sim = self.average_cosine_similarity(forerunner_features, mean_pool_layer)
                cosine_vs_mean_pool.append(cos_sim)
                
                # Kernel alignment (compare with SAME BGE similarity graph for this k)
                forerunner_sim_graph = self.sim_graph(forerunner_features)
                ka_mean, _, _ = self.kernel_alignment(forerunner_sim_graph, bge_sim_graph, k=self.k_neighbors)
                kernel_align_vs_bge.append(ka_mean)
            
            results[k] = {
                'cosine_vs_mean_pool': cosine_vs_mean_pool,
                'kernel_align_vs_bge': kernel_align_vs_bge,
                'n_samples': n_samples,
                'random_baseline': self.k_neighbors / n_samples
            }
            
            print(f"k={k}: Layer 0 kernel_align={kernel_align_vs_bge[0]:.4f}, Random baseline={self.k_neighbors/n_samples:.4f}")
        
        return results

def plot_token_kernel_alignment(comparison_results: Dict, save_path: str = "figs/token_kernel_alignment.png"):
    plt.figure(figsize=(10, 6))
    
    colors = {
        'forerunner': '#1f77b4',
        'last_input_text': '#ff7f0e', 
        'label': '#2ca02c'
    }
    
    labels = {
        'forerunner': 'Forerunner Token of Label',
        'last_input_text': 'Last Token of Input Text',
        'label': 'Label Token'
    }
    
    for token_type, results in comparison_results.items():
        if 'kernel_align_vs_bge' in results:
            alignments = results['kernel_align_vs_bge']
            layers = list(range(len(alignments)))
            plt.plot(layers, alignments, color=colors[token_type], 
                    label=labels[token_type], linewidth=2.5, marker='o', markersize=4)

    if comparison_results:
        n_layers = len(list(comparison_results.values())[0]['kernel_align_vs_bge'])
        random_baseline = 8/60
        layers = list(range(n_layers))
        plt.axhline(y=random_baseline, color='black', linestyle='--', 
                   linewidth=2, label='Random Baseline')
    
    plt.xlabel('Transformer Block Number', fontsize=20)
    plt.ylabel('Kernel Alignment', fontsize=20)
    plt.legend(fontsize=18, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_k_value_kernel_alignment(k_analysis_results: Dict, save_path: str = "figs/k_value_kernel_alignment.png"):
    plt.figure(figsize=(10, 6))
    
    colors = ['#8e44ad', '#e67e22', '#27ae60', '#3498db', '#e74c3c']

    for i, (k, results) in enumerate(sorted(k_analysis_results.items())):
        if 'kernel_align_vs_bge' in results:
            alignments = results['kernel_align_vs_bge']
            layers = list(range(len(alignments)))
            plt.plot(layers, alignments, color=colors[i % len(colors)], 
                    label=f'k={k}', linewidth=2.5, marker='o', markersize=4)
    
    if k_analysis_results:
        first_k = list(k_analysis_results.keys())[0]
        random_baseline = k_analysis_results[first_k]['random_baseline']
        n_layers = len(k_analysis_results[first_k]['kernel_align_vs_bge'])
        layers = list(range(n_layers))
        plt.axhline(y=random_baseline, color='black', linestyle='--', 
                   linewidth=2, label='Random Baseline')
    
    plt.xlabel('Transformer Block Number', fontsize=20)
    plt.ylabel('Kernel Alignment', fontsize=20)
    plt.legend(fontsize=18, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/InternVL3-8B-Instruct/vlicl_hidden_states_final.pkl")
    parser.add_argument("--k_neighbors", type=int, default=8)
    
    args = parser.parse_args()
    
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    analyzer = KernelAlignmentAnalyzer(k_neighbors=args.k_neighbors)
    
    if 4 in data['data']:
        print("\n=== Token Type Analysis ===")
        comparison_results = analyzer.analyze_token_types(data['data'][4])
        plot_token_kernel_alignment(comparison_results)
    
    print("\n=== K Value Analysis ===")
    k_analysis_results = analyzer.analyze_k_values(data)
    plot_k_value_kernel_alignment(k_analysis_results)

if __name__ == "__main__":
    main()