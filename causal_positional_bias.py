import torch
import numpy as np
import json
import os
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import re

from models.model_factory import create_model
from tasks.i2t_tasks import OperatorInductionTask

def build_vl_icl_prompt(task, demonstrations, query, mode="constrained", debug=False, max_images=6):
    """Build prompt following VL-ICL standard format with image limit handling"""
    
    prompt_parts = [task.get_task_instruction()]
    
    if demonstrations:
        total_demo_images = sum(len(demo.get('image', [])) for demo in demonstrations)
        query_images = len(query.get('image', [])) if isinstance(query.get('image'), list) else (1 if query.get('image') else 0)
        total_images = total_demo_images + query_images
        
        if total_images > max_images:
            available_for_demos = max_images - query_images
            if available_for_demos > 0:
                demonstrations = demonstrations[-available_for_demos:]
        
        support_parts = []
        for demo in demonstrations:
            demo_text = task.format_demonstration(demo, include_image_token=True, mode=mode)
            support_parts.append(demo_text)
        
        support_set = "\n\n".join(support_parts) if mode == "free" else "\n".join(support_parts)
        prompt_parts.append(f"Support Set:\n{support_set}")
    
    query_text = task.format_query(query, include_image_token=True, mode=mode)
    prompt_parts.append(f"Query:\n{query_text}")
    prompt_parts.append("Answer:")
    
    full_prompt = "\n\n".join(prompt_parts)
    return full_prompt

def collect_images_for_prompt(task, demonstrations, query, debug=False, max_images=6):
    """Collect all images in the order they appear in the prompt, respecting limits"""
    images = []
    
    for demo in demonstrations:
        if 'image' in demo:
            for img_path in demo['image']:
                if len(images) < max_images - 1:
                    images.append(task.load_image(img_path))
                else:
                    break
            if len(images) >= max_images - 1:
                break
    
    if 'image' in query and len(images) < max_images:
        if isinstance(query['image'], list):
            for img_path in query['image']:
                if len(images) < max_images:
                    images.append(task.load_image(img_path))
        else:
            images.append(task.load_image(query['image']))
    
    return images

class PositionalInterventionExperiment:
    def __init__(self, model_name: str, data_dir: str):
        self.model_name = model_name
        self.data_dir = data_dir
        self.model = None
        self.task = None
        
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        self.model = create_model('internvl', self.model_name)
        self.task = OperatorInductionTask(self.data_dir)
    
    def extract_numbers_from_path(self, image_path: str) -> Tuple[int, int]:
        filename = os.path.basename(image_path)
        numbers = re.findall(r'\d+', filename)
        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        return None, None
    
    def compute_answer(self, num1: int, num2: int, operator: str) -> int:
        if operator == '+':
            return num1 + num2
        elif operator == '-':
            return num1 - num2
        elif operator == 'x':
            return num1 * num2
        return 0
        
    def create_corrupted_demonstrations(self, query: Dict, n_shot: int = 4, 
                                      corruption_position: int = None, corruption_operator: str = None):
        demonstrations = self.task.select_demonstrations(query, n_shot)
        
        if corruption_position is not None and corruption_operator is not None:
            if 0 <= corruption_position < len(demonstrations):
                demo = demonstrations[corruption_position]
                if 'image' in demo and len(demo['image']) > 0:
                    num1, num2 = self.extract_numbers_from_path(demo['image'][0])
                    if num1 is not None and num2 is not None:
                        corrupted_answer = self.compute_answer(num1, num2, corruption_operator)
                        demo['answer'] = corrupted_answer
        
        return demonstrations
    
    def evaluate_single_query(self, query: Dict, n_shot: int = 4, 
                            corruption_position: int = None, corruption_operator: str = None):
        try:
            if corruption_position is not None and corruption_operator is not None:
                demonstrations = self.create_corrupted_demonstrations(query, n_shot, corruption_position, corruption_operator)
            else:
                demonstrations = self.task.select_demonstrations(query, n_shot)
            
            prompt = build_vl_icl_prompt(
                self.task, demonstrations, query, 
                mode="constrained",
                max_images=8
            )
            
            images = collect_images_for_prompt(
                self.task, demonstrations, query,
                max_images=8
            )
            
            response = self.model.generate_text(
                images=images,
                prompt=prompt,
                max_new_tokens=128,
                debug=False
            )
            
            is_correct = self.task.evaluate_response(query, response, mode="constrained")
            
            return is_correct
                
        except Exception as e:
            print(f"Error in evaluation: {e}")
            return False
    
    def run_intervention_experiment(self, num_samples: int = 60, n_shot: int = 4, num_rollouts: int = 5):
        if self.model is None:
            self.load_model()
        
        query_samples = self.task.query_data[:num_samples]
        all_operators = ['+', '-', 'x']
        intervention_positions = list(range(n_shot))
        
        print(f"Running intervention experiment:")
        print(f"  Samples: {num_samples}")
        print(f"  Shots: {n_shot}")
        print(f"  Rollouts: {num_rollouts}")
        print(f"  Positions: {intervention_positions}")
        
        # Baseline: No corruption
        print("Evaluating baseline...")
        baseline_results = {}
        for operator in all_operators:
            operator_queries = [q for q in query_samples if q['operator'] == operator]
            correct_count = 0
            for query in tqdm(operator_queries, desc=f"Baseline-{operator}"):
                if self.evaluate_single_query(query, n_shot):
                    correct_count += 1
            baseline_results[operator] = correct_count / len(operator_queries)
            print(f"  {operator}: {baseline_results[operator]:.3f}")
        
        # Intervention experiments
        results = {}
        
        for rollout in range(num_rollouts):
            print(f"\nRollout {rollout + 1}/{num_rollouts}")
            
            for operator in all_operators:
                operator_queries = [q for q in query_samples if q['operator'] == operator]
                other_operators = [op for op in all_operators if op != operator]
                
                for pos in intervention_positions:
                    corruption_op = random.choice(other_operators)
                    
                    correct_count = 0
                    for query in tqdm(operator_queries, desc=f"R{rollout+1}-{operator}-pos{pos}"):
                        if self.evaluate_single_query(query, n_shot, pos, corruption_op):
                            correct_count += 1
                    
                    accuracy = correct_count / len(operator_queries)
                    key = f"{operator}_pos_{pos}"
                    
                    if key not in results:
                        results[key] = []
                    results[key].append(accuracy)  # Append each rollout result separately
        
        return baseline_results, results
    
    def aggregate_results_across_operators(self, baseline_results: Dict, intervention_results: Dict, n_shot: int, num_rollouts: int):
        all_operators = ['+', '-', 'x']
        positions = list(range(n_shot))
        
        # Calculate overall baseline
        overall_baseline = np.mean(list(baseline_results.values()))
        
        # Properly aggregate: first average across operators for each rollout, then collect rollout values
        position_rollout_averages = {}
        for pos in positions:
            rollout_averages = []
            
            # For each rollout, average across all operators
            for rollout in range(num_rollouts):
                rollout_operator_accs = []
                for operator in all_operators:
                    key = f"{operator}_pos_{pos}"
                    if key in intervention_results:
                        # Get the specific rollout value (each key should have 1 value per rollout)
                        if rollout < len(intervention_results[key]):
                            rollout_operator_accs.append(intervention_results[key][rollout])
                
                if rollout_operator_accs:
                    rollout_avg = np.mean(rollout_operator_accs)
                    rollout_averages.append(rollout_avg)
            
            position_rollout_averages[pos] = rollout_averages
        
        return overall_baseline, position_rollout_averages
    
    def plot_results(self, baseline_results: Dict, intervention_results: Dict, n_shot: int, num_rollouts: int = 5):
        overall_baseline, position_rollout_averages = self.aggregate_results_across_operators(
            baseline_results, intervention_results, n_shot, num_rollouts
        )
        
        positions = list(range(n_shot))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        position_means = []
        position_stds = []
        
        for pos in positions:
            if pos in position_rollout_averages and position_rollout_averages[pos]:
                rollout_values = position_rollout_averages[pos]
                mean_acc = np.mean(rollout_values)
                std_acc = np.std(rollout_values) if len(rollout_values) > 1 else 0
                position_means.append(mean_acc)
                position_stds.append(std_acc)
                print(f"Position {pos}: {len(rollout_values)} rollouts, mean={mean_acc:.3f}, std={std_acc:.3f}")
            else:
                position_means.append(0)
                position_stds.append(0)
        
        # Plot intervention results with confidence intervals
        ax.errorbar(positions, position_means, yerr=position_stds, 
                   marker='o', linewidth=4, markersize=12, capsize=8, capthick=3,
                   label='Intervention', color='steelblue')
        
        # Plot baseline
        ax.axhline(y=overall_baseline, linestyle='--', linewidth=4, alpha=0.8, 
                  color='red', label='Baseline')
        
        # Formatting
        ax.set_xlabel('Demonstration Position', fontsize=22, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=22, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.legend(fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(positions)
        
        # Set y-axis to 0-1 range
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        model_short = self.model_name.split('/')[-1]
        filename = f"figs/causal_positional_bias_{model_short}_{n_shot}shot_cot.pdf"
        os.makedirs("figs", exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Figure saved to {filename}")
    
    def analyze_results(self, baseline_results: Dict, intervention_results: Dict, n_shot: int, num_rollouts: int = 5):
        print(f"\n{'='*50}")
        print("RESULTS ANALYSIS")
        print(f"{'='*50}")
        
        all_operators = ['+', '-', 'x']
        positions = list(range(n_shot))
        
        # Per-operator analysis
        for operator in all_operators:
            print(f"\nOperator: {operator}")
            baseline = baseline_results[operator]
            print(f"  Baseline accuracy: {baseline:.3f}")
            
            degradations = []
            for pos in positions:
                key = f"{operator}_pos_{pos}"
                if key in intervention_results:
                    mean_acc = np.mean(intervention_results[key])
                    degradation = baseline - mean_acc
                    degradations.append(degradation)
                    print(f"  Position {pos}: {mean_acc:.3f} (degradation: {degradation:.3f})")
            
            if len(degradations) >= 2:
                early_deg = np.mean(degradations[:2])
                late_deg = np.mean(degradations[-2:])
                print(f"  Early vs Late: {early_deg:.3f} vs {late_deg:.3f} (diff: {late_deg - early_deg:.3f})")
        
        # Overall analysis
        print(f"\nOVERALL (Aggregated across operators):")
        overall_baseline, position_rollout_averages = self.aggregate_results_across_operators(
            baseline_results, intervention_results, n_shot, num_rollouts
        )
        print(f"  Overall baseline: {overall_baseline:.3f}")
        
        overall_degradations = []
        for pos in positions:
            if pos in position_rollout_averages and position_rollout_averages[pos]:
                mean_acc = np.mean(position_rollout_averages[pos])
                degradation = overall_baseline - mean_acc
                overall_degradations.append(degradation)
                print(f"  Position {pos}: {mean_acc:.3f} (degradation: {degradation:.3f})")
        
        if len(overall_degradations) >= 2:
            early_deg = np.mean(overall_degradations[:2])
            late_deg = np.mean(overall_degradations[-2:])
            print(f"  Early vs Late: {early_deg:.3f} vs {late_deg:.3f} (diff: {late_deg - early_deg:.3f})")
            
            if late_deg > early_deg:
                print(f"  → Recency bias detected (later positions cause more damage)")
            else:
                print(f"  → Primacy bias or uniform influence")

def main():
    model_name = "OpenGVLab/InternVL3-38B-Instruct"
    data_dir = "./VL-ICL"
    num_rollouts = 5
    
    experiment = PositionalInterventionExperiment(model_name, data_dir)
    
    baseline_results, intervention_results = experiment.run_intervention_experiment(
        num_samples=60, n_shot=4, num_rollouts=num_rollouts
    )
    
    experiment.analyze_results(baseline_results, intervention_results, n_shot=4, num_rollouts=num_rollouts)
    experiment.plot_results(baseline_results, intervention_results, n_shot=4, num_rollouts=num_rollouts)
    
    return baseline_results, intervention_results

if __name__ == "__main__":
    results = main()