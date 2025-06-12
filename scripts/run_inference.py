import argparse
import json
import os
from tqdm import tqdm
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model
from tasks.i2t_tasks import OperatorInductionTask, OpenMiniImageNetTask, CLEVRTask, TextOCRTask
from utils.common import set_random_seed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='OpenGVLab/InternVL3-8B', type=str)
    parser.add_argument('--model_type', default='internvl', choices=['internvl'])
    parser.add_argument('--data_dir', default='./VL-ICL', type=str)
    parser.add_argument('--dataset', default='operator_induction', type=str, 
                       choices=['operator_induction', 'open_mi', 'clevr', 'textocr'])
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", type=int)
    parser.add_argument('--max_new_tokens', default=50, type=int)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--use_vllm', action='store_true')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--debug_samples', default=3, type=int, help='Number of samples to debug')
    return parser.parse_args()

def create_task(dataset_name: str, data_dir: str):
    task_map = {
        'operator_induction': OperatorInductionTask,
        'open_mi': OpenMiniImageNetTask,
        'clevr': CLEVRTask,
        'textocr': TextOCRTask
    }
    return task_map[dataset_name](data_dir)

def run_inference(args):
    set_random_seed(args.seed)
    
    logger.info(f"Loading model: {args.model_name}")
    model = create_model(args.model_type, args.model_name, use_vllm=args.use_vllm)
    
    logger.info(f"Loading task: {args.dataset}")
    task = create_task(args.dataset, args.data_dir)
    
    logger.info(f"Total queries: {len(task.query_data)}")
    logger.info(f"Total support examples: {len(task.support_data)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for n_shot in args.n_shot:
        logger.info(f"Running {args.dataset} with {n_shot} shots...")
        results = []
        debug_count = 0
        
        for query_idx, query in enumerate(tqdm(task.query_data, desc=f"{n_shot}-shot")):
            try:
                demonstrations = task.select_demonstrations(query, n_shot)
                
                images = []
                prompt_parts = [task.get_task_instruction() + "\n"]
                
                # Add demonstration examples
                for demo_idx, demo in enumerate(demonstrations):
                    if 'image' in demo:
                        for img_path in demo['image']:
                            images.append(task.load_image(img_path))
                    prompt_parts.append(task.format_demonstration(demo))
                
                # Add query image
                if 'image' in query:
                    if isinstance(query['image'], list):
                        for img_path in query['image']:
                            images.append(task.load_image(img_path))
                    else:
                        images.append(task.load_image(query['image']))
                
                prompt_parts.append(task.format_query(query))
                full_prompt = "\n".join(prompt_parts)
                
                # Debug logging for first few samples
                debug_this = args.debug and debug_count < args.debug_samples
                if debug_this:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"DEBUG SAMPLE {debug_count + 1} (Query {query_idx + 1})")
                    logger.info(f"{'='*50}")
                    logger.info(f"Number of demonstration examples: {n_shot}")
                    logger.info(f"Number of images total: {len(images)}")
                    logger.info(f"Query: {query}")
                    logger.info(f"Demonstrations: {demonstrations}")
                    logger.info(f"Full prompt:\n{full_prompt}")
                    logger.info(f"{'='*50}")
                    debug_count += 1
                
                response = model.generate_text(
                    images=images,
                    prompt=full_prompt,
                    max_new_tokens=args.max_new_tokens,
                    debug=debug_this
                )
                
                is_correct = task.evaluate_response(query, response)
                
                if debug_this:
                    logger.info(f"Model response: '{response}'")
                    logger.info(f"Expected answer: {query.get('answer', 'N/A')}")
                    logger.info(f"Evaluation result: {is_correct}")
                    logger.info(f"{'='*50}\n")
                
                result = {
                    'query_id': query.get('id', query_idx),
                    'query': query,
                    'demonstrations': demonstrations,
                    'n_shot': n_shot,
                    'prompt': full_prompt,
                    'response': response,
                    'correct': is_correct
                }
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing query {query.get('id', query_idx)}: {e}")
                if args.debug:
                    import traceback
                    logger.error(traceback.format_exc())
                continue
        
        # Save results
        output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.model_type}_{n_shot}shot.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Calculate and report accuracy
        if results:
            accuracy = sum(r['correct'] for r in results) / len(results)
            correct_count = sum(r['correct'] for r in results)
            total_count = len(results)
            logger.info(f"{n_shot}-shot results: {correct_count}/{total_count} correct ({accuracy:.3f} accuracy)")
        else:
            logger.warning(f"No results for {n_shot}-shot")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)