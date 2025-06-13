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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='OpenGVLab/InternVL3-8B-Instruct', type=str)
    parser.add_argument('--model_type', default='internvl', choices=['internvl'])
    parser.add_argument('--data_dir', default='./VL-ICL', type=str)
    parser.add_argument('--dataset', default='operator_induction', type=str, 
                       choices=['operator_induction', 'open_mi', 'clevr', 'textocr'])
    parser.add_argument('--n_shot', default=[0, 1, 2, 4, 8], nargs="+", type=int)
    parser.add_argument('--max_new_tokens', default=None, type=int)
    parser.add_argument('--output_dir', default='./results', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_samples', default=3, type=int)
    return parser.parse_args()

def create_task(dataset_name: str, data_dir: str):
    task_map = {
        'operator_induction': OperatorInductionTask,
        'open_mi': OpenMiniImageNetTask,
        'clevr': CLEVRTask,
        'textocr': TextOCRTask
    }
    return task_map[dataset_name](data_dir)

def get_max_tokens_for_task(dataset_name: str, user_specified: int = None):
    if user_specified is not None:
        return user_specified
    
    # Following VL-ICL paper's token limits
    task_tokens = {
        'operator_induction': 15,
        'clevr': 15,
        'textocr': 30,
        'open_mi': 20
    }
    return task_tokens.get(dataset_name, 20)

def build_vl_icl_prompt(task, demonstrations, query, debug=False):
    """Build prompt following VL-ICL standard format"""
    
    # Task Description
    prompt_parts = [task.get_task_instruction()]
    
    # Support Set: [Image][Question][Answer] (n-shot)
    if demonstrations:
        support_parts = []
        for demo in demonstrations:
            demo_text = task.format_demonstration(demo, include_image_token=False)
            support_parts.append(demo_text)
        
        support_set = "\n".join(support_parts)
        prompt_parts.append(f"Support Set:\n{support_set}")
    
    # Query: [Image][Question]
    query_text = task.format_query(query, include_image_token=False)
    prompt_parts.append(f"Query:\n{query_text}")
    
    # Prediction: [Answer]
    prompt_parts.append("Prediction:")
    
    full_prompt = "\n\n".join(prompt_parts)
    
    if debug:
        logger.info(f"Built VL-ICL prompt:\n{full_prompt}")
    
    return full_prompt

def collect_images_for_prompt(task, demonstrations, query, debug=False):
    """Collect all images in the order they appear in the prompt"""
    images = []
    
    # Add demonstration images
    for demo in demonstrations:
        if 'image' in demo:
            for img_path in demo['image']:
                images.append(task.load_image(img_path))
    
    # Add query image(s)
    if 'image' in query:
        if isinstance(query['image'], list):
            for img_path in query['image']:
                images.append(task.load_image(img_path))
        else:
            images.append(task.load_image(query['image']))
    
    if debug:
        logger.info(f"Collected {len(images)} images total")
    
    return images

def run_inference(args):
    set_random_seed(args.seed)
    
    logger.info(f"Loading model: {args.model_name}")
    model = create_model(args.model_type, args.model_name)
    
    logger.info(f"Loading task: {args.dataset}")
    task = create_task(args.dataset, args.data_dir)
    
    max_new_tokens = get_max_tokens_for_task(args.dataset, args.max_new_tokens)
    logger.info(f"Using max_new_tokens: {max_new_tokens}")
    
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
                
                # Build VL-ICL standard prompt
                prompt = build_vl_icl_prompt(
                    task, demonstrations, query, 
                    debug=(args.debug and debug_count < args.debug_samples)
                )
                
                # Collect images in order
                images = collect_images_for_prompt(
                    task, demonstrations, query,
                    debug=(args.debug and debug_count < args.debug_samples)
                )
                
                # Debug logging
                debug_this = args.debug and debug_count < args.debug_samples
                if debug_this:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"DEBUG SAMPLE {debug_count + 1} (Query {query_idx + 1})")
                    logger.info(f"{'='*60}")
                    logger.info(f"N-shot: {n_shot}")
                    logger.info(f"Number of images: {len(images)}")
                    logger.info(f"Query ID: {query.get('id', 'N/A')}")
                    logger.info(f"Expected answer: {query.get('answer', 'N/A')}")
                    if demonstrations:
                        logger.info(f"Sample demonstration: {demonstrations[0]}")
                    logger.info(f"{'='*60}")
                    debug_count += 1
                
                # Generate response
                response = model.generate_text(
                    images=images,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    debug=debug_this
                )
                
                # Evaluate
                is_correct = task.evaluate_response(query, response)
                
                if debug_this:
                    logger.info(f"Final evaluation: {is_correct}")
                    logger.info(f"{'='*60}\n")
                
                result = {
                    'query_id': query.get('id', query_idx),
                    'query': query,
                    'demonstrations': demonstrations,
                    'n_shot': n_shot,
                    'prompt': prompt,
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
        output_file = os.path.join(
            args.output_dir, 
            f"{args.dataset}_{args.model_type.replace('/', '_')}_{n_shot}shot.json"
        )
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Report results
        if results:
            accuracy = sum(r['correct'] for r in results) / len(results)
            correct_count = sum(r['correct'] for r in results)
            total_count = len(results)
            logger.info(f"{n_shot}-shot: {correct_count}/{total_count} = {accuracy:.3f}")
        else:
            logger.warning(f"No results for {n_shot}-shot")

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)