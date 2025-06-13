import random
import re
import copy
from typing import List, Dict
from .base_task import BaseTask
import logging

logger = logging.getLogger(__name__)

class OperatorInductionTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("operator_induction", data_dir)
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def get_task_instruction(self) -> str:
        # Exact prompt from VL-ICL appendix
        return ("The image contains two digit numbers and a ? representing the mathematical operator. "
                "Induce the mathematical operator (addition, multiplication, minus) according to the "
                "results of the in-context examples and calculate the result."
                "Answer with only the final number.")
    
    def format_demonstration(self, support_item: Dict, include_image_token=True) -> str:
        # VL-ICL format: [Image][Question][Answer]
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}What is the result of the following mathematical expression?\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        if n_shot == 0:
            return []
            
        operator_index = {'+': 0, '-': 1, 'x': 2}
        operator = query['operator']
        operator_idx = operator_index[operator]
        
        selected = random.sample(self.support_data, n_shot)
        demonstrations = []
        
        for support in selected:
            demo = copy.deepcopy(support)
            if isinstance(demo['answer'], list):
                demo['answer'] = demo['answer'][operator_idx]
            demonstrations.append(demo)
        
        return demonstrations
    
    def format_query(self, query: Dict, include_image_token=True) -> str:
        # VL-ICL format: Query: [Image][Question]
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}What is the result of the following mathematical expression?"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip()
        
        # Extract numbers from response
        numbers = re.findall(r'\d+', response)
        if not numbers:
            return False
        
        try:
            predicted = int(numbers[-1])  # Take the last number
        except ValueError:
            return False
        
        # Get ground truth
        ground_truth = query['answer']
        if isinstance(ground_truth, list):
            operator_index = {'+': 0, '-': 1, 'x': 2}
            operator = query['operator']
            operator_idx = operator_index[operator]
            ground_truth = ground_truth[operator_idx]
        
        return predicted == ground_truth

class OpenMiniImageNetTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("open_mi", data_dir)
    
    def get_task_instruction(self) -> str:
        # Following VL-ICL appendix
        return "Induce the concept from the in-context examples. Answer the question with a single word or phase."
    
    def format_demonstration(self, support_item: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}This is a\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        if n_shot == 0:
            return []
            
        query_class = query['answer']
        other_class = random.choice([cls for cls in query['classes'] if cls != query_class])
        order_keys = [query_class, other_class] if random.choice([True, False]) else [other_class, query_class]
        
        demonstrations = []
        for i in range(n_shot):
            for key in order_keys:
                support = {
                    'image': [query['support'][key]['images'][i]], 
                    'answer': key,
                    'question': "This is a"
                }
                demonstrations.append(support)
        return demonstrations
    
    def format_query(self, query: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}This is a"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip().lower()
        answer = query['answer'].lower()
        return answer in response

class CLEVRTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("clevr", data_dir)
    
    def get_task_instruction(self) -> str:
        # Following VL-ICL appendix
        return ("The image contains objects of different shapes, colors, sizes and materials. "
                "The question describes the attribute and its value. You need to find all objects "
                "within the image that satisfy the condition. You should induce what operation to use "
                "according to the results of the in-context examples and then calculate the result.")
    
    def format_demonstration(self, support_item: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}{support_item['question']}\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        if n_shot == 0:
            return []
        return random.sample(self.support_data, n_shot)
    
    def format_query(self, query: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}{query['question']}"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip()
        numbers = re.findall(r'\d+', response)
        if not numbers:
            return False
        
        try:
            predicted = int(numbers[-1])
        except ValueError:
            return False
        
        return predicted == query['answer']

class TextOCRTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("textocr", data_dir)
    
    def get_task_instruction(self) -> str:
        # Following VL-ICL appendix
        return ("An image will be provided where a red box is drawn around the text of interest. "
                "Answer with the text inside the red box. Ensure that the transcription is precise, "
                "reflecting the exact characters, including letters, numbers, symbols.")
    
    def format_demonstration(self, support_item: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}Answer with the text inside the red box.\n{support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        if n_shot == 0:
            return []
        return random.sample(self.support_data, n_shot)
    
    def format_query(self, query: Dict, include_image_token=True) -> str:
        image_part = "<image>" if include_image_token else ""
        return f"{image_part}Answer with the text inside the red box."
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip().lower()
        ground_truth = query['answer'].strip().lower()
        return ground_truth in response