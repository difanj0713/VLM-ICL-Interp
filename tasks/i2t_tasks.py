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
        # Debug: inspect data structure
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def get_task_instruction(self) -> str:
        return ("The image contains two digit numbers and a ? representing the mathematical operator. "
                "Induce the mathematical operator (addition, multiplication, minus) according to the "
                "results of the in-context examples and calculate the result.")
    
    def format_demonstration(self, support_item: Dict) -> str:
        return f"Answer: {support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        operator_index = {'+': 0, '-': 1, 'x': 2}
        operator = query['operator']
        operator_idx = operator_index[operator]
        
        selected = random.sample(self.support_data, n_shot)
        demonstrations = []
        
        for support in selected:
            demo = copy.deepcopy(support)
            # Fix: Handle both list and int cases
            if isinstance(demo['answer'], list):
                demo['answer'] = demo['answer'][operator_idx]
            else:
                # If answer is already an int, assume it's for the current operator
                demo['answer'] = demo['answer']
            demonstrations.append(demo)
        
        return demonstrations
    
    def format_query(self, query: Dict) -> str:
        return "Answer:"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip()
        # Extract first number from response
        match = re.search(r'\d+', response)
        if match:
            predicted = int(match.group())
        else:
            return False
        
        # Handle both list and int cases for ground truth
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
        # Debug: inspect data structure
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def get_task_instruction(self) -> str:
        return "Induce the concept from the in-context examples. Answer the question with a single word or phrase."
    
    def format_demonstration(self, support_item: Dict) -> str:
        return f"This is a {support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
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
    
    def format_query(self, query: Dict) -> str:
        return "This is a"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip().lower()
        return query['answer'].lower() in response

class CLEVRTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("clevr", data_dir)
        # Debug: inspect data structure
        if self.support_data:
            logger.info(f"Sample support data: {self.support_data[0]}")
        if self.query_data:
            logger.info(f"Sample query data: {self.query_data[0]}")
    
    def get_task_instruction(self) -> str:
        return ("The image contains objects of different shapes, colors, sizes and materials. "
                "You should induce what operation to use according to the results of the "
                "in-context examples and then calculate the result.")
    
    def format_demonstration(self, support_item: Dict) -> str:
        return f"{support_item['question']}\nAnswer: {support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        return random.sample(self.support_data, n_shot)
    
    def format_query(self, query: Dict) -> str:
        return f"{query['question']}\nAnswer:"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip()
        match = re.search(r'\d+', response)
        if match:
            predicted = int(match.group())
        else:
            return False
        return predicted == query['answer']

class TextOCRTask(BaseTask):
    def __init__(self, data_dir: str):
        super().__init__("textocr", data_dir)
    
    def get_task_instruction(self) -> str:
        return ("An image will be provided where a red box is drawn around the text of interest. "
                "Answer with the text inside the red box. Ensure that the transcription is precise.")
    
    def format_demonstration(self, support_item: Dict) -> str:
        return f"Answer: {support_item['answer']}"
    
    def select_demonstrations(self, query: Dict, n_shot: int) -> List[Dict]:
        return random.sample(self.support_data, n_shot)
    
    def format_query(self, query: Dict) -> str:
        return "Answer:"
    
    def evaluate_response(self, query: Dict, response: str) -> bool:
        response = response.strip().lower()
        ground_truth = query['answer'].strip().lower()
        return response == ground_truth