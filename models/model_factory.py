from .internvl_model import InternVLModel

def create_model(model_type: str, model_name: str, **kwargs):
    if model_type == 'internvl':
        return InternVLModel(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")