from .cnn_model import CNNModel
from src.pretrained.MedViT import MedViT
import torch

def model_factory(model_name, models_config):
    model_info = models_config['models'].get(model_name)
    if model_name == "MedViT":
        model = MedViT.MedViT_small
        model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=14, bias=True)
    else:
        model_class = globals()[model_info['model_class']]  # Dynamically load class
        model = model_class(num_classes=model_info['params']['num_classes'])
    


    return model
