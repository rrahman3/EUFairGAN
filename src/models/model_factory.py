from .cnn_model import CNNModel
# from src.pretrained.MedViT import MedViT
from src.models.cnn_model import CNNModel
from src.models.transformer_model import ViT, ViT2
from src.models.resnet50 import ResNet50

import torch

def model_factory(model_name, models_config):
    # Get model info
    model_info = models_config['models'].get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

    # Initialize the corresponding model dynamically
    print(f"Initializing {model_name} model...")
    # model_class = globals()[model_info['model_class']]  # Dynamically load class
    # model = model_class(num_classes=model_info['params']['num_classes'])
    # model = model.to(device)

    model_info = models_config['models'].get(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(model_info)

    if model_info['model_class'] == "MedViT":
        from src.pretrained.MedViT.MedViT import MedViT_small as tiny
        from src.pretrained.MedViT.MedViT import MedViT_base as base
        from src.pretrained.MedViT.MedViT import MedViT_large as large

        model = base()  
        # for NIHCC Chest Xray, out_features = 20
        model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=model_info['params']['num_classes'], bias=True)
    elif model_info['model_class'] == "BNN_MedViT":
        from src.pretrained.MedViT.BNN_MedViT import BNN_MedViT_small as tiny
        from src.pretrained.MedViT.BNN_MedViT import BNN_MedViT_base as base
        from src.pretrained.MedViT.BNN_MedViT import BNN_MedViT_large as large
        model_pth = 'outputs/bnn_medvit_base_nihcccheset/medvit_mnist__base_wt74.pt'

        model = base()  
        # for NIHCC Chest Xray, out_features = 20
        model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=model_info['params']['num_classes'], bias=True)
        model.load_model(model_pth)

    elif model_info['model_class'] == "ResNet50":
        # model_class = globals()[model_info['model_class']]
        model = ResNet50(num_classes=model_info['params']['num_classes'], pretrained=True)
        # model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=model_info['params']['num_classes'], bias=True)
        # model.load_model(model_pth)

    else:
        model_class = globals()[model_info['model_class']]  # Dynamically load class
        model = model_class(num_classes=model_info['params']['num_classes'])

    model = model.to(device)
    return model
