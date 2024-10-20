import argparse
import torch
from src.dataloader.celeba_loader import CelebADataset
from src.models.cnn_model import CNNModel
from src.models.transformer_model import ViT, ViT2
from src.pretrained.MedViT import MedViT
# from src.models.medical_transformer import MedViT
from src.training.trainer import Trainer
from src.evaluations.evaluator import Evaluator
from src.utils.config_reader import ConfigReader
from src.dataloader.dataloader_factory import dataloader_factory
from src.utils.filename_manager import FilenameManager
from src.evaluations.monte_carlo import MonteCarloPrediction


def main(args):
    dataset_name = args.dataset
    model_name = args.model
    task_name = args.task
    task_config_name = args.task_config
    file_manger = FilenameManager(model_name=model_name, dataset_name=dataset_name, task_name=task_name)


    # Load configuration files
    configs = ConfigReader().load_all_configs()
    datasets_config = configs['datasets']
    models_config = configs['models']
    config = configs['project']

    # Device configuration (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Initializing DataLoader for {dataset_name}...")
    # Initialize DataLoader for the specified dataset
    dataset_info = datasets_config['datasets'].get(dataset_name)
    print(dataset_info)
    print(dataset_info['train']['img_dim'])
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
    
    # Create the DataLoaders for train, validation, and test splits
    train_loader = dataloader_factory(dataset_name, 'train', dataset_info)
    val_loader = dataloader_factory(dataset_name, 'val', dataset_info)

    male_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group=0)
    female_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group=1)

    # Get model info
    model_info = models_config['models'].get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

    # Initialize the corresponding model dynamically
    print(f"Initializing {model_name} model...")
    model_class = globals()[model_info['model_class']]  # Dynamically load class
    model = model_class(num_classes=model_info['params']['num_classes'])
    model = model.to(device)
    print(model)
    
    if task_name == 'train_bnn':

        # Initialize Trainer
        print(f"Training {model_name} on {dataset_name}...")
        trainer = Trainer(
            model=model,
            dataloader=train_loader,
            config=config
        )

        # Train the model
        trainer.train(val_loader)

    elif task_name == 'test_bnn':
        task_config = config[task_name][task_config_name]
        print(task_config)

        model_saved_location = task_config['bnn_model_location']
        model.load_model(model_saved_location)        

        print('Male test')
        male_monte_carlo = MonteCarloPrediction(model=model, dataloader=male_test_loader)
        male_monte_carlo.asfsdgd()

        print('Female test')
        female_monte_carlo = MonteCarloPrediction(model=model, dataloader=female_test_loader)
        female_monte_carlo.asfsdgd()

    

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a specified dataset.")
    parser.add_argument('--model', default='chestmnist_transformer',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--dataset', default='ChestMNIST', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')
    parser.add_argument('--task', default='quick_check',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--task_config', default='utkface_gender',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')

    args = parser.parse_args()
    if args.task == 'quick_check':
        from src.pretrained.MedViT import check_medvit
    else:
        main(args)



# Git Token
# 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