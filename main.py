import argparse
import torch
from src.dataloader.celeba_loader import CelebADataset
from src.models.cnn_model import CNNModel
from src.training.trainer import Trainer
from src.evaluations.evaluator import Evaluator
from src.utils.config_reader import ConfigReader
from src.dataloader.dataloader_factory import dataloader_factory
from src.utils.filename_manager import FilenameManager


def main(model_name, dataset_name):
    # Load configuration files
    configs = ConfigReader().load_all_configs()
    datasets_config = configs['datasets']
    models_config = configs['models']
    config = configs['project']

    file_manger = FilenameManager(model_name=model_name, dataset_name=dataset_name, task_name='train_bnn')

    # Device configuration (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Initializing DataLoader for {dataset_name}...")
    # Initialize DataLoader for the specified dataset
    dataset_info = datasets_config['datasets'].get(dataset_name)
    if dataset_info is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration.")
    
    # Create the DataLoaders for train, validation, and test splits
    train_loader = dataloader_factory(dataset_name, 'train', dataset_info)
    val_loader = dataloader_factory(dataset_name, 'val', dataset_info)
    male_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group='male')
    female_test_loader = dataloader_factory(dataset_name, 'test', dataset_info, group='female')

    # Get model info
    model_info = models_config['models'].get(model_name)
    if model_info is None:
        raise ValueError(f"Model '{model_name}' not found in configuration.")

    # Initialize the corresponding model dynamically
    print(f"Initializing {model_name} model...")
    model_class = globals()[model_info['model_class']]  # Dynamically load class
    model = model_class(output_class=model_info['params']['num_classes'])
    model = model.to(device)

    # Initialize Trainer
    print(f"Training {model_name} on {dataset_name}...")
    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        config=config
    )

    # Train the model
    trainer.train(val_loader)
    

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a specified dataset.")
    parser.add_argument('--model', default='utkface_cnn',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--dataset', default='UTKFace', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')

    args = parser.parse_args()
    main(args.model, args.dataset)


# Git Token
# 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