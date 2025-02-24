import argparse
import torch
# from src.dataloader.celeba_loader import CelebADataset
# from src.models.cnn_model import CNNModel
# from src.models.transformer_model import ViT, ViT2
# from src.pretrained.MedViT import MedViT
# from src.models.medical_transformer import MedViT
# from src.training.trainer import Trainer
# from src.training.bnn_trainer import Trainer
from src.evaluations.evaluator import Evaluator
from src.utils.config_reader import ConfigReader
from src.dataloader.dataloader_factory import dataloader_factory
from src.utils.filename_manager import FilenameManager
from src.evaluations.monte_carlo import MonteCarloPrediction
from src.models.model_factory import model_factory
from src.training.bnn_trainer import Trainer
from src.training.traditional_trainer import TraditionalTrainer

def main(args):
    dataset_name = args.dataset
    model_name = args.model
    task_name = args.task
    task_config_name = args.task_config

    print(args)
    
    file_manger = FilenameManager(model_name=model_name, dataset_name=dataset_name, task_name=task_name)


    # Load configuration files
    configs = ConfigReader().load_all_configs()
    datasets_config = configs['datasets']
    models_config = configs['models']
    config = configs['project']

    if args.task == 'check_medvit_train':
        print(f'task: {args.task}')
        from src.pretrained.MedViT import check_medvit
        check_medvit.train()

    elif args.task == 'check_medvit_test':
        model_pth = config[task_name]['model_pth']
        print(f'task: {args.task}')
        from src.pretrained.MedViT import check_medvit
        print('Male Testing')
        check_medvit.test(model_pth=model_pth, sensitive_group='male')
        print('Female Testing')
        check_medvit.test(model_pth=model_pth, sensitive_group='female')
    
    elif args.task == 'check_bnn_medvit_train':
        print(f'task: {args.task}')
        from src.pretrained.MedViT import BNN_MedViT_Training
        BNN_MedViT_Training.train()

    elif args.task == 'check_bnn_medvit_test':
        model_pth = config[task_name]['model_pth']
        print(f'task: {args.task}')
        from src.pretrained.MedViT import BNN_MedViT_Training
        print("Age Testing")
        print('Below 65 Testing')
        BNN_MedViT_Training.test(model_pth=model_pth, sensitive_group='male')
        print('Above 65 Testing')
        BNN_MedViT_Training.test(model_pth=model_pth, sensitive_group='female')

    elif task_name == 'fine_tune_biggan':
            from src.training.finetune_biggan import finetune_biggan
            finetune_biggan(train_loader)

    else:


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
        print("Model Config", models_config)
        model = model_factory(model_name=model_name, models_config=models_config)
        print(model)

        N_MonteCarloSimulation = config['N_MonteCarloSimulation']
        
        if task_name == 'train_bnn':

            # Initialize Trainer
            print(f"Training {model_name} on {dataset_name}...")
            trainer = TraditionalTrainer(
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
            male_monte_carlo = MonteCarloPrediction(model=model, dataloader=male_test_loader, N=N_MonteCarloSimulation)
            male_monte_carlo.asfsdgd()

            print('Female test')
            female_monte_carlo = MonteCarloPrediction(model=model, dataloader=female_test_loader, N=N_MonteCarloSimulation)
            female_monte_carlo.asfsdgd()
            
        
    
'''
dataset:
UTKFace
CelebA
CheXpert
NIHChestXray
ChestMNIST

model
CNNModel
task:
train_bnn
'''


if __name__ == "__main__":
    import os
    print(os.getcwd())
    # Argument parsing
    
    parser = argparse.ArgumentParser(description="Train and evaluate a model on a specified dataset.")
    parser.add_argument('--model', default='nihccchest_transformer',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--dataset', default='NIHChestXray', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')
    # parser.add_argument('--dataset', default='CheXpert', type=str, required=False, help='Name of the dataset to use (e.g., dataset1, dataset2)')
    # parser.add_argument('--task', default='train_bnn',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    # parser.add_argument('--task', default='check_medvit_test',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--task', default='fine_tune_biggan',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')
    parser.add_argument('--task_config', default='medvit_test',  type=str, required=False, help='Name of the model to train (e.g., cnn, resnet)')

    args = parser.parse_args()
    print(args)

    main(args)
    from src.pretrained.BigGAN_PyTorch import BigGANdeep
    # from BigGAN



# Git Token
# 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

# git clone https://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@github.com/rrahman3/EUFairGAN.git