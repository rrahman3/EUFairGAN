from torch.utils.data import DataLoader
from .celeba_loader import CelebADataset
from .utkface_loader import UTKFaceDataset
from .chestmnist_dataset import *

def dataloader_factory(dataset_name, split, config, group=None): #group nust be 'male/female, black/white
    if dataset_name == "UTKFace":
        dataset = UTKFaceDataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            image_dim=eval(config[split]['img_dim']),
            frac=config[split]['frac']
        )
        if split == 'test':
            print(config[split]['groups'])
            print(type(config[split]['groups']))
            groups = config[split]['groups']
            if groups[0] == "male" or groups[0] =="female":
                dataset = dataset.filter_by_gender(groups[group])
            elif groups[0] == "black" or groups[0] =="white":
                dataset = dataset.filter_by_race(groups[group])

    elif dataset_name == "CelebA":
        dataset = CelebADataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            image_dim=eval(config[split]['img_dim']),
            frac=config[split]['frac']
        )
        if split == 'test':
            groups = config[split]['groups']
            if groups[0] == "male" or groups[0] =="female":
                dataset = dataset.filter_by_gender(groups[group])
    
    elif dataset_name == "ChestMNIST":
        dataset = CelebADataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            image_dim=eval(config[split]['img_dim']),
            frac=config[split]['frac']
        )
        if split == 'train':
            dataset = chestmnist_dataset.train_dataset
        if split == 'val':
            dataset = chestmnist_dataset.train_dataset
        if split == 'test':
            dataset = chestmnist_dataset.test_dataset
            # groups = config[split]['groups']
            # if groups[0] == "male" or groups[0] =="female":
            #     dataset = dataset.filter_by_gender(groups[group])
    
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    


    dataloader = DataLoader(
        dataset,
        batch_size=config[split]['batch_size'],
        shuffle=config[split]['shuffle'],
        num_workers=config[split]['num_workers']
    )
    print(f"Total samples in this dataloader: {len(dataloader.dataset)}")
    return dataloader
