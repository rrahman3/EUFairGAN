from torch.utils.data import DataLoader
from .celeba_loader import CelebADataset
from .utkface_loader import UTKFaceDataset
from .chestmnist_dataset import train_loader, train_loader_at_eval, test_loader
from .medical_dataset import NIHChestXrayDataset

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
    
    elif dataset_name == "NIHChestXray":
        isTest = True if split == 'test' else False
        dataset = NIHChestXrayDataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            # image_dim=eval(config[split]['img_dim']),
            frac=config[split]['frac'],
            isTest=isTest,
        )

        if split == 'test':
            groups = config[split]['groups']
            if groups[0] == "male" or groups[0] =="female":
                dataset = dataset.filter_by_gender(groups[group])
            elif groups[0] == 'age':
                age_threshold = int(groups[1])
                if group == 0:
                    dataset = dataset.filter_by_NIH_age(age_threshold=age_threshold, below_threshold=True)
                elif group == 1:
                    dataset = dataset.filter_by_NIH_age(age_threshold=age_threshold, below_threshold=False)
                else:
                    raise ValueError("group is not 0 or 1")

    elif dataset_name == "ChestMNIST":
        if split == 'train':
            return train_loader

        elif split == 'val':
            return train_loader_at_eval

        elif split == 'test':
            return test_loader


    
    
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
