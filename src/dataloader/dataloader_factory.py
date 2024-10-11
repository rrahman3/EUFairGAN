from torch.utils.data import DataLoader
from src.dataloader.celeba_loader import CelebADataset
from src.dataloader.utkface_loader import UTKFaceDataset

def dataloader_factory(dataset_name, split, config, group=None): #group nust be 'male/female, black/white
    if dataset_name == "UTKFace":
        dataset = UTKFaceDataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            frac=config[split]['frac']
        )
    elif dataset_name == "CelebA":
        dataset = CelebADataset(
            metadata_file=config[split]['metadata_file'],
            image_dir=config[split]['img_dir'],
            frac=config[split]['frac']
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if group is not None:
        dataset = dataset.filter_by_gender(group)

    dataloader = DataLoader(
        dataset,
        batch_size=config[split]['batch_size'],
        shuffle=config[split]['shuffle'],
        num_workers=config[split]['num_workers']
    )
    print(f"Total samples in this dataloader: {len(dataloader.dataset)}")
    return dataloader
