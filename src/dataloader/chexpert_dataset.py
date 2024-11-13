import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
from .custom_dataset import CustomDataset
import numpy as np
import torchvision

class CheXpertDataset(CustomDataset):
    def __init__(self, image_dir, metadata_file, image_dim=(128, 128), frac=None, isTest=False, is_MNIST_like=True):
        print("-------------------------------------------Loading CheXpert dataset------------------------------------------")

        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.metadata = self._process_csv()
        if frac is not None:
            self.metadata = self.metadata.sample(frac=frac, random_state=42).reset_index(drop=True)
        
        self.image_dim = image_dim
        print(len(self.metadata))
        self.model_input_image_dim = (128, 128)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.AugMix(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.Resize((224, 224)),
            # transforms.Lambda(lambda image: image.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),         # Resize the images to 128x128
            transforms.RandomHorizontalFlip(),     # Apply random horizontal flip
            transforms.ToTensor(),                 # Convert image to PyTorch tensor
            transforms.Normalize((0.5, 0.5, 0.5), # Normalize the images with mean and std
                                (0.5, 0.5, 0.5)) 
        ])
        self.isTest = isTest
        self.is_MNIST_like = is_MNIST_like

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.FullPath[idx]
        # image = Image.open(img_name)
        image = Image.open(img_name).convert('RGB')
        # lr_image = self._process_raw_image(image, self.image_dim)
        lr_image = self.test_transform(image) if self.isTest else self.train_transform(image)

        # gender = np.array([self.metadata.Male[idx]]).astype(np.float32)
        y_label = self.get_labels(idx)
        # print(y_label)
        
        sample = {
                    'lr_image': lr_image,
                    # 'gender': gender, 
                    'y_label': y_label, 
                }
        
        return sample['lr_image'], sample['y_label']
    
    def get_labels(self, idx):
        # 'Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 
        
        labels_column = [
            'No Finding',
            'Enlarged Cardiomediastinum', 
            'Cardiomegaly', 
            'Lung Opacity',
            'Lung Lesion', 
            'Edema', 
            'Consolidation', 
            'Pneumonia', 
            'Atelectasis',
            'Pneumothorax', 
            'Pleural Effusion', 
            'Pleural Other', 
            'Fracture',
            'Support Devices'    
        ]


        labels = self.metadata[labels_column].loc[idx]
        labels_numpy = np.array(labels).astype(np.float32)
        # print(labels_numpy)
        return labels_numpy

    
    def _process_raw_image(self, img, image_dim):
        img = img.resize(image_dim)
        # if image_dim != self.model_input_image_dim:
        #     img = img.resize(self.model_input_image_dim)
        temp = np.array(img.copy())
        temp = temp/255.0
        temp = temp.transpose((2, 0, 1))
        temp = torch.tensor(np.array(temp)).float()
        return temp

    def _process_csv(self):
        metadata = pd.read_csv(self.metadata_file)
        print(metadata.columns)        
        metadata = metadata.replace(-1, 0)
        metadata = metadata.fillna(0)
        metadata['FullPath'] = metadata['Path'].apply(lambda x: os.path.join(self.image_dir, str(x)))
        return metadata

    
    def filter_by_NIH_age(self, age_threshold, below_threshold=True):
        if below_threshold:
           age_indices = self.metadata[self.metadata['Age'] < age_threshold].index.tolist()
        else:
            age_indices = self.metadata[self.metadata['Age'] >= age_threshold].index.tolist()
        return torch.utils.data.Subset(self, age_indices)


    def filter_by_gender(self, gender_type):        
        if gender_type == 'male':
            gender_value = 'Male'
        elif gender_type == 'female':
            gender_value = 'Female'
        else:
            raise Exception(f"Invalid gender type: {gender_type}") 
               
        gender_indices = self.metadata[self.metadata['Sex'] == gender_value].index.tolist()
        return torch.utils.data.Subset(self, gender_indices)
    
    def filter_dataset(self, column_name, column_value): #column_name == Male, column value 0 for female, 1, for male       
        if column_name not in self.metadata.columns:
            raise Exception(f"Invalid column name: {column_name}") 

        elif column_value not in self.metadata[column_name].unique():
            raise Exception(f"Invalid column value: {column_name}") 
                
        else:               
            indices = self.metadata[self.metadata[column_name] == column_value].index.tolist()
            return torch.utils.data.Subset(self, indices)
