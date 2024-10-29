import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
from .custom_dataset import CustomDataset
import numpy as np
import torchvision
    
class NIHChestXrayDataset(CustomDataset):
    def __init__(self, image_dir, metadata_file, image_dim=(128, 128), frac=None, isTest=False, is_MNIST_like=True):
        print("-------------------------------------------Loading NIHCC Chest Xray dataset------------------------------------------")

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
        if self.is_MNIST_like:    
            labels_column = ['Atelectasis', 
                         'Cardiomegaly',
                         'Effusion', 
                         'Infiltration', 
                         'Mass',
                         'Nodule', 
                         'Pneumonia', 
                         'Pneumothorax',
                         'Consolidation',
                         'Edema',
                         'Emphysema', 
                         'Fibrosis', 
                         'Pleural Thickening', 
                         'Hernia', 
                    ]
        else:
            labels_column = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                            'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
                            'Nodule', 'Pleural Thickening', 'Pneumonia', 'Pneumothorax',
                            'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema',
                            'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding'
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
        # metadata = metadata.replace(-1, 0)
        metadata['FullPath'] = metadata['id'].apply(lambda x: os.path.join(self.image_dir, str(x)))
        return metadata
    
    def filter_by_NIH_age(self, age_threshold, below_threshold=True):
        if below_threshold:
           age_indices = self.metadata[self.metadata['Patient Age'] < age_threshold].index.tolist()
        else:
            age_indices = self.metadata[self.metadata['Patient Age'] >= age_threshold].index.tolist()
        return torch.utils.data.Subset(self, age_indices)


    def filter_by_gender(self, gender_type):        
        if gender_type == 'male':
            gender_value = 'M'
        elif gender_type == 'female':
            gender_value = 'F'
        else:
            raise Exception(f"Invalid gender type: {gender_type}") 
               
        gender_indices = self.metadata[self.metadata['Patient Gender'] == gender_value].index.tolist()
        return torch.utils.data.Subset(self, gender_indices)
    
    def filter_dataset(self, column_name, column_value): #column_name == Male, column value 0 for female, 1, for male       
        if column_name not in self.metadata.columns:
            raise Exception(f"Invalid column name: {column_name}") 

        elif column_value not in self.metadata[column_name].unique():
            raise Exception(f"Invalid column value: {column_name}") 
                
        else:               
            indices = self.metadata[self.metadata[column_name] == column_value].index.tolist()
            return torch.utils.data.Subset(self, indices)

class NIHChestXrayESRGANDataset(NIHChestXrayDataset):
    def __init__(self, image_dir, metadata_file, lr_image_dim=(64, 64), hr_image_dim=(128, 128), frac=None):
        super().__init__(image_dir, metadata_file, lr_image_dim, frac)
        self.lr_image_dim = lr_image_dim
        self.hr_image_dim = hr_image_dim

    def __getitem__(self, idx):

        img_name = self.metadata.FullPath[idx]
        image = Image.open(img_name).convert('RGB')

        lr_image = self._process_raw_image(image, self.lr_image_dim)
        hr_image = self._process_raw_image(image, self.hr_image_dim)

        gender = np.array([self.metadata.Male[idx]]).astype(np.float32)
        y_label = np.eye(2)[self.metadata.Smiling[idx]].reshape(-1).astype(np.float32)
        
        sample = {
                    'lr_image': lr_image,
                    'hr_image': hr_image,
                    'gender': gender, 
                    'y_label': y_label, 
                }
            
        return sample['lr_image'], sample['hr_image'], sample['gender'], sample['y_label']
        
    
class NIHChestXrayHRGeneratorDataset(NIHChestXrayDataset):
    def __init__(self, image_dir, metadata_file, model, image_dim=(64, 64), input_dim=(128, 128), frac=None):
        super().__init__(image_dir, metadata_file, image_dim, frac)
        self.model = model
        self.input_dim = input_dim

    def __getitem__(self, idx):
        img_name = self.metadata.FullPath[idx]
        image = Image.open(img_name).convert('RGB')
        lr_image = self._process_raw_image(image, self.image_dim).unsqueeze(dim=0)
        lr_image = torch.FloatTensor(lr_image).to(utils.device).detach()
        hr_image = self.model(lr_image).detach().squeeze() #(1, 3, 128, 128) --> squeezed to (2, 128, 128)
        hr_image_dim = (hr_image.shape[-2], hr_image.shape[-1])
        print(hr_image_dim, hr_image.shape, lr_image.shape)

        if hr_image_dim != self.input_dim:
            hr_image = self._resize_image(hr_image).squeeze()

        gender = np.array([self.metadata.Male[idx]]).astype(np.float32)
        y_label = np.eye(2)[self.metadata.Smiling[idx]].reshape(-1).astype(np.float32)        

        sample = {
                    'hr_image': hr_image,
                    'gender': gender, 
                    'y_label': y_label, 
                }
            
        return sample['hr_image'], sample['gender'], sample['y_label']
    
    def _resize_image(self, image):
        # print(f"Before CPU: {type(image)}, {image.device}")

        # output from ESRGAN is (3, H, W), but the Image takes the input of shape (H, W, 3) to resize
        image = image.squeeze(1).cpu().numpy().transpose(1, 2, 0) * 255 
        # print(image.shape)
        image = image.astype(np.uint8)
        img = Image.fromarray(image)
        return self._process_raw_image(img, self.input_dim)

