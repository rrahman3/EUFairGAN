import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
import numpy as np
    
class UTKFaceDataset(Dataset):
    def __init__(self, image_dir, metadata_file, image_dim=(128, 128), frac=None, task=None):
        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.metadata = self._process_csv()
        if frac is not None:
            self.metadata = self.metadata.sample(frac=frac, random_state=42).reset_index(drop=True)
        self.image_dim = image_dim
        # print(len(self.metadata))
        self.model_input_image_dim = (128, 128)
        self.task = task

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get image filename
        try:
            img_name = os.path.join(self.image_dir, self.metadata.filename[idx])
            image = Image.open(img_name).convert('RGB')

            lr_image = self._process_raw_image(image, self.image_dim)
            hr_image = self._process_raw_image(image, (128, 128))

            gender = np.array([self.metadata.Gender[idx]]).astype(np.float32)
            race =  np.array([self.metadata.Race[idx]]).astype(np.float32)
            age_regression = np.array([self.metadata.Age[idx]]).astype(np.float32)
            age_classification = np.eye(3)[self._convert_age_for_classification(self.metadata.Age[idx])].reshape(-1).astype(np.float32)
            
            sample = {
                        # 'image': image, 
                        'lr_image': lr_image,
                        'hr_image': hr_image,
                        'gender': gender, 
                        'race': race, 
                        'age_classification': age_classification, 
                        'age_regression': age_regression
                    }
            
            # return sample['lr_image'], sample['hr_image'], sample['gender'], sample['age_classification']
            if self.task == 2:
                return sample['lr_image'], sample['age_classification']
            return sample['lr_image'], sample['gender'], sample['age_classification']
        
        except KeyError as e:
            print(idx)
            print(self.metadata[idx])
            print(f"KeyError for index {idx}: {e}")
            raise

    def _convert_age_for_classification(self, age):
        # return age
        if age >=0 and age <=19:
            return 0    #Children

        elif age >=20 and age <=59:
            return 1    #Adults
        
        else:
            return 2 
        
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
        metadata = pd.read_csv(self.metadata_file, nrows=None)
        metadata[['Age', 'Gender', 'Race', 'image']] = metadata['filename'].str.split("_", expand=True)
        metadata['FullPath'] = os.path.join(self.image_dir) + metadata['filename']
        metadata['Age'] = metadata['Age'].replace('', np.nan)
        metadata['Gender'] = metadata['Gender'].replace('', np.nan)
        metadata['Race'] = metadata['Race'].replace('', np.nan)
        metadata['image'] = metadata['image'].replace('', np.nan)
        metadata.dropna(subset=['Age', 'Gender', 'Race', 'image'], inplace=True)
        metadata.drop(columns=['image'], inplace=True)
        metadata['Age'] = metadata['Age'].astype(int)
        metadata['Gender'] = metadata['Gender'].astype(int)
        metadata['Race'] = metadata['Race'].astype(int)
        print(len(metadata))
        return metadata
 
    def filter_by_gender(self, gender_type):        
        if gender_type == 'male':
            gender_value = 0
        elif gender_type == 'female':
            gender_value = 1
        else:
            raise Exception(f"Invalid gender type: {gender_type}") 
               
        gender_indices = self.metadata[self.metadata['Gender'] == gender_value].index.tolist()
        return torch.utils.data.Subset(self, gender_indices)

    def fileter_only_black_and_white_race(self):
        racial_indices_white = self.metadata[self.metadata['Race'] == 0].index.tolist()
        racial_indices_black = self.metadata[self.metadata['Race'] == 1].index.tolist()
        # print(racial_indices_white)        
        # print(racial_indices_black)        
        racial_indices = []
        racial_indices.extend(racial_indices_white)
        racial_indices.extend(racial_indices_black)
        print(f"Total black and white images: {len(racial_indices)}")
        return torch.utils.data.Subset(self, racial_indices)

    def filter_by_race(self, racial_type):        
        if racial_type == 'white':
            race_value = 0
        elif racial_type == 'black':
            race_value = 1
        elif racial_type == 'asian':
            race_value = 2
        elif racial_type == 'indian':
            race_value = 3
        elif racial_type == 'others':
            race_value = 4
        else:
            raise Exception(f"Invalid racial type: {racial_type}") 
               
        racial_indices = self.metadata[self.metadata['Race'] == race_value].index.tolist()
        return torch.utils.data.Subset(self, racial_indices)
    
    def filter_by_sensitive_attribute_value(self, sensitive_attribute_name = 'race', senstitive_attribute_value='white'):

        if sensitive_attribute_name == 'gender':
            if senstitive_attribute_value == 'male':
                gender_value = 0
            elif senstitive_attribute_value == 'female':
                gender_value = 1
            else:
                raise Exception(f"Invalid gender type: {senstitive_attribute_value}") 
                
            indices = self.metadata[self.metadata['Gender'] == gender_value].index.tolist()
    
        elif sensitive_attribute_name == 'race':
            if senstitive_attribute_value == 'white':
                race_value = 0
            elif senstitive_attribute_value == 'black':
                race_value = 1
            elif senstitive_attribute_value == 'asian':
                race_value = 2
            elif senstitive_attribute_value == 'indian':
                race_value = 3
            elif senstitive_attribute_value == 'others':
                race_value = 4
            else:
                raise Exception(f"Invalid racial type: {senstitive_attribute_value}") 
                
            indices = self.metadata[self.metadata['Race'] == race_value].index.tolist()

        return torch.utils.data.Subset(self, indices)

class UTKFaceAgePredictionDataset(UTKFaceDataset):
    def __init__(self, image_dir, metadata_file, image_dim, transform=None, num_finetune_images=None):
        super().__init__(image_dir, metadata_file, transform, num_finetune_images)
        self.image_dim = image_dim
        self.model_input_image_dim = (128, 128)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.metadata.filename[idx])
        image = Image.open(img_name).convert('RGB')
        lr_image = self._process_raw_image(image, self.image_dim)

        gender = np.array([self.metadata.Gender[idx]]).astype(np.float32)
        age_classification = np.eye(utkface_dataloader.class_types)[self._convert_age_for_classification(self.metadata.Age[idx])].reshape(-1).astype(np.float32)
        
        sample = {
                    'lr_image': lr_image,
                    'gender': gender, 
                    'age_classification': age_classification, 
                }
        
        return sample['lr_image'], sample['gender'], sample['age_classification']
    
    def _process_raw_image(self, img, image_dim):
        # return 0
        img = img.resize(image_dim)
        if image_dim != self.model_input_image_dim:
            img = img.resize(self.model_input_image_dim)
        temp = np.array(img.copy())
        temp = temp/255.0
        temp = temp.transpose((2, 0, 1))
        temp = torch.tensor(np.array(temp)).float()
        return temp
    
class UTKFaceAgePredictionHRDataset(UTKFaceAgePredictionDataset):
    def __init__(self, image_dir, metadata_file, image_dim, model, transform=None, num_finetune_images=None):
        super().__init__(image_dir, metadata_file, image_dim, transform, num_finetune_images)
        self.model = model

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.metadata.filename[idx])
        image = Image.open(img_name).convert('RGB')
        lr_image = self._process_raw_image(image, self.image_dim)
        # print(f' lr_image: {lr_image.shape}')
        lr_image = torch.FloatTensor(lr_image).to(utils.device).detach()
        hr_image = self.model(lr_image).detach().squeeze()
        # print(f'before hr_image: {hr_image.shape}')
        if self.image_dim == (128, 128):
            hr_image = self._resize_image(hr_image).squeeze()
            # print(f'after hr_image: {hr_image.shape}')

        gender = np.array([self.metadata.Gender[idx]]).astype(np.float32)

        age_labels = np.array(self._convert_age_for_classification(self.metadata.Age[idx]))        
        age_classification = np.eye(utkface_dataloader.class_types)[age_labels.reshape(-1)].reshape(-1).astype(np.float32)
        
        sample = {
                    'hr_image': hr_image,
                    'gender': gender, 
                    'age_classification': age_classification, 
                }
        
        return sample['hr_image'], sample['gender'], sample['age_classification']
    
    def _process_raw_image(self, img, image_dim):
        img = img.resize(image_dim)
        temp = np.array(img.copy())
        temp = temp/255.0
        temp = temp.transpose((2, 0, 1))
        # temp = np.expand_dims(temp, axis=0)
        temp = torch.tensor(np.array(temp)).float().unsqueeze(dim=0)
        return temp
    
    def _resize_image(self, image):
        # print(f"Before CPU: {type(image)}, {image.device}")
        image = image.squeeze(1).cpu().numpy().transpose(1, 2, 0) * 255
        # print(image.shape)
        image = image.astype(np.uint8)
        img = Image.fromarray(image)
        return self._process_raw_image(img, (128, 128))

