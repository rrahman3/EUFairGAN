import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision import transforms
from src.dataloader.custom_dataset import CustomDataset
import numpy as np
    
class CelebADataset(CustomDataset):
    def __init__(self, image_dir, metadata_file, image_dim=(128, 128), frac=None):

        self.image_dir = image_dir
        self.metadata_file = metadata_file
        self.metadata = self._process_csv()
        if frac is not None:
            self.metadata = self.metadata.sample(frac=frac, random_state=42).reset_index(drop=True)
        self.image_dim = image_dim
        print(len(self.metadata))
        self.model_input_image_dim = (128, 128)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.FullPath[idx]
        image = Image.open(img_name).convert('RGB')
        lr_image = self._process_raw_image(image, self.image_dim)

        gender = np.array([self.metadata.Male[idx]]).astype(np.float32)
        y_label = np.eye(2)[self.metadata.Smiling[idx]].reshape(-1).astype(np.float32)
        
        sample = {
                    'lr_image': lr_image,
                    'gender': gender, 
                    'y_label': y_label, 
                }
        
        return sample['lr_image'], sample['gender'], sample['y_label']
    
    def _process_raw_image(self, img, image_dim):
        img = img.resize(image_dim)
        if image_dim != self.model_input_image_dim:
            img = img.resize(self.model_input_image_dim)
        temp = np.array(img.copy())
        temp = temp/255.0
        temp = temp.transpose((2, 0, 1))
        temp = torch.tensor(np.array(temp)).float()
        return temp

    def _process_csv(self):
        metadata = pd.read_csv(self.metadata_file)
        metadata = metadata.replace(-1, 0)
        metadata['FullPath'] = os.path.join(self.image_dir) + metadata.image_file
        return metadata
    
    def filter_by_gender(self, gender_type):        
        if gender_type == 'male':
            gender_value = 1
        elif gender_type == 'female':
            gender_value = 0
        else:
            raise Exception(f"Invalid gender type: {gender_type}") 
               
        gender_indices = self.metadata[self.metadata.Male == gender_value].index.tolist()
        return torch.utils.data.Subset(self, gender_indices)
    
    def filter_dataset(self, column_name, column_value): #column_name == Male, column value 0 for female, 1, for male       
        if column_name not in self.metadata.columns:
            raise Exception(f"Invalid column name: {column_name}") 

        elif column_value not in self.metadata[column_name].unique():
            raise Exception(f"Invalid column value: {column_name}") 
                
        else:               
            indices = self.metadata[self.metadata[column_name] == column_value].index.tolist()
            return torch.utils.data.Subset(self, indices)

class CelebAESRGANDataset(CelebADataset):
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
        
    
class CelebAHRGeneratorDataset(CelebADataset):
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

class CelebADataLoader():
    def __init__(self, dataset_name, model=None, lr_image_dim=(64, 64), hr_image_dim=(128, 128), frac=None, batch_size=32, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.model = model
        self.lr_image_dim = lr_image_dim
        self.hr_image_dim = hr_image_dim
        self.frac = frac
        self._initialize()
    
    def _initialize(self):

        if self.dataset_name == 'CelebADataset':
            self.train_dataset = CelebADataset(
                    metadata_file=celebAConfig.celeba_train_filenames, 
                    image_dim=self.lr_image_dim, 
                    frac = self.frac
                )
            self.val_dataset = CelebADataset(
                    metadata_file=celebAConfig.celeba_val_filenames, 
                    image_dim=self.lr_image_dim, 
                    frac = 1.0
                )
            self.test_dataset = CelebADataset(
                    metadata_file=celebAConfig.celeba_test_filenames, 
                    image_dim=self.lr_image_dim, 
                    frac = 1.0
                )

        elif self.dataset_name == 'CelebAHRGeneratorDataset':
            if self.model == None:
                print("Must pass the ESRGAN model as the parameter")
                return
            self.train_dataset = CelebAHRGeneratorDataset(
                    metadata_file=celebAConfig.celeba_train_filenames, 
                    image_dim=self.lr_image_dim, 
                    model=self.model,
                    frac = self.frac
                )
            self.val_dataset = CelebAHRGeneratorDataset(
                    metadata_file=celebAConfig.celeba_val_filenames, 
                    image_dim=self.lr_image_dim, 
                    model=self.model,
                    frac = 1.0
                )
            self.test_dataset = CelebAHRGeneratorDataset(
                    metadata_file=celebAConfig.celeba_test_filenames, 
                    image_dim=self.lr_image_dim, 
                    model=self.model,
                    frac = 1.0
                )

        elif self.dataset_name == 'CelebAESRGANDataset':

            self.train_dataset = CelebAESRGANDataset(
                    metadata_file=celebAConfig.celeba_train_filenames, 
                    lr_image_dim=self.lr_image_dim, 
                    hr_image_dim=self.hr_image_dim, 
                    frac = self.frac
                )
            self.val_dataset = CelebAESRGANDataset(
                    metadata_file=celebAConfig.celeba_val_filenames, 
                    lr_image_dim=self.lr_image_dim, 
                    hr_image_dim=self.hr_image_dim, 
                    frac = 1.0
                )
            self.test_dataset = CelebAESRGANDataset(
                    metadata_file=celebAConfig.celeba_test_filenames, 
                    lr_image_dim=self.lr_image_dim, 
                    hr_image_dim=self.hr_image_dim, 
                    frac = 1.0
                )
        else:
            print('No dataset class selected.')

        self.train_dataloader = DataLoader(self.train_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=self.shuffle, 
                                    num_workers=self.num_workers
                                )

        self.val_dataloader = DataLoader(
                                    self.val_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=self.shuffle, 
                                    num_workers=self.num_workers
                                )

        self.male_val_dataset = self.val_dataset.filter_by_gender('male')
        self.male_val_dataloader = DataLoader(
                                    self.male_val_dataset, 
                                    batch_size=self.batch_size, 
                                    shuffle=self.shuffle, 
                                    num_workers=self.num_workers
                                )

        self.female_val_dataset = self.val_dataset.filter_by_gender('female')
        self.female_val_dataloader = DataLoader(
                                        self.female_val_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=self.shuffle, 
                                        num_workers=self.num_workers
                                )
        
        self.test_dataloader = DataLoader(
                                        self.test_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=self.shuffle, 
                                        num_workers=self.num_workers
                                )

        self.male_test_dataset = self.test_dataset.filter_by_gender('male')
        self.male_test_dataloader = DataLoader(
                                        self.male_test_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=self.shuffle, 
                                        num_workers=self.num_workers
                                )

        self.female_test_dataset = self.test_dataset.filter_by_gender('female')
        self.female_test_dataloader = DataLoader(
                                        self.female_test_dataset, 
                                        batch_size=self.batch_size, 
                                        shuffle=self.shuffle, 
                                        num_workers=self.num_workers
                                )
        
    def __str__(self):
        data_len = f"Train: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}, Test: {len(self.test_dataset)}\n"
        data_len += f"Validation Male: {len(self.male_val_dataset)}, Validation Female: {len(self.female_val_dataset)}\n"
        data_len += f"Test Male: {len(self.male_test_dataset)}, Test Female: {len(self.female_test_dataset)}\n"
        return data_len



if __name__=="__main__":
    print("Hello Nuha & Ruhani")
    # from RealESRGAN import RealESRGAN
    # device = utils.device
    # import celeba_settings as celebAConfig
    # print('Loading ESRGAN model')
    # ESRGAN_model = RealESRGAN(device, scale=int(2))
    # hrgan_model = ESRGAN_model.model
    # loadnet = torch.load(celebAConfig.esrgan_aufairgan_model_pth, map_location=torch.device(device))
    # if 'params' in loadnet:
    #     hrgan_model.load_state_dict(loadnet['params'], strict=True)
    # elif 'params_ema' in loadnet:
    #     hrgan_model.load_state_dict(loadnet['params_ema'], strict=True)
    # else:
    #     hrgan_model.load_state_dict(loadnet, strict=True)
    # hrgan_model.eval()
    # hrgan_model.to(device)
    celeba_loader = CelebADataLoader(dataset_name='CelebADataset', lr_image_dim=(128, 128), frac=1, batch_size=32)
    # celeba_loader = CelebADataLoader(dataset_name='CelebAESRGANDataset', lr_image_dim=(64, 64), hr_image_dim=(128, 128), frac=0.99, batch_size=32)
    # celeba_loader = CelebADataLoader(dataset_name='CelebAHRGeneratorDataset', lr_image_dim=(64, 64), model=hrgan_model, frac=0.99, batch_size=32)
    # for step, (a, b, c) in enumerate(celeba_loader.test_dataloader):
    #     print(step, a.shape, b.shape, c.shape)
    #     break

    print(celeba_loader)
    print(len(celeba_loader.train_dataset.filter_by_gender('male')))
    print(len(celeba_loader.train_dataset.filter_by_gender('female')))
    