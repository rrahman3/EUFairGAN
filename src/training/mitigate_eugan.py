# trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from src.evaluations.evaluator import Evaluator
from src.utils.losses import crossentropy_loss
from src.utils.filename_manager import FilenameManager
from src.utils.results_writer import MetricsTracker

class Trainer:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.num_samples = len(self.dataloader.dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Load configuration settings
        self.training_config = config['training']
        self.optimizer_config = self.training_config['optimizer']
        self.learning_rate = self.training_config['learning_rate']
        self.loss_function_config = self.training_config['loss_function']
        self.num_epochs = self.training_config['num_epochs']

        # Initialize optimizer and loss function
        self.optimizer = self._initialize_optimizer()
        self.loss_function = self._initialize_loss_function()

        self.evaluation_metrics = Evaluator()
        self.log_file_path = FilenameManager().get_filename('training_log')
        self.results_writer = MetricsTracker(self.log_file_path)

    def _initialize_optimizer(self):
        if self.optimizer_config == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_config}")

    def _initialize_loss_function(self):
        if self.loss_function_config == "cross_entropy":
            return crossentropy_loss
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function_config}")

    def train(self, val_loader):

        for epoch in range(1, self.num_epochs+1):
            print(f"Epoch [{epoch}/{self.num_epochs}]")

            train_loss, train_metrics = self.train_epoch(epoch=epoch)
            val_loss, val_metrics = self.validate_epoch(val_loader=val_loader)

            model_saved_path = FilenameManager().generate_model_filename(epoch=epoch, learning_rate=self.learning_rate, extension='pth')            
            self.model.save_model(model_saved_path)

            self.results_writer.update(epoch=epoch, batch=None, train_loss=train_loss, val_loss=val_loss, train_metrics=train_metrics, val_metrics=val_metrics)
            self.results_writer.save()



    def train_epoch(self, epoch):
        self.evaluation_metrics.reset_metrics()
        self.model.train()

        running_loss = 0.0

        for batch, (images, genders, y)  in enumerate(self.dataloader):
            images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
            if epoch == 1 and batch == 0:
                print(f'{images.shape}, {genders.shape}, {y.shape}')

            self.model.train()
            self.optimizer.zero_grad()

            # Compute prediction and loss
            pred, var = self.model(images, genders)
            loss = self.loss_function(y, pred, var)
            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.evaluation_metrics.update_metrics(y_true=y, y_pred=pred, y_variance=var)
        
        epoch_loss = running_loss / self.num_samples
        print(f"Epoch [{epoch}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics
        

    def validate_epoch(self, val_loader):
        self.model.eval()
        self.evaluation_metrics.reset_metrics()

        running_loss = 0.0
        with torch.no_grad():
            for batch, (images, genders, y)  in enumerate(val_loader):
                images, genders, y = images.to(self.device), genders.to(self.device), y.to(self.device)
                if batch == 0:
                    print(f'{images.shape}, {genders.shape}, {y.shape}')

                # Compute prediction and loss
                pred, var = self.model(images, genders)
                loss = self.loss_function(y, pred, var)
                running_loss += loss.item()

                self.evaluation_metrics.update_metrics(y_true=y, y_pred=pred, y_variance=var)

            
        epoch_loss = running_loss / len(val_loader)
        print(f"Validation Loss: {epoch_loss:.4f}")
        epoch_metrics = self.evaluation_metrics.compute_epoch_metrics()
        print(f"Validation:\n")
        self.evaluation_metrics.print_metrics()
        self.evaluation_metrics.reset_metrics()

        return epoch_loss, epoch_metrics




from RealESRGAN import RealESRGAN
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import utils
from utils import debug_print
from torch.utils.data import DataLoader, Subset
from torchvision.models import vgg19
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from ufairgan_models import ESRGANDiscriminator, GenderDiscriminator
from celeba_models import AgePredictionModel
import celeba_settings as celebAConfig
from celeba_dataloader import CelebADataLoader
import pandas as pd


def validate_esrgan(generator, cnn_model, dataloader, esrgan_loss_fn,  d_loss_fn):
    au_list = []
    generator.model.eval()
    with torch.no_grad():
        for step, (lr_images, hr_images, gender_labels, ages) in enumerate(dataloader):
            lr_images, hr_images, ages, gender_labels = lr_images.to(device), hr_images.to(device), ages.to(device), gender_labels.to(device)

            # GAN Discriminator, decides fake vs real image
            fake_hr_images = generator.model(lr_images)
            d_fake = esrgan_discriminator(fake_hr_images).to(device)
            valid_labels = torch.ones(hr_images.size(0), 1).to(device)
            g_adversarial_loss = d_loss_fn(d_fake, valid_labels)

            pred_age, au = cnn_model(fake_hr_images, gender_labels)
            
            g_loss = esrgan_loss_fn(fake_hr_images, hr_images) + 0.0001 * g_adversarial_loss

            # GAN Gender Discriminator, decides fake vs real gender from the fake image
            # need to flip the gender labels

            # print(f'step [{step+1}], d_loss: {g_loss.item():.8f}, g_adversarial_loss: {g_adversarial_loss.item():.8f}')
            # print(au.mean().item())
            au_list.extend(au.cpu().numpy())
        
    print(f'Average AU GAP in validation: {np.mean(au_list)}')
    return np.mean(au_list)
            

d_losses = []
g_losses = []
gd_losses = []
steps = []
epoch_steps = []

def train_esrgan(cnn_model, generator, esrgan_discriminator, gender_discriminator, num_epochs=25):

    esrgan_loss_fn = ESRGANLoss().to(device)
    gender_loss_fn = nn.BCEWithLogitsLoss()
    d_loss_fn = AdversarialLoss().to(device)

    optimizer_g = optim.Adam(generator.model.parameters(), lr=celebAConfig.mitigation_lr_esrgan_gen, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(esrgan_discriminator.parameters(), lr=celebAConfig.mitigation_lr_esrgan_disc, betas=(0.5, 0.999))
    optimizer_gd = optim.Adam(gender_discriminator.parameters(), lr=celebAConfig.mitigation_lr_gender_disc, betas=(0.5, 0.999))

    generator.model.train()

    for epoch in range(num_epochs):
        generator.model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        for step, (lr_images, hr_images, gender_labels, ages) in enumerate(celeba_loader.train_dataloader):
            generator.model.train()
            lr_images, hr_images, ages, gender_labels = lr_images.to(device), hr_images.to(device), ages.to(device), gender_labels.to(device)
            debug_print(lr_images.shape)
            debug_print(hr_images.shape)

            # ---- Train ESRGAN Discriminator ----
            optimizer_d.zero_grad()
            # fake_hr_images = generator(lr_images).detach()
            fake_hr_images = generator.model(lr_images)
            debug_print(f"before: {fake_hr_images.shape}")
            real_labels = torch.ones(hr_images.size(0), 1).to(device)
            fake_labels = torch.zeros(hr_images.size(0), 1).to(device)

            d_real = esrgan_discriminator(hr_images).to(device)
            d_fake = esrgan_discriminator(fake_hr_images).to(device)

            d_loss_real = d_loss_fn(d_real, real_labels)
            d_loss_fake = d_loss_fn(d_fake, fake_labels)
            debug_print(f"d_loss_real {d_loss_real}")
            debug_print(f"d_loss_fake {d_loss_fake}")

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()


            with torch.no_grad():
                pred_age, fake_au = cnn_model(fake_hr_images, gender_labels)

            with torch.no_grad():
                pred_age, real_au = cnn_model(hr_images, gender_labels)

            # ---- Train GenderDiscriminator ----
            optimizer_gd.zero_grad()
            fake_gender_output = gender_discriminator(fake_au)
            # print(gender_output, gender_labels)
            real_gender_output = gender_discriminator(real_au)
            debug_print(gender_labels.shape)
            debug_print(fake_gender_output.shape)

            gd_loss_fake = gender_loss_fn(fake_gender_output, gender_labels)
            gd_loss_real = gender_loss_fn(real_gender_output, gender_labels)
            gd_loss = gd_loss_fake + gd_loss_real
            gd_loss.backward()
            optimizer_gd.step()


            # ---- Train ESRGAN Generator ----
            optimizer_g.zero_grad()

            # GAN Discriminator, decides fake vs real image
            fake_hr_images = generator.model(lr_images)
            d_fake = esrgan_discriminator(fake_hr_images).to(device)
            valid_labels = torch.ones(hr_images.size(0), 1).to(device)
            g_adversarial_loss = d_loss_fn(d_fake, valid_labels)
            
            # GAN Gender Discriminator, decides fake vs real gender from the fake image
            # need to flip the gender labels
            _, cnn_fake_au = cnn_model(fake_hr_images, gender_labels)
            d_fake_gender = gender_discriminator(cnn_fake_au)
            debug_print(d_fake_gender.shape)
            toggled_gender_labels = 1 - gender_labels
            debug_print(f'gender labels vs toggled, {gender_labels}, {toggled_gender_labels}')
            debug_print(toggled_gender_labels.shape)
            d_gender_loss = gender_loss_fn(d_fake_gender, toggled_gender_labels)
            esrgan_loss = esrgan_loss_fn(fake_hr_images, hr_images)
            g_loss = esrgan_loss + celebAConfig.mitigation_adv_coefficient * g_adversarial_loss + celebAConfig.mitigation_mu_coefficient * d_gender_loss

            # g_loss = esrgan_loss_fn(fake_hr_images, hr_images) + 0.001 * g_adversarial_loss
            g_loss.backward()
            optimizer_g.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            gd_losses.append(gd_loss.item())
            steps.append(step)
            epoch_steps.append(epoch)

            print(f'step [{step+1}], d_loss: {d_loss.item():.8f}, g_loss: {g_loss.item():.8f}, gd_loss: {gd_loss.item():.8f}')
            print(f"ESRGAN train losses: {esrgan_loss.item()}, {g_adversarial_loss.item()}, {d_gender_loss.item()}")
            # break


        output_img_path = utils.get_directory_to_save_file(folder_name = base_results_folder, file_name=f'image_{epoch:05d}.jpg', type='images')
        utils.generate_and_save_images(generator.model, celeba_loader.test_dataloader, device, random_indices, output_img_path, num_samples=num_samples)

        model_saved_location = utils.get_directory_to_save_file(folder_name=base_results_folder, file_name=f'model_weights_{epoch:05d}.pth', type='models')
        torch.save(generator.model.state_dict(), model_saved_location)

        male_val_au = validate_esrgan(generator, cnn_model, celeba_loader.male_val_dataloader, esrgan_loss_fn,  d_loss_fn)
        female_val_au = validate_esrgan(generator, cnn_model, celeba_loader.female_val_dataloader, esrgan_loss_fn,  d_loss_fn)
        print(f"Epoch [{epoch+1}/{num_epochs}]:: Validation AU: Male: {male_val_au}, Female: {female_val_au},  Valiation AU Gap {male_val_au-female_val_au}")

    
        df = pd.DataFrame({
            'epochs':epoch_steps,
            'steps': steps,
            'g_loss': g_losses,
            'd_loss': d_losses,
            'gd_loss': gd_losses
        })
        df.to_csv(loss_saved_file_name, index=False)
    

