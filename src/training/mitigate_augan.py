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

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_layer=20):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:vgg_layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
    
    def forward(self, generated, target):
        features_generated = self.vgg(generated)
        features_target = self.vgg(target)
        return nn.functional.mse_loss(features_generated, features_target)

class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        # print(f"y_true {y_true.shape}\n{y_true}")
        # print(f"y_pred {y_pred.shape}\n{y_pred}")

        return F.binary_cross_entropy_with_logits(y_pred, y_true)

def ragan_discriminator_loss(D_real, D_fake):
    D_real_rel = torch.sigmoid(D_real - D_fake.mean(dim=0, keepdim=True))
    D_fake_rel = torch.sigmoid(D_fake - D_real.mean(dim=0, keepdim=True))
    
    loss_real = F.binary_cross_entropy(D_real_rel, torch.ones_like(D_real))
    loss_fake = F.binary_cross_entropy(D_fake_rel, torch.zeros_like(D_fake))

    return loss_real + loss_fake


def discriminator_loss(D_real, D_fake):

    loss_real = F.binary_cross_entropy(D_real, torch.ones_like(D_real))
    loss_fake = F.binary_cross_entropy(D_fake, torch.zeros_like(D_fake))

    return loss_real + loss_fake

def ragan_generator_loss(D_real, D_fake):
    """
    Relativistic Average GAN (RaGAN) Generator Loss.
    """
    D_real_rel = torch.sigmoid(D_real - D_fake.mean(dim=0, keepdim=True))
    D_fake_rel = torch.sigmoid(D_fake - D_real.mean(dim=0, keepdim=True))

    loss_real = F.binary_cross_entropy(D_real_rel, torch.zeros_like(D_real))
    loss_fake = F.binary_cross_entropy(D_fake_rel, torch.ones_like(D_fake))

    return loss_real + loss_fake



class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, generated, target):
        return nn.functional.l1_loss(generated, target)

class ESRGANLoss(nn.Module):
    def __init__(self, vgg_layer=20):
        super(ESRGANLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(vgg_layer)
        self.l1_loss = L1Loss()
    
    def forward(self, generated, target):
        perceptual = self.perceptual_loss(generated, target)
        l1 = self.l1_loss(generated, target)
        return perceptual + 0.006 * l1  # Example weights

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

            # d_loss = d_loss_real + d_loss_fake
            d_loss = ragan_discriminator_loss(d_real, d_fake)
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
            # g_adversarial_loss = d_loss_fn(d_fake, valid_labels)
            g_adversarial_loss = ragan_generator_loss(d_real, d_fake)
            
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
    
        

if __name__=="__main__":
    base_results_folder = celebAConfig.mitigation_model_saved_pth
    loss_saved_file_name = utils.get_directory_to_save_file(folder_name=celebAConfig.celeba_results_folder, 
                                                            file_name=celebAConfig.celeba_esrgan_mitigation_loss_log_filename, type="results")
    print(f"results of each epoch will be saved in the csv files located at {loss_saved_file_name}")
    
    args = utils.parse_args()
    utils.DEBUG = args.debug_print
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # device = 'cpu'
    print('device:', device)
    celeba_loader = CelebADataLoader(
                            dataset_name='CelebAESRGANDataset', 
                            lr_image_dim=(64, 64), 
                            hr_image_dim=(128, 128), 
                            frac=celebAConfig.mitigation_training_data_frac, 
                            batch_size=args.batch_size
                        )

    # Loading ESRGAN pretrained model
    ESRGAN_model = RealESRGAN(device, scale=2)    
    loadnet = torch.load(celebAConfig.esrgan_finetune_model_pth, map_location=torch.device(device))
    if 'params' in loadnet: 
        ESRGAN_model.model.load_state_dict(loadnet['params'], strict=True)
    elif 'params_ema' in loadnet:
        ESRGAN_model.model.load_state_dict(loadnet['params_ema'], strict=True)
    else:
        ESRGAN_model.model.load_state_dict(loadnet, strict=True)
    ESRGAN_model.model.eval()
    ESRGAN_model.model.to(device)

    # saveing the generated image
    output_img_path = utils.get_directory_to_save_file(folder_name = base_results_folder, file_name='before_finetuning_image.jpg', type='images')
    import random
    num_samples = 25
    random_indices = random.sample(range(len(celeba_loader.test_dataloader.dataset)), num_samples)
    utils.generate_and_save_images(ESRGAN_model.model, celeba_loader.test_dataloader, device, random_indices, output_img_path, num_samples=num_samples)

    # Loading pretrained classfication model
    prediction_model = torch.load(celebAConfig.cnn_model_pth, map_location=torch.device(device)).to(device)

    esrgan_discriminator = ESRGANDiscriminator().to(device)
    gender_discriminator = GenderDiscriminator().to(device)

    train_esrgan(prediction_model, ESRGAN_model, esrgan_discriminator, gender_discriminator, num_epochs=args.num_epochs)


