import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
from src.utils.filename_manager import FilenameManager
from src.pretrained.pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, n_classes=14, embed_dim=128):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        # Feature extractor for images
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fully connected layers
        self.fc_real_fake = nn.Linear(512 * (image_size // 16) ** 2, 1)  # Real vs fake
        self.fc_class_embed = nn.Linear(embed_dim, 512 * (image_size // 16) ** 2)

        # Class embedding
        self.class_embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, images, labels):
        # Extract image features
        features = self.conv_layers(images)  # Shape: [batch_size, 512, image_size//16, image_size//16]
        features = features.view(features.size(0), -1)  # Flatten: [batch_size, 512 * (image_size // 16)^2]

        # Embed labels
        # labels = labels.long()  # Ensure labels are integers
        # label_embed = self.class_embedding(labels)  # Shape: [batch_size, embed_dim]
        # label_embed = self.fc_class_embed(label_embed)  # Shape: [batch_size, 512 * (image_size // 16)^2]
        print(features.shape)
            #   , label_embed.shape)
        # Combine features and label embeddings
        combined = features 
        # + label_embed  # Element-wise addition

        # Output real/fake prediction
        output = self.fc_real_fake(combined)  # Shape: [batch_size, 1]
        return output

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

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, generated, target):
        return nn.functional.l1_loss(generated, target)

class BigGANLoss(nn.Module):
    def __init__(self, vgg_layer=20):
        super(BigGANLoss, self).__init__()
        self.perceptual_loss = PerceptualLoss(vgg_layer)
        self.l1_loss = L1Loss()
    
    def forward(self, generated, target):
        perceptual = self.perceptual_loss(generated, target)
        l1 = self.l1_loss(generated, target)
        return perceptual + 0.006 * l1  # Example weights
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

# discriminator = Discriminator(image_size=128)

# from src.pretrained.BigGAN_PyTorch.BigGANdeep import Discriminator
discriminator = Discriminator(n_classes=14)
discriminator = discriminator.to(device)

from src.pretrained.BigGAN_PyTorch.BigGANdeep import Generator
# biggan_model = Generator(n_classes=14)
biggan_model = BigGAN.from_pretrained('biggan-deep-128')
biggan_model.config.num_classes = 14
biggan_model.embeddings = torch.nn.Linear(in_features=14, out_features=128, bias=True)
load_path = 'outputs/fine_tune_biggan_nihccchest_transformer_NIHChestXray_20250201_121241/models/model_weights_epoch77_lr0.0001_20250201_121241.pth'
biggan_model.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))
print(f"Model loaded from {load_path}")
biggan_model = biggan_model.to(device)
# for param in biggan_model.parameters():
    # print(param.device)
# print("BigGAN model is on device:", next(biggan_model.parameters()).device)
# biggan_model.embeddings = torch.nn.Linear(in_features=14, out_features=128, bias=True)
# biggan_model.config.num_classes = 14

num_epochs = 1000
# Initialize optimizers for generator and discriminator
generator_learning_rate = 1e-4
discriminator_learning_rate = 1e-4

optimizer_G = Adam(biggan_model.parameters(), lr=generator_learning_rate)
optimizer_D = Adam(discriminator.parameters(), lr=discriminator_learning_rate)

criterion = BCEWithLogitsLoss()
# Training loop

from src.pretrained.BigGAN_PyTorch.utils import prepare_z_y

def finetune_biggan(dataloader):
    truncation = 0.6
    batch_size = 32
    kkk = 0
    for epoch in range(num_epochs):
        batch_num = 1
        for images, labels in dataloader:
            batch_size = images.shape[0]
            real_images = images.to(device)
            labels = labels.to(device)
            # labels = labels.unsqueeze(-1)
            print(f'Image Shape: {images.shape}, Label Shape: {labels.shape}')
            # z_, y_ = prepare_z_y(batch_size, biggan_model.dim_z, nclasses=14,
            #                  device=device)
            # print(f'z_ Shape: {z_.shape}, y_ Shape: {z_.shape}')
            # --- Update Discriminator ---
            optimizer_D.zero_grad()

            # Generate fake images
            for i in range(2):
                noise_vector = truncated_noise_sample(truncation=truncation, dim_z=128, batch_size=batch_size)
                noise_vector = torch.from_numpy(noise_vector).float().to(device) 
                print(f'Noise vector {noise_vector.shape}')

                # torch.randn(images.size(0), biggan_model.config.latent_dim).to(device)
                fake_images = biggan_model(noise_vector, labels.float(), truncation=truncation)
                # fake_images = biggan_model(noise_vector, labels.unsqueeze(-1).float(), truncation=truncation)
                # fake_images = biggan_model(z_, y_)

                # Discriminator predictions
                #
                #
                #
            
                real_preds = discriminator(real_images, labels.unsqueeze(-1).long())
                fake_preds = discriminator(fake_images.detach(), labels.unsqueeze(-1).long())


                # real_preds = discriminator(real_images)
                # fake_preds = discriminator(fake_images.detach())

                # Discriminator loss
                real_loss = criterion(real_preds, torch.ones_like(real_preds).to(device))  # Use ones_like on the same device Real images as 1
                fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds).to(device))  # Use zeros_like on the same device, Fake iamge as 0
                d_loss = real_loss + fake_loss

                d_loss.backward()
                optimizer_D.step()

            for i in range(3):
                # --- Update Generator ---
                optimizer_G.zero_grad()

                noise_vector = truncated_noise_sample(truncation=truncation, dim_z=128, batch_size=batch_size)
                noise_vector = torch.from_numpy(noise_vector).float().to(device) 
                print(f'Noise vector {noise_vector.shape}')

                fake_images = biggan_model(noise_vector, labels.float(), truncation=truncation)
                # Generate new fake images
                fake_preds = discriminator(fake_images, labels.unsqueeze(-1).long())

                # Generator loss (tries to fool the discriminator)
                g_loss = criterion(fake_preds, torch.ones_like(fake_preds))  # Fake images as 1

                g_loss.backward()
                optimizer_G.step()

            print(f"Epoch {epoch+1}/{num_epochs}: Batch {batch_num}: , D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
            save_image_grid(fake_images, save_path=f"outputs/images/generated_image_grid_{kkk}.png", grid_size=(8, 4))
            kkk += 1
            if kkk == 100:
                kkk = 0
        model_saved_path = FilenameManager().generate_model_filename(epoch=epoch, learning_rate=generator_learning_rate, extension='pth')            
        torch.save(biggan_model.state_dict(), model_saved_path)
        print(f"Model saved to {model_saved_path}")
        image_saved_path = FilenameManager().get_filename("images")
        save_image_grid(fake_images, save_path=f"{image_saved_path}epoch_{epoch}.png", grid_size=(8, 4))
        print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")


from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
def save_image_grid(fake_images, save_path="generated_image_grid.png", grid_size=(8, 4)):
    fake_images = fake_images.detach().cpu()  # Make sure the tensor is on CPU

    grid = vutils.make_grid(fake_images, nrow=grid_size[0], padding=2, normalize=True)
    
    grid_image = transforms.ToPILImage()(grid)
    
    grid_image.save(save_path)