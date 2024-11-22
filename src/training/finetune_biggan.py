import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
from src.pretrained.pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, display_in_terminal)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, num_classes=14, embed_dim=128):
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
        self.class_embedding = nn.Embedding(num_classes, embed_dim)

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

from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss

discriminator = Discriminator(image_size=128)
biggan_model = BigGAN.from_pretrained('biggan-deep-128')
biggan_model.embeddings = torch.nn.Linear(in_features=14, out_features=128, bias=True)
biggan_model.config.num_classes = 14

num_epochs = 100
# Initialize optimizers for generator and discriminator
optimizer_G = Adam(biggan_model.parameters(), lr=1e-4)
optimizer_D = Adam(discriminator.parameters(), lr=1e-4)

criterion = BCEWithLogitsLoss()
# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def finetune_biggan(dataloader):
    truncation = 0.6
    batch_size = 32
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            batch_size = images.shape[0]
            real_images = images.to(device)
            labels = labels.to(device)

            # --- Update Discriminator ---
            optimizer_D.zero_grad()

            # Generate fake images
            noise_vector = truncated_noise_sample(truncation=truncation, dim_z=128, batch_size=batch_size)
            noise_vector = torch.from_numpy(noise_vector).float()
            # torch.randn(images.size(0), biggan_model.config.latent_dim).to(device)
            fake_images = biggan_model(noise_vector, labels, truncation)

            # Discriminator predictions
            real_preds = discriminator(real_images, labels.long())
            fake_preds = discriminator(fake_images.detach(), labels.long())

            # Discriminator loss
            real_loss = criterion(real_preds, torch.ones_like(real_preds))  # Real images as 1
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))  # Fake images as 0
            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_D.step()

            # --- Update Generator ---
            optimizer_G.zero_grad()

            # Generate new fake images
            fake_preds = discriminator(fake_images, labels)

            # Generator loss (tries to fool the discriminator)
            g_loss = criterion(fake_preds, torch.ones_like(fake_preds))  # Fake images as 1

            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch {epoch+1}/{num_epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
