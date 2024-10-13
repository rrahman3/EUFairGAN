import torch
import torch.nn as nn
import torch.optim as optim
from src.models.base_model import BaseModel

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CelebA

# Vision Transformer Model from Scratch
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)  # Flatten the patches into sequences
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class ViT2(BaseModel):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=3, embed_dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT2, self).__init__(model_name='ViT2')
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)

        self.flatten = nn.Flatten()
        self.variance = nn.LazyLinear(1)

        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, x, y):
        batch_size = x.size(0)
        
        # Patch + Position Embedding
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # Concatenate CLS token to patch embeddings
        x = x + self.position_embedding  # Add position embeddings
        x = self.dropout(x)
        
        # Transformer Encoder
        x = self.transformer(x)
        
        # Classification using the CLS token output

        cls_output = x[:, 0]  # Take the CLS token output
        cls_output = self.norm(cls_output)

        variance = self.flatten(cls_output)
        variance = self.variance(variance)
        variance = nn.functional.softplus(variance)
        return self.mlp_head(cls_output), variance
    
# Define Vision Transformer model
class ViT(BaseModel):
    def __init__(self, img_size=224, patch_size=16, num_classes=3, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT, self).__init__(model_name="ViT Model")
        
        # Patch embedding
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size ** 2  # 3 channels * patch_size^2

        self.patch_to_embedding = nn.Linear(self.patch_dim, dim)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder blocks
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout), 
            num_layers=depth
        )
        self.flatten = nn.Flatten()
        self.variance = nn.LazyLinear(1)
        # MLP Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, y):
        # Split image into patches
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels, height // self.patch_size, self.patch_size, width // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(batch_size, -1, self.patch_dim)
        
        # Apply patch embeddings and add position embeddings
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.position_embedding[:, :x.size(1)]
        x = self.dropout(x)
        
        # Apply transformer layers
        x = self.transformer(x)
        print(x.shape)
        
        # Classification head
        cls_output = x.mean(dim=1) 
        print('cls_output', cls_output.shape)
        variance = self.variance(cls_output)
        variance = nn.functional.softplus(variance)
        print('variance', variance.shape)
        out = self.mlp_head(cls_output)
        print('out', out.shape, out)
        
        return out, variance
