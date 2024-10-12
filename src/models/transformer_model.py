import torch
import torch.nn as nn
import torch.optim as optim
from src.models.base_model import BaseModel

# Define Vision Transformer model
class ViT(BaseModel):
    def __init__(self, img_size=224, patch_size=16, output_class=3, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super(ViT, self).__init__()
        
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
            nn.Linear(dim, output_class)
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
        cls_output = x[:, 0]  # Take CLS token output
        print('cls_output', cls_output.shape)
        variance = self.variance(cls_output)
        variance = nn.functional.softplus(variance)
        print('variance', variance.shape)
        out = self.mlp_head(cls_output)
        print('out', out.shape)
        
        return out, variance
