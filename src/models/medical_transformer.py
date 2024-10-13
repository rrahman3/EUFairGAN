import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from src.models.base_model import BaseModel

# Helper function: Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        super().__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_emb = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.patch_emb(x)  # [B, emb_size, n_patches**0.5, n_patches**0.5]
        x = x.flatten(2)  # Flatten height and width
        x = x.transpose(1, 2)  # [B, n_patches, emb_size]
        return x

# Helper function: Multi-Head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_size=768, num_heads=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        assert self.head_dim * num_heads == emb_size, "Embedding size should be divisible by num_heads"

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.fc = nn.Linear(emb_size, emb_size)
        self.scale = self.head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)  # [B, N, emb_size] -> 3 * [B, N, emb_size]
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv)

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        return self.fc(attn_output)

# Helper function: Transformer Encoder Block
class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size=768, num_heads=8, mlp_dim=2048, dropout_rate=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.msa = MultiHeadSelfAttention(emb_size, num_heads)
        self.ln2 = nn.LayerNorm(emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, emb_size),
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x + self.dropout(self.msa(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x

# Vision Transformer (ViT) for Medical Images
class MedViT(BaseModel):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, emb_size=768, depth=12, num_heads=8, mlp_dim=2048, num_classes=1000, dropout_rate=0.1):
        super().__init__(model_name="MedViT")
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + 1, emb_size))
        self.dropout = nn.Dropout(dropout_rate)

        # Transformer Encoder layers
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(emb_size, num_heads, mlp_dim, dropout_rate)
            for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes)
        self.variance = nn.LazyLinear(1)

    def forward(self, x, y):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=B)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for encoder_block in self.encoder:
            x = encoder_block(x)

        x = self.ln(x[:, 0])

        print('x', x.shape, '\n', x)
        y = self.head(x)
        print('y', y.shape, '\n', y)
        variance = self.variance(x)
        variance = nn.functional.softplus(variance)
        print('variance', variance.shape, '\n', variance)
        return y, variance

# # Example Usage
# model = MedViT(img_size=224, patch_size=16, in_channels=3, num_classes=2)  # Binary classification (e.g., disease detection)
# x = torch.randn(8, 3, 224, 224)  # Batch of 8 images (3 channels, 224x224)
# output = model(x)
# print(output.shape)  # Output: [8, 2]
