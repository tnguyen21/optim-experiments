import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, H//patch_size, W//patch_size)
        x = self.projection(x)
        # Flatten spatial dimensions: (B, embed_dim, num_patches)
        x = x.flatten(2)
        # Transpose: (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim=192, num_heads=3):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim**-0.5

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn  # Return attention weights for analysis


class MLP(nn.Module):
    """Feed-forward network"""

    def __init__(self, embed_dim=192, mlp_ratio=4.0):
        super(MLP, self).__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""

    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)

    def forward(self, x):
        # Self-attention with residual connection
        attn_input = self.norm1(x)
        attn_output, attn_weights = self.attn(attn_input)
        x = x + attn_output

        # MLP with residual connection
        mlp_input = self.norm2(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x, attn_weights


class ViTTiny(nn.Module):
    """Vision Transformer Tiny for CIFAR-10

    Architecture optimized for Muon optimizer:
    - Patch size: 4x4 (gives 8x8 = 64 patches for 32x32 images)
    - Embed dim: 192
    - Depth: 6 transformer blocks
    - Heads: 3
    - MLP ratio: 4
    - Parameters: ~5.5M (mostly in linear layers, perfect for Muon)
    """

    def __init__(self, img_size=32, patch_size=4, num_classes=10, embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0):
        super(ViTTiny, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights following ViT paper"""
        # Initialize position embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # Add position embeddings
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x, _ = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Extract class token
        x = self.head(cls_token_final)

        return x

    def get_features(self, x):
        """Extract intermediate features for analysis"""
        features = {}
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        features["patch_embed"] = x.clone()

        # Add class token and position embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        features["pos_embed"] = x.clone()

        # Apply transformer blocks and collect intermediate features
        attn_weights = []
        for i, block in enumerate(self.blocks):
            x, attn = block(x)
            features[f"block_{i}"] = x.clone()
            attn_weights.append(attn)

            # Store features at key layers
            if i in [2, 5, 8, 11]:  # Quarter, half, 3/4, and final
                features[f"block_{i}_detailed"] = {
                    "tokens": x.clone(),
                    "cls_token": x[:, 0].clone(),
                    "patch_tokens": x[:, 1:].clone(),
                    "attention_weights": attn,
                }

        # Final features
        x_norm = self.norm(x)
        features["norm"] = x_norm.clone()
        features["cls_token_final"] = x_norm[:, 0].clone()

        # Classification
        output = self.head(x_norm[:, 0])
        features["output"] = output.clone()
        features["attention_weights"] = attn_weights

        return output, features
