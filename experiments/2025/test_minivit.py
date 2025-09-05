#!/usr/bin/env python3
"""
Test script to verify MiniViT model parameters
"""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import numpy as np

class MiniViT(nn.Module):
    """
    Mini Vision Transformer adapted for the 1989 LeCun dataset
    Designed to have approximately 10,000 parameters while keeping ViT architecture
    """
    def __init__(self, *, image_size=16, patch_size=4, num_classes=10, dim=32, depth=2, heads=2, mlp_dim=32, channels=1):
        super().__init__()
        
        # Calculate patches and embedding dimensions
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size * patch_size
        
        # Patch embedding - convert patches to vectors
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        # Positional embeddings and class token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.ReLU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # Calculate and store parameter count
        self.param_count = sum(p.numel() for p in self.parameters())
        
    def forward(self, img):
        # Poor man's data augmentation by 1 pixel (matching LeCun approach)
        if self.training:
            shift_x, shift_y = np.random.randint(-1, 2, size=2)
            img = torch.roll(img, (shift_x, shift_y), (2, 3))
        
        # Convert to patches and add positional embeddings
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        
        # Add class token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        
        # Apply transformer layers
        for norm1, attn, norm2, mlp in self.layers:
            # Self-attention with residual connection
            normed = norm1(x)
            attn_out, _ = attn(normed, normed, normed)
            x = x + attn_out
            
            # MLP with residual connection
            normed = norm2(x)
            mlp_out = mlp(normed)
            x = x + mlp_out
        
        # Classification using the class token
        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

if __name__ == "__main__":
    # Test the model
    model = MiniViT()
    print(f"MiniViT model parameter count: {model.param_count}")
    
    # Test forward pass with dummy data (16x16 image like LeCun dataset)
    dummy_input = torch.randn(1, 1, 16, 16)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Try different configurations to get close to 10,000 params
    configs = [
        {"dim": 24, "depth": 2, "heads": 2, "mlp_dim": 24},
        {"dim": 28, "depth": 2, "heads": 2, "mlp_dim": 28},
        {"dim": 32, "depth": 2, "heads": 2, "mlp_dim": 32},
        {"dim": 36, "depth": 2, "heads": 2, "mlp_dim": 36},
        {"dim": 40, "depth": 2, "heads": 2, "mlp_dim": 40},
    ]
    
    print("\nTesting different configurations:")
    for i, config in enumerate(configs):
        test_model = MiniViT(**config)
        param_count = sum(p.numel() for p in test_model.parameters())
        print(f"Config {i+1}: dim={config['dim']}, params={param_count}")
