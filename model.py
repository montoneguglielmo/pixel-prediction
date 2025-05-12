import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTransformer(nn.Module):
    def __init__(self, patch_dim=16, embed_dim=128, num_heads=4, depth=4):
        super().__init__()
        self.embed = nn.Linear(patch_dim, embed_dim)
        self.max_seq_len = 48
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.predict = nn.Linear(embed_dim, patch_dim)
        
    def generate_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    def forward(self, x):
        x = self.embed(x) + self.pos_embed
        #print('Input shape:', x.shape)
        mask = self.generate_causal_mask(x.size(0), x.device)  # (seq_len, seq_len)
        #print('Mask shape:', mask.shape)
        x = self.transformer(x, mask=mask)  # Enforce causal attention
        out = self.predict(x)
        #print('Output shape:', out.shape)
        return out