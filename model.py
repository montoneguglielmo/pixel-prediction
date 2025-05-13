import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    EMBED_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT,
    NUM_PATCHES
)

class PatchTransformer(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=EMBED_DIM, num_heads=NUM_HEADS, 
                 num_layers=NUM_LAYERS, max_len=NUM_PATCHES):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.max_seq_len = max_len
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=DROPOUT
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.predict = nn.Linear(embed_dim, vocab_size)
        
    def generate_causal_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask.to(device)

    def forward(self, x):
        #print('x shape:', x.shape)
        #print('embed shape:', self.embed(x).shape)
        x = self.embed(x) +  self.pos_embed[:, :x.size(1)]
        #print('Input shape:', x.shape)
        mask = self.generate_causal_mask(x.size(1), x.device)  # (seq_len, seq_len)
        #print('Mask shape:', mask.shape)
        x = self.transformer(x, mask=mask)  # Enforce causal attention
        out = self.predict(x)
        #print('Output shape:', out.shape)
        return out