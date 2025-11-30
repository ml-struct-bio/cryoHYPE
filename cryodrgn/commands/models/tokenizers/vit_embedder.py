import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from cryodrgn.commands.models import register
from .positional_embedding import PositionalEmbedding
from .base_tokenizer import BaseTokenizer
    

@register('vit_embedder')
class Embedder(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=1, pe_type='gaussian', extent=1, feat_sigma=1, concat=False, **kwargs):
        super().__init__()
        self.tokenizer = BaseTokenizer(input_size, patch_size, dim, padding, img_channels)
        self.pos_embedding = PositionalEmbedding(dim, input_size, patch_size, pe_type=pe_type, extent=extent, feat_sigma=feat_sigma)
        self.concat = concat

    def forward(self, data):
        x = self.tokenizer(data)
        if self.concat:
            # interleaved concat
            pe = self.pos_embedding().repeat(x.shape[0], 1, 1)  # (B, N, dim)
            x = torch.stack([x, pe], dim=-2)
            x = torch.flatten(x, start_dim=-3, end_dim=-2)
        else:
            x = x + self.pos_embedding(x)
        return x