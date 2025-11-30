from math import pi

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    
    def __init__(self, dim, input_size, patch_size, pe_type='gaussian', extent=1, feat_sigma=1):
        super().__init__()
        self.dim = dim
        self.pe_type = pe_type
        self.input_size = input_size
        self.patch_size = patch_size
        self.patches = input_size // patch_size
        self.num_patches = (input_size // patch_size) ** 2 
        self.extent = abs(extent)
        self.feat_sigma = feat_sigma
        if self.pe_type == 'gaussian':
            self.rand_freqs = torch.randn((self.dim // 2, 2)) * feat_sigma

            coords = torch.stack(torch.meshgrid(torch.linspace(-self.extent, self.extent, self.patches), torch.linspace(-self.extent, extent, self.patches)), dim=-1)
            coords = coords.reshape(-1, 2)  # (N, 2)
            
            # Compute the Gaussian random features
            proj = coords @ self.rand_freqs.T  # (N, dim//2)
            pe = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, dim)
            pe = pe.unsqueeze(0)
            self.register_buffer('pos_embedding', pe)
        elif self.pe_type == 'rope':
            # self.rand_freqs = torch.randn((self.dim // 2, 2)) * feat_sigma # cryodrgn default (gaussian)
            # positions = torch.linspace(-self.extent, self.extent, self.patches)
            self.rand_freqs = torch.linspace(1., self.extent, self.patches) * pi # frequencies used in lucidrains' rope implementation for pixel data
            positions = torch.linspace(-self.extent, self.extent, self.patches)
            
            proj = positions @ self.rand_freqs.T  # Outer product, shape: (N, dim//2)
            proj = torch.cat([proj, proj], dim=-1)  # (N, dim)
            
            sin = torch.sin(proj).unsqueeze(0) # (1, N, dim)
            cos = torch.cos(proj).unsqueeze(0) # (1, N, dim)
            self.register_buffer('sin', sin)
            self.register_buffer('cos', cos)
        elif self.pe_type == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, self.dim))
        elif self.pe_type == 'sinusoidal': 
            pe = torch.zeros((self.dim, self.patches, self.patches))  # (dim, H, W)
            coords = torch.linspace(-self.extent, self.extent, self.patches).unsqueeze(1) # (patches, 1)
            d_half = self.dim // 2
            denominator = torch.pow(10000, torch.arange(0, d_half, 2) / d_half) # (dim/2,)
            pe[0:d_half:2, :, :] = torch.sin(coords * denominator).transpose(0, 1).unsqueeze(1).repeat(1, self.patches, 1) 
            pe[1:d_half:2, :, :] = torch.cos(coords * denominator).transpose(0, 1).unsqueeze(1).repeat(1, self.patches, 1) 
            pe[d_half::2, :, :] = torch.sin(coords * denominator).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.patches) 
            pe[d_half + 1::2, :, :] = torch.cos(coords * denominator).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.patches)
            pe = pe.view(self.dim, -1).T.unsqueeze(0) 
            self.register_buffer('pos_embedding', pe)
        else:  
            raise ValueError(f"Unknown pe_type: {self.pe_type}")
        
    def _rotate_half(self, x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x):
        '''
        x: (B, N, dim)
        '''
        if self.pe_type == 'rope':
            return x * self.cos + self._rotate_half(x) * self.sin
        return self.pos_embedding