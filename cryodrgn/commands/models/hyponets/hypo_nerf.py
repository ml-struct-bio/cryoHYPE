import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import numpy as np

from cryodrgn.commands.models import register
from .layers import batched_linear_mm


@register('hypo_nerf')
class HypoNerf(nn.Module):

    def __init__(self, use_viewdirs=False, in_dim=3, out_dim=1, depth=6, hidden_dim=256, use_pe=True, pe_dim=40, pe_type='gaussian', feat_sigma=1, residual=False):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.depth = depth
        self.use_pe = use_pe
        self.pe_dim = pe_dim
        self.pe_type = pe_type
        self.feat_sigma = feat_sigma
        self.residual = residual
        self.param_shapes = dict()
        
        if self.pe_type == 'gaussian' and self.use_pe:
            rand_freqs = (
                torch.randn((3 * (self.pe_dim//2), 3), dtype=torch.float) * self.feat_sigma
            )
            self.register_buffer('rand_freqs', rand_freqs)

        if use_pe:
            last_dim = in_dim * pe_dim
        else:
            last_dim = in_dim
        for i in range(depth - 1):
            self.param_shapes[f'wb{i}'] = (last_dim + 1, hidden_dim)
            last_dim = hidden_dim

        if self.use_viewdirs:
            self.param_shapes['viewdirs_fc'] = (3 + 1, hidden_dim // 2)
            self.param_shapes['density_fc'] = (hidden_dim + 1, 1)
            self.param_shapes['rgb_fc1'] = (hidden_dim + hidden_dim // 2 + 1, hidden_dim)
            self.param_shapes['rgb_fc2'] = (hidden_dim + 1, 3)
        else:
            self.param_shapes['rgb_density_fc'] = (hidden_dim + 1, out_dim)

        self.relu = nn.ReLU()
        self.params = None

    def set_params(self, params):
        self.params = params

    def convert_posenc(self, x):
        if self.pe_type == 'cryodrgn':
            x_shape = x.shape
            freqs = torch.arange(1, (self.pe_dim // 2) + 1, dtype=torch.float, device=x.device)
            freqs = freqs.view(*[1] * len(x.shape), -1)  # 1 x 1 x D2
            x = x.unsqueeze(-1)  # B x 3 x 1
            k = x[..., 0:3, :] * freqs  # B x 3 x D2
            s = torch.sin(k)  # B x 3 x D2
            c = torch.cos(k)  # B x 3 x D2
            x = torch.cat([s, c], -1).view(*x_shape[:-1], -1)  # B x 3 x D
        elif self.pe_type == 'gaussian':
            freqs = self.rand_freqs.view(*[1] * (len(x.shape) - 1), -1, 3) * (self.pe_dim // 2)
            kxkykz = x[..., None, 0:3] * freqs  # compute the x,y,z components of k
            k = kxkykz.sum(-1)  # compute k
            s = torch.sin(k)
            c = torch.cos(k)
            x = torch.cat([s, c], -1)
        else:
            w = 2**torch.linspace(0, 8, self.pe_dim // 2, device=x.device)
            x = torch.matmul(x.unsqueeze(-1), w.unsqueeze(0)).view(*x.shape[:-1], -1)
            x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x

    def forward(self, x, viewdirs=None):
        B, query_shape = x.shape[0], x.shape[1: -1]

        # x = x.view(B, -1, 3)
        if self.use_pe:
            x = self.convert_posenc(x)

        if self.use_viewdirs:
            viewdirs = viewdirs.contiguous().view(B, -1, 3)
            viewdirs = F.normalize(viewdirs, dim=-1)
            viewdirs = batched_linear_mm(viewdirs, self.params['viewdirs_fc'])
            viewdirs = self.relu(viewdirs)

        for i in range(self.depth - 1):
            if self.residual and i > 0:
                x = batched_linear_mm(x, self.params[f'wb{i}']) + x
            else:
                x = batched_linear_mm(x, self.params[f'wb{i}'])
            x = self.relu(x)

        if self.use_viewdirs:
            density = batched_linear_mm(x, self.params['density_fc'])
            x = torch.cat([x, viewdirs], dim=-1)
            x = batched_linear_mm(x, self.params['rgb_fc1'])
            x = self.relu(x)
            rgb = batched_linear_mm(x, self.params['rgb_fc2'])
            out = torch.cat([rgb, density], dim=-1)
        else:
            out = batched_linear_mm(x, self.params['rgb_density_fc'])

        if len(out.shape) == 3:
            out = out.view(B, *query_shape, -1)
        elif len(out.shape) == 4:
            out = einops.rearrange(out, 'b v h w -> (b v) (h w)')
            
        return out
