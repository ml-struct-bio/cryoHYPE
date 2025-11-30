import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import cryodrgn.commands.models as models
from cryodrgn.commands.models import register


def init_wb(shape):
    weight = torch.empty(shape[1], shape[0] - 1)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    bias = torch.empty(shape[1], 1)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    return torch.cat([weight, bias], dim=1).t().detach()


@register('trans_inr')
class TransInr(nn.Module):

    def __init__(self, tokenizer, hyponet, n_groups, transformer_encoder, norm=True, add=False, residual=True, wdropout=0., drop_type='token', load_base_params=None):
        super().__init__()
        dim = transformer_encoder['args']['dim']
        
        if tokenizer['args'].get('filter_type', None) is not None:
            filter_type = tokenizer['args']['filter_type']
            cutoff = tokenizer['args']['cutoff']
        else:
            filter_type = None
            cutoff = None
        self.tokenizer = models.make(tokenizer, args={'dim': dim, 'filter_type': filter_type, 'cutoff':cutoff})
        self.hyponet = models.make(hyponet)
        self.transformer_encoder = models.make(transformer_encoder)
        self.norm = norm
        self.add = add
        self.residual = residual
        
        if load_base_params is not None:
            ckpt = torch.load(load_base_params, weights_only=True)
            if hasattr(self.hyponet, 'rand_freqs') and 'model.hyponet.rand_freqs' in ckpt['state_dict']:
                self.hyponet.rand_freqs = ckpt['state_dict']['model.hyponet.rand_freqs']

        self.base_params = nn.ParameterDict()
        n_wtokens = 0
        self.wtoken_postfc = nn.ModuleDict()
        self.wtoken_rng = dict()
        for name, shape in self.hyponet.param_shapes.items():
            if load_base_params is not None:
                assert 'model.base_params.' + name in ckpt['state_dict']
                assert ckpt['state_dict']['model.base_params.' + name].shape == shape
                self.base_params[name] = ckpt['state_dict']['model.base_params.' + name]
            else:
                self.base_params[name] = nn.Parameter(init_wb(shape))
            g = min(n_groups, shape[1])
            assert shape[1] % g == 0
            self.wtoken_postfc[name] = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, shape[0] - 1),
            )
            self.wtoken_rng[name] = (n_wtokens, n_wtokens + g)
            n_wtokens += g
        self.wtokens = nn.Parameter(torch.randn(n_wtokens, dim))
        
        # weight token dropout
        if wdropout > 0:
            if drop_type == 'token':
                self.wdropout = nn.Dropout1d(p=wdropout)
            elif drop_type == 'weight':
                self.wdropout = nn.Dropout(p=wdropout)
        else:
            self.wdropout = nn.Identity()

    def forward(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        wtoken = self.wdropout(wtokens)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]

        params = dict()
        for name, shape in self.hyponet.param_shapes.items():
            wb = einops.repeat(self.base_params[name], 'n m -> b n m', b=B)
            w, b = wb[:, :-1, :], wb[:, -1:, :]

            l, r = self.wtoken_rng[name]
            x = self.wtoken_postfc[name](trans_out[:, l: r, :])
            x = x.transpose(-1, -2) # (B, shape[0] - 1, g)
            if self.residual:
                if self.add:
                    w = w + x.repeat(1, 1, w.shape[2] // x.shape[2])
                else:
                    w = w * x.repeat(1, 1, w.shape[2] // x.shape[2])
            else:
                w = x.repeat(1, 1, w.shape[2] // x.shape[2])
            if self.norm:
                w = F.normalize(w, dim=1)

            wb = torch.cat([w, b], dim=1)
            params[name] = wb

        self.hyponet.set_params(params)
        return self.hyponet

    def get_z(self, data):
        dtokens = self.tokenizer(data)
        B = dtokens.shape[0]
        wtokens = einops.repeat(self.wtokens, 'n d -> b n d', b=B)
        wtoken = self.wdropout(wtokens)
        trans_out = self.transformer_encoder(torch.cat([dtokens, wtokens], dim=1))
        trans_out = trans_out[:, -len(self.wtokens):, :]
        return trans_out