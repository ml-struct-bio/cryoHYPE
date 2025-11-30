import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from cryodrgn.commands.models import register


class Attention(nn.Module):

    def __init__(self, dim, n_head, head_dim, dropout=0.):
        super().__init__()
        self.n_head = n_head
        inner_dim = n_head * head_dim
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.scale = head_dim ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, fr, to=None, return_attention=False):
        if to is None:
            to = fr
        q = self.to_q(fr)
        k, v = self.to_kv(to).chunk(2, dim=-1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h=self.n_head), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(dots, dim=-1) # b h n n
        out = torch.matmul(attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)')
        if return_attention:
            return self.to_out(out), attn
        else:
            return self.to_out(out)


class FeedForward(nn.Module):

    def __init__(self, dim, ff_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, return_attention=False):
        if return_attention:
            return self.fn(self.norm(x), return_attention=True)
        else:
            return self.fn(self.norm(x))


@register('transformer_encoder')
class TransformerEncoder(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
            ]))

    def forward(self, x, return_attention=False):
        attentions = []
        for norm_attn, norm_ff in self.layers:
            if return_attention:
                x_res, attn = norm_attn(x, return_attention=True)
                attentions.append(attn)
                x = x + x_res
            else:
                x = x + norm_attn(x)
            x = x + norm_ff(x)
        if return_attention:
            return x, attentions
        else:
            return x
    
    
def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, p, **kwargs):
        super().__init__()
        self.p = p

    def forward(self, x):
        x = drop_path(x, self.p, self.training)
        return x


@register('transformer_encoder_dp')
class TransformerEncoderDP(nn.Module):

    def __init__(self, dim, depth, n_head, head_dim, ff_dim, dropout=0., sd=0.):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, n_head, head_dim, dropout=dropout)),
                DropPath(sd) if sd > 0. else nn.Identity(),
                PreNorm(dim, FeedForward(dim, ff_dim, dropout=dropout)),
                DropPath(sd) if sd > 0. else nn.Identity(),
            ]))

    def forward(self, x):
        for norm_attn, dp1, norm_ff, dp2 in self.layers:
            x = x + dp1(norm_attn(x))
            x = x + dp2(norm_ff(x))
        return x