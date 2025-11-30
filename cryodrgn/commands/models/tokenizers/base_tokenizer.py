import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from cryodrgn.commands.models import register
            

@register('base_tokenizer')
class BaseTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=1, **kwargs):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * (img_channels), dim)
        self.grid_shape = ((input_size[0] + padding[0] * 2) // patch_size[0],
                           (input_size[1] + padding[1] * 2) // patch_size[1])

    def forward(self, data):
        imgs = data
        B = imgs.shape[0]
        H, W = imgs.shape[-2:]

        x = imgs
        x = einops.rearrange(x, 'b n d h w -> (b n) d h w')
        p = self.patch_size
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = einops.rearrange(x, '(b n) ppd l -> b (n l) ppd', b=B)

        x = self.prefc(x)
        return x