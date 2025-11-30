import torch
from torch import Tensor
from typing import Tuple
import torch.nn as nn
import torch.nn.functional as F
import einops
from cryodrgn.commands.models import register

def _get_center_distance(size: Tuple[int], device: str = 'cpu') -> Tensor:
    """Compute the distance of each matrix element to the center.

    Args:
        size (Tuple[int]): [m, n].
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [m, n].
    """    
    m, n = size
    i_ind = torch.tile(
                torch.tensor([[[i]] for i in range(m)], device=device),
                dims=[1, n, 1]).float()  # [m, n, 1]
    j_ind = torch.tile(
                torch.tensor([[[i] for i in range(n)]], device=device),
                dims=[m, 1, 1]).float()  # [m, n, 1]
    ij_ind = torch.cat([i_ind, j_ind], dim=-1)  # [m, n, 2]
    ij_ind = ij_ind.reshape([m * n, 1, 2])  # [m * n, 1, 2]
    center_ij = torch.tensor(((m - 1) / 2, (n - 1) / 2), device=device).reshape(1, 2)
    center_ij = torch.tile(center_ij, dims=[m * n, 1, 1])
    dist = torch.cdist(ij_ind, center_ij, p=2).reshape([m, n])
    return dist


def _get_gaussian_weights(size: Tuple[int], D0: float, device: str = 'cpu') -> Tensor:
    """Get H(u, v) of Gaussian filter.

    Args:
        size (Tuple[int]): [H, W].
        D0 (float): The cutoff frequency.
        device (str, optional): cpu/cuda. Defaults to 'cpu'.

    Returns:
        Tensor: [H, W].
    """    
    center_distance = _get_center_distance(size=size, device=device)
    weights = torch.exp(- (torch.square(center_distance) / (2 * D0 ** 2)))
    return weights


def gaussian(image: Tensor, D0: float, is_highpass=False) -> Tensor:
    """Gaussian low-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (int): Cutoff frequency.

    Returns:
        Tensor: [B, C, H, W].
    """    
    weights = _get_gaussian_weights(image.shape[-2:], D0=D0, device=image.device)
    if is_highpass:
        weights = 1-weights
    # image_fft = _to_freq(image)
    image_fft = image * weights
    # image = _to_space(image_fft)
    return image_fft

# def _get_center_distance_highpass(size: Tuple[int], device: str = 'cpu') -> Tensor:
#     """Calculate the distance of each pixel from the center of the frequency domain image."""
#     H, W = size
#     u = torch.arange(H, device=device).float() - H // 2
#     v = torch.arange(W, device=device).float() - W // 2
#     U, V = torch.meshgrid(u, v, indexing='ij')
#     center_distance = torch.sqrt(U ** 2 + V ** 2)
#     return center_distance

# High pass filter

def _get_gaussian_weights_highpass(shape, D0, device):
    """Generates Gaussian low-pass filter weights."""
    H, W = shape
    u = torch.arange(-H//2, H//2, device=device).unsqueeze(1).repeat(1, W)
    v = torch.arange(-W//2, W//2, device=device).unsqueeze(0).repeat(H, 1)
    D_uv = torch.sqrt(u**2 + v**2)
    weights_lp = torch.exp(-(D_uv**2) / (2 * (D0**2)))
    return torch.fft.fftshift(weights_lp)

def gaussian_highpass(image: torch.Tensor, D0: float) -> torch.Tensor:
    """Gaussian high-pass filter for images.

    Args:
        image (Tensor): [B, C, H, W].
        D0 (float): Cutoff frequency.

    Returns:
        Tensor: [B, C, H, W].
    """    
    B, C, H, W = image.shape
    device = image.device
    # Compute the Gaussian low-pass filter weights
    weights_lp = _get_gaussian_weights_highpass((H, W), D0=D0, device=device)
    # Transform the low-pass filter into a high-pass filter
    weights_hp = 1 - weights_lp
    # Apply the Fourier Transform to the image
    # image_fft = _to_freq(image)
    # Expand weights to match the batch and channel dimensions
    weights_hp = weights_hp.unsqueeze(0).unsqueeze(0)
    # Apply the high-pass filter in the frequency domain
    image_fft_filtered = image * weights_hp
    # Transform back to the spatial domain
    # image_filtered = _to_space(image_fft_filtered)
    return image_fft_filtered

@register('filter_tokenizer')
class FilterTokenizer(nn.Module):

    def __init__(self, input_size, patch_size, dim, padding=0, img_channels=3, filter_type=None, cutoff=None):
        super().__init__()
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.patch_size = patch_size
        self.padding = padding
        
        self.filter_type = filter_type
        self.cutoff = cutoff

        self.prefc = nn.Linear(patch_size[0] * patch_size[1] * (img_channels), dim)
        self.grid_shape = ((input_size[0] + padding[0] * 2) // patch_size[0],
                           (input_size[1] + padding[1] * 2) // patch_size[1])

    def forward(self, data):
        imgs = data
        # imgs = data['support_imgs']
        B = imgs.shape[0]
        H, W = imgs.shape[-2:]
        # rays_o, rays_d = poses_to_rays(data['support_poses'], H, W, data['support_focals'])
        # rays_o = einops.rearrange(rays_o, 'b n h w c -> b n c h w')
        # rays_d = einops.rearrange(rays_d, 'b n h w c -> b n c h w')

        # x = torch.cat([imgs, rays_o, rays_d], dim=2)
        x = imgs
        x = einops.rearrange(x, 'b n d h w -> (b n) d h w')
        p = self.patch_size
        if self.filter_type == 'gaussian_highpass':
            x = gaussian_highpass(x, self.cutoff)
        elif self.filter_type == 'gaussian':
            x = gaussian(x, self.cutoff)
        x = F.unfold(x, p, stride=p, padding=self.padding)
        x = einops.rearrange(x, '(b n) ppd l -> b (n l) ppd', b=B)

        x = self.prefc(x)
        return x
