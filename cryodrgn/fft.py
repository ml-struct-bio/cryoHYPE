"""Utility functions used in Fast Fourier Transform calculations on image tensors."""

import logging
import numpy as np
import torch
from torch.fft import fftshift, ifftshift, fft2, fftn, ifftn
from typing import Optional

logger = logging.getLogger(__name__)


def normalize(
    img: torch.Tensor,
    mean: float = 0,
    std: Optional[float] = None,
    std_n: Optional[int] = None,
) -> torch.Tensor:
    """Normalize an image tensors to z-scores using the first `std_n` samples.

    Note that since taking the standard deviation is a memory-consuming process,
    we here use the first `std_n` samples for its calculation.

    """
    if std is None:
        std = torch.std(img[:std_n, ...])

    logger.info(f"Normalized by {mean} +/- {std}")
    return (img - mean) / std


def fft2_center(img: torch.Tensor) -> torch.Tensor:
    return fftshift(fft2(fftshift(img, dim=(-1, -2))), dim=(-1, -2))


def fftn_center(img: torch.Tensor) -> torch.Tensor:
    return fftshift(fftn(fftshift(img)))


def ifftn_center(img: torch.Tensor) -> torch.Tensor:
    x = ifftshift(img)
    y = ifftn(x)
    z = ifftshift(y)
    return z


def ht2_center(img: torch.Tensor) -> torch.Tensor:
    _img = fft2_center(img)
    return _img.real - _img.imag


def htn_center(img: torch.Tensor) -> torch.Tensor:
    _img = fftshift(fftn(fftshift(img)))
    return _img.real - _img.imag


def iht2_center(img: torch.Tensor) -> torch.Tensor:
    img = fft2_center(img)
    img /= img.shape[-1] * img.shape[-2]
    return img.real - img.imag

def batch_ihtn_center_loop(batch_ht: torch.Tensor) -> torch.Tensor:
    """
    batch_ht: [B, H, W] complex half‑plane tensor
    returns: [B, H, W] real center‑inverted images
    """
    outs = []
    for ht in batch_ht:
        # ht is [H, W], so this matches your single‑image ihtn_center
        img = ihtn_center(ht)
        outs.append(img)
    return torch.stack(outs, dim=0)

def batch_htn_center_loop(batch_ht: torch.Tensor) -> torch.Tensor:
    """
    batch_ht: [B, H, W] complex half‑plane tensor
    returns: [B, H, W] real center‑inverted images
    """
    outs = []
    for ht in batch_ht:
        # ht is [H, W], so this matches your single‑image ihtn_center
        img = ht2_center(ht)
        outs.append(img)
    return torch.stack(outs, dim=0)

# New functions for explicit Fourier transforms
def batch_fft_center_loop(batch_real: torch.Tensor) -> torch.Tensor:
    """
    batch_real: [B, H, W] real space images
    returns: [B, H, W] Fourier transformed images
    """
    outs = []
    for img in batch_real:
        # Apply centered Fourier transform
        ft_img = fft2_center(img)
        outs.append(ft_img)
    return torch.stack(outs, dim=0)

def batch_ifft_center_loop(batch_ft: torch.Tensor) -> torch.Tensor:
    """
    batch_ft: [B, H, W] Fourier space images
    returns: [B, H, W] real space images
    """
    outs = []
    for ft in batch_ft:
        # Apply centered inverse Fourier transform
        img = ifftn_center(ft)
        # Take real part only as the result should be real
        img = img.real
        outs.append(img)
    return torch.stack(outs, dim=0)

def ihtn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional inverse discrete Hartley transform with origin at center."""
    img = fftn_center(img)
    img /= torch.prod(torch.tensor(img.shape, device=img.device))
    return img.real - img.imag

def fftn_center(img: torch.Tensor) -> torch.Tensor:
    """N-dimensional discrete Fourier transform reordered with origin at center."""
    return fftshift(fftn(fftshift(img)))

def symmetrize_ht(ht: torch.Tensor) -> torch.Tensor:
    if ht.ndim == 2:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 3
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0, :]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]  # last corner is first corner

    if n == 1:
        sym_ht = sym_ht[0, ...]

    return sym_ht