"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from .math import complex_abs_sq, complex_conj, complex_mul
from .fft import fft2c, ifft2c

def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))

def sense_expand(x: torch.Tensor, sens_maps: torch.Tensor):
    return complex_mul(x, sens_maps)

def sense_reduce(x: torch.Tensor, sens_maps: torch.Tensor, COIL_DIM=0):
    return complex_mul(x, complex_conj(sens_maps)).sum(
        dim=COIL_DIM, keepdim=True
    )