"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional

import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse("1.7.0"):
    import torch.fft  # type: ignore


def fft2(image):
    if image.dtype in (torch.complex32, torch.complex64):
        return torch.fft.fftn(image.contiguous(), dim=(-2, -1))
    else:
        return torch.view_as_real(
                torch.fft.fftn(  # type: ignore
                    torch.view_as_complex(image.contiguous()), dim=(-2, -1)
                )
        )


def ifft2(image):
    if image.dtype in (torch.complex32, torch.complex64):
        return torch.fft.ifftn(image.contiguous(), dim=(-2, -1))
    else:
        return torch.view_as_real(
            torch.fft.ifftn(  # type: ignore
                torch.view_as_complex(image.contiguous()), dim=(-2, -1)
            )
        )


def fft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2 dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The FFT of the input.
    """
    if data.shape[-1] == 2:
        data = ifftshift(data, dim=[-3, -2])
        data = fft2(data)
        data = fftshift(data, dim=[-3, -2])
    elif data.dtype in (torch.complex32, torch.complex64):
        data = ifftshift(data, dim=[-2, -1])
        data = fft2(data)
        data = fftshift(data, dim=[-2, -1])
    else:
        raise ValueError("Tensor does not have separate complex dim.")
    return data


def ifft2c(data: torch.Tensor) -> torch.Tensor:
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data containing at least 3 dimensions:
            dimensions -3 & -2 are spatial dimensions and dimension -1 has size
            2. All other dimensions are assumed to be batch dimensions.

    Returns:
        The IFFT of the input.
    """
    if data.shape[-1] == 2:
        data = ifftshift(data, dim=[-3, -2])
        data = ifft2(data)
        data = fftshift(data, dim=[-3, -2])
    elif data.dtype in (torch.complex32, torch.complex64):
        data = ifftshift(data, dim=[-2, -1])
        data = ifft2(data)
        data = fftshift(data, dim=[-2, -1])
    else:
        raise ValueError("Tensor does not have separate complex dim.")
    return data


# Helper functions


def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)
