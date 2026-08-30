#!/usr/bin/env python
"""
Torch port of `wavelet2d1d.py`: 2D-1D undecimated (a trous) starlet transform
for spectral cubes, running on whatever device the input tensor lives on
(cpu / cuda / mps).

Same conventions as the numpy version:
  - cube shape is (nz, ny, nx); axis 0 is spectral, axes (1, 2) are spatial;
  - `starlet_forward` returns `num_scales + 1` planes (details fine->coarse,
    then the coarse approximation) whose sum is exactly the input;
  - the 2D-1D transform is separable: a 2D starlet over (1, 2), then a 1D
    starlet over (0,) applied to each 2D plane, so sub-band (j2, j1) has the
    same shape as the cube and `sum over (j2, j1) == cube`.

Two things exist here that the numpy version does not have, both driven by the
FISTA deconvolution loop:

  - `prox_2d1d`, which fuses forward transform + per-sub-band soft threshold +
    reconstruction into one call. It never materializes the full
    (J2+1, J1+1, nz, ny, nx) coefficient array -- it holds at most (J2+1) plus
    (J1+1) full cubes at a time -- which matters for real ALMA cube sizes
    (100 x 256 x 256 x 20 sub-bands is ~0.5 GB in float32 alone).

  - `subband_mad_noise`, the per-sub-band noise level measured directly as the
    MAD of each sub-band of a cube (in practice the dirty cube). Robust to the
    signal sitting in the sub-band, and needs nothing but the data.
"""

import numpy as np
import torch

_B3_SPLINE = (1.0, 4.0, 6.0, 4.0, 1.0)


def _reflect_index(n, pad, device):
    """Indices realizing numpy's `mode='reflect'` padding by `pad` on both sides."""
    idx = np.pad(np.arange(n), pad, mode="reflect")
    return torch.as_tensor(idx, dtype=torch.long, device=device)


def _smooth_along_axis(x, step, axis):
    """Convolve `x` along `axis` with the a trous-dilated B3-spline kernel."""
    n = x.shape[axis]
    pad = 2 * step
    xp = x.index_select(axis, _reflect_index(n, pad, x.device))

    out = None
    for k, w in enumerate(_B3_SPLINE):
        piece = xp.narrow(axis, k * step, n)
        out = piece * w if out is None else out + piece * w
    return out / 16.0


def starlet_forward(cube, num_scales, axes):
    """
    A trous starlet decomposition of `cube` along `axes`.

    Returns a tensor of shape (num_scales + 1,) + cube.shape; planes[:-1] are
    detail planes (fine to coarse), planes[-1] is the coarse approximation, and
    planes.sum(0) reconstructs the input exactly.
    """
    c = cube
    planes = torch.empty((num_scales + 1,) + tuple(cube.shape),
                         dtype=cube.dtype, device=cube.device)
    for j in range(num_scales):
        step = 2 ** j
        cs = c
        for ax in axes:
            cs = _smooth_along_axis(cs, step, ax)
        planes[j] = c - cs
        c = cs
    planes[num_scales] = c
    return planes


def transform_2d1d(cube, num_scales_2d, num_scales_1d):
    """
    Full 2D (axes 1,2) x 1D (axis 0) starlet transform.

    Returns shape (num_scales_2d + 1, num_scales_1d + 1, nz, ny, nx). Only use
    this for inspection/diagnostics -- `prox_2d1d` is the memory-light path.
    """
    planes_2d = starlet_forward(cube, num_scales_2d, axes=(1, 2))
    subbands = torch.empty((num_scales_2d + 1, num_scales_1d + 1) + tuple(cube.shape),
                           dtype=cube.dtype, device=cube.device)
    for j2 in range(planes_2d.shape[0]):
        subbands[j2] = starlet_forward(planes_2d[j2], num_scales_1d, axes=(0,))
    return subbands


def inverse_2d1d(subbands):
    """Exact inverse of `transform_2d1d`: sum over both scale axes."""
    return subbands.sum(dim=(0, 1))


def soft_threshold(x, thr):
    """Elementwise soft threshold (shrinkage) by `thr`."""
    return torch.sign(x) * torch.clamp(x.abs() - thr, min=0.0)


def prox_2d1d(cube, thresholds, num_scales_2d, num_scales_1d,
              keep_coarse=True):
    """
    Soft-threshold `cube` in the 2D-1D starlet domain and reconstruct.

    Parameters
    ----------
    cube : (nz, ny, nx) real tensor
    thresholds : (num_scales_2d + 1, num_scales_1d + 1) array/tensor
        Threshold for sub-band (j2, j1), in the units of `cube`.
    keep_coarse : bool
        If True the coarsest-of-coarse band (j2 = J2, j1 = J1) -- the smooth
        background carrying the cube's total flux -- is passed through
        untouched, which is the usual starlet-deconvolution convention.

    Returns
    -------
    (nz, ny, nx) tensor: the thresholded cube.
    """
    thr = torch.as_tensor(thresholds, dtype=cube.dtype, device=cube.device)
    planes_2d = starlet_forward(cube, num_scales_2d, axes=(1, 2))

    out = torch.zeros_like(cube)
    for j2 in range(num_scales_2d + 1):
        planes_1d = starlet_forward(planes_2d[j2], num_scales_1d, axes=(0,))
        for j1 in range(num_scales_1d + 1):
            band = planes_1d[j1]
            if keep_coarse and j2 == num_scales_2d and j1 == num_scales_1d:
                out += band
            else:
                out += soft_threshold(band, thr[j2, j1])
    return out


def mad_sigma(x):
    """
    Robust noise sigma of a tensor: median(|x - median(x)|) / 0.6745, the
    Gaussian-consistent MAD estimator. Robust to the signal in the band --
    bright emission moves few enough voxels to leave the median deviation
    alone, which a plain std would not survive.
    """
    med = torch.median(x)
    return torch.median(torch.abs(x - med)) / 0.6745


def subband_mad_noise(cube, num_scales_2d, num_scales_1d):
    """
    Per-sub-band noise level of `cube`, measured as the MAD of each 2D-1D
    starlet sub-band. Typically called once on the dirty cube; the resulting
    levels scale the thresholds as k * sigma[j2, j1].

    Like `prox_2d1d`, this streams the sub-bands rather than materializing the
    whole coefficient array.

    Returns
    -------
    (num_scales_2d + 1, num_scales_1d + 1) tensor of noise levels.
    """
    planes_2d = starlet_forward(cube, num_scales_2d, axes=(1, 2))
    sigma = torch.empty((num_scales_2d + 1, num_scales_1d + 1),
                        dtype=cube.dtype, device=cube.device)
    for j2 in range(num_scales_2d + 1):
        planes_1d = starlet_forward(planes_2d[j2], num_scales_1d, axes=(0,))
        for j1 in range(num_scales_1d + 1):
            sigma[j2, j1] = mad_sigma(planes_1d[j1])
    return sigma
