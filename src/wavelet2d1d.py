#!/usr/bin/env python
"""
Self-contained 2D-1D starlet (a trous) wavelet transform for spectral cubes.

This is a standalone re-implementation of the transform used by
Denoiser2D1D-improved's `Wavelet2D1DTransform` (which wraps CosmoStat's
compiled `pysparse`/Sparse2D `MR2D1D`). It exists so this repository has no
external astronomy-specific dependency: everything below is pure numpy.

Design choices relative to the original wrapper
-------------------------------------------------
- 2D spatial part: isotropic undecimated (a trous) starlet transform using the
  classic B3-spline scaling function, applied via separable 1D smoothing
  along the y and x axes. This matches `transform_type=2`'s spatial part in
  the original module (isotropic 2D starlets).
- 1D spectral part: the original uses a *decimated* biorthogonal 7/9
  (CDF 9/7) wavelet along the spectral axis, which requires bookkeeping
  variable-length sub-bands (see `_extract_metadata` in the original
  `wavelet_denoising.py`). Here the spectral direction instead uses the same
  undecimated a trous B3-spline transform as the spatial part. This keeps
  every sub-band the same shape as the input cube, which means:
    1. reconstruction is *exact* by construction: sum of all sub-bands
       (all 2D scales x all 1D scales, including the coarsest) equals the
       input cube, with no separate inverse-transform bookkeeping needed.
    2. the whole (n_2d_scales+1) x (n_1d_scales+1) coefficient array is a
       single dense numpy array, which is what a gradient-based deconvolution
       iteration (see `deconvolver.py`) needs to threshold every iteration.
  The trade-off is a larger memory footprint per iteration than the
  decimated original (all sub-bands keep the full cube's voxel count), which
  is fine at the toy sizes used in this repo but would need revisiting for
  large production cubes -- at that point, swapping back to `pysparse`'s
  decimated MR2D1D behind the same `forward`/`inverse` interface used here
  is the natural next step.

Both directions are still "2D scale x 1D scale" multi-resolution
decompositions of the cube, i.e. conceptually the same 2D-1D transform as the
original -- spatial morphology captured by 2D scales, spectral/kinematic
structure (line profiles, velocity splitting) captured by 1D scales -- just
with an undecimated spectral transform instead of a decimated one.
"""

import numpy as np

# B3-spline scaling function taps, the standard starlet smoothing kernel.
_B3_SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _dilate_kernel(kernel, step):
    """Insert (step - 1) zeros between kernel taps ('a trous' dilation)."""
    if step == 1:
        return kernel
    n = len(kernel)
    dilated = np.zeros((n - 1) * step + 1)
    dilated[::step] = kernel
    return dilated


def _smooth_along_axis(arr, step, axis):
    """
    Convolve `arr` along `axis` with the dilated B3-spline kernel, using
    reflective boundary conditions, via a vectorized FFT convolution (no
    Python-level loop over the other axes).

    Returns an array the same shape as `arr` ('same'/'valid'-after-padding
    convolution).
    """
    kernel = _dilate_kernel(_B3_SPLINE, step)
    klen = len(kernel)
    pad = klen // 2

    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    arr_p = np.pad(arr, pad_width, mode="reflect")

    npad = arr_p.shape[axis]
    nfft = npad + klen - 1

    kshape = [1] * arr.ndim
    kshape[axis] = -1
    K = np.fft.rfft(kernel, n=nfft).reshape(kshape)
    A = np.fft.rfft(arr_p, n=nfft, axis=axis)
    full = np.fft.irfft(A * K, n=nfft, axis=axis)

    valid_len = npad - klen + 1  # == arr.shape[axis]
    start = klen - 1
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice(start, start + valid_len)
    return full[tuple(sl)]


def starlet_forward(cube, num_scales, axes):
    """
    Generic a trous starlet decomposition along one or more axes.

    Parameters
    ----------
    cube : ndarray
    num_scales : int
        Number of detail scales (there will be num_scales + 1 output planes,
        the last one being the coarse/residual approximation).
    axes : tuple of int
        Axes to smooth over at each scale (e.g. (1, 2) for a 2D spatial
        transform on a (nz, ny, nx) cube, or (0,) for a 1D spectral one).

    Returns
    -------
    planes : ndarray, shape (num_scales + 1,) + cube.shape
        planes[:-1] are wavelet detail planes (fine to coarse), planes[-1]
        is the final coarse approximation. `planes.sum(axis=0) == cube`
        exactly (up to floating point round-off).
    """
    c = cube.astype(np.float64, copy=True)
    planes = np.empty((num_scales + 1,) + cube.shape, dtype=np.float64)
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
    Forward 2D (spatial, axes 1,2) x 1D (spectral, axis 0) starlet transform.

    Parameters
    ----------
    cube : ndarray, shape (nz, ny, nx)
    num_scales_2d, num_scales_1d : int

    Returns
    -------
    subbands : ndarray, shape (num_scales_2d + 1, num_scales_1d + 1, nz, ny, nx)
        subbands[j2, j1] is the (2D scale j2, 1D scale j1) sub-band.
        subbands.sum(axis=(0, 1)) == cube exactly.
    """
    planes_2d = starlet_forward(cube, num_scales_2d, axes=(1, 2))  # (J2+1, nz, ny, nx)
    n2, nz, ny, nx = planes_2d.shape
    subbands = np.empty((n2, num_scales_1d + 1, nz, ny, nx), dtype=np.float64)
    for j2 in range(n2):
        subbands[j2] = starlet_forward(planes_2d[j2], num_scales_1d, axes=(0,))
    return subbands


def inverse_2d1d(subbands):
    """Exact inverse of `transform_2d1d`: sum over both scale axes."""
    return subbands.sum(axis=(0, 1))


def estimate_subband_noise(sigma, shape, num_scales_2d, num_scales_1d,
                            n_mc=4, seed=0):
    """
    Monte-Carlo estimate of the per-sub-band noise standard deviation that
    results from propagating white Gaussian noise of std `sigma` through
    `transform_2d1d`. Mirrors `Denoiser2D1D._decompose_and_estimate_noise`'s
    `noise_cube` branch in the original module (there, an *independent* noise
    realization is transformed and its per-band std taken directly; here we
    average over a few realizations for a more stable estimate since we no
    longer have the original's pre-tabulated `NOISE_TAB`).

    Returns
    -------
    noise_levels : ndarray, shape (num_scales_2d + 1, num_scales_1d + 1)
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((num_scales_2d + 1, num_scales_1d + 1))
    for _ in range(n_mc):
        noise = rng.normal(0.0, sigma, size=shape)
        sb = transform_2d1d(noise, num_scales_2d, num_scales_1d)
        acc += sb.std(axis=(2, 3, 4))
    return acc / n_mc
