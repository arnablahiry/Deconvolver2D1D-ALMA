#!/usr/bin/env python
"""
Classic Hogbom CLEAN, implemented from scratch, as the baseline this whole
repo is being compared against: iterative Dirac-delta (point-source)
component fitting, run independently per spectral channel, versus
`Deconvolver2D1D`'s single joint 2D-1D-wavelet-sparse fit across the whole
cube at once.

Algorithm (per channel, standard Hogbom 1974)
-----------------------------------------------
residual <- dirty channel; model <- 0
repeat:
    (y0, x0) <- location of max(|residual|)
    if |residual[y0, x0]| <= threshold: stop
    amp <- gain * residual[y0, x0]
    model[y0, x0] += amp
    residual -= amp * (dirty beam recentered on (y0, x0))
until max iterations or threshold reached

The `gain` (loop gain, typically 0.1-0.3) is what keeps Hogbom CLEAN from
ever "running away": every subtraction only removes a small fraction of the
current peak, so unlike the wavelet deconvolver's un-penalized coarse band
(see `deconvolver.py`'s note on the missing zero-spacing flux), there's no
step that can add unbounded flux in one iteration -- but by the same token,
CLEAN also has no explicit way to model genuinely extended/diffuse
structure: everything is built out of delta functions, so smooth or
multi-scale emission (like the toy ring source in this repo) gets
approximated by a pile of point components rather than represented
natively, which is precisely the practical motivation for the wavelet
approach.

`restore_clean_cube` produces the conventional CLEAN "restored image": the
delta-function model, convolved with a Gaussian "clean beam" fit to the
dirty beam's own main lobe, plus the leftover residual map added back --
this is what makes CLEAN output comparable in units/resolution to a
wavelet-deconvolved cube instead of a sparse set of infinitely-sharp spikes.
"""

import numpy as np

from deconvolver import convolve_cube
from psf import beam_fwhm_pixels


def _subtract_shifted_psf(residual, psf, y0, x0, amp):
    """In-place: residual -= amp * psf, with psf's own center pixel placed
    at (y0, x0) in `residual` (clipped at the array edges)."""
    ny, nx = residual.shape
    py, px = psf.shape
    cy, cx = py // 2, px // 2

    y_lo, y_hi = max(0, y0 - cy), min(ny, y0 - cy + py)
    x_lo, x_hi = max(0, x0 - cx), min(nx, x0 - cx + px)
    py_lo = y_lo - (y0 - cy)
    px_lo = x_lo - (x0 - cx)
    py_hi = py_lo + (y_hi - y_lo)
    px_hi = px_lo + (x_hi - x_lo)

    residual[y_lo:y_hi, x_lo:x_hi] -= amp * psf[py_lo:py_hi, px_lo:px_hi]


def hogbom_clean_channel(dirty, psf, gain=0.15, threshold=0.0, n_iter_max=500):
    """
    Hogbom CLEAN on a single 2D channel.

    Returns
    -------
    model : ndarray, shape dirty.shape
        Sparse delta-function ("clean components") model.
    residual : ndarray, shape dirty.shape
        What's left after subtracting the model's beam response.
    n_used : int
        Number of components actually placed.
    """
    residual = dirty.astype(np.float64).copy()
    model = np.zeros_like(residual)

    n_used = 0
    for _ in range(n_iter_max):
        idx = np.argmax(np.abs(residual))
        y0, x0 = np.unravel_index(idx, residual.shape)
        peak_val = residual[y0, x0]
        if abs(peak_val) <= threshold:
            break
        amp = gain * peak_val
        model[y0, x0] += amp
        _subtract_shifted_psf(residual, psf, y0, x0, amp)
        n_used += 1

    return model, residual, n_used


def hogbom_clean_cube(dirty_cube, psf, sigma_noise, gain=0.15,
                       threshold_sigma=3.0, n_iter_max=500, verbose=True):
    """
    Run `hogbom_clean_channel` independently on every channel of a cube --
    CLEAN has no notion of the spectral axis, each channel is deconvolved
    completely separately (unlike `Deconvolver2D1D`, which fits 2D spatial
    *and* 1D spectral structure jointly).

    `psf` may be a single (py, px) beam shared by every channel, or a
    (nz, py, px) per-channel beam cube (real data, e.g. CASA's `.psf`
    product) -- each channel is then cleaned with its own dirty beam.

    Returns
    -------
    model_cube, residual_cube : ndarray, shape dirty_cube.shape
    n_components : list of int, length nz
    """
    threshold = threshold_sigma * sigma_noise
    nz = dirty_cube.shape[0]
    model_cube = np.zeros_like(dirty_cube)
    residual_cube = np.zeros_like(dirty_cube)
    n_components = []

    per_channel_psf = (psf.ndim == 3)
    for k in range(nz):
        model, residual, n_used = hogbom_clean_channel(
            dirty_cube[k], psf[k] if per_channel_psf else psf,
            gain=gain, threshold=threshold, n_iter_max=n_iter_max
        )
        model_cube[k] = model
        residual_cube[k] = residual
        n_components.append(n_used)

    if verbose:
        total = sum(n_components)
        print(f"[Hogbom CLEAN] {total} components across {nz} channels "
              f"({total / nz:.1f}/channel avg), gain={gain}, "
              f"threshold={threshold:.4g} ({threshold_sigma} sigma)")

    return model_cube, residual_cube, n_components


def make_gaussian_beam(shape, fwhm_pixels, normalize="peak"):
    """
    A Gaussian 'clean beam' matching the dirty beam's main-lobe FWHM.

    normalize : {'peak', 'sum'}
        'peak' (peak = 1) is the standard radio-astronomy convention: the
        restored map ends up in the same "Jy/beam" units as the dirty map,
        and recovering a genuine total flux from it requires dividing by
        the beam's solid angle (area in pixels) -- exactly the same
        peak = 1 convention used for `psf.mock_alma_dirty_beam` itself.
        'sum' (flux-conserving, sums to 1) instead keeps convolution flux-
        preserving, which is what this repo uses for `restore_clean_cube`
        so that CLEAN's output lands on the same flux-density pixel scale
        as `Deconvolver2D1D`'s output and the true cube, making a direct
        pixel-by-pixel / RMSE comparison meaningful without also having to
        carry a beam-area (pixels-per-beam) conversion factor around.
    """
    ny, nx = shape
    sigma = fwhm_pixels / 2.3548200450309493  # FWHM -> sigma
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    Y, X = np.meshgrid(y, x, indexing="ij")
    g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    if normalize == "peak":
        return g / g.max()
    elif normalize == "sum":
        return g / g.sum()
    raise ValueError(f"Unknown normalize={normalize!r}")


def restore_clean_cube(model_cube, residual_cube, dirty_psf, clean_beam_shape=None):
    """
    Standard CLEAN restoration: model (*) clean-beam + residual, using a
    flux-conserving (sum = 1) Gaussian clean beam -- see `make_gaussian_beam`
    docstring for why 'sum' rather than the usual radio-astronomy 'peak'
    convention is used here.

    `dirty_psf` may be a single 2D beam or a (nz, py, px) per-channel cube;
    in the latter case its central channel sets the (single, shared)
    restoring beam's FWHM -- a single common clean beam per cube is the
    standard convention (real per-channel beam variation is small enough
    that CASA does the same by default).
    """
    psf_for_fwhm = dirty_psf if dirty_psf.ndim == 2 else dirty_psf[dirty_psf.shape[0] // 2]
    if clean_beam_shape is None:
        clean_beam_shape = psf_for_fwhm.shape
    fwhm = beam_fwhm_pixels(psf_for_fwhm)
    clean_beam = make_gaussian_beam(clean_beam_shape, fwhm, normalize="sum")
    restored = convolve_cube(model_cube, clean_beam) + residual_cube
    return restored, clean_beam
