#!/usr/bin/env python
"""
Loader for real ALMA data products: a genuine dirty cube + per-channel dirty
beam (CASA `tclean` with `niter=0`), used to test `Deconvolver2D1D` and
`clean.py` on actual interferometric data instead of the toy simulations
elsewhere in this repo.

The reference dataset is TW Hya (a well-studied protoplanetary disk), CO(3-2)
at 345.796 GHz, imaged from the calibrated ALMA measurement set with:

    tclean(..., specmode='cube', nchan=40, imsize=[256, 256], cell='0.08arcsec',
           weighting='briggs', robust=0.5, niter=0)               # dirty cube + psf
    tclean(..., deconvolver='multiscale', scales=[0, 5, 15],
           niter=5000, threshold='15mJy')                          # CASA benchmark

then exported to FITS with `exportfits`. See `data/prep_wavelet_data.py` for
the exact CASA commands used to produce the files this module loads.

Requires `astropy` (only used here, for FITS I/O -- the rest of the repo is
pure numpy/matplotlib).
"""

import os

import numpy as np
from astropy.io import fits


def _load_cube(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float64)
    # exportfits sometimes leaves degenerate leading axes (e.g. Stokes) even
    # with dropstokes=True depending on CASA/astropy version; squeeze them.
    while data.ndim > 3:
        data = data[0]
    return data


def _central_crop(cube, size):
    """Crop the last two (spatial) axes of a (nz, ny, nx) cube to a centered
    (size, size) region. `size` must be <= the smaller spatial dimension."""
    nz, ny, nx = cube.shape
    y0 = (ny - size) // 2
    x0 = (nx - size) // 2
    return cube[:, y0:y0 + size, x0:x0 + size]


def taper_psf(psf, flat_frac=0.0):
    """
    Apply a radial raised-cosine (Hann-family) taper to a dirty beam,
    forcing it smoothly to zero at the edge of its own array instead of
    being hard-truncated there.

    This matters specifically for `Deconvolver2D1D`'s gradient-based
    (ISTA/FISTA) iteration, not for `clean.py`'s Hogbom CLEAN (which only
    ever does local spatial-domain subtraction and has no FFT-based
    Lipschitz/step-size dependence on the beam's global shape). Real ALMA
    dirty beams -- unlike the compact mock beams in `psf.py` -- have
    sidelobe structure that does not decay to ~0 within the imaged field of
    view (see the demo notebook: even the edge of the *full* 256x256 CASA
    image is still ~5-9% of the peak). Convolving with that abruptly-
    truncated kernel via FFT (zero-padding then cropping, as
    `deconvolver.convolve_cube` does) introduces a sharp edge discontinuity
    that inflates the operator's Lipschitz constant enormously (observed:
    dropping from L ~ 2.7e4 untapered to L ~ 5.7e3 with a full-width taper
    on the TW Hya beam), forcing an ISTA/FISTA step size small enough that
    convergence becomes impractically slow. Tapering trades a small amount
    of far-sidelobe fidelity (which mostly sits below the map's own noise
    level anyway) for a well-conditioned operator that actually converges
    in a tractable number of iterations.

    Parameters
    ----------
    psf : ndarray, shape (py, px) or (nz, py, px)
    flat_frac : float in [0, 1)
        Fraction of the radius (from center to edge) left untouched
        (taper = 1) before the raised-cosine roll-off to 0 begins.
        0 = taper starts at the very center (most aggressive, smallest
        resulting Lipschitz constant); closer to 1 preserves more of the
        beam's true sidelobe structure at the cost of a larger, more
        slowly-convergent Lipschitz constant.
    """
    spatial_shape = psf.shape[-2:]
    ny, nx = spatial_shape
    y = np.arange(ny) - ny / 2
    x = np.arange(nx) - nx / 2
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    r_max = ny / 2
    r_flat = flat_frac * r_max

    taper = np.ones_like(R)
    roll = R > r_flat
    frac = np.clip((R[roll] - r_flat) / max(r_max - r_flat, 1e-9), 0, 1)
    taper[roll] = 0.5 * (1 + np.cos(np.pi * frac))
    taper[R > r_max] = 0.0

    if psf.ndim == 2:
        return psf * taper
    return psf * taper[None, :, :]


def crop_psf_support(psf, support):
    """
    Crop a PSF (2D or per-channel 3D) down to a centered (support, support)
    window before tapering. Combined with `taper_psf`, this controls the
    ISTA/FISTA Lipschitz constant much more effectively than tapering alone
    (measured on the TW Hya beam, 128-pixel crop: L ~ 2.7e4 untapered,
    ~5.7e3 tapered at full 128 support, ~1.7e3 with a 41-pixel support +
    taper) -- a smaller-support kernel is inherently better-conditioned as a
    convolution operator. A 41-pixel window at this dataset's 0.08 arcsec/pix
    still comfortably covers the main lobe and several sidelobe rings
    (~1.6 arcsec radius); the truncated, very-low-level far sidelobes barely
    change the beam's practical effect but disproportionately drive up the
    operator norm.
    """
    py, px = psf.shape[-2:]
    cy, cx = py // 2, px // 2
    half = support // 2
    if psf.ndim == 2:
        return psf[cy - half:cy + half + 1, cx - half:cx + half + 1]
    return psf[:, cy - half:cy + half + 1, cx - half:cx + half + 1]


def estimate_noise_mad(cube, mask=None):
    """
    Robust per-voxel noise std estimate via median absolute deviation,
    scaled by the usual 1.4826 factor (consistent estimator for Gaussian
    noise). Uses the whole (masked) cube rather than a hand-picked
    'empty' patch, since the compact source only occupies a small fraction
    of voxels and the MAD is robust to that minority of high-signal pixels.
    """
    vals = cube[mask] if mask is not None else cube.ravel()
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return 1.4826 * mad


def load_twhya_data(data_dir, crop=128, load_benchmark=True, load_residual=False):
    """
    Load (and centrally crop) the TW Hya dirty cube, per-channel dirty beam,
    and (optionally) the CASA multiscale-CLEAN benchmark/residual for
    comparison.

    Parameters
    ----------
    data_dir : str
        Directory containing twhya_dirty_cube.fits, twhya_psf_cube.fits,
        and (if requested) twhya_clean_benchmark.fits / twhya_residual_cube.fits.
    crop : int or None
        Central spatial crop size (pixels). The TW Hya disk is compact
        relative to the full 256x256/~20.5-arcsec CASA field of view
        (see `data/prep_wavelet_data.py`'s cell size), so a crop keeps the
        real science while cutting compute substantially: at 0.08 arcsec/pix,
        crop=128 keeps a ~10.2 arcsec (~5.1 arcsec radius) field, comfortably
        larger than TW Hya's few-arcsec CO(3-2) disk. Pass None for no crop.
    load_benchmark, load_residual : bool
        Whether to also load the CASA multiscale-CLEAN benchmark image and
        its residual, for comparison against this repo's own deconvolution.

    Returns
    -------
    dict with keys:
        'dirty'      : (nz, ny, nx) dirty cube, Jy/beam, NaN-masked pixels
                       (outside the CASA pblimit) replaced with 0.
        'psf'        : (nz, py, px) per-channel dirty beam, peak-normalized to 1.
        'valid_mask' : (nz, ny, nx) bool, True where the original data was
                       *not* NaN (i.e. inside the primary-beam-limited region).
        'sigma_noise': robust MAD noise estimate over the (cropped) dirty cube.
        'benchmark'  : (nz, ny, nx) CASA multiscale-CLEAN image, or None.
        'residual'   : (nz, ny, nx) CASA CLEAN residual, or None.
        'cell_deg'   : pixel scale in degrees (from the FITS header).
        'restfreq_hz': rest frequency in Hz.
    """
    dirty = _load_cube(os.path.join(data_dir, "twhya_dirty_cube.fits"))
    psf = _load_cube(os.path.join(data_dir, "twhya_psf_cube.fits"))

    with fits.open(os.path.join(data_dir, "twhya_dirty_cube.fits")) as hdul:
        header = hdul[0].header
    cell_deg = abs(float(header.get("CDELT1", np.nan)))
    restfreq_hz = float(header.get("RESTFRQ", np.nan))

    valid_mask = ~np.isnan(dirty)
    dirty = np.nan_to_num(dirty, nan=0.0)

    benchmark = residual = None
    if load_benchmark:
        benchmark = _load_cube(os.path.join(data_dir, "twhya_clean_benchmark.fits"))
        benchmark = np.nan_to_num(benchmark, nan=0.0)
    if load_residual:
        residual = _load_cube(os.path.join(data_dir, "twhya_residual_cube.fits"))
        residual = np.nan_to_num(residual, nan=0.0)

    if crop is not None:
        dirty = _central_crop(dirty, crop)
        psf = _central_crop(psf, crop)
        valid_mask = _central_crop(valid_mask, crop)
        if benchmark is not None:
            benchmark = _central_crop(benchmark, crop)
        if residual is not None:
            residual = _central_crop(residual, crop)

    sigma_noise = estimate_noise_mad(dirty, mask=valid_mask)

    return {
        "dirty": dirty,
        "psf": psf,
        "valid_mask": valid_mask,
        "sigma_noise": sigma_noise,
        "benchmark": benchmark,
        "residual": residual,
        "cell_deg": cell_deg,
        "restfreq_hz": restfreq_hz,
    }
