#!/usr/bin/env python
"""
Loader for `data/export_visibilities.py`'s output (`twhya_visibilities.npz`)
-- real, averaged-down TW Hya visibilities, for use with
`uv_operator.NonUniformFourierOperator` / `uv_deconvolver.UVDeconvolver2D1D`.

Status: run and verified against real data, see
`notebooks/real_uv_deconvolution_demo.ipynb`. `export_visibilities.py`
needs CASA's `casatools`, not available in the sandbox this repo's other
real-data tooling runs in (same constraint noted in `io_fits.py`'s and
`run_real_pipeline.py`'s module docstrings) -- it was run in a separate
`casa_env` conda environment to produce `twhya_visibilities.npz`. That run
surfaced two real bugs in `export_visibilities.py` (a raw-vs-regridded
channel index mismatch, and `ms.getdata`'s `axis_info` not reflecting
channel averaging), both fixed and documented in that script directly.

Single-operator (shared-uv) simplification
-------------------------------------------
`export_visibilities.py` saves per-channel (u, v) in wavelength units
(`u_lambda`, `v_lambda`, shape `(n_chan, n_uv)`), since they genuinely do
differ channel to channel with real frequency. Building a separate dense
`NonUniformFourierOperator` *per channel*, though, would need
`n_chan` times the memory and compute of the single-operator toy demo --
for typical real ALMA uv counts/image sizes this becomes impractical for a
dense (non-gridding) operator (see `export_visibilities.py`'s docstring).
This loader instead builds **one shared operator from the central channel's
(u, v)** -- the same simplification level `psf.mock_alma_dirty_beam_rings`
already makes for the image-domain PSF elsewhere in this repo. This is a
reasonable approximation specifically because the 40-channel window here
spans a narrow spectral line (CO(3-2), `width=5` raw channels each, see
`prep_wavelet_data.py`), so the fractional bandwidth -- and hence the
channel-to-channel (u, v) drift in wavelength units -- is small compared to
the array's own antenna-position jitter. A fully accurate per-channel
version (a list of `NonUniformFourierOperator`s, with `UVDeconvolver2D1D`
generalized to accept one operator per channel the way `deconvolver.py`
already accepts a per-channel PSF cube) is the natural next step if that
approximation turns out to matter in practice -- it hasn't been tested
either way yet.
"""

import numpy as np

from uv_operator import NonUniformFourierOperator, pixel_scale_for_uv


def load_twhya_visibilities(npz_path, ny, nx, oversample=1.1):
    """
    Parameters
    ----------
    npz_path : str
        Path to `twhya_visibilities.npz` (see `data/export_visibilities.py`).
    ny, nx : int
        Desired image shape for the `NonUniformFourierOperator` -- unlike
        the image-domain path (`io_fits.load_twhya_data`, crop=128), this
        needs to be chosen small enough to keep the dense operator's
        `(n_uv, ny*nx)` matrix tractable -- see `uv_operator.py`'s and
        `export_visibilities.py`'s docstrings for the scaling argument.

    Returns
    -------
    dict with keys:
        'vis'      : (n_chan, n_uv) complex, the observed visibilities.
        'operator' : NonUniformFourierOperator, built from the central
                     channel's (u, v) -- see module docstring.
        'sigma_vis': per-visibility noise standard deviation estimate.
        'freqs_hz' : (n_chan,) channel frequencies.
    """
    z = np.load(npz_path)
    u_lambda = z["u_lambda"]  # (n_chan, n_uv)
    v_lambda = z["v_lambda"]
    vis = z["vis"]
    freqs_hz = z["freqs_hz"]
    sigma_vis = float(z["sigma_vis"])

    n_chan = u_lambda.shape[0]
    k0 = n_chan // 2
    u0, v0 = u_lambda[k0], v_lambda[k0]

    pixel_scale = pixel_scale_for_uv(u0, v0, oversample=oversample)
    operator = NonUniformFourierOperator(ny, nx, u0, v0, pixel_scale)

    return {
        "vis": vis,
        "operator": operator,
        "sigma_vis": sigma_vis,
        "freqs_hz": freqs_hz,
        "pixel_scale": pixel_scale,
    }
