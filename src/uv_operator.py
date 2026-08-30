#!/usr/bin/env python
"""
Non-uniform Fourier ("uv-plane") measurement operator for interferometric
deconvolution -- the genuine alternative to `deconvolver.py`'s image-domain
dirty-beam convolution, not just a different implementation of the same
thing.

Where this differs from everything else in this repo
------------------------------------------------------
`deconvolver.py` and `clean.py` both work in image (pixel/"Dirac") space:
they take as given a *dirty image* and a *dirty beam*, i.e. the visibilities
have already been gridded onto a regular array and inverse-FFTed exactly
once (that is what CASA's `tclean(..., niter=0)` does to produce
`twhya_dirty_cube.fits`/`twhya_psf_cube.fits`). The forward operator `H` in
that formulation is then a single, fixed, stationary 2D convolution with
that one dirty beam.

Here, the forward operator instead maps a sky-brightness image directly onto
the scattered, irregularly-sampled (u, v) points an interferometer actually
measures, via the van Cittert-Zernike relation

    V(u, v) = integral  I(l, m) * exp(-2*pi*i*(u*l + v*m))  dl dm

(a genuine, non-uniform 2D Fourier transform), and the deconvolution runs
directly against those visibilities -- there is no dirty beam, no gridding,
and no single fixed convolution kernel anywhere in this module. This is the
formulation actual compressed-sensing radio-interferometric imagers
(MORESANE, PURIFY/SARA, and friends -- see the earlier literature-search
discussion in this repo's development) use, typically via an accelerated
non-uniform FFT (NUFFT) for the forward/adjoint operator.

`NonUniformFourierOperator` below implements that same forward/adjoint pair
as an explicit dense matrix instead of an accelerated NUFFT, since the toy
image sizes and uv-point counts used in this repo's demo (thousands of
pixels x thousands of visibilities) are small enough for a dense matrix to
be entirely tractable, and it makes the linear-algebra correctness of the
operator (in particular, its adjoint) trivial to verify directly rather than
trusting a gridding-kernel implementation. Scaling this approach to a real
production-size image (10^5-10^6 pixels, 10^5-10^7 visibilities) would need
a real NUFFT library instead -- see the module docstring note in
`uv_deconvolver.py` for how the real-TW-Hya-data path handles that.
"""

import numpy as np

from psf import ring_antenna_layout, uv_coverage


def toy_uv_points(n_rings=4, antennas_per_ring=7, max_radius=120.0, jitter=0.03,
                   hour_angles_deg=np.linspace(-90, 90, 25), seed=3):
    """
    The exact same mock antenna layout + Earth-rotation-synthesis uv-coverage
    machinery as `psf.mock_alma_dirty_beam_rings`, but returning the
    scattered (u, v) sample points directly instead of gridding them into a
    dirty beam -- the entire point of the uv-plane approach is to *not*
    collapse them onto a fixed image-domain kernel before deconvolving.
    """
    ax, ay = ring_antenna_layout(n_rings, antennas_per_ring, max_radius,
                                  jitter=jitter, seed=seed)
    u, v = uv_coverage(ax, ay, hour_angles_deg)
    return u, v


def pixel_scale_for_uv(u, v, oversample=1.1):
    """
    Nyquist-matched image pixel scale for a given set of (u, v) sample
    points, using the same `extent = oversample * max(|u|, |v|)` convention
    as `psf.dirty_beam_from_uv`'s default gridding -- so a
    `NonUniformFourierOperator` built with this pixel scale covers a
    field of view/resolution directly comparable to the gridded, FFT-based
    dirty beam used elsewhere in this repo (useful for the sanity check in
    the demo notebook: on a *regular* subset of uv points, this operator's
    adjoint should reproduce `psf.dirty_beam_from_uv`'s FFT result).
    """
    extent = oversample * max(np.abs(u).max(), np.abs(v).max())
    return 1.0 / (2.0 * extent)


class NonUniformFourierOperator:
    """
    Dense non-uniform 2D Fourier measurement operator: maps a real
    image (ny, nx) -- the sky model -- to complex visibilities at an
    arbitrary, fixed set of (u, v) points, and back (its adjoint).

    Parameters
    ----------
    ny, nx : int
        Image shape.
    u, v : ndarray, shape (n_uv,)
        Fixed sample points (baseline components), in the same length^-1
        units as `1 / pixel_scale`.
    pixel_scale : float
        Angular size of one image pixel, in the same length units `1/u, 1/v`
        are expressed in (see `pixel_scale_for_uv` for a sensible default).

    The forward matrix is built once at construction (`Fr`, `Fi`, the real
    and imaginary parts of the (n_uv, ny*nx) non-uniform DFT matrix) and
    reused for every `forward`/`adjoint` call -- this is what a NUFFT
    library would instead recompute on the fly via gridding, in exchange for
    not needing O(n_uv * n_pix) memory.
    """

    def __init__(self, ny, nx, u, v, pixel_scale):
        self.ny, self.nx = ny, nx
        self.u = np.asarray(u, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.pixel_scale = float(pixel_scale)
        self.n_uv = self.u.shape[0]

        l = (np.arange(nx) - nx // 2) * pixel_scale   # x/column angular offset
        m = (np.arange(ny) - ny // 2) * pixel_scale   # y/row angular offset
        LL, MM = np.meshgrid(l, m, indexing="xy")      # both shape (ny, nx)

        # phase[k, p] = -2*pi*(u_k * l_p + v_k * m_p), p = flattened (row-major) pixel index
        phase = -2.0 * np.pi * (
            self.u[:, None] * LL.ravel()[None, :] +
            self.v[:, None] * MM.ravel()[None, :]
        )
        F = np.exp(1j * phase)          # (n_uv, n_pix) complex
        self.Fr = F.real.copy()
        self.Fi = F.imag.copy()

    # -- single image (2D) -----------------------------------------------
    def forward(self, x):
        """x: (ny, nx) real -> visibilities (n_uv,) complex."""
        xr = np.asarray(x, dtype=np.float64).ravel()
        return (self.Fr @ xr) + 1j * (self.Fi @ xr)

    def adjoint(self, vis):
        """
        visibilities (n_uv,) complex -> image (ny, nx) real:
        Re(F^H vis) = Fr^T Re(vis) + Fi^T Im(vis), i.e. the "dirty image"
        analogue -- backprojecting the visibilities straight onto the sky,
        with none of the gridding/tapering/weighting a real imager applies.
        """
        vis = np.asarray(vis, dtype=np.complex128)
        img = self.Fr.T @ vis.real + self.Fi.T @ vis.imag
        return img.reshape(self.ny, self.nx)

    # -- per-channel cube (nz, ny, nx), same uv sampling for every channel
    #    (same simplification level `psf.mock_alma_dirty_beam_rings` makes
    #    for the image-domain PSF: no per-channel chromatic uv rescaling) --
    def forward_cube(self, cube):
        """cube: (nz, ny, nx) real -> visibilities (nz, n_uv) complex."""
        nz = cube.shape[0]
        X = cube.reshape(nz, -1)
        Vr = X @ self.Fr.T
        Vi = X @ self.Fi.T
        return Vr + 1j * Vi

    def adjoint_cube(self, vis_cube):
        """visibilities (nz, n_uv) complex -> cube (nz, ny, nx) real."""
        Vr, Vi = vis_cube.real, vis_cube.imag
        img = Vr @ self.Fr + Vi @ self.Fi
        return img.reshape(-1, self.ny, self.nx)

    def lipschitz_constant(self, n_iter=40, seed=0):
        """
        Power iteration for the largest eigenvalue of `adjoint(forward(.))`,
        i.e. the Lipschitz constant L of the data-fidelity gradient here --
        the uv-plane analogue of `deconvolver.lipschitz_constant`'s
        `max|FFT(psf)|^2` shortcut, which has no equivalent closed form once
        the operator is a non-uniform (scattered-point) transform instead of
        a stationary convolution. Reuses `forward`/`adjoint` directly rather
        than forming `Fr^T Fr + Fi^T Fi` explicitly, so it is exactly the
        operator actually used by the FISTA iteration, not an approximation
        of it.
        """
        rng = np.random.default_rng(seed)
        xk = rng.normal(size=(self.ny, self.nx))
        xk /= np.linalg.norm(xk)
        eigval = 0.0
        for _ in range(n_iter):
            yk = self.adjoint(self.forward(xk))
            eigval = float(np.linalg.norm(yk))
            if eigval < 1e-300:
                break
            xk = yk / eigval
        return eigval
