"""
The measurement operator: how a sky image relates to interferometer
visibilities, and how that relation is applied in the deconvolution loop.

======================================================================
THE PHYSICS, IN ONE EQUATION
======================================================================
An interferometer measures visibilities V(u, v) -- one complex number per
antenna pair per instant -- related to the true sky brightness I(l, m) by

    V(u, v) = Integral  I(l, m) * exp(-2*pi*i*(u*l + v*m))  dl dm

i.e. V is literally the 2D Fourier transform of the sky, sampled only at the
scattered (u, v) points the antennas happen to measure that day (baseline
vector / wavelength, in units of wavelengths). That scatter -- not a full,
regular Fourier grid -- is the entire reason interferometric imaging is a
non-trivial inverse problem, rather than a single inverse FFT.

======================================================================
GRIDDING: TURNING SCATTERED SAMPLES INTO SOMETHING FFT-ABLE
======================================================================
To use an FFT at all, the scattered visibilities first have to be resampled
("gridded") onto a regular (u, v) grid. Conceptually:

  1. Build the "sampling function" W(u, v): a regular grid the same size as
     the image, valued at each cell by the sum of imaging weights of every
     visibility that fell in it (zero where nothing was measured -- and
     nothing is EVER measured at u=v=0, the "zero-spacing" baseline, because
     no antenna sits on top of another). This is what `im.weight()` /
     `synthesisimager.setweighting()` compute, and it depends only on the
     (u, v) coverage and chosen weighting scheme (this dataset: Briggs,
     robust=0.5), not on the sky.

  2. Convolve the scattered visibilities onto that grid with a small
     interpolation kernel (this is the actual "gridding" step -- CASA calls
     it once per major cycle, in `synthesisimager.executemajorcycle`).

  3. Divide by W to compensate for the uneven sample density, then inverse
     FFT the gridded, weighted visibilities:

         dirty_image = IFFT2( W(u, v) * V_gridded(u, v) )

That produces the "dirty image": the true sky I convolved with a single
fixed kernel, the "dirty beam" or point spread function (PSF):

         dirty_image = I  (*)  PSF                 (convolution theorem)
         PSF         = IFFT2( W(u, v) )             (the response to a POINT source)

PSF = IFFT2(W) is exactly the response you get by feeding a single Dirac
delta (unit point source, flat spectrum) through the same gridding process --
which is why CASA can compute the PSF once, cheaply, from the (u, v)
coverage alone.

======================================================================
WHAT THIS MODULE ACTUALLY DOES
======================================================================
Gridding real ALMA visibilities (steps 1-3 above) needs CASA's own gridder --
that part is done once, offline, by `grid_with_casa.py`, which calls
`casatools.synthesisimager` to write out two cubes:

    psf.npy    (nz, ny, nx)  -- the PSF, peak-normalized to 1.0 per channel
    dirty.npy  (nz, ny, nx)  -- the dirty image, Jy/beam

Everything below this point never touches a visibility again. Instead it
uses the convolution-theorem identity above directly: since

    dirty_image = I (*) PSF        (exact, for a single un-mosaicked pointing)

the same relation holds for *any* trial image x during the solver's search,
not just the true sky I:

    N(x)  :=  x (*) PSF  =  IFFT2( OTF * FFT2(x) ),    OTF := FFT2(PSF)

`N` is the "normal operator" A^H W A (measure x, then grid the result back
into an image) -- computing it via one forward + one inverse FFT is
mathematically identical to a full CASA degrid-then-grid major cycle, and
about a million times faster, because no scattered-point interpolation is
needed once the OTF has been precomputed. This equivalence is what every
line below implements; nothing here approximates the physics beyond the
same single-pointing/shift-invariant assumption already baked into using one
fixed PSF per channel.

The solver's gradient step, every iteration, is just:

    residual  = dirty_image - N(x)          # image domain, no visibilities
    gradient  = -residual                    # since N is self-adjoint
"""

import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2


def image_to_uv(image_padded):
    """
    Step 1 of N: image -> its 2D Fourier transform (rfft2 for real input).

    This *is* the van Cittert-Zernike relation from the module docstring,
    V(u,v) = FT[I(l,m)], evaluated on every (u, v) grid cell at once instead
    of only the scattered points a real antenna pair measures.
    """
    return rfft2(image_padded, axes=(-2, -1))


def apply_sampling_response(uv_plane, otf):
    """
    Step 2: multiply by the transfer function OTF = FFT(PSF).

    This single multiplication stands in for "grid the visibilities, weight
    them, divide by the weight sum" -- OTF already contains the (u, v)
    coverage, the Briggs weighting, and the correct normalization, because
    it was derived from the real, CASA-gridded PSF. Where OTF is near zero
    (baselines nobody measured), this multiplication destroys that spatial
    frequency -- exactly as a real missing baseline would.
    """
    return uv_plane * otf


def uv_to_image(uv_plane, out_shape):
    """Step 3: inverse FFT back to the image domain -- the gridded dirty image."""
    return irfft2(uv_plane, axes=(-2, -1), s=out_shape)


class ShiftInvariantOperator:
    """
    The normal operator N = A^H W A, applied per spectral channel via the
    three-step image -> uv -> image round trip above.

    Construction pads every channel by `pad` (default 2x) before the FFT
    so the *linear* convolution with the PSF is exact -- an un-padded FFT
    convolution is circular and would wrap the PSF's sidelobes around the
    edges of the field.
    """

    def __init__(self, psf, dirty_image, pad=2.0):
        psf = np.asarray(psf, dtype=np.float64)
        dirty_image = np.asarray(dirty_image, dtype=np.float64)
        assert psf.shape == dirty_image.shape
        self.nz, self.ny, self.nx = psf.shape
        self.dirty_image = dirty_image

        # Re-normalize the PSF to peak exactly 1.0 per channel: that peak
        # defines the Jy/pixel <-> Jy/beam unit relation the whole solver
        # relies on (see deconvolve.py's docstring on units).
        peak = psf.max(axis=(1, 2), keepdims=True)
        self.psf = psf / peak

        self.pad_y = next_fast_len(int(self.ny * pad), real=True)
        self.pad_x = next_fast_len(int(self.nx * pad), real=True)

        # Move the PSF's peak to pixel (0, 0) before transforming, so that
        # convolving with it introduces no spatial shift.
        py, px = np.unravel_index(np.argmax(self.psf[0]), self.psf[0].shape)
        big_psf = np.zeros((self.nz, self.pad_y, self.pad_x))
        big_psf[:, : self.ny, : self.nx] = self.psf
        big_psf = np.roll(big_psf, (-py, -px), axis=(1, 2))
        self.otf = image_to_uv(big_psf)

        # max|OTF| is the exact Lipschitz constant of N -- since N is a pure
        # convolution, its eigenvalues ARE the OTF values, no power
        # iteration needed.
        self.lipschitz = float(np.abs(self.otf).max())

    def apply(self, x):
        """N(x): image -> uv -> image, per channel. See module docstring."""
        big = np.zeros((self.nz, self.pad_y, self.pad_x))
        big[:, : self.ny, : self.nx] = x
        uv = image_to_uv(big)
        uv = apply_sampling_response(uv, self.otf)
        out = uv_to_image(uv, (self.pad_y, self.pad_x))
        return out[:, : self.ny, : self.nx]

    def residual(self, x):
        """d - N(x): what's left of the dirty image once x is subtracted."""
        return self.dirty_image - self.apply(x)

    def gradient(self, x):
        """d/dx of (1/2)||N(x) - d||^2 = N(x) - d = -residual(x)."""
        return -self.residual(x)
