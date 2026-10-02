"""
Restoration: turning the deconvolved model back into an image with a finite,
well-behaved resolution.

`deconvolve.py` returns a model x in Jy/PIXEL: the sparsest sky consistent
with the data, with no beam at all. Displaying it as-is over-interprets it --
structure finer than the data actually constrain looks as real as structure
they do. The standard fix (the "R" in CLEAN's restore step) is to convolve x
with a *clean beam*: an elliptical Gaussian, peak 1, usually fitted to the
main lobe of the dirty beam. With a peak-1 kernel, the result is in Jy/BEAM
of that beam -- the same units as the dirty image, but without the dirty
beam's sidelobes.

The clean beam does not have to be the data's own. Convolving a model with a
beam *smaller* than its own dirty-beam main lobe asks what the model says at
finer resolution than the data were taken at -- only as trustworthy as the
sparsity prior that produced the model -- which is exactly the comparison
the deconvolution notebook makes between each galaxy's compact
configuration (restored with the extended configuration's beam) and its
extended one.

Position angle convention (CASA / FITS): measured from north through east.
With arrays laid out (row = Dec increasing upward, col = RA pixel, RA
increasing to the LEFT, i.e. east = decreasing column index), a +PA rotates
the major axis from +row toward -col.
"""

import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2


def gaussian_beam(shape, cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg, center=None):
    """
    Peak-1 elliptical Gaussian on a (ny, nx) grid, FWHMs bmaj x bmin, position
    angle bpa (north through east). Centred on pixel `center` = (row, col),
    default (ny // 2, nx // 2) -- the phase-centre pixel of a CASA image of
    even size.
    """
    ny, nx = shape
    cy, cx = (ny // 2, nx // 2) if center is None else center
    rows, cols = np.mgrid[:ny, :nx]
    north = (rows - cy) * cell_arcsec
    east = -(cols - cx) * cell_arcsec           # RA increases to the left
    pa = np.deg2rad(bpa_deg)
    along_major = east * np.sin(pa) + north * np.cos(pa)
    along_minor = east * np.cos(pa) - north * np.sin(pa)
    return np.exp(-4.0 * np.log(2.0) * ((along_major / bmaj_arcsec) ** 2
                                        + (along_minor / bmin_arcsec) ** 2))


def convolve_with_beam(cube, cell_arcsec, bmaj_arcsec, bmin_arcsec, bpa_deg):
    """
    Convolve every channel of `cube` (nz, ny, nx), Jy/pixel, with the peak-1
    clean beam -> Jy/beam of that beam. Linear (zero-padded) FFT
    convolution, with the kernel's peak placed at pixel (0, 0) so the result
    is not shifted.
    """
    cube = np.asarray(cube, dtype=np.float64)
    nz, ny, nx = cube.shape
    pad_y = next_fast_len(2 * ny, real=True)
    pad_x = next_fast_len(2 * nx, real=True)

    kernel = gaussian_beam((pad_y, pad_x), cell_arcsec, bmaj_arcsec, bmin_arcsec,
                           bpa_deg, center=(pad_y // 2, pad_x // 2))
    kernel = np.roll(kernel, (-(pad_y // 2), -(pad_x // 2)), axis=(0, 1))
    kernel_ft = rfft2(kernel)

    big = np.zeros((nz, pad_y, pad_x))
    big[:, :ny, :nx] = cube
    out = irfft2(rfft2(big, axes=(-2, -1)) * kernel_ft, s=(pad_y, pad_x), axes=(-2, -1))
    return out[:, :ny, :nx]


if __name__ == "__main__":
    # Self-test: a unit point source restores to exactly the beam, peak 1 at
    # the source pixel, and a +45 deg beam is elongated along NE-SW (up-left).
    cell = 0.1
    cube = np.zeros((1, 64, 64))
    cube[0, 32, 32] = 1.0
    restored = convolve_with_beam(cube, cell, 1.0, 0.5, 45.0)[0]
    beam = gaussian_beam((64, 64), cell, 1.0, 0.5, 45.0)
    print(f"point source -> beam: max err = {np.abs(restored - beam).max():.2e}, "
          f"peak {restored.max():.3f} at {np.unravel_index(restored.argmax(), restored.shape)}")
    ne, nw = beam[32 + 3, 32 - 3], beam[32 + 3, 32 + 3]   # (up, left) = NE; (up, right) = NW
    print(f"PA 45: response NE {ne:.3f} > NW {nw:.3f}: {ne > nw}")
