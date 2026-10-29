"""
Beam helpers shared by the comparison figures: a cube restored with one Gaussian beam brought to a larger one (e.g. an
extended CLEAN residual to the compact clean beam), using convolve_with_beam's own axis and PA conventions.
Needs simple/ on sys.path (restore.convolve_with_beam).
"""
import numpy as np
from scipy.signal import fftconvolve

from restore import convolve_with_beam


def omega(b):
    """solid angle of beam b = (bmaj, bmin, bpa), arcsec^2"""
    return np.pi * b[0] * b[1] / (4 * np.log(2))


def cov(b, cell):
    """pixel covariance of beam b, measured on convolve_with_beam's own kernel"""
    h = int(np.ceil(3 * b[0] / cell)) + 8
    d = np.zeros((1, 2 * h + 1, 2 * h + 1)); d[0, h, h] = 1.0
    g = convolve_with_beam(d, cell, *b)[0]; g /= g.sum()
    yy, xx = np.indices(g.shape) - h
    return np.array([[(g * yy * yy).sum(), (g * yy * xx).sum()], [(g * yy * xx).sum(), (g * xx * xx).sum()]])


def to_beam(cube, b_from, b, cell):
    """`cube` (nchan, ny, nx) in Jy/beam of b_from -> Jy/beam of b: convolved with the Gaussian whose covariance is
    the difference of the two beams' (clipped to be positive), then x omega_b / omega_from"""
    if tuple(b) == tuple(b_from):
        return cube
    S = cov(b, cell) - cov(b_from, cell)
    w, V = np.linalg.eigh(S); S = V @ np.diag(np.clip(w, 1e-3, None)) @ V.T
    h = int(np.ceil(4 * np.sqrt(S.diagonal().max())))
    r = np.stack(np.indices((2 * h + 1, 2 * h + 1)) - h, -1)
    k = np.exp(-0.5 * np.einsum("...i,ij,...j", r, np.linalg.inv(S), r)); k /= k.sum()
    return np.stack([fftconvolve(c, k, mode="same") for c in cube]) * omega(b) / omega(b_from)
