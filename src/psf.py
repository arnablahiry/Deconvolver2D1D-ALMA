#!/usr/bin/env python
"""
Toy but structurally realistic ALMA-like dirty beam (point spread function).

Simulates the actual origin of a radio-interferometric dirty beam rather than
hand-drawing an Airy-disk-like blob: random antenna positions -> baseline
vectors for every antenna pair -> multiple "hour angle" rotations of the
array to mimic Earth-rotation aperture synthesis -> grid the resulting
(u, v) sample points -> the dirty beam is the inverse Fourier transform of
that (u, v) sampling function.

Because baselines always occur in +/- pairs (antenna i to j and j to i),
the sampling function is automatically point-symmetric about the origin,
so the resulting dirty beam is real-valued and centro-symmetric
(psf[y, x] == psf[-y, -x]). That matters for `deconvolver.py`: it means the
beam-convolution operator is self-adjoint, so the same convolution routine
can be reused for both the forward model and its adjoint in the gradient
step, exactly as it would be for a real ALMA dirty beam derived from a
Hermitian visibility set.
"""

import numpy as np


def random_antenna_layout(n_ant=28, max_radius=120.0, seed=0):
    """Antenna positions drawn from an annulus/disk, loosely mimicking a
    compact ALMA-like configuration (denser toward the center, some antennas
    further out to fill in longer baselines)."""
    rng = np.random.default_rng(seed)
    r = max_radius * np.sqrt(rng.uniform(0.05, 1.0, n_ant))
    theta = rng.uniform(0, 2 * np.pi, n_ant)
    return r * np.cos(theta), r * np.sin(theta)


def uv_coverage(ax, ay, hour_angles_deg=np.linspace(-45, 45, 9)):
    """
    Baselines for every ordered antenna pair, repeated at several rotation
    ("hour") angles of the array to emulate Earth-rotation synthesis.
    Ordered pairs (i, j) and (j, i) are both included, which is what makes
    the (u, v) point set symmetric about the origin.
    """
    n_ant = len(ax)
    us, vs = [], []
    for ha in np.deg2rad(hour_angles_deg):
        c, s = np.cos(ha), np.sin(ha)
        rx = ax * c - ay * s
        ry = ax * s + ay * c
        du = rx[:, None] - rx[None, :]
        dv = ry[:, None] - ry[None, :]
        mask = ~np.eye(n_ant, dtype=bool)
        us.append(du[mask])
        vs.append(dv[mask])
    return np.concatenate(us), np.concatenate(vs)


def dirty_beam_from_uv(u, v, grid_size=81, extent=None, natural_weighting=False):
    """
    Grid (u, v) sample points onto a regular array and inverse-FFT to get the
    dirty beam, normalized to a peak of 1.

    Parameters
    ----------
    natural_weighting : bool
        If True, weight each uv-cell by the number of samples that fall in
        it (natural weighting -> more sensitive, broader/lower sidelobes).
        If False, binary sampling (uniform weighting -> higher resolution,
        more prominent sidelobes) -- closer to what makes CLEAN's job hard
        and is the more interesting case for a deconvolution demo.

    Returns
    -------
    beam : ndarray, shape (grid_size, grid_size), peak-normalized, real.
    sampling : ndarray, shape (grid_size, grid_size)
        The gridded (u, v) sampling function, for visualization.
    """
    if extent is None:
        extent = 1.1 * max(np.abs(u).max(), np.abs(v).max())
    edges = np.linspace(-extent, extent, grid_size + 1)
    counts, _, _ = np.histogram2d(u, v, bins=[edges, edges])
    sampling = counts if natural_weighting else (counts > 0).astype(float)

    beam = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(sampling))).real
    beam = beam / beam.max()
    return beam, sampling


def mock_alma_dirty_beam(n_ant=28, max_radius=120.0,
                          hour_angles_deg=np.linspace(-45, 45, 9),
                          grid_size=81, natural_weighting=False, seed=0):
    """Convenience wrapper: antenna layout -> uv coverage -> dirty beam."""
    ax, ay = random_antenna_layout(n_ant, max_radius, seed=seed)
    u, v = uv_coverage(ax, ay, hour_angles_deg)
    beam, sampling = dirty_beam_from_uv(u, v, grid_size=grid_size,
                                         natural_weighting=natural_weighting)
    return beam, sampling, (ax, ay), (u, v)


def beam_fwhm_pixels(beam):
    """Rough main-lobe FWHM (in pixels) along the row/column through the
    peak, averaged, for a quick sanity check of beam size."""
    cy, cx = np.unravel_index(np.argmax(beam), beam.shape)
    row = beam[cy, :]
    col = beam[:, cx]

    def _fwhm_1d(profile, peak_idx):
        half = profile[peak_idx] / 2.0
        left = peak_idx
        while left > 0 and profile[left] > half:
            left -= 1
        right = peak_idx
        while right < len(profile) - 1 and profile[right] > half:
            right += 1
        return right - left

    return 0.5 * (_fwhm_1d(row, cx) + _fwhm_1d(col, cy))
