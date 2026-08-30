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


def ring_antenna_layout(n_rings=4, antennas_per_ring=7, max_radius=120.0,
                         jitter=0.03, seed=0):
    """
    Antennas placed on a handful of discrete concentric rings instead of at
    continuously random radii. This is what produces the concentric-fringe
    ("target"/Airy-like) dirty beam real ALMA images often show, rather than
    the more speckle-like sidelobe field `random_antenna_layout` gives:
    with only a few distinct radii, the pairwise baseline lengths cluster
    onto a few dominant shells, and (especially once rotated through many
    hour angles, see `uv_coverage`) each shell is close to a fully-sampled
    annulus in the (u, v) plane. The dirty beam is then close to a sum of a
    handful of Bessel-J0-like ring patterns at those baseline lengths,
    which is exactly what produces concentric rings in the image domain
    (this is the same reason a circular aperture's diffraction pattern is
    the Airy pattern -- a filled disk is, among other things, a continuum
    of such rings).

    Parameters
    ----------
    n_rings : int
        Number of discrete radii.
    antennas_per_ring : int
        Antennas placed (evenly, before jitter) around each ring.
    max_radius : float
        Outermost ring radius.
    jitter : float
        Fractional random perturbation of each antenna's radius and angle
        (0 = perfectly regular rings -> very clean, strongly periodic
        fringes; a little jitter breaks the array's exact symmetry so the
        fringe pattern doesn't tile perfectly, closer to how a real,
        imperfectly-regular antenna pad layout looks).
    """
    rng = np.random.default_rng(seed)
    radii = np.linspace(max_radius / n_rings, max_radius, n_rings)
    ax_list, ay_list = [], []
    for r0 in radii:
        thetas = np.linspace(0, 2 * np.pi, antennas_per_ring, endpoint=False)
        thetas = thetas + rng.uniform(-jitter, jitter, antennas_per_ring)
        r = r0 * (1.0 + rng.uniform(-jitter, jitter, antennas_per_ring))
        ax_list.append(r * np.cos(thetas))
        ay_list.append(r * np.sin(thetas))
    return np.concatenate(ax_list), np.concatenate(ay_list)


def azimuthal_ring_coherence(beam, n_bins=40):
    """
    Diagnostic for 'how ring-like vs. speckle-like is this beam': at each
    radius, the ratio |azimuthal mean| / (azimuthal std + eps) of the beam
    values on that ring. High values at many radii = coherent concentric
    fringes (the sidelobe level is nearly constant with azimuth at fixed
    radius); values near 1 = incoherent/speckled sidelobes (as much
    variation *around* a ring as *along* it).
    """
    ny, nx = beam.shape
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2)
    r_max = R.max()
    edges = np.linspace(0, r_max, n_bins + 1)
    coherence = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (R >= edges[i]) & (R < edges[i + 1])
        vals = beam[mask]
        if vals.size < 4:
            continue
        coherence[i] = np.abs(vals.mean()) / (vals.std() + 1e-6)
    return coherence


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


def dirty_beam_from_uv(u, v, grid_size=81, extent=None, natural_weighting=False,
                       weights=None, robust=None):
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
        Ignored when `robust` is given.
    weights : ndarray, shape (n_uv,), optional
        Per-visibility natural weights (e.g. CASA's data weights, exported in
        `twhya_visibilities.npz`). If given, cells accumulate summed weight
        rather than sample counts -- required for a faithful Briggs/robust
        weighting (`robust`), and used as the natural-weighting density
        otherwise. If None, every point is weighted equally (weight 1).
    robust : float, optional
        Briggs robust parameter (Briggs 1995), matching CASA `tclean`'s
        `weighting='briggs', robust=R`. When given, each uv cell `i` is
        weighted by `W_i / (1 + W_i * f^2)` with `W_i` the summed natural
        weight in that cell and

            f^2 = (5 * 10^{-R})^2 / (sum_i W_i^2 / sum_k w_k)

        the standard AIPS/CASA robustness normalization. `robust` large
        (+2) -> natural weighting; small (-2) -> uniform; intermediate values
        (e.g. 0.5) trade resolution against sidelobe level, and reproduce the
        weighting CASA used for the TW Hya benchmark. Overrides
        `natural_weighting`.

    Returns
    -------
    beam : ndarray, shape (grid_size, grid_size), peak-normalized, real.
    sampling : ndarray, shape (grid_size, grid_size)
        The gridded (u, v) sampling function actually used (post-weighting),
        for visualization.
    """
    if extent is None:
        extent = 1.1 * max(np.abs(u).max(), np.abs(v).max())
    edges = np.linspace(-extent, extent, grid_size + 1)

    if weights is None and robust is None:
        counts, _, _ = np.histogram2d(u, v, bins=[edges, edges])
        sampling = counts if natural_weighting else (counts > 0).astype(float)
    else:
        w = np.ones(len(u)) if weights is None else np.asarray(weights, dtype=float)
        Wgrid, _, _ = np.histogram2d(u, v, bins=[edges, edges], weights=w)  # summed natural weight per cell
        if robust is None:
            sampling = Wgrid if natural_weighting else (Wgrid > 0).astype(float)
        else:
            # Briggs/robust: down-weight densely-sampled cells. f^2 sets how
            # aggressively, normalized so robust=+/-2 approach natural/uniform.
            f2 = (5.0 * 10.0 ** (-robust)) ** 2 / (np.sum(Wgrid ** 2) / np.sum(w))
            sampling = Wgrid / (1.0 + Wgrid * f2)

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


def mock_alma_dirty_beam_rings(n_rings=4, antennas_per_ring=7, max_radius=120.0,
                                jitter=0.03,
                                hour_angles_deg=np.linspace(-90, 90, 25),
                                grid_size=81, natural_weighting=False, seed=0):
    """
    Same idea as `mock_alma_dirty_beam`, but using `ring_antenna_layout`
    (a handful of discrete antenna radii) and a wider hour-angle sweep, which
    together give a dirty beam with the concentric-fringe ("target"-like)
    look real ALMA dirty beam images often have, instead of the more
    speckled sidelobe field a fully random antenna layout produces (see the
    `ring_antenna_layout` and `azimuthal_ring_coherence` docstrings).
    """
    ax, ay = ring_antenna_layout(n_rings, antennas_per_ring, max_radius,
                                  jitter=jitter, seed=seed)
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
