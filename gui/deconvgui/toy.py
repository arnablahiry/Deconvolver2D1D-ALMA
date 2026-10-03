"""
A synthetic example dataset, so the whole GUI can be tried without CASA or
real data: a rotating, clumpy disc observed with a random 12m-like (u, v)
coverage, giving a dirty cube with realistic sidelobes and correlated noise.
"""

import numpy as np


def make_toy(ny=128, nz=40, cell_arcsec=0.1, n_ant=40, max_baseline_px=None, seed=1, noise=0.02,
             hours=6.0, dec=-30.0):
    rng = np.random.default_rng(seed)
    nx = ny
    # ---- sky: inclined rotating disc + clumps, Jy/pixel --------------------
    yy, xx = np.mgrid[:ny, :nx] - ny / 2
    inc, pa = np.radians(55), np.radians(30)
    xr = xx * np.cos(pa) + yy * np.sin(pa)
    yr = (-xx * np.sin(pa) + yy * np.cos(pa)) / np.cos(inc)
    r = np.hypot(xr, yr)
    phi = np.arctan2(yr, xr)
    surf = np.exp(-r / (ny / 12)) + 0.6 * np.exp(-((r - ny / 6) ** 2) / (2 * (ny / 40) ** 2))
    for _ in range(12):
        cx, cy = rng.normal(0, ny / 7, 2)
        surf += rng.uniform(0.3, 1.0) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.5 ** 2))
    vrot = 200 * np.tanh(r / 6) * np.cos(phi) * np.sin(inc)          # km/s
    v = np.linspace(-260, 260, nz)
    sigma_v = 25.0
    sky = surf[None] * np.exp(-((v[:, None, None] - vrot[None]) ** 2) / (2 * sigma_v ** 2))
    sky *= 0.05 / sky.max()

    # ---- PSF from an Earth-rotation-synthesis (u, v) coverage, 2x the image --
    # n_ant antennas in a centrally condensed layout, tracked over +-hours
    # (6 h by default) at declination dec: dense, ALMA-like coverage with a
    # natural-weighting density. A single snapshot would leave large holes in
    # the (u, v) plane, i.e. sky components the data do not constrain at all.
    P = 2 * ny
    if max_baseline_px is None:
        max_baseline_px = P // 5
    r_ant = max_baseline_px / 2.0 * np.sqrt(rng.uniform(0, 1, n_ant)) ** 1.2
    th = rng.uniform(0, 2 * np.pi, n_ant)
    ant = np.c_[r_ant * np.cos(th), r_ant * np.sin(th)]
    i, j = np.triu_indices(n_ant, 1)
    bx, by = (ant[i] - ant[j]).T
    sd = np.sin(np.radians(dec))
    grid = np.zeros((P, P))
    for H in np.radians(np.linspace(-15 * hours / 2, 15 * hours / 2, 60)):
        u = bx * np.cos(H) - by * np.sin(H)
        w = sd * (bx * np.sin(H) + by * np.cos(H))
        for s in (1, -1):
            np.add.at(grid, (np.round(s * w).astype(int) % P, np.round(s * u).astype(int) % P), 1.0)
    grid[0, 0] = 0.0                                    # no zero spacing (no single dish)
    psf2d = np.fft.fftshift(np.real(np.fft.ifft2(grid)))
    psf2d /= psf2d.max()
    psf = np.repeat(psf2d[None], nz, axis=0)

    # ---- dirty = N(sky) + noise -------------------------------------------
    # Dirty-image noise of natural weighting has covariance N itself, i.e. a
    # power spectrum proportional to the (u, v) weight density (the OTF), not
    # to its square (which is what convolving white noise with the PSF gives).
    import os, sys
    from scipy.fft import rfft2, irfft2
    from .store import SIMPLE_DIR
    if SIMPLE_DIR not in sys.path:
        sys.path.insert(0, SIMPLE_DIR)
    from uv_to_image import ShiftInvariantOperator
    op = ShiftInvariantOperator(psf, np.zeros((nz, ny, nx)))
    clean = op.apply(sky)
    white = rng.normal(0, 1, (nz, op.pad_y, op.pad_x))
    n = irfft2(rfft2(white, axes=(-2, -1)) * np.sqrt(np.clip(op.otf.real, 0, None)),
               s=(op.pad_y, op.pad_x), axes=(-2, -1))[:, :ny, :nx]
    dirty = clean + noise * clean.max() * n / n.std()

    # clean-beam estimate: FWHM of the PSF main lobe along north (rows) and east (columns)
    c = P // 2
    fy = 2 * np.argmax(psf2d[c:, c] < 0.5) * cell_arcsec
    fx = 2 * np.argmax(psf2d[c, c:] < 0.5) * cell_arcsec
    beam = [max(fx, fy), min(fx, fy), 0.0 if fy >= fx else 90.0]
    meta = dict(cell_arcsec=cell_arcsec, beam=beam, width_kms=float(v[1] - v[0]),
                start_kms=float(v[0]), nchan=nz, imsize=ny, restfreq_ghz=230.538,
                source="toy", note="Synthetic rotating disc; true sky saved as truth.npy")
    return dirty.astype(np.float32), psf.astype(np.float32), sky.astype(np.float32), meta
