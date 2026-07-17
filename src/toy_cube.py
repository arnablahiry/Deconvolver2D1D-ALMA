#!/usr/bin/env python
"""
Toy spatially-varying (non-stationary) spectral cube: a rotating ring.

Unlike the point/box sources used in Denoiser2D1D-improved's toy training
data (see `experiment/toy_train_compare.py`'s `make_cube`, or the fuller
Sersic-disk generator in `src/data/toy_cube_dataset.py`), the emission line
profile here is *not* the same shape at every spatial pixel: each point on
the ring has its own line-of-sight velocity set by a simple rotation field,
so the spectral line center v(x, y) varies continuously across the source
(hence "non-stationary"). This is what makes a single, spatially connected
source (the ring) appear as two spatially separated blobs in some velocity
channels: at a fixed channel velocity v0, only the two points on the ring
whose local velocity equals v0 light up, and for most v0 those are two
distinct locations (near +/-90 degrees around the ring, symmetric about the
kinematic minor axis). The two blobs merge into one only right at the
velocity extremes (+-v_max), at the top/bottom of the ring.

This is the same basic morphology as the classic "double-horn" / spider
diagram seen in ALMA channel maps of rotating disks and rings.
"""

import numpy as np


def rotating_ring_cube(nz=41, ny=80, nx=80, v_max=220.0, line_sigma=18.0,
                        ring_radius=11.0, ring_sigma=1.6, peak_flux=1.0,
                        vel_range_factor=1.3):
    """
    Parameters
    ----------
    nz, ny, nx : int
        Spectral (velocity channels) and spatial dimensions.
    v_max : float
        Peak rotation speed (km/s) of the ring.
    line_sigma : float
        Intrinsic + instrumental line width (km/s) at each spatial pixel.
    ring_radius, ring_sigma : float
        Ring radius and radial (Gaussian) thickness, in pixels.
    peak_flux : float
        Peak flux density of the (noiseless, beam-unconvolved) cube.
    vel_range_factor : float
        Velocity axis spans +/- vel_range_factor * v_max.

    Returns
    -------
    cube : ndarray, shape (nz, ny, nx)
        True (noiseless, unconvolved) sky cube.
    velocities : ndarray, shape (nz,)
        Channel velocities in km/s.
    v_los : ndarray, shape (ny, nx)
        The underlying spatially-varying line-of-sight velocity field, for
        diagnostic plotting.
    """
    y = np.arange(ny) - (ny - 1) / 2.0
    x = np.arange(nx) - (nx - 1) / 2.0
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    radial_profile = np.exp(-0.5 * ((R - ring_radius) / ring_sigma) ** 2)
    v_los = v_max * np.sin(theta)  # spatially-varying (non-stationary) kinematics

    velocities = np.linspace(-vel_range_factor * v_max, vel_range_factor * v_max, nz)

    # Broadcast: (nz, 1, 1) velocities against (ny, nx) v_los field.
    line = np.exp(-0.5 * ((velocities[:, None, None] - v_los[None, :, :]) / line_sigma) ** 2)
    cube = peak_flux * radial_profile[None, :, :] * line

    return cube, velocities, v_los


def count_blobs(image, threshold):
    """
    Trivial 4-connectivity connected-component count above `threshold`, used
    as a quick, dependency-free sanity check that a channel map shows one
    vs. two separated blobs (no scipy.ndimage available in this repo).

    Returns
    -------
    n_blobs : int
    labels : ndarray, same shape as image, 0 = background, 1..n = blobs.
    """
    mask = image > threshold
    labels = np.zeros(image.shape, dtype=int)
    n_labels = 0
    ny, nx = image.shape
    for y0 in range(ny):
        for x0 in range(nx):
            if mask[y0, x0] and labels[y0, x0] == 0:
                n_labels += 1
                stack = [(y0, x0)]
                labels[y0, x0] = n_labels
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny2, nx2 = y + dy, x + dx
                        if 0 <= ny2 < ny and 0 <= nx2 < nx and mask[ny2, nx2] and labels[ny2, nx2] == 0:
                            labels[ny2, nx2] = n_labels
                            stack.append((ny2, nx2))
    return n_labels, labels
