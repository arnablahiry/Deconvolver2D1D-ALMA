"""
The sparsity dictionary: a 2D transform across the sky (spatial) times a 1D
transform across velocity/frequency (spectral). A cube is sparse in this
combined basis when most of its energy in the transformed domain concentrates
into a few large coefficients -- that's the assumption compressed-sensing
deconvolution leans on instead of CLEAN's iterative point-source fitting.

Two DIFFERENT wavelets are used, deliberately:

  * SPATIAL (2D): the "a trous" / starlet transform, UNDECIMATED (every
    scale has the same pixel grid as the input). Chosen because it is
    isotropic and shift-invariant, so a source doesn't shift or ring
    differently depending on where it lands in the pixel grid -- important
    for imaging, where source positions are arbitrary.

  * SPECTRAL (1D): the CDF 9/7 biorthogonal wavelet (the JPEG2000
    "irreversible" transform), DECIMATED (each scale is half the length of
    the one before). Chosen because along the spectral axis there is no
    shift-invariance requirement -- channels are a fixed, ordered grid -- so
    a compact, perfect-reconstruction, non-redundant wavelet is the more
    standard, more efficient choice, and it is what most sparse spectral-line
    deconvolution literature actually uses for the velocity axis.

Every forward transform here has an exact inverse (sum of planes for the
starlet; lifting inverse for CDF 9/7) -- both are verified by the
self-tests at the bottom of this file (`python simple/wavelets.py`).
"""

import numpy as np


# ======================================================================
# 2D starlet (spatial), undecimated
# ======================================================================
_B3_SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _smooth_2d(plane, step):
    """Separable convolution with the dyadically-dilated B3-spline kernel."""
    out = plane
    for axis in (0, 1):
        n = out.shape[axis]
        pad = 2 * step
        idx = np.pad(np.arange(n), pad, mode="reflect")
        padded = np.take(out, idx, axis=axis)
        acc = np.zeros_like(out)
        for k, w in enumerate(_B3_SPLINE):
            acc = acc + w * np.take(padded, np.arange(n) + k * step, axis=axis)
        out = acc
    return out


def starlet_2d_forward(cube, num_scales):
    """
    cube: (nz, ny, nx). Decompose each spectral channel's image over `axes
    (1, 2)` into `num_scales` detail planes plus one coarse plane.

    Returns planes, shape (num_scales + 1, nz, ny, nx); `planes.sum(0)`
    reconstructs the input exactly (this transform has no lossy step).
    """
    nz = cube.shape[0]
    planes = np.empty((num_scales + 1,) + cube.shape)
    coarse = cube.copy()
    for j in range(num_scales):
        step = 2 ** j
        smoothed = np.stack([_smooth_2d(coarse[z], step) for z in range(nz)])
        planes[j] = coarse - smoothed
        coarse = smoothed
    planes[num_scales] = coarse
    return planes


def starlet_2d_inverse(planes):
    """Exact inverse of `starlet_2d_forward`: just sum the planes."""
    return planes.sum(axis=0)


# ======================================================================
# 1D CDF 9/7 (spectral), decimated, via the standard lifting scheme
# ======================================================================
# The four lifting-step coefficients and the final scaling factor, as
# standardized for the JPEG2000 "9/7 irreversible" transform.
_ALPHA = -1.586134342059924
_BETA = -0.052980118572961
_GAMMA = 0.882911075530934
_DELTA = 0.443506852043971
_ZETA = 1.230174104914001


def _lift(x, coeff, parity):
    """
    One lifting step along axis 0: add coeff*(left neighbor + right
    neighbor) to every sample of the given parity. Boundaries use
    whole-sample symmetric reflection (numpy's `mode="reflect"`), the
    standard extension for biorthogonal wavelets.
    """
    padded = np.pad(x, [(1, 1)] + [(0, 0)] * (x.ndim - 1), mode="reflect")
    left = padded[0:-2]     # left[i]  == x[i - 1] (reflected at the start)
    right = padded[2:]      # right[i] == x[i + 1] (reflected at the end)
    out = x.copy()
    out[parity::2] += coeff * (left[parity::2] + right[parity::2])
    return out


def _pad_to_even(x):
    """Whole-sample-symmetric pad by one sample along axis 0 if length is odd."""
    if x.shape[0] % 2 == 0:
        return x, False
    return np.pad(x, [(0, 1)] + [(0, 0)] * (x.ndim - 1), mode="reflect"), True


def cdf97_forward_1level(x):
    """One level: (nz, ...) -> approx (ceil(nz/2), ...), detail (nz//2, ...)."""
    x, _ = _pad_to_even(x)
    x = _lift(x, _ALPHA, parity=1)     # predict odd samples from even neighbors
    x = _lift(x, _BETA, parity=0)      # update even samples from odd neighbors
    x = _lift(x, _GAMMA, parity=1)     # predict again (sharper high-pass)
    x = _lift(x, _DELTA, parity=0)     # update again (smoother low-pass)
    approx = x[0::2] / _ZETA
    detail = x[1::2] * _ZETA
    return approx, detail


def cdf97_inverse_1level(approx, detail):
    """Exact inverse of `cdf97_forward_1level`."""
    n = approx.shape[0] + detail.shape[0]
    x = np.empty((n,) + approx.shape[1:])
    x[0::2] = approx * _ZETA
    x[1::2] = detail / _ZETA
    x = _lift(x, -_DELTA, parity=0)
    x = _lift(x, -_GAMMA, parity=1)
    x = _lift(x, -_BETA, parity=0)
    x = _lift(x, -_ALPHA, parity=1)
    return x


def cdf97_forward(x, levels):
    """Multi-level decomposition along axis 0. Returns (details, approx, orig_lens)
    where `details` is a list, fine to coarse, and `orig_lens[i]` is the exact
    pre-padding length at level i (needed to undo the odd-length padding on
    the way back)."""
    details, orig_lens = [], []
    approx = x
    for _ in range(levels):
        orig_lens.append(approx.shape[0])
        approx, detail = cdf97_forward_1level(approx)
        details.append(detail)
    return details, approx, orig_lens


def cdf97_inverse(details, approx, orig_lens):
    """Exact inverse of `cdf97_forward`."""
    for detail, orig_len in zip(reversed(details), reversed(orig_lens)):
        approx = cdf97_inverse_1level(approx, detail)
        approx = approx[:orig_len]
    return approx


# ======================================================================
# Combined 2D (spatial) x 1D (spectral) transform
# ======================================================================
def analyze(cube, num_scales_2d, num_levels_1d):
    """
    Full decomposition: 2D starlet over (y, x), then CDF 9/7 over the
    spectral axis of each resulting plane.

    Returns a list of length `num_scales_2d + 1` (one per spatial scale,
    fine to coarse); each entry is `(details, approx, orig_lens)` as
    returned by `cdf97_forward`, so different spatial scales are decomposed
    independently along the spectral axis.
    """
    planes = starlet_2d_forward(cube, num_scales_2d)
    return [cdf97_forward(plane, num_levels_1d) for plane in planes]


def synthesize(coeffs):
    """Exact inverse of `analyze`."""
    planes = np.stack([cdf97_inverse(*c) for c in coeffs])
    return starlet_2d_inverse(planes)


def soft_threshold_all(coeffs, thresholds, keep_coarsest=True):
    """
    Soft-threshold every detail sub-band coefficient array in `coeffs`
    (mutates a copy, returns it). `thresholds[j2][l]` is the scalar
    threshold for spatial scale `j2`, spectral level `l` (l = len(details)
    is the spectral approx -- the coarsest spectral band of that spatial
    scale, thresholded with `thresholds[j2][-1]` unless it is also the
    single coarsest overall sub-band, in which case `keep_coarsest` decides
    whether to leave it untouched (see `deconvolve.py` for why that matters:
    an interferometer often -- but not always -- has zero response to the
    sky's total flux, and this is the sub-band that carries it).
    """
    out = []
    n2 = len(coeffs)
    for j2, (details, approx, orig_lens) in enumerate(coeffs):
        new_details = []
        for l, d in enumerate(details):
            t = thresholds[j2][l]
            new_details.append(np.sign(d) * np.maximum(np.abs(d) - t, 0.0))
        is_coarsest = j2 == n2 - 1
        if is_coarsest and keep_coarsest:
            new_approx = approx
        else:
            t = thresholds[j2][-1]
            new_approx = np.sign(approx) * np.maximum(np.abs(approx) - t, 0.0)
        out.append((new_details, new_approx, orig_lens))
    return out


def mad_sigma(coeffs):
    """
    Per-sub-band noise estimate: MAD(coefficients) / 0.6745, the usual
    robust-to-outliers (i.e. robust to the source itself) standard-deviation
    estimator for wavelet-domain noise. Returns thresholds[j2][l] in the
    same nested-list shape `soft_threshold_all` expects.
    """
    out = []
    for details, approx, _ in coeffs:
        levels = [_mad(d) for d in details] + [_mad(approx)]
        out.append(levels)
    return out


def _mad(a):
    med = np.median(a)
    return np.median(np.abs(a - med)) / 0.6745


# ======================================================================
# Self-tests: run this file directly to verify perfect reconstruction.
# ======================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    x = rng.normal(size=(37,))
    a, d, ol = cdf97_forward(x, levels=3)
    x_rec = cdf97_inverse(a, d, ol)
    print(f"CDF 9/7 1D round trip, odd length: max err = {np.abs(x - x_rec).max():.2e}")

    cube = rng.normal(size=(90, 24, 24))
    coeffs = analyze(cube, num_scales_2d=4, num_levels_1d=3)
    cube_rec = synthesize(coeffs)
    print(f"2D-1D round trip: max err = {np.abs(cube - cube_rec).max():.2e}")
