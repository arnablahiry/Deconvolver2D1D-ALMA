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

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# Threads for the independent per-channel / per-plane work below: numpy
# releases the GIL inside these large array operations, so this parallelizes
# with results bit-identical to a serial run. Env SIMPLE_NUM_THREADS overrides.
NUM_THREADS = int(os.environ.get("SIMPLE_NUM_THREADS", 8))
_pool = None


def _map(fn, items):
    global _pool
    if NUM_THREADS <= 1:
        return list(map(fn, items))
    if _pool is None:
        _pool = ThreadPoolExecutor(NUM_THREADS)
    return list(_pool.map(fn, items))


# ======================================================================
# 2D starlet (spatial), undecimated
# ======================================================================
_B3_SPLINE = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


def _smooth_2d(plane, step):
    """Separable convolution with the dyadically-dilated B3-spline kernel,
    reflect boundaries. The 5 taps are read as slice views of one padded
    copy per axis (no per-tap copies)."""
    out = plane
    for axis in (0, 1):
        n = out.shape[axis]
        width = [(0, 0), (0, 0)]
        width[axis] = (2 * step, 2 * step)
        padded = np.pad(out, width, mode="reflect")
        acc = np.zeros_like(out)
        for k, w in enumerate(_B3_SPLINE):
            window = [slice(None), slice(None)]
            window[axis] = slice(k * step, k * step + n)
            acc += w * padded[tuple(window)]
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
        smoothed = np.empty_like(coarse)

        def smooth_channel(z):
            smoothed[z] = _smooth_2d(coarse[z], step)

        _map(smooth_channel, range(nz))
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
    One lifting step along axis 0, IN PLACE on `x` (even length): add
    coeff*(left neighbor + right neighbor) to every sample of the given
    parity. Boundaries use whole-sample symmetric reflection (numpy's
    `mode="reflect"`: x[-1] -> x[1], x[n] -> x[n-2]), the standard extension
    for biorthogonal wavelets. The neighbours of one parity are all of the
    other parity, which this step does not change -- so updating in place
    is exact, and saves the pad + copy of the whole array a functional
    version needs.
    """
    even, odd = x[0::2], x[1::2]
    if parity == 1:                  # odd i: left = x[i-1], right = x[i+1]
        pair = np.empty_like(even)
        np.add(even[:-1], even[1:], out=pair[:-1])
        np.add(even[-1:], even[-1:], out=pair[-1:])     # x[n] -> x[n-2]
        odd += coeff * pair
    else:                            # even i: left = x[i-1], right = x[i+1]
        pair = np.empty_like(odd)
        np.add(odd[:1], odd[:1], out=pair[:1])           # x[-1] -> x[1]
        np.add(odd[:-1], odd[1:], out=pair[1:])
        even += coeff * pair


def _pad_to_even(x):
    """Whole-sample-symmetric pad by one sample along axis 0 if length is odd."""
    if x.shape[0] % 2 == 0:
        return x, False
    return np.pad(x, [(0, 1)] + [(0, 0)] * (x.ndim - 1), mode="reflect"), True


def cdf97_forward_1level(x):
    """One level: (nz, ...) -> approx (ceil(nz/2), ...), detail (nz//2, ...)."""
    x, padded = _pad_to_even(x)
    if not padded:
        x = x.copy()                   # the lifting steps below work in place
    _lift(x, _ALPHA, parity=1)         # predict odd samples from even neighbors
    _lift(x, _BETA, parity=0)          # update even samples from odd neighbors
    _lift(x, _GAMMA, parity=1)         # predict again (sharper high-pass)
    _lift(x, _DELTA, parity=0)         # update again (smoother low-pass)
    approx = x[0::2] / _ZETA
    detail = x[1::2] * _ZETA
    return approx, detail


def cdf97_inverse_1level(approx, detail):
    """Exact inverse of `cdf97_forward_1level`."""
    n = approx.shape[0] + detail.shape[0]
    x = np.empty((n,) + approx.shape[1:])
    x[0::2] = approx * _ZETA
    x[1::2] = detail / _ZETA
    _lift(x, -_DELTA, parity=0)
    _lift(x, -_GAMMA, parity=1)
    _lift(x, -_BETA, parity=0)
    _lift(x, -_ALPHA, parity=1)
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


def max_scales_2d(ny, nx):
    """Deepest starlet for a (ny, nx) image: scale j smooths with the B3
    kernel dilated by 2**j, support 4 * 2**j + 1 pixels, which must still fit
    inside the image (beyond that, reflect-padding just folds the image onto
    itself). 800 px -> 8 scales, the last one 128-256 px."""
    return int(np.floor(np.log2((min(ny, nx) - 1) / 4))) + 1


def max_levels_1d(nz):
    """Deepest CDF 9/7 decomposition of `nz` channels: halve (rounding up)
    until the approximation is a single channel. 32 channels -> 5 levels."""
    return max(1, int(np.ceil(np.log2(nz))))


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
    return _map(lambda plane: cdf97_forward(plane, num_levels_1d), planes)


def synthesize(coeffs):
    """Exact inverse of `analyze`."""
    planes = np.stack(_map(lambda c: cdf97_inverse(*c), coeffs))
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
# Adjoints (transposes) of the forward transforms, for solvers that need
# W^T -- e.g. the primal-dual solver in deconvolve_pd.py. These are NOT the
# inverses above: the starlet and CDF 9/7 are redundant / biorthogonal, so
# W^T != W^{-1}. Verified by the dot-product test at the bottom of this file.
# ======================================================================
def _smooth_1d_adjoint(y, step, axis):
    """Transpose of one axis of `_smooth_2d`: scatter each tap back, then
    fold the reflect-padded margins onto the samples they were copied from
    (numpy "reflect": index -q -> q, index n-1+q -> n-1-q)."""
    n = y.shape[axis]
    p = 2 * step

    def along(a, sl):
        index = [slice(None)] * a.ndim
        index[axis] = sl
        return a[tuple(index)]

    shape = list(y.shape)
    shape[axis] = n + 2 * p
    z = np.zeros(shape, dtype=y.dtype)
    for k, w in enumerate(_B3_SPLINE):
        along(z, slice(k * step, k * step + n))[...] += w * y
    out = along(z, slice(p, p + n)).copy()
    along(out, slice(1, p + 1))[...] += np.flip(along(z, slice(0, p)), axis=axis)
    along(out, slice(n - 1 - p, n - 1))[...] += np.flip(along(z, slice(p + n, 2 * p + n)), axis=axis)
    return out


def _smooth_2d_adjoint(plane, step):
    return _smooth_1d_adjoint(_smooth_1d_adjoint(plane, step, 1), step, 0)


def starlet_2d_adjoint(planes):
    """Transpose of `starlet_2d_forward`: (num_scales + 1, nz, ny, nx) -> (nz, ny, nx).
    Forward: c_{j+1} = H_j c_j, w_j = c_j - c_{j+1}, coarse = c_J; so the
    transpose runs back down the scales: g_j = w_j + H_j^T (g_{j+1} - w_j)."""
    num_scales = planes.shape[0] - 1
    g = planes[num_scales].copy()
    for j in reversed(range(num_scales)):
        step = 2 ** j
        diff = g - planes[j]
        smoothed = np.empty_like(diff)

        def smooth_channel(z):
            smoothed[z] = _smooth_2d_adjoint(diff[z], step)

        _map(smooth_channel, range(diff.shape[0]))
        g = planes[j] + smoothed
    return g


def _lift_adjoint(x, coeff, parity):
    """Transpose of `_lift` (in place). `_lift(parity=1)` is odd += c P even,
    so its transpose is even += c P^T odd, and vice versa for parity 0 --
    with P, Q the reflect-boundary neighbour-sum matrices of `_lift`."""
    even, odd = x[0::2], x[1::2]
    m = even.shape[0]
    if parity == 1:                  # even += c P^T odd;  P: (P e)[i] = e[i] + e[i+1], last = 2 e[m-1]
        tr = odd.copy()              # the "e[i]" term of every row
        if m > 1:
            tr[1:] += odd[:-1]       # the "e[i+1]" term of row i lands on column i+1
        tr[-1] += odd[-1]            # last row counts e[m-1] twice
        even += coeff * tr
    else:                            # odd += c Q^T even;  Q: (Q o)[0] = 2 o[0], (Q o)[i] = o[i-1] + o[i]
        tr = even.copy()             # the "o[i]" term
        if m > 1:
            tr[:-1] += even[1:]      # the "o[i-1]" term of row i lands on column i-1
        tr[0] += even[0]             # row 0 counts o[0] twice
        odd += coeff * tr


def cdf97_adjoint_1level(approx, detail, orig_len):
    """Transpose of `cdf97_forward_1level` for an input of length `orig_len`."""
    n = approx.shape[0] + detail.shape[0]
    x = np.empty((n,) + approx.shape[1:], dtype=approx.dtype)
    x[0::2] = approx / _ZETA
    x[1::2] = detail * _ZETA
    _lift_adjoint(x, _DELTA, parity=0)
    _lift_adjoint(x, _GAMMA, parity=1)
    _lift_adjoint(x, _BETA, parity=0)
    _lift_adjoint(x, _ALPHA, parity=1)
    if orig_len == n:
        return x
    out = x[:orig_len].copy()        # undo _pad_to_even: x[n-1] was a copy of x[orig_len - 2]
    out[max(orig_len - 2, 0)] += x[orig_len]
    return out


def cdf97_adjoint(details, approx, orig_lens):
    """Transpose of `cdf97_forward`."""
    for detail, orig_len in zip(reversed(details), reversed(orig_lens)):
        approx = cdf97_adjoint_1level(approx, detail, orig_len)
    return approx


def analyze_adjoint(coeffs):
    """Transpose W^T of `analyze` (W): coefficients -> cube."""
    planes = np.stack(_map(lambda c: cdf97_adjoint(*c), coeffs))
    return starlet_2d_adjoint(planes)


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

    # dot-product test of the adjoints: <W x, u> == <x, W^T u>
    for shape, J, L in (((16, 40, 40), 3, 4), ((33, 64, 48), 4, 6), ((1, 21, 21), 2, 1)):
        xx = rng.normal(size=shape)
        wx = analyze(xx, J, L)
        u = [([rng.normal(size=d.shape) for d in det], rng.normal(size=ap.shape), ol) for det, ap, ol in wx]
        lhs = sum((d * e).sum() for (dw, aw, _), (du, au, _) in zip(wx, u) for d, e in zip(dw + [aw], du + [au]))
        rhs = (xx * analyze_adjoint(u)).sum()
        print(f"adjoint dot-product test {shape} J={J} L={L}: relative mismatch {abs(lhs - rhs) / abs(lhs):.1e}")
