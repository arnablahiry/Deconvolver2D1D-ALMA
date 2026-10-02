"""
2D-1D wavelet deconvolution as a proper convex problem, solved with a
primal-dual algorithm -- the replacement for deconvolve.py's iteration.

======================================================================
THE PROBLEM
======================================================================
    minimize_x   1/2 x^T N x - d^T x   +   sum_b lambda_b || W_b x ||_1   subject to x >= 0
                 ___________________       ___________________________
                 = 1/2 ||A x - y||_W^2      analysis L1 over the 2D-1D
                   + const: the natural-    dictionary (2D starlet x 1D
                   weighting visibility     CDF 9/7), one weight per
                   chi^2 (gradient N x - d, sub-band b
                   uv_to_image.py)

x is the sky model (Jy/pixel), d the dirty cube, N = A^H W A the normal
operator (PSF convolution). The data term is unchanged from deconvolve.py.

======================================================================
WHY NOT deconvolve.py's ITERATION
======================================================================
deconvolve.py soft-thresholds the model's own coefficients in a redundant,
non-orthogonal starlet frame, then clips negatives, reweights and adds FISTA
momentum. That combination is not the proximal operator of any objective,
so it has no fixed point to converge to. Measured on ground-truth
simulations with the real PSFs (and the real extended noise):
  * its result swings from a pedestal filling the field (k = 4, 3.7x the
    true flux) to almost nothing (k = 6, 0.08x): no usable threshold;
  * soft-thresholding weak NEGATIVE lobes of coarse starlet coefficients
    leaves the coarse plane uncancelled -> smooth positive halos, which the
    extended array (blind to its largest scales) never corrects;
  * the same scheme with a masked update (multiresolution support)
    diverges past ~60 iterations.
Here the analysis L1 and the positivity are handled exactly -- the L1
through its dual variable, positivity as a projection -- so the iteration
converges to the minimizer of the problem above (Condat 2013; Vu 2013), the
same class of solver as SARA / PURIFY.

======================================================================
THE WEIGHTS lambda_b
======================================================================
lambda_b = k_sigma * sigma_b, sigma_b the MAD, in wavelet space, of the
NOISE's coefficients in sub-band b (per 2D scale x 1D level):
  * noise = a realization shaped like the real dirty-cube noise: white noise
    filtered by sqrt(OTF), so its power spectrum follows the (u, v) weight
    density (natural weighting), scaled to the dirty cube's robust pixel
    sigma. On the real cubes the noise at 0.3-5" scales is 10-200x what
    WHITE noise of the same pixel sigma predicts -- thresholds that ignore
    this fit that correlated noise as emission;
  * floored at the white-noise MAD of the same sub-band: in sub-bands the
    PSF does not sample, the correlated noise (and the data) have ~no power,
    so without a floor lambda_b ~ 0 exactly where the data constrain
    nothing -- the original arc/knot artifacts.

======================================================================
THE ITERATION (Condat-Vu, relaxation 1)
======================================================================
    x+ = P_{x>=0}( x - tau (N x - d + W^T u) )
    u+ = clip( u + sigma W (2 x+ - x),  -lambda, +lambda )
with tau = 1 / beta (beta = the largest eigenvalue of N on the field, see
field_lipschitz) and
sigma = beta / (2 ||W||^2), which satisfies 1/tau - sigma ||W||^2 >= beta/2.
W^T is the exact adjoint (wavelets.analyze_adjoint), ||W||^2 is measured by
power iteration. Optional reweighted L1: every `reweight_every` iterations
after `burn_in_iters`, lambda -> lambda / (|W x| / lambda + eps) per
coefficient.
"""

import numpy as np
from scipy.fft import rfft2, irfft2

from uv_to_image import ShiftInvariantOperator
from wavelets import analyze, analyze_adjoint, mad_sigma, max_scales_2d, max_levels_1d
from deconvolve import real_noise_ratios


# ---------------------------------------------------------------- config --
N_ITER = 300              # maximum iterations (stops earlier at `tol`)
K_SIGMA = 3.0             # lambda_b = K_SIGMA x noise MAD of sub-band b
NUM_SCALES_2D = None      # spatial starlet scales; None = deepest for the cube
NUM_LEVELS_1D = None      # spectral CDF-9/7 levels; None = deepest for the cube
REWEIGHT = False          # reweighted L1 after BURN_IN_ITERS
BURN_IN_ITERS = 100
REWEIGHT_EVERY = 25
REWEIGHT_EPS = 1e-2
POSITIVITY = True
TOL = 1e-4                # stop when ||x+ - x|| / ||x+|| < TOL


def _mad(a):
    return 1.4826 * float(np.median(np.abs(a - np.median(a))))


def noise_sigmas(operator, dirty, num_scales_2d, num_levels_1d, seed=0):
    """Per-sub-band noise level sigma_b (see module docstring): MAD of the
    coefficients of correlated noise shaped like the data's, floored at the
    MAD of white noise, both at the dirty cube's robust pixel sigma."""
    nz, ny, nx = dirty.shape
    s_pix = _mad(dirty)
    rng = np.random.default_rng(seed)
    white_big = rng.normal(size=(nz, operator.pad_y, operator.pad_x))
    shaped = irfft2(rfft2(white_big, axes=(-2, -1)) * np.sqrt(np.clip(operator.otf.real, 0.0, None)),
                    s=(operator.pad_y, operator.pad_x), axes=(-2, -1))[:, :ny, :nx]
    shaped *= s_pix / shaped.std()
    sig_shaped = mad_sigma(analyze(shaped, num_scales_2d, num_levels_1d))
    del shaped, white_big
    sig_white = mad_sigma(analyze(rng.normal(0.0, s_pix, dirty.shape), num_scales_2d, num_levels_1d))
    return [[max(a, b) for a, b in zip(la, lb)] for la, lb in zip(sig_shaped, sig_white)], s_pix


def analysis_norm_sq(shape, num_scales_2d, num_levels_1d, n_power=60, seed=0):
    """||W||^2 = largest eigenvalue of W^T W, by power iteration. It converges
    slowly (15 steps under-estimate it by ~10%, enough to break the step-size
    condition), hence 60 steps and a 10% margin. The top eigenvector is a
    smooth interior mode, so the norm does not depend on the image size once
    the deepest scale fits: it is measured on a field just large enough."""
    nz, ny, nx = shape
    side = min(ny, nx, 4 * 2 ** (num_scales_2d - 1) + 9)
    v = np.random.default_rng(seed).normal(size=(nz, side, side))
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(n_power):
        w = analyze_adjoint(analyze(v, num_scales_2d, num_levels_1d))
        lam = float(np.linalg.norm(w))
        v = w / lam
    return lam * 1.1


def field_lipschitz(operator, shape, n_power=40, margin=1.15, seed=0):
    """Largest eigenvalue of N restricted to the image field, by power
    iteration (x `margin`, since power iteration approaches it from below),
    capped at max|OTF| -- the rigorous bound, but 1.7-2.5x too loose for a
    2x PSF here, which would shrink every step by that much."""
    v = np.random.default_rng(seed).normal(size=shape)
    lam = 0.0
    for _ in range(n_power):
        w = operator.apply(v)
        lam = float(np.linalg.norm(w) / np.linalg.norm(v))
        v = w / np.linalg.norm(w)
    return min(lam * margin, operator.lipschitz)


def deconvolve(psf, dirty, operator=None, n_iter=N_ITER, k_sigma=K_SIGMA,
               num_scales_2d=NUM_SCALES_2D, num_levels_1d=NUM_LEVELS_1D,
               reweight=REWEIGHT, burn_in_iters=BURN_IN_ITERS, reweight_every=REWEIGHT_EVERY,
               reweight_eps=REWEIGHT_EPS, positivity=POSITIVITY, tol=TOL, noise_channels=None,
               verbose=True, print_every=10, log=print, label=""):
    """
    Same interface as deconvolve.deconvolve: `psf` and `dirty` (nz, ny, nx),
    PSF peak 1. Returns (x, history): the model in Jy/pixel and one dict per
    iteration (iter, residual_rms, flux, active, peak_ratio, rel_change).
    """
    def say(msg):
        if verbose:
            log(f"{label}{msg}")

    fast = ShiftInvariantOperator(psf, dirty)
    if operator is None:
        operator = fast
    nz, ny, nx = dirty.shape
    if num_scales_2d is None:
        num_scales_2d = max_scales_2d(ny, nx)
    if num_levels_1d is None:
        num_levels_1d = max_levels_1d(nz)

    beta = field_lipschitz(operator, dirty.shape) if operator is fast else operator.lipschitz
    if noise_channels is None:
        sigmas, s_pix = noise_sigmas(fast, dirty, num_scales_2d, num_levels_1d)
    else:
        # the REAL noise per spatial scale (line-free channels), white-floored
        s_pix = _mad(dirty)
        white = mad_sigma(analyze(np.random.default_rng(0).normal(0.0, s_pix, dirty.shape),
                                  num_scales_2d, num_levels_1d))
        ratios = real_noise_ratios(dirty, noise_channels, num_scales_2d)
        sigmas = [[s * max(1.0, r) for s in levels] for levels, r in zip(white, ratios)]
        say(f"[setup] real/white noise per spatial scale (from {len(noise_channels)} line-free "
            f"channels): " + " ".join(f"{r:.1f}" for r in ratios))
    lam = [[k_sigma * s for s in levels] for levels in sigmas]          # lam[j2][l]
    norm_sq = analysis_norm_sq(dirty.shape, num_scales_2d, num_levels_1d)
    tau, sig = 1.0 / beta, beta / (2.0 * norm_sq)
    say(f"[setup] {num_scales_2d} starlet scales x {num_levels_1d} CDF 9/7 levels; pixel noise "
        f"{s_pix:.3e}; lambda = {k_sigma} x sub-band noise MAD: {min(min(l) for l in lam):.2e} .. "
        f"{max(max(l) for l in lam):.2e}; ||W||^2 = {norm_sq:.3f}; tau = {tau:.3e}, sigma = {sig:.3e}; "
        f"up to {n_iter} iterations (tol {tol}), reweighting "
        f"{f'from {burn_in_iters} every {reweight_every}' if reweight else 'off'}, positivity={positivity}")

    x = np.zeros_like(dirty, dtype=np.float64)
    u = analyze(np.zeros_like(x), num_scales_2d, num_levels_1d)         # dual variable, starts at 0
    # bounds of the dual variable: |u| <= lambda (scalars per sub-band until reweighted)
    bounds = [(lv[:-1], lv[-1]) for lv in lam]
    history = []
    for it in range(n_iter):
        grad = operator.gradient(x)                       # N x - d
        x_new = x - tau * (grad + analyze_adjoint(u))
        if positivity:
            np.maximum(x_new, 0.0, out=x_new)

        if reweight and it >= burn_in_iters and (it - burn_in_iters) % reweight_every == 0:
            wx = analyze(x_new, num_scales_2d, num_levels_1d)
            bounds = [([L / (np.abs(d) / L + reweight_eps) for d, L in zip(det, lv[:-1])],
                       lv[-1] / (np.abs(ap) / lv[-1] + reweight_eps))
                      for (det, ap, _), lv in zip(wx, lam)]
            del wx

        v = analyze(2.0 * x_new - x, num_scales_2d, num_levels_1d)
        u = [([np.clip(ud + sig * vd, -b, b) for ud, vd, b in zip(du, dv, bd)],
              np.clip(ua + sig * va, -ba, ba), ol)
             for (du, ua, ol), (dv, va, _), (bd, ba) in zip(u, v, bounds)]
        del v

        residual = -grad                                  # d - N x, at the previous x
        peak_ratio = (dirty - residual).max() / dirty.max()
        rel_change = float(np.linalg.norm(x_new - x) / max(np.linalg.norm(x_new), 1e-30))
        history.append(dict(iter=it, residual_rms=float(residual.std()), flux=float(x_new.sum()),
                            active=int(np.count_nonzero(x_new)), peak_ratio=float(peak_ratio),
                            rel_change=rel_change))
        x = x_new
        if it % print_every == 0 or it == n_iter - 1:
            say(f"iter {it:3d}  residual_rms={residual.std():.4e}  flux={x.sum():8.2f} Jy  "
                f"active={np.count_nonzero(x)}  N(model)/dirty peak={peak_ratio:.3f}  "
                f"change={rel_change:.2e}")
        if it > 10 and rel_change < tol:
            say(f"converged at iter {it}: relative change {rel_change:.2e} < tol {tol}")
            break

    return x, history
