"""
The entire deconvolution algorithm, as one readable loop.

Run `python simple/grid_with_casa.py` first to produce simple/data/psf.npy
and dirty.npy (see that file's docstring, and uv_to_image.py's, for how a real
ALMA measurement set becomes those two arrays). Everything below is pure
numpy from there on.

======================================================================
THE OPTIMIZATION PROBLEM
======================================================================
We want the sky image x (Jy/pixel) that best explains the dirty image d
(Jy/beam), penalizing anything not sparse in the 2D-1D wavelet dictionary:

    minimize_x   1/2 || N(x) - d ||^2   +   lambda * || W(x) ||_1
                 ________________            _____________
                 data fidelity:                sparsity prior:
                 x must reproduce               most wavelet coefficients
                 the dirty image when            of the true sky should be
                 measured the same way            ~zero (compact/sparse
                 the real array measured           structure, not noise)
                 the true sky (see uv_to_image.py)

`N(x)` is the normal operator from uv_to_image.py: convolving x with the PSF is
mathematically the same as degridding x into visibilities and gridding the
result back, so this line is really "x must look like the dirty image once
it's been through the same measurement process".

This is solved with FISTA (Beck & Teboulle 2009): alternate a gradient step
on the smooth data term with a "proximal" step on the non-smooth L1 term,
which for L1 is just soft-thresholding in the wavelet domain. FISTA adds a
momentum term (`t`, `z`) that makes this converge in O(1/k^2) instead of
O(1/k).

======================================================================
WHY THE FINAL MODEL HAS LOWER AMPLITUDE THAN THE DIRTY IMAGE
======================================================================
Two genuinely different reasons, and only one of them is a defect:

1. UNITS. `dirty.npy` is in Jy/BEAM: every pixel already represents flux
   integrated over the beam's solid angle. The model `x` this script
   produces is in Jy/PIXEL: undiluted point-by-point sky brightness. There
   are ~15 pixels per beam here (pi * bmaj * bmin / (4 ln 2) / cell^2), so
   for extended emission the model's peak in Jy/pixel is naturally much
   smaller than the dirty peak in Jy/beam -- they are not the same
   quantity. The apples-to-apples check is `N(x)` (which IS in Jy/beam,
   since it re-applies the beam) against `dirty_image`, not `x` against
   `dirty_image` directly. This script prints that check every iteration.

2. L1 SHRINKAGE BIAS. Plain soft-thresholding does two jobs when it zeroes
   small coefficients: it *selects* which structure survives (wanted), and
   it *shrinks every surviving coefficient by the same amount* (an
   unwanted side effect -- the proximal operator of the L1 norm is
   literally `sign(a) * max(|a| - T, 0)`, which subtracts T from
   everything that survives). Measured on this dataset: with plain soft
   thresholding, `N(model)` reached only 79% of the true dirty peak -- a
   large real bias, not a units artifact.

   The fix implemented below is REWEIGHTED L1 (Candes, Wakin, Boyd 2008):
   after an initial "burn-in" using a fixed threshold T, later iterations
   use a threshold `T / (|previous coefficient| / T + eps)` instead of a
   flat T -- so a coefficient already many multiples of T above the noise
   is barely shrunk, while a coefficient near zero is still crushed. This
   removed essentially all of the bias here (`N(model)/dirty` peak ratio:
   0.79 without reweighting -> 0.99 with it, printed every 5 iterations
   below).
"""

import numpy as np

from uv_to_image import ShiftInvariantOperator
from wavelets import analyze, synthesize, soft_threshold_all, mad_sigma


# ---------------------------------------------------------------- config --
PSF_PATH = "simple/data/psf.npy"
DIRTY_PATH = "simple/data/dirty.npy"
OUT_PATH = "simple/data/model.npy"

NUM_SCALES_2D = 4        # spatial starlet detail scales
NUM_LEVELS_1D = 3         # spectral CDF-9/7 decomposition levels
N_ITER = 60
K_SIGMA = 4.0             # detection threshold, in units of the noise sigma
BURN_IN_ITERS = 20        # plain soft-threshold iterations before reweighting
REWEIGHT_EPS = 1e-2       # reweighting floor (see module docstring, part 2)
POSITIVITY = True         # project the model to x >= 0 every iteration

# The exact operator (exact_operator.ExactOperator) computes the gradient on
# the REAL visibility residual, `Vis - A(x)`, via one CASA degrid+grid pass
# per iteration -- see exact_operator.py's docstring for exactly why that is
# unavoidable for a mosaic (a single fixed PSF cannot represent 3 pointings'
# distinct primary beams). It reproduces the fast operator's result to
# 0.55% rms, with the difference concentrated at the mosaic's outer edge,
# and costs roughly 60-140s PER ITERATION instead of ~0.2s -- 60 iterations
# is ~5 minutes with the fast operator, ~1-2 HOURS with the exact one.
USE_EXACT_OPERATOR = False


def keep_coarsest_band(psf, lipschitz):
    """
    Should the very coarsest wavelet sub-band (smoothest in both space and
    spectrum) be zeroed instead of fit?

    An interferometer never samples the true zero-length baseline, so it
    can be completely blind to the sky's total flux -- if so, that sub-band
    is an unconstrained direction the solver could run away in, and must be
    zeroed. But whether that is actually true here is a property of the
    IMAGE GRID, not just the array: the (u, v) cell size is 1/field_of_view,
    so a small enough field makes even the shortest real baseline fall
    inside the sampled central cell. Check directly rather than assume:
    """
    dc_response = np.abs(psf.sum(axis=(1, 2))).max()
    measured = dc_response > 1e-3 * lipschitz
    print(f"[setup] zero-spacing response |sum(PSF)| = {dc_response:.2f} "
          f"({'MEASURED -> keep coarsest band' if measured else 'ZERO -> drop coarsest band'})")
    return not measured


def main():
    psf = np.load(PSF_PATH)
    dirty = np.load(DIRTY_PATH)

    # The fast operator is always built: it supplies the Lipschitz constant
    # (exact for itself; a very close, ~0.55%-rms-accurate stand-in for the
    # exact operator, whose true constant would cost ~20 major cycles to
    # measure directly) and the zero-spacing check either way.
    fast_operator = ShiftInvariantOperator(psf, dirty)
    keep_coarsest = keep_coarsest_band(psf, fast_operator.lipschitz)

    if USE_EXACT_OPERATOR:
        from exact_operator import ExactOperator
        operator = ExactOperator(dirty, lipschitz_estimate=fast_operator.lipschitz)
        print("[setup] using the EXACT operator: one real CASA degrid+grid "
              "pass per iteration (slow, mosaic-accurate)")
    else:
        operator = fast_operator
        print("[setup] using the FAST operator: PSF convolution via FFT "
              "(instant, exact for a single pointing, ~0.55% rms off at "
              "this mosaic's edge)")
    step_size = 1.0 / operator.lipschitz

    # Noise level per wavelet sub-band, estimated ONCE from the dirty image
    # itself and held fixed for the whole run. Re-estimating it every
    # iteration from the shrinking residual (tempting, and what the
    # `uv_deconvolver.py` version of this pipeline originally did) is
    # degenerate: as the fit improves the residual shrinks, so the
    # threshold shrinks, so more coefficients pass, so the residual shrinks
    # further -- a runaway whose fixed point is fitting pure noise. Fixing
    # the threshold to the data's own noise level (computed here, from the
    # dirty image, before any fitting) removes that runaway.
    # NOTE the `step_size *` here: thresholds are compared against `v = z -
    # step_size * grad` every iteration (see the loop below), which is
    # `step_size` times smaller than the raw dirty image. Estimating sigma
    # from `dirty` directly instead of `step_size * dirty` makes every
    # threshold ~1/step_size too large -- everything gets zeroed on
    # iteration 0, x never leaves zero, and the loop silently does nothing
    # forever (residual_rms flat, active=0, every iteration identical).
    noise_coeffs = analyze(step_size * dirty, NUM_SCALES_2D, NUM_LEVELS_1D)
    sigma = mad_sigma(noise_coeffs)                      # sigma[j2][l]
    thresholds = [[K_SIGMA * s for s in levels] for levels in sigma]

    x = np.zeros_like(dirty)          # the model, Jy/pixel
    z = x.copy()                      # FISTA's momentum variable
    t = 1.0
    prev_coeffs = None                # for reweighting: previous model's coefficients

    for it in range(N_ITER):
        # --- 1. gradient step on the smooth data term ---------------------
        grad = operator.gradient(z)              # N(z) - dirty
        v = z - step_size * grad
        residual = -grad                          # dirty - N(z), for diagnostics

        # --- 2. proximal step: analyze, threshold, synthesize -------------
        coeffs = analyze(v, NUM_SCALES_2D, NUM_LEVELS_1D)
        if it >= BURN_IN_ITERS:
            # reweighted thresholds: T / (|prev coeff| / T + eps), per
            # sub-band, computed against last iteration's model
            rw_thresholds = []
            for j2, (details, approx, _) in enumerate(coeffs):
                p_details, p_approx, _ = prev_coeffs[j2]
                levels = []
                for l, (T, p) in enumerate(zip(thresholds[j2][:-1], p_details)):
                    levels.append(T / (np.abs(p) / T + REWEIGHT_EPS))
                T = thresholds[j2][-1]
                levels.append(T / (np.abs(p_approx) / T + REWEIGHT_EPS))
                rw_thresholds.append(levels)
            new_coeffs = _soft_threshold_reweighted(coeffs, rw_thresholds, keep_coarsest)
        else:
            new_coeffs = soft_threshold_all(coeffs, thresholds, keep_coarsest)

        x_new = synthesize(new_coeffs)
        if POSITIVITY:
            np.maximum(x_new, 0.0, out=x_new)
        prev_coeffs = analyze(x_new, NUM_SCALES_2D, NUM_LEVELS_1D)

        # --- 3. FISTA momentum ---------------------------------------------
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

        if it % 5 == 0 or it == N_ITER - 1:
            # Reuse `residual` (= dirty - N(z), already computed this
            # iteration for free) instead of a fresh `operator.apply(x)`
            # call -- for the exact operator that would be one extra
            # ~60-140s CASA major cycle per printout, purely for a diagnostic.
            peak_ratio = (dirty - residual).max() / dirty.max()
            print(f"iter {it:3d}  residual_rms={residual.std():.4e}  "
                  f"flux={x.sum():8.2f} Jy  active={np.count_nonzero(x)}  "
                  f"N(model)/dirty peak={peak_ratio:.3f}")

    if USE_EXACT_OPERATOR:
        operator.done()

    np.save(OUT_PATH, x)
    print(f"\nsaved {OUT_PATH}, shape {x.shape}")
    print(f"model is in Jy/pixel; N(model) [= model convolved with the PSF, "
          f"in Jy/beam] should closely match dirty.npy if the fit converged")


def _soft_threshold_reweighted(coeffs, rw_thresholds, keep_coarsest):
    out = []
    n2 = len(coeffs)
    for j2, (details, approx, orig_lens) in enumerate(coeffs):
        new_details = [np.sign(d) * np.maximum(np.abs(d) - t, 0.0)
                       for d, t in zip(details, rw_thresholds[j2][:-1])]
        if j2 == n2 - 1 and keep_coarsest:
            new_approx = approx
        else:
            t = rw_thresholds[j2][-1]
            new_approx = np.sign(approx) * np.maximum(np.abs(approx) - t, 0.0)
        out.append((new_details, new_approx, orig_lens))
    return out


if __name__ == "__main__":
    main()
