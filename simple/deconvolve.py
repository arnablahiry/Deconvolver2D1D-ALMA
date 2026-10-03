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
from wavelets import (analyze, synthesize, soft_threshold_all, mad_sigma, _mad,
                      max_scales_2d, max_levels_1d, starlet_2d_forward)


# ---------------------------------------------------------------- config --
PSF_PATH = "simple/data/psf.npy"
DIRTY_PATH = "simple/data/dirty.npy"
OUT_PATH = "simple/data/model.npy"

NUM_SCALES_2D = None      # spatial starlet detail scales; None = deepest for the cube
NUM_LEVELS_1D = None      # spectral CDF-9/7 levels; None = deepest for the cube
N_ITER = 60
K_SIGMA = 4.0             # detection threshold, in units of the noise sigma
BURN_IN_ITERS = 20        # plain soft-threshold iterations before reweighting
REWEIGHT = True           # reweighted L1 after BURN_IN_ITERS (False: plain soft-thresholding throughout)
REWEIGHT_EPS = 1e-2       # reweighting floor (see module docstring, part 2)
POSITIVITY = True         # project the model to x >= 0 every iteration

# Which reweighting to use once BURN_IN_ITERS is passed:
#   "threshold"  -- fold the weight into the threshold (original; debiases
#                   well but loosens detection, which rings on noiseless data)
#   "bias-only"  -- keep detection at the plain K_SIGMA threshold and reweight
#                   only the shrinkage (see _soft_threshold_bias_corrected)
#   "off"        -- plain soft-thresholding throughout
REWEIGHT_MODE = "threshold"

# Extra unpenalized gradient steps after the main loop, restricted to the
# FIXED support the main loop converged to (x > 0 at the last iteration).
# This removes L1 shrinkage bias -- the same problem reweighting is for --
# without either of reweighting's failure modes: it cannot admit new
# coefficients (the support is frozen, taken from whichever run this follows,
# reweighted or not), so it cannot ring on a noiseless or low-S/N problem the
# way reweighting does. It also cannot fix a support that was wrong to begin
# with -- run this after a clean (e.g. non-reweighted) main loop, not instead
# of getting detection right. 0 = off (default; existing behaviour unchanged).
DEBIAS_ITERS = 0

# The exact operator (exact_operator.ExactOperator) computes the gradient on
# the REAL visibility residual, `Vis - A(x)`, via one CASA degrid+grid pass
# per iteration -- see exact_operator.py's docstring for exactly why that is
# unavoidable for a mosaic (a single fixed PSF cannot represent 3 pointings'
# distinct primary beams). It reproduces the fast operator's result to
# 0.55% rms, with the difference concentrated at the mosaic's outer edge,
# and costs roughly 60-140s PER ITERATION instead of ~0.2s -- 60 iterations
# is ~5 minutes with the fast operator, ~1-2 HOURS with the exact one.
USE_EXACT_OPERATOR = False


def real_noise_ratios(dirty, noise_channels, num_scales_2d, seed=0):
    """
    How much stronger the REAL noise is than white noise of the same pixel
    sigma, per 2D starlet scale: MAD of each spatial plane of the line-free
    `noise_channels` (per-pixel mean over those channels removed, i.e. the
    spectrally flat continuum), divided by the same for white noise.

    Real interferometric noise is correlated: measured on these cubes it is
    10-60x (extended) and up to 200x (compact) above white at 0.3-5" scales
    -- and noise shaped from the PSF's power spectrum alone still falls short
    of the real large-scale noise by ~1.5-4x (extended). Thresholds that
    miss this let that noise through as diffuse "emission". Returns one ratio
    per spatial plane (num_scales_2d details + coarse).
    """
    noise = dirty[list(noise_channels)]
    noise = noise - noise.mean(axis=0, keepdims=True)
    s_pix = 1.4826 * float(np.median(np.abs(noise - np.median(noise))))
    white = np.random.default_rng(seed).normal(0.0, s_pix, noise.shape)
    mad = lambda a: float(np.median(np.abs(a - np.median(a))))
    return [mad(a) / mad(b) for a, b in zip(starlet_2d_forward(noise, num_scales_2d),
                                            starlet_2d_forward(white, num_scales_2d))]


def keep_coarsest_band(psf, lipschitz, log=print):
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
    log(f"[setup] zero-spacing response |sum(PSF)| = {dc_response:.2f} "
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

    # Pass the module globals explicitly: deconvolve()'s defaults are bound at
    # import time, so callers that set e.g. `deconvolve.K_SIGMA` and then call
    # main() would otherwise be silently ignored.
    x, _ = deconvolve(psf, dirty, operator=operator, n_iter=N_ITER, k_sigma=K_SIGMA,
                      num_scales_2d=NUM_SCALES_2D, num_levels_1d=NUM_LEVELS_1D,
                      reweight=REWEIGHT, reweight_mode=REWEIGHT_MODE,
                      burn_in_iters=BURN_IN_ITERS, reweight_eps=REWEIGHT_EPS,
                      positivity=POSITIVITY, debias_iters=DEBIAS_ITERS)

    if USE_EXACT_OPERATOR:
        operator.done()

    np.save(OUT_PATH, x)
    print(f"\nsaved {OUT_PATH}, shape {x.shape}")
    print(f"model is in Jy/pixel; N(model) [= model convolved with the PSF, "
          f"in Jy/beam] should closely match dirty.npy if the fit converged")


def deconvolve(psf, dirty, operator=None, n_iter=N_ITER, k_sigma=K_SIGMA,
               num_scales_2d=NUM_SCALES_2D, num_levels_1d=NUM_LEVELS_1D,
               reweight=REWEIGHT, reweight_mode=REWEIGHT_MODE, burn_in_iters=BURN_IN_ITERS, reweight_eps=REWEIGHT_EPS,
               positivity=POSITIVITY, noise_channels=None, max_scale_px=None,
               debias_iters=DEBIAS_ITERS, verbose=True, print_every=5, log=print, label=""):
    """
    The solver loop on in-memory cubes: `psf` and `dirty`, both (nz, ny, nx),
    PSF peak-normalized with its peak at the same pixel in every channel.
    `operator` defaults to the fast `ShiftInvariantOperator(psf, dirty)`.

    With `verbose`, the setup and every `print_every`-th iteration are sent
    to `log` (default print), each line prefixed by `label` -- e.g. a
    queue's `put` to stream progress out of a worker process.

    `reweight_mode` picks the reweighting once `burn_in_iters` is passed (see
    REWEIGHT_MODE); `debias_iters` > 0 runs the fixed-support amplitude refit
    after the main loop (see DEBIAS_ITERS).

    Returns (x, history): the model cube in Jy/pixel, and one dict per
    iteration with the diagnostics.
    """
    def say(msg):
        if verbose:
            log(f"{label}{msg}")

    if operator is None:
        operator = ShiftInvariantOperator(psf, dirty)
    keep_coarsest = keep_coarsest_band(psf, operator.lipschitz, log=say)
    step_size = 1.0 / operator.lipschitz
    nz, ny, nx = dirty.shape
    if num_scales_2d is None:
        num_scales_2d = max_scales_2d(ny, nx)
    if num_levels_1d is None:
        num_levels_1d = max_levels_1d(nz)
    if reweight_mode not in ("threshold", "bias-only", "off"):
        raise ValueError(f"reweight_mode must be 'threshold', 'bias-only' or 'off', got {reweight_mode!r}")
    reweight = reweight and reweight_mode != "off"

    # Largest recoverable scale: with `max_scale_px` (the array's LRS, in
    # pixels), every starlet plane whose central scale 2**(j + 1/2) px lies
    # beyond it -- and the coarse plane, centred at 2**(J + 1/2) -- is ZEROED
    # every iteration, not thresholded. The data measure nothing there (no
    # baselines short enough), so any flux in those planes is prior-made; and
    # soft-thresholding them is worse than useless: it removes their weak
    # negative lobes and leaves a smooth positive halo/plateau. The planes
    # that remain are zero-mean band-passes, which cannot build a pedestal.
    dropped = [False] * (num_scales_2d + 1)
    if max_scale_px is not None:
        dropped = [2 ** (j + 0.5) > max_scale_px for j in range(num_scales_2d + 1)]

    # Thresholds are fixed ONCE, before any fitting. Re-estimating them every
    # iteration from the shrinking residual is degenerate: as the fit
    # improves the residual shrinks, so the threshold shrinks, so more
    # coefficients pass, so the residual shrinks further -- a runaway whose
    # fixed point is fitting pure noise.
    #
    # Threshold of sub-band b = k_sigma * sigma_img * ||psi_b||: the
    # image-domain noise level of the gradient step, times that sub-band's
    # response to WHITE noise of unit variance -- i.e. one uniform lambda
    # (k_sigma * sigma_img) on the normalized dictionary.
    #
    # NOT each sub-band's own MAD of the dirty cube, which this solver used
    # to do and which fails both ways at once (measured on a ground-truth
    # simulation with the real NGC 3110 compact PSF: 24 sigma residuals AND
    # 10x the true fine-scale energy):
    #   * in sub-bands the PSF does not sample, the dirty cube has ~no power,
    #     so the threshold was ~0 exactly where the data constrain nothing --
    #     anything the iteration put there (positivity-clipping edges,
    #     momentum overshoot) was never removed: contour-like arcs and knots;
    #   * in sub-bands the source dominates, the MAD measures the source, not
    #     the noise, so thresholds were inflated and real flux was withheld.
    # sigma_img is robust (MAD) over the whole cube, where the source fills
    # few voxels. `step_size *` because thresholds are compared against
    # `v = z - step_size * grad`, i.e. step_size times the dirty cube's units.
    sigma_img = step_size * 1.4826 * np.median(np.abs(dirty - np.median(dirty)))
    white = np.random.default_rng(0).normal(0.0, sigma_img, dirty.shape)
    sigma = mad_sigma(analyze(white, num_scales_2d, num_levels_1d))   # sigma[j2][l]
    del white
    # With line-free `noise_channels`: raise each spatial scale's thresholds
    # to the REAL noise there (never below the white floor).
    if noise_channels is not None:
        ratios = real_noise_ratios(dirty, noise_channels, num_scales_2d)
        sigma = [[s * max(1.0, r) for s in levels] for levels, r in zip(sigma, ratios)]
        say(f"[setup] real/white noise per spatial scale (from {len(noise_channels)} line-free "
            f"channels): " + " ".join(f"{r:.1f}" for r in ratios))
    thresholds = [[k_sigma * s for s in levels] for levels in sigma]
    if any(dropped):
        say(f"[setup] largest recoverable scale {max_scale_px:.0f} px: spatial planes "
            f"{[j for j, d in enumerate(dropped) if d]} zeroed (of 0..{num_scales_2d}, {num_scales_2d} = coarse)")
    say(f"[setup] {num_scales_2d} starlet scales x {num_levels_1d} CDF 9/7 levels; "
        f"step 1/L = {step_size:.3e}; image noise (step units) = {sigma_img:.3e}; "
        f"thresholds {k_sigma} sigma: {min(min(l) for l in thresholds):.2e} .. "
        f"{max(max(l) for l in thresholds):.2e}; {n_iter} iterations, reweighting "
        f"{f'{reweight_mode} from iteration {burn_in_iters}' if reweight else 'off'}, positivity={positivity}")

    x = np.zeros_like(dirty)          # the model, Jy/pixel
    z = x.copy()                      # FISTA's momentum variable
    t = 1.0
    prev_coeffs = None                # for reweighting: previous model's coefficients
    history = []

    for it in range(n_iter):
        # --- 1. gradient step on the smooth data term ---------------------
        grad = operator.gradient(z)              # N(z) - dirty
        v = z - step_size * grad
        residual = -grad                          # dirty - N(z), for diagnostics

        # --- 2. proximal step: analyze, threshold, synthesize -------------
        coeffs = analyze(v, num_scales_2d, num_levels_1d)
        if reweight and it >= burn_in_iters and reweight_mode == "bias-only":
            new_coeffs = _soft_threshold_bias_corrected(
                coeffs, thresholds, prev_coeffs, keep_coarsest, k_sigma)
        elif reweight and it >= burn_in_iters:
            # reweighted thresholds: T / (|prev coeff| / T + eps), per
            # sub-band, computed against last iteration's model
            rw_thresholds = []
            for j2, (details, approx, _) in enumerate(coeffs):
                p_details, p_approx, _ = prev_coeffs[j2]
                levels = []
                for l, (T, p) in enumerate(zip(thresholds[j2][:-1], p_details)):
                    levels.append(T / (np.abs(p) / T + reweight_eps))
                T = thresholds[j2][-1]
                levels.append(T / (np.abs(p_approx) / T + reweight_eps))
                rw_thresholds.append(levels)
            new_coeffs = _soft_threshold_reweighted(coeffs, rw_thresholds, keep_coarsest)
        else:
            new_coeffs = soft_threshold_all(coeffs, thresholds, keep_coarsest)

        for j2, drop in enumerate(dropped):
            if drop:
                det, ap, ol = new_coeffs[j2]
                new_coeffs[j2] = ([np.zeros_like(d) for d in det], np.zeros_like(ap), ol)
        x_new = synthesize(new_coeffs)
        if positivity:
            np.maximum(x_new, 0.0, out=x_new)
        if reweight:              # only reweighting needs the model's own coefficients
            prev_coeffs = analyze(x_new, num_scales_2d, num_levels_1d)

        # --- 3. FISTA momentum ---------------------------------------------
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new

        # Reuse `residual` (= dirty - N(z), already computed this iteration
        # for free) instead of a fresh `operator.apply(x)` call -- for the
        # exact operator that would be one extra ~60-140s CASA major cycle
        # per iteration, purely for a diagnostic.
        peak_ratio = (dirty - residual).max() / dirty.max()
        history.append(dict(iter=it, residual_rms=float(residual.std()),
                            flux=float(x.sum()), active=int(np.count_nonzero(x)),
                            peak_ratio=float(peak_ratio)))
        if it % print_every == 0 or it == n_iter - 1:
            say(f"iter {it:3d}  residual_rms={residual.std():.4e}  "
                f"flux={x.sum():8.2f} Jy  active={np.count_nonzero(x)}  "
                f"N(model)/dirty peak={peak_ratio:.3f}")

    if debias_iters > 0:
        support = x > 0
        say(f"[debias] refitting amplitudes on the fixed {np.count_nonzero(support)}-pixel "
              f"support with no L1 penalty, {debias_iters} unpenalized (accelerated) "
              f"gradient steps")
        x_d, z_d, t_d = x.copy(), x.copy(), 1.0
        for it in range(debias_iters):
            grad = operator.gradient(z_d)
            x_new = z_d - step_size * grad
            x_new[~support] = 0.0                 # support frozen: no new pixels can appear
            if positivity:
                np.maximum(x_new, 0.0, out=x_new)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_d * t_d))
            z_d = x_new + ((t_d - 1.0) / t_new) * (x_new - x_d)
            x_d, t_d = x_new, t_new
            if it % 5 == 0 or it == debias_iters - 1:
                pred = operator.apply(x_d)
                say(f"[debias] iter {it:3d}  residual_rms={(dirty - pred).std():.4e}  "
                      f"flux={x_d.sum():8.2f} Jy  N(model)/dirty peak={pred.max() / dirty.max():.3f}")
        x = x_d


    return x, history


def _soft_threshold_bias_corrected(coeffs, thresholds, prev_coeffs, keep_coarsest,
                                   k_sigma=K_SIGMA):
    """
    Reweighting that removes the L1 shrinkage bias WITHOUT loosening detection.

    The scheme above (`_soft_threshold_reweighted`) folds the weight into the
    threshold itself, so a large previous coefficient both shrinks less *and*
    gates less -- the solver can then admit structure the plain threshold
    would have rejected. On a noiseless, sub-beam problem that is what
    produces high-frequency ringing (measured on the FIRE cube: 3.5x the
    truth's sub-beam power, vs 1.2x with reweighting off).

    This variant, following the iterative-soft scheme in
    `Denoiser3D-IFU/src/wavelet_denoising.py::_denoise_iterative_soft`, keeps
    the two jobs separate:

        mask   = |c| > T                            (selection: plain T)
        w      = k_sigma * s_p / (|p| + s_p * 1e-6) (s_p = MAD noise of the
                                                     previous model's band)
        out    = sign(c) * max(|c| - w * T, 0)  where mask, else 0

    so which coefficients survive is unchanged, and only the amount
    subtracted from the survivors shrinks as the model grows. The epsilon is
    scaled to the sub-band's own noise so near-zero coefficients cannot blow
    the weight up arbitrarily.
    """
    out = []
    n2 = len(coeffs)
    for j2, (details, approx, orig_lens) in enumerate(coeffs):
        p_details, p_approx, _ = prev_coeffs[j2]

        def shrink(c, p, T):
            s_p = _mad(p)          # per-sub-band MAD of the previous model
            if s_p <= 0.0:
                # Sub-band still identically zero in the model (common in the
                # first reweighted iterations, and for bands the prior has
                # emptied): there is nothing to debias against, so fall back
                # to the plain unweighted shrinkage rather than 0/0.
                w = np.ones_like(c)
            else:
                w = k_sigma * s_p / (np.abs(p) + s_p * 1e-6)
            keep = np.abs(c) > T
            res = np.zeros_like(c)
            res[keep] = np.sign(c[keep]) * np.maximum(
                np.abs(c[keep]) - w[keep] * T, 0.0)
            return res

        new_details = [shrink(d, p, T) for d, p, T
                       in zip(details, p_details, thresholds[j2][:-1])]
        if j2 == n2 - 1 and keep_coarsest:
            new_approx = approx
        else:
            new_approx = shrink(approx, p_approx, thresholds[j2][-1])
        out.append((new_details, new_approx, orig_lens))
    return out


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
