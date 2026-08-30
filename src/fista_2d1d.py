#!/usr/bin/env python
"""
FISTA with 2D-1D starlet sparsity, driven by a normal operator `N = A^H W A`.

This is the same algorithm as `uv_deconvolver.UVDeconvolver2D1D` -- FISTA on

    min_x  1/2 || A x - y ||^2_W  +  lambda || Phi^T x ||_1 ,   x >= 0

with per-sub-band MAD noise estimation following Garsden et al. (2015) -- but
written against the `N x` / `d` formulation of `alma_fourier` instead of
against a separate `forward_cube`/`adjoint_cube` pair. The gradient step is

    grad f(x) = N x - d

which is a single call on either `PSFNormalOperator` (two FFTs per channel) or
`CASAImager` (one real degrid+grid major cycle). Nothing else in the loop
changes between them, so the cheap operator can be used to find the solution
and the exact one to confirm it.

Differences from `uv_deconvolver.py`, and why
---------------------------------------------
* It takes an operator exposing `gradient(x)` and `lipschitz_constant()`,
  not `forward_cube`/`adjoint_cube`. There is no step-size guesswork: for the
  PSF operator `L = max|OTF|` is exact, so `mu = 1/L` is exactly the largest
  stable step.
* A non-negativity projection is applied after thresholding. Sky brightness
  in a continuum-subtracted line cube can legitimately be negative
  (absorption), so this is optional and off by default -- but it is by far
  the strongest constraint available on real data when you know there is no
  absorption, so `positivity=True` is worth trying.
* The coarsest sub-band is zeroed rather than passed through, for the reason
  spelled out in `alma_fourier.PSFNormalOperator.zero_spacing_response`: an
  interferometer has no zero-spacing baseline, so that mode is in the null
  space of `N` and, left free, grows without bound.
* Iteration history (residual RMS, model flux, active coefficient count) is
  recorded so convergence can be checked instead of assumed.
"""

from __future__ import annotations

import numpy as np

from wavelet2d1d import transform_2d1d, inverse_2d1d


def _mad_sigma(a, axis):
    med = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - med), axis=axis) / 0.6745


def estimate_subband_noise_mad(res_cube, num_scales_2d, num_scales_1d):
    """Per-sub-band sigma from the MAD of the transformed residual cube.

    Garsden et al. (2015) Eq. 11: sigma_j = MAD(alpha_j) / 0.6745, evaluated
    on the residual (dirty image of the residual visibilities) at the current
    iterate, so the threshold tracks the falling noise floor automatically.
    """
    sb = transform_2d1d(res_cube, num_scales_2d, num_scales_1d)
    return _mad_sigma(sb, axis=(2, 3, 4))


def threshold_subbands(subbands, thresholds, zero_coarse=True):
    """Soft-threshold every detail sub-band.

    `thresholds` is (J2+1, J1+1), broadcast over each sub-band's pixels. The
    coarsest (J2, J1) sub-band is zeroed when `zero_coarse` -- see module
    docstring.

    Soft thresholding only. It is the proximal operator of the L1 norm, which
    is what makes the iteration an actual FISTA on a well-defined objective
    with convergence guarantees. Hard thresholding is not a proximal step and
    was measured to be strictly worse here: on this dataset it overfitted
    harder (residual 0.86x the noise floor vs 0.96x) while producing a *less*
    sparse model (25.1% of pixels active vs 16.7%), because it removes the
    shrinkage that was the only thing limiting the runaway.
    """
    out = subbands.copy()
    n2, n1 = out.shape[:2]
    for j2 in range(n2):
        for j1 in range(n1):
            if zero_coarse and j2 == n2 - 1 and j1 == n1 - 1:
                out[j2, j1] = 0.0
                continue
            band = out[j2, j1]
            out[j2, j1] = np.sign(band) * np.maximum(
                np.abs(band) - thresholds[j2, j1], 0.0)
    return out


def threshold_subbands_reweighted(subbands, prev_subbands, thresholds,
                                  eps=1e-2, zero_coarse=True):
    """Soft threshold with per-coefficient reweighting (Candes/Wakin/Boyd 2008).

    Plain soft thresholding shrinks *every* surviving coefficient by the same
    threshold `T`, which is where the amplitude bias comes from: on this
    dataset the model reproduced only 0.789 of the dirty peak. Reweighted L1
    replaces the single threshold with a per-coefficient one derived from the
    previous iterate,

        w_i = 1 / (|alpha_i^(prev)| / T + eps)
        T_i = T * w_i

    so a coefficient already far above the noise (|alpha| = 10 T) is shrunk by
    only ~T/10 -- the bias on bright structure nearly vanishes -- while a
    coefficient near zero gets `T_i -> T/eps`, i.e. 100x the nominal threshold
    at the default `eps`, and is killed. The result is simultaneously less
    biased *and* sparser, which is exactly the combination that the
    unconstrained support refit in `debias_on_support` could not achieve
    without overfitting: here the threshold is bounded below by `eps * T`
    rather than going to zero, so the iteration cannot chase the noise.

    This approximates the non-convex log-sum penalty by a sequence of convex
    weighted-L1 problems. Each iteration is still a proper proximal step (the
    weighted soft threshold is the prox of the weighted L1 norm), so the FISTA
    machinery is unchanged; only the threshold varies between iterations.
    """
    out = subbands.copy()
    n2, n1 = out.shape[:2]
    for j2 in range(n2):
        for j1 in range(n1):
            if zero_coarse and j2 == n2 - 1 and j1 == n1 - 1:
                out[j2, j1] = 0.0
                continue
            T = thresholds[j2, j1]
            if T <= 0:
                continue
            # weight from the previous model's coefficients, computed in
            # place per sub-band so no second full-size array is held
            w = 1.0 / (np.abs(prev_subbands[j2, j1]) / T + eps)
            band = out[j2, j1]
            out[j2, j1] = np.sign(band) * np.maximum(np.abs(band) - T * w, 0.0)
    return out


class FISTA2D1D:
    """Sparse 2D-1D starlet deconvolution of an interferometric cube.

    Parameters
    ----------
    num_scales_2d, num_scales_1d : int
        Starlet detail scales, spatial and spectral. The spatial transform
        resolves structures up to 2**num_scales_2d pixels; the spectral one
        needs `nz > 2**num_scales_1d`.
    k_sigma : float
        Detection threshold in units of the per-sub-band MAD noise.
    k_extra_finescale : float
        Extra sigma on the finest spatial scale, which is dominated by noise
        and PSF sidelobe residuals.
    positivity : bool
        Project onto x >= 0 after each thresholding step.
    noise_mode : {'fixed', 'dynamic'}
        'fixed' estimates the per-sub-band noise once, at iteration 0, from
        the dirty image and holds it. 'dynamic' re-estimates it from the
        shrinking residual every iteration and is degenerate -- see
        `deconvolve`. Always prefer 'fixed'.
    reweight : bool
        Enable iteratively-reweighted L1 after `burn_in_iters` iterations, to
        remove soft thresholding's amplitude bias. See
        `threshold_subbands_reweighted`.
    burn_in_iters : int or None
        Iterations of plain soft thresholding before reweighting starts. The
        weights are derived from the current model, so it needs to be a
        sensible estimate first. Defaults to a third of `n_iter`.
    eps : float
        Reweighting floor. The effective threshold on a zero coefficient is
        `T / eps`, so smaller `eps` means a sparser, more aggressive solution.
    """

    def __init__(self, num_scales_2d=4, num_scales_1d=2, k_sigma=3.0,
                 k_extra_finescale=1.0, positivity=True,
                 zero_coarse=True, noise_mode="fixed", reweight=True,
                 burn_in_iters=None, eps=1e-2, verbose=True):
        self.num_scales_2d = num_scales_2d
        self.num_scales_1d = num_scales_1d
        self.k_sigma = k_sigma
        self.k_extra_finescale = k_extra_finescale
        self.positivity = positivity
        self.zero_coarse = zero_coarse
        self.noise_mode = noise_mode
        self.reweight = reweight
        self.burn_in_iters = burn_in_iters
        self.eps = eps
        self.verbose = verbose
        self.history = []

    def _thresholds(self, sigma):
        t = self.k_sigma * sigma
        t[0, :] += self.k_extra_finescale * sigma[0, :]
        return t

    def deconvolve(self, operator, n_iter=100, x0=None, step_safety=1.0,
                   callback=None):
        """Run FISTA.

        `operator` must provide `gradient(x)`, `residual(x)`, `dirty` and
        `lipschitz_constant()` -- satisfied by
        `alma_fourier.PSFNormalOperator`.

        Returns the model cube (same units as the dirty image's Jy/pixel
        convention: convolving it with the PSF reproduces the dirty cube).
        """
        d = operator.dirty
        nz, ny, nx = d.shape
        if nz <= 2 ** self.num_scales_1d:
            raise ValueError(
                f"nz={nz} too small for num_scales_1d={self.num_scales_1d}")

        L = operator.lipschitz_constant()
        mu = step_safety / L
        if self.verbose:
            print(f"[fista] cube {d.shape}  L={L:.6g}  step mu={mu:.6g}  "
                  f"soft threshold, noise={self.noise_mode}, k={self.k_sigma}")
        burn_in = (self.burn_in_iters if self.burn_in_iters is not None
                   else max(n_iter // 3, 1))
        if self.verbose and self.reweight:
            print(f"[fista] reweighted L1 after {burn_in} burn-in iterations "
                  f"(eps={self.eps})")

        sigma0 = None
        x = np.zeros_like(d) if x0 is None else np.array(x0, dtype=np.float64)
        z = x.copy()
        t = 1.0
        self.history = []

        for it in range(n_iter):
            grad = operator.gradient(z)          # N z - d
            v = z - mu * grad
            # The residual is just -grad; computing it separately would double
            # the operator cost per iteration, which matters a lot when the
            # operator is a CASA major cycle rather than a pair of FFTs.
            res = -grad                          # d - N z

            if self.noise_mode == "fixed":
                # Estimate the noise ONCE, at iteration 0, where the residual
                # is the dirty image (MAD is robust enough that the source
                # does not bias it), and hold the threshold there.
                #
                # `noise_mode="dynamic"` -- re-estimating sigma from
                # `mu * residual` every iteration, as Garsden et al. (2015)
                # Eq. 11 is implemented in `uv_deconvolver.py` -- is
                # degenerate in this formulation: as the residual shrinks the
                # threshold shrinks with it, so more coefficients survive,
                # so the residual shrinks further. Its fixed point is
                # residual -> 0, i.e. fitting the noise. Measured on this
                # dataset against a 7.89e-4 Jy/beam source-free noise floor:
                #     soft + dynamic -> residual 0.96x noise, support 16.7%
                #     hard + dynamic -> residual 0.86x noise, support 25.1%
                # Soft thresholding only looks stable there because its
                # shrinkage partially self-limits; hard thresholding has
                # nothing holding it back and overfits harder while ending up
                # *less* sparse. The threshold has to be tied to the noise in
                # the data, which is a fixed quantity.
                if sigma0 is None:
                    sigma0 = estimate_subband_noise_mad(
                        mu * res, self.num_scales_2d, self.num_scales_1d)
                sigma = sigma0
            else:
                sigma = estimate_subband_noise_mad(mu * res, self.num_scales_2d,
                                                   self.num_scales_1d)

            sb = transform_2d1d(v, self.num_scales_2d, self.num_scales_1d)
            thr = self._thresholds(sigma)
            if self.reweight and it >= burn_in:
                # weights from the previous model, not from `v`: they must
                # reflect the current best estimate of which coefficients are
                # real, not the noisy gradient step
                prev_sb = transform_2d1d(x, self.num_scales_2d,
                                         self.num_scales_1d)
                sb = threshold_subbands_reweighted(sb, prev_sb, thr, self.eps,
                                                   self.zero_coarse)
                del prev_sb
            else:
                sb = threshold_subbands(sb, thr, self.zero_coarse)
            x_new = inverse_2d1d(sb)
            if self.positivity:
                np.maximum(x_new, 0.0, out=x_new)

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)

            rel = (np.linalg.norm(x_new - x)
                   / max(np.linalg.norm(x_new), 1e-300))
            rec = {
                "iter": it,
                "residual_rms": float(res.std()),
                "residual_peak": float(np.abs(res).max()),
                "model_flux": float(x_new.sum()),
                "model_peak": float(x_new.max()),
                "n_active": int(np.count_nonzero(x_new)),
                "rel_change": float(rel),
            }
            self.history.append(rec)
            if self.verbose and (it % 5 == 0 or it == n_iter - 1):
                print(f"[fista] {it:4d}  res_rms={rec['residual_rms']:.4e} "
                      f"res_peak={rec['residual_peak']:.4e} "
                      f"flux={rec['model_flux']:.4f} "
                      f"active={rec['n_active']:8d} "
                      f"dx={rec['rel_change']:.3e}", flush=True)
            if callback is not None:
                callback(it, x_new, rec)

            x, t = x_new, t_new

        self.model = x
        self.residual = operator.residual(x)
        return x


def debias_on_support(operator, model, n_iter=40, positivity=True, verbose=True):
    """Remove soft-thresholding shrinkage bias by refitting amplitudes.

    Why this is needed if you intend to use the model *directly* rather than
    restoring it with a clean beam. Soft thresholding is the proximal
    operator of the L1 norm, so it does two jobs at once: it selects a
    support (which coefficients are non-zero) and it shrinks every surviving
    coefficient by the threshold. The selection is what you want; the
    shrinkage is a pure amplitude bias. On this dataset it cost 37% of the
    peak -- `model (*) beam` reached only 0.628 of the dirty peak, with the
    missing flux sitting in the residual. That is invisible in a CLEAN-style
    restored image (`model (*) beam + residual` puts it back) but is a direct
    error in the model itself.

    The standard fix (LASSO debiasing / "hybrid" estimator) is to keep the
    support that L1 selected and re-solve the *unregularized* least-squares
    problem restricted to it:

        min_x || N x - d ||^2   subject to   supp(x) = supp(model), x >= 0

    That is a convex problem on a convex set, so accelerated projected
    gradient with the exact step `1/L` converges. No thresholding happens
    here, so nothing is shrunk and no new pixels are switched on -- the
    sparsity and the resolution of the wavelet solution are preserved
    exactly, only the amplitudes change.

    *** This only works on a genuinely sparse support. ***
    Measured on this dataset with the soft-threshold solution, whose support
    was 1,541,365 pixels (16.7% of the cube): the refit drove the residual
    rms to 5.30e-4, *below* the 7.89e-4 source-free noise -- i.e. it fitted
    the noise -- and doubled the flux from 81 to 158 Jy while the peak bias
    barely moved (0.628 -> 0.701). With ~1.5e6 free parameters and an
    operator whose condition number is 8e7, the restricted problem is not
    well posed. Check `residual.std()` against the source-free noise
    afterwards: if it drops below, the support was too large. Get a sparser
    support first (a larger `k_sigma`, or `noise_mode="fixed"`) and
    debias that.
    """
    support = model > 0
    L = operator.lipschitz_constant()
    mu = 1.0 / L
    x = model.copy()
    z = x.copy()
    t = 1.0
    if verbose:
        print(f"[debias] refitting {support.sum()} active pixels, step {mu:.4g}")
    for it in range(n_iter):
        x_new = z - mu * operator.gradient(z)
        x_new[~support] = 0.0
        if positivity:
            np.maximum(x_new, 0.0, out=x_new)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x, t = x_new, t_new
        if verbose and (it % 10 == 0 or it == n_iter - 1):
            r = operator.residual(x)
            print(f"[debias] {it:3d}  res_rms={r.std():.4e}  flux={x.sum():.3f}",
                  flush=True)
    return x


def restore(model, residual, clean_beam):
    """CLEAN-style restored cube: model (*) clean beam + residual.

    `clean_beam` is a (ny, nx) normalized Gaussian with peak 1, so the output
    is in Jy/beam and directly comparable to the dirty cube.
    """
    from scipy.signal import fftconvolve

    out = np.empty_like(model)
    for k in range(model.shape[0]):
        out[k] = fftconvolve(model[k], clean_beam, mode="same")
    return out + residual


def gaussian_beam(shape, bmaj_pix, bmin_pix, bpa_deg):
    """Peak-1 elliptical Gaussian on a (ny, nx) grid, centred.

    `bmaj_pix`/`bmin_pix` are FWHM in pixels; `bpa_deg` is the CASA/FITS beam
    position angle, measured North through East.

    The angle is negated below because of how the sky maps onto the array:
    RA has a negative pixel increment, so the +x array direction points West,
    not East, and a rotation from North toward East runs the opposite way in
    array coordinates. Getting this backwards is silent -- it produces a
    perfectly plausible beam at the mirrored angle -- so it is checked
    against the PSF main lobe in the export stage (0.010 vs 0.144 max
    absolute error for the right and wrong sign on this dataset).
    """
    ny, nx = shape
    y = np.arange(ny) - ny // 2
    x = np.arange(nx) - nx // 2
    X, Y = np.meshgrid(x, y)
    pa = np.deg2rad(-bpa_deg)
    # rotate into the beam frame: major axis along the rotated y
    xr = X * np.cos(pa) - Y * np.sin(pa)
    yr = X * np.sin(pa) + Y * np.cos(pa)
    f = 2.0 * np.sqrt(2.0 * np.log(2.0))
    smaj, smin = bmaj_pix / f, bmin_pix / f
    return np.exp(-0.5 * ((yr / smaj) ** 2 + (xr / smin) ** 2))
