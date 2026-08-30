#!/usr/bin/env python
"""
UVDeconvolver2D1D: sparse 2D-1D wavelet deconvolution run directly against
visibilities in the uv plane, via a `uv_operator.NonUniformFourierOperator`.

This version implements dynamic per-iteration, per-subband noise estimation
using the Median Absolute Deviation (MAD) of the back-projected visibility
residual image, following Garsden et al. (2015) Eq. 11. It applies a fixed
detection threshold multiplier (k) at every scale in every iteration,
allowing the absolute threshold to decay naturally as the residual noise
floor drops.

The coarsest wavelet sub-band -- where essentially all of an extended
source's mean flux lives -- is always left completely untouched by
thresholding here, at every iteration, matching `deconvolver._threshold_subbands`'s
convention. There is no "subtract the coarse component from the data" step:
gradient pressure into this coefficient comes from the real, unmodified visibilities
every iteration (`resid = vis_obs - operator.forward_cube(x_aux)`).

Reweighting (bias correction for soft thresholding)
----------------------------------------------------
`deconvolve`'s `reweight=True` option adds an iteratively-reweighted-L1
correction (Candes/Wakin/Boyd 2008) for soft-thresholding's amplitude bias,
folded directly into the single FISTA loop. For the first `burn_in_iters`
iterations, the algorithm runs standard soft-thresholding using the fixed `k`
multiplier and dynamic MAD noise estimates. Every iteration after that, the
threshold for each coefficient is re-derived continuously from the immediately
preceding iteration's model via `deconvolver._reweighted_threshold_subbands`,
using the dynamic MAD estimate for sigma^(n) and sigma^(n-1).
"""

import numpy as np

from wavelet2d1d import transform_2d1d, inverse_2d1d
from deconvolver import _threshold_subbands, _reweighted_threshold_subbands


def _estimate_subband_noise_mad(res_img, num_scales_2d, num_scales_1d):
    """
    Estimate per-subband noise dynamically using the Median Absolute Deviation (MAD)
    of the back-projected visibility residual image: R^(n) = mu * operator.adjoint_cube(resid).
    
    Formula from Garsden et al. (2015) Eq. 11:
        sigma_j = MAD(alpha_j) / 0.6745
    """
    sb_res = transform_2d1d(res_img, num_scales_2d, num_scales_1d)
    # Compute median and MAD along spatial and spectral pixel axes (2, 3, 4)
    med = np.median(sb_res, axis=(2, 3, 4), keepdims=True)
    mad = np.median(np.abs(sb_res - med), axis=(2, 3, 4))
    return mad / 0.6745


class UVDeconvolver2D1D:
    """
    Sparse 2D-1D wavelet deconvolution via ISTA/FISTA, run directly against
    visibilities through a `NonUniformFourierOperator`. Uses dynamic MAD-based
    thresholding at every iteration with a fixed detection multiplier.
    """

    def __init__(self, num_scales_2d=None, num_scales_1d=None,
                 threshold_type="soft", positivity=True, verbose=True):
        self.num_scales_2d = num_scales_2d
        self.num_scales_1d = num_scales_1d
        self.threshold_type = threshold_type
        self.positivity = positivity
        self.verbose = verbose

    def _resolve_scales(self, cube_shape):
        max2d = int(np.log2(min(cube_shape[1], cube_shape[2])))
        max1d = int(np.log2(cube_shape[0]))
        n2d = self.num_scales_2d or max2d
        n1d = self.num_scales_1d or max1d
        n2d = min(max(n2d, 2), max2d)
        n1d = min(max(n1d, 2), max1d)
        return n2d, n1d

    def deconvolve(self, vis_obs, operator, sigma_vis, cube_shape, n_iter=150,
                    k=3.0, k_extra_finescale=1.0, fista=True, x0=None,
                    reweight=False, burn_in_iters=None, lam=3.0, eps=1e-2):
        """
        Parameters
        ----------
        vis_obs : ndarray, shape (nz, n_uv), complex
            Observed visibilities, Y in the forward model Y = Phi(X) + N.
        operator : uv_operator.NonUniformFourierOperator
            The (shared, per-channel) non-uniform Fourier measurement
            operator -- provides `forward_cube`/`adjoint_cube`.
        sigma_vis : float or None
            Kept in the signature for positional backward-compatibility with
            existing scripts, but is completely ignored now that dynamic MAD
            estimation is active.
        cube_shape : tuple (nz, ny, nx)
            Shape of the image-space model to reconstruct.
        n_iter : int
            Number of ISTA/FISTA iterations.
        k : float
            Fixed detection threshold multiplier (in sigma units) applied across
            all iterations. Absolute thresholding scales dynamically via MAD.
        k_extra_finescale, fista, x0 :
            Same meaning as `Deconvolver2D1D.deconvolve`.
        reweight : bool
            If False (default): plain FISTA for the whole run using fixed `k`
            and dynamic per-iteration MAD thresholding.
            If True: corrects soft-thresholding's amplitude bias via
            iteratively-reweighted L1. For the first `burn_in_iters` iterations,
            behaves like the plain path (fixed `k` with MAD noise); every
            iteration after that, the threshold for each coefficient is
            re-derived from the immediately preceding iteration's model.
        burn_in_iters : int or None
            Only used when `reweight=True`. Number of initial iterations
            using standard thresholding before continuous reweighting begins.
            Defaults to `n_iter // 3` if not given.
        lam : float
            Only used when `reweight=True`. The reweighting scale parameter
            used in the per-iteration reweighting formula after burn-in.
        eps : float
            Only used when `reweight=True`. Small fraction added to the
            reweighting ratio's denominator (default 1e-2), avoiding division
            by ~0 for coefficients that were essentially zero the iteration before.

        Returns
        -------
        model : ndarray, shape cube_shape
        history : dict with 'residual_std' (list of len n_iter): the
            std of the stacked real/imaginary visibility residual,
            std(dirty - H(model)) computed in the uv plane, at each iteration.
            'reweight_start_iter': present only when `reweight=True` -- the
            iteration index continuous reweighting began at (== burn_in_iters).
        """
        if reweight and self.threshold_type != "soft":
            raise ValueError("reweight=True only makes sense with threshold_type='soft'")

        n2d, n1d = self._resolve_scales(cube_shape)
        if self.verbose:
            print(f"[UVDeconvolver2D1D] {n2d} spatial scales x {n1d} spectral "
                  f"scales, {n_iter} {'FISTA' if fista else 'ISTA'} iterations "
                  f"({self.threshold_type} thresholding), n_uv={operator.n_uv}, "
                  f"reweight={reweight}, coarse band always skipped")

        L = operator.lipschitz_constant()
        mu = 1.0 / L
        if self.verbose:
            print(f"[UVDeconvolver2D1D] Lipschitz constant L={L:.4g} (power iteration), "
                  f"step size mu={mu:.4g}")
            print(f"[UVDeconvolver2D1D] using dynamic per-iteration MAD noise estimation "
                  f"with fixed multiplier k={k:.2f}")

        def _vis_std(resid_vis):
            return float(np.std(np.concatenate([resid_vis.real.ravel(),
                                                  resid_vis.imag.ravel()])))

        x = np.zeros(cube_shape) if x0 is None else x0.copy()
        x_aux = x.copy()
        t = 1.0
        residual_std_history = []

        if not reweight:
            # Plain path with dynamic MAD thresholding and fixed k multiplier
            for it in range(n_iter):
                resid = vis_obs - operator.forward_cube(x_aux)
                res_img = mu * operator.adjoint_cube(resid)
                grad_step = x_aux + res_img

                # Dynamically estimate noise per-subband from the back-projected residual
                sigma_curr = _estimate_subband_noise_mad(res_img, n2d, n1d)

                subbands = transform_2d1d(grad_step, n2d, n1d)
                subbands = _threshold_subbands(subbands, sigma_curr, k,
                                                k_extra_finescale, self.threshold_type)
                x_new = inverse_2d1d(subbands)
                if self.positivity:
                    x_new = np.maximum(x_new, 0.0)

                if fista:
                    t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t ** 2))
                    x_aux = x_new + ((t - 1.0) / t_new) * (x_new - x)
                    t = t_new
                else:
                    x_aux = x_new

                x = x_new
                res_std = _vis_std(vis_obs - operator.forward_cube(x))
                residual_std_history.append(res_std)

                if self.verbose and (it % max(n_iter // 10, 1) == 0 or it == n_iter - 1):
                    print(f"  iter {it + 1:4d}/{n_iter}  mad_sigma_fine={sigma_curr[0, 0]:.3g}  "
                          f"residual_std={res_std:.4g}  flux={x.sum():.4g}")

            return x, {"residual_std": residual_std_history}

        # --- reweight=True: continuous, per-iteration reweighting with dynamic MAD ---
        burn_in = burn_in_iters if burn_in_iters is not None else max(n_iter // 3, 1)
        burn_in = min(burn_in, n_iter)
        if self.verbose:
            print(f"[UVDeconvolver2D1D] reweight=True: burn_in={burn_in} iterations "
                  f"(fixed k={k} with MAD noise), then continuous per-iteration "
                  f"reweighting (scale lam={lam}) for the remaining {n_iter - burn_in} iterations.")

        sigma_prev_state = None

        for it in range(n_iter):
            resid = vis_obs - operator.forward_cube(x_aux)
            res_img = mu * operator.adjoint_cube(resid)
            grad_step = x_aux + res_img

            # Dynamically estimate noise per-subband from the back-projected residual
            sigma_curr = _estimate_subband_noise_mad(res_img, n2d, n1d)

            subbands = transform_2d1d(grad_step, n2d, n1d)
            if it < burn_in:
                subbands = _threshold_subbands(subbands, sigma_curr, k,
                                                k_extra_finescale, "soft")
            else:
                sigma_prev = sigma_prev_state if sigma_prev_state is not None else sigma_curr
                prev_subbands = transform_2d1d(x, n2d, n1d)
                subbands = _reweighted_threshold_subbands(subbands, prev_subbands, sigma_curr,
                                                           sigma_prev, lam, eps, "soft")
            sigma_prev_state = sigma_curr

            x_new = inverse_2d1d(subbands)
            if self.positivity:
                x_new = np.maximum(x_new, 0.0)

            if fista:
                t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t ** 2))
                x_aux = x_new + ((t - 1.0) / t_new) * (x_new - x)
                t = t_new
            else:
                x_aux = x_new

            x = x_new
            res_std = _vis_std(vis_obs - operator.forward_cube(x))
            residual_std_history.append(res_std)

            if self.verbose and (it % max(n_iter // 10, 1) == 0 or it == n_iter - 1):
                print(f"  iter {it + 1:4d}/{n_iter}  mad_sigma_fine={sigma_curr[0, 0]:.3g}  "
                          f"residual_std={res_std:.4g}  flux={x.sum():.4g}")

        return x, {"residual_std": residual_std_history, "reweight_start_iter": burn_in}