#!/usr/bin/env python
"""
Deconvolver2D1D: sparse 2D-1D wavelet deconvolution for ALMA-like dirty
spectral cubes.

This is the deconvolution counterpart of Denoiser2D1D-improved's
`Denoiser2D1D` (`src/wavelet_denoising.py`). See the repo README for the
full derivation; the short version:

Denoising (the original repo)
    Forward model:  Y = X + N                (H = identity)
    `_denoise_iterative_hard` re-estimates a per-sub-band significance mask
    from the current model, then simply reapplies that mask to the *data's*
    wavelet coefficients and reconstructs -- there is no explicit gradient
    step because, when H = I, the gradient of the data-fidelity term
    ||Y - X||^2 w.r.t. X evaluated at the current model is just (X - Y), so
    "use the data coefficients directly" already *is* the gradient step.

Deconvolution (this module)
    Forward model:  Y = H(X) + N             (H = convolution with the dirty
                                                beam / PSF, per channel)
    H is no longer the identity, so the shortcut above no longer applies.
    Each iteration must now do an explicit forward-backward (ISTA/FISTA)
    step:
        1. residual        r_k      = Y - H(x_aux)
        2. gradient step    z_k      = x_aux + mu * H^T(r_k)
        3. sparsify         w        = Wavelet2D1D_forward(z_k)
                             w_thresh = threshold(w, per-sub-band noise, k_sigma)
        4. reconstruct       x_{k+1}  = Wavelet2D1D_inverse(w_thresh)
        5. positivity        x_{k+1}  = max(0, x_{k+1})
        6. (FISTA) momentum-extrapolate x_aux for the next iteration
    which is exactly proximal-gradient descent on
        argmin_x  (1/2)||Y - H(x)||^2  +  lambda * ||W(x)||_1
    i.e. the sparse-deconvolution objective used by MORESANE/PURIFY/SARA,
    just built on this repo's own 2D-1D starlet dictionary instead of a
    single wavelet basis or a Dirac+wavelet dictionary.

    Because our mock dirty beam is centro-symmetric (see `psf.py`), H is
    self-adjoint (H^T = H), so `convolve_cube` below is reused for both the
    forward operator and its adjoint in step 2.
"""

import numpy as np

from wavelet2d1d import transform_2d1d, inverse_2d1d


def convolve_cube(cube, psf):
    """Per-channel 2D linear convolution of a (nz, ny, nx) cube with a single
    (py, px) PSF, vectorized over the channel axis via FFT, 'same'-shaped
    output centered on the PSF's own center pixel."""
    nz, ny, nx = cube.shape
    py, px = psf.shape
    fy, fx = ny + py - 1, nx + px - 1
    Fpsf = np.fft.rfft2(psf, s=(fy, fx))
    Fcube = np.fft.rfft2(cube, s=(fy, fx), axes=(1, 2))
    full = np.fft.irfft2(Fcube * Fpsf[None, :, :], s=(fy, fx), axes=(1, 2))
    y0, x0 = py // 2, px // 2
    return full[:, y0:y0 + ny, x0:x0 + nx]


def lipschitz_constant(psf, spatial_shape):
    """Operator norm^2 of the per-channel convolution-by-psf operator, i.e.
    the Lipschitz constant L of the data-fidelity gradient. Sets the largest
    stable ISTA/FISTA step size mu = 1 / L."""
    ny, nx = spatial_shape
    py, px = psf.shape
    fy, fx = ny + py - 1, nx + px - 1
    Fpsf = np.fft.rfft2(psf, s=(fy, fx))
    return float(np.max(np.abs(Fpsf)) ** 2)


def _threshold_subbands(subbands, noise_levels, k_sigma, k_extra_finescale, mode):
    """
    Apply a per-sub-band threshold k_sigma * noise_levels[j2, j1] to every
    detail sub-band. The finest spatial scale (j2 = 0) gets an extra
    `k_extra_finescale` sigma, exactly as in the original `Denoiser2D1D`
    (`threshold_increment_high_freq`), since it is dominated by noise/PSF-
    sidelobe residuals.

    The coarsest sub-band (j2 = J2, j1 = J1) is set to *zero*, not passed
    through untouched. This differs from the original denoiser, where the
    coarse band is always kept as-is (it just carries whatever smooth
    background was in the data, Y = X + N, harmless). Here it matters for a
    physical reason specific to interferometry: our dirty beam has zero DC
    response (`beam.sum() == 0`, verified in the demo notebook) because no
    antenna pair ever measures the true zero-length/zero-spacing baseline.
    That means the data-fidelity term ||Y - H(x)||^2 is completely blind to
    any flux added in the coarsest, most-diffuse mode of x -- it is a null
    space of H. Leaving that sub-band unthresholded (as in the denoiser)
    gives the FISTA iteration an unconstrained, unpenalized direction to
    grow in every single step, which does not converge: the reconstructed
    flux diverges monotonically instead of stabilizing. Zeroing that band
    removes the free direction and matches the physical fact that an
    interferometer fundamentally cannot recover the sky's total/zero-
    spacing flux -- CLEAN has the same blind spot, it just does not have an
    explicit degree of freedom that can run away with it.
    """
    n2, n1 = subbands.shape[:2]
    out = subbands.copy()
    for j2 in range(n2):
        for j1 in range(n1):
            if j2 == n2 - 1 and j1 == n1 - 1:
                out[j2, j1] = 0.0  # zero-spacing / null-space mode: unrecoverable, drop it
                continue
            thresh = k_sigma * noise_levels[j2, j1]
            if j2 == 0:
                thresh += k_extra_finescale * noise_levels[j2, j1]
            band = out[j2, j1]
            if mode == "hard":
                band[np.abs(band) <= thresh] = 0.0
            elif mode == "soft":
                out[j2, j1] = np.sign(band) * np.maximum(np.abs(band) - thresh, 0.0)
            else:
                raise ValueError(f"Unknown threshold mode {mode!r}")
    return out


def _estimate_subband_noise_after_gradient(sigma_noise, psf, mu, shape,
                                            num_scales_2d, num_scales_1d,
                                            n_mc=4, seed=0):
    """
    Correct per-sub-band noise-level calibration for the deconvolution
    gradient step (see the ISTA/FISTA loop in `Deconvolver2D1D.deconvolve`):
    white detector noise of std `sigma_noise` is first hit by the adjoint
    operator H^T (= `convolve_cube(..., psf)`, since our PSF is centro-
    symmetric) and scaled by the step size `mu` *before* it reaches the
    wavelet transform, i.e. it is no longer white once it gets there.

    This differs from `wavelet2d1d.estimate_subband_noise`, which assumes
    the identity forward model (H = I) appropriate for pure denoising in
    the original `Denoiser2D1D` -- reusing it here without first convolving
    by the PSF would under-estimate the true noise level in every sub-band
    (H redistributes/correlates the noise, generally *increasing* its
    variance through the sidelobe structure) and let the iteration fit
    sidelobe noise as if it were signal, so it is re-derived here with the
    PSF convolution included in the propagation chain.
    """
    rng = np.random.default_rng(seed)
    acc = np.zeros((num_scales_2d + 1, num_scales_1d + 1))
    for _ in range(n_mc):
        white = rng.normal(0.0, sigma_noise, size=shape)
        propagated = mu * convolve_cube(white, psf)
        sb = transform_2d1d(propagated, num_scales_2d, num_scales_1d)
        acc += sb.std(axis=(2, 3, 4))
    return acc / n_mc


class Deconvolver2D1D:
    """
    Sparse 2D-1D wavelet deconvolution via ISTA/FISTA, as described in the
    module docstring above.

    Parameters
    ----------
    num_scales_2d, num_scales_1d : int, optional
        Number of spatial / spectral starlet scales. Defaults to the maximum
        allowed by the cube's dimensions (log2 of size), same convention as
        the original `Denoiser2D1D.denoise`.
    threshold_type : {'soft', 'hard'}
    positivity : bool
    verbose : bool
    """

    def __init__(self, num_scales_2d=None, num_scales_1d=None,
                 threshold_type="soft", positivity=True, verbose=True):
        self.num_scales_2d = num_scales_2d
        self.num_scales_1d = num_scales_1d
        self.threshold_type = threshold_type
        self.positivity = positivity
        self.verbose = verbose

    def _resolve_scales(self, cube):
        max2d = int(np.log2(min(cube.shape[1], cube.shape[2])))
        max1d = int(np.log2(cube.shape[0]))
        n2d = self.num_scales_2d or max2d
        n1d = self.num_scales_1d or max1d
        n2d = min(max(n2d, 2), max2d)
        n1d = min(max(n1d, 2), max1d)
        return n2d, n1d

    def deconvolve(self, dirty, psf, sigma_noise, n_iter=150,
                    k_start=6.0, k_end=3.0, k_extra_finescale=1.0,
                    fista=True, x0=None):
        """
        Parameters
        ----------
        dirty : ndarray, shape (nz, ny, nx)
            Dirty (PSF-convolved, noisy) cube, Y in the forward model.
        psf : ndarray, shape (py, px)
            Dirty beam, assumed the same for every channel and centro-
            symmetric (see `psf.py`).
        sigma_noise : float
            Estimated per-voxel noise standard deviation of `dirty` (used
            only to set the wavelet detection thresholds, not injected).
        n_iter : int
            Number of ISTA/FISTA iterations.
        k_start, k_end : float
            The detection threshold (in sigma) decays linearly from
            k_start down to k_end over the iterations -- start conservative
            (avoid fitting sidelobes/noise as if they were signal) and relax
            as the model firms up, same philosophy as the original
            `Denoiser2D1D`'s plateau/iteration schedule, but explicit here
            instead of adaptive.
        k_extra_finescale : float
            Extra sigma added to the finest spatial scale's threshold.
        fista : bool
            Use Nesterov/FISTA momentum (recommended; much faster
            convergence than plain ISTA).
        x0 : ndarray or None
            Initial model. Defaults to zeros.

        Returns
        -------
        model : ndarray, shape (nz, ny, nx)
            Deconvolved ("clean") cube estimate.
        history : dict
            'residual_std': list of len n_iter, std(dirty - H(model)) per
            iteration, for convergence diagnostics.
        """
        n2d, n1d = self._resolve_scales(dirty)
        if self.verbose:
            print(f"[Deconvolver2D1D] {n2d} spatial scales x {n1d} spectral "
                  f"scales, {n_iter} {'FISTA' if fista else 'ISTA'} iterations "
                  f"({self.threshold_type} thresholding)")

        L = lipschitz_constant(psf, dirty.shape[1:])
        mu = 1.0 / L
        if self.verbose:
            print(f"[Deconvolver2D1D] Lipschitz constant L={L:.4g}, step size mu={mu:.4g}")

        # Noise entering the wavelet domain at each gradient step has passed
        # through H^T (convolution with psf) and been scaled by mu; propagate
        # a white-noise realization through that exact chain once, up front.
        noise_levels = _estimate_subband_noise_after_gradient(
            sigma_noise, psf, mu, dirty.shape, n2d, n1d, n_mc=4
        )
        if self.verbose:
            print(f"[Deconvolver2D1D] per-subband noise levels estimated "
                  f"(finest subband sigma={noise_levels[0, 0]:.3g})")

        x = np.zeros_like(dirty) if x0 is None else x0.copy()
        x_aux = x.copy()
        t = 1.0
        residual_std_history = []

        for it in range(n_iter):
            resid = dirty - convolve_cube(x_aux, psf)          # H self-adjoint => H^T == convolve_cube
            grad_step = x_aux + mu * convolve_cube(resid, psf)

            subbands = transform_2d1d(grad_step, n2d, n1d)
            frac = it / max(n_iter - 1, 1)
            k_sigma = k_start + (k_end - k_start) * frac
            subbands = _threshold_subbands(subbands, noise_levels, k_sigma,
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
            res_std = float(np.std(dirty - convolve_cube(x, psf)))
            residual_std_history.append(res_std)

            if self.verbose and (it % max(n_iter // 10, 1) == 0 or it == n_iter - 1):
                print(f"  iter {it + 1:4d}/{n_iter}  k_sigma={k_sigma:.2f}  "
                      f"residual_std={res_std:.4g}  flux={x.sum():.4g}")

        return x, {"residual_std": residual_std_history}
