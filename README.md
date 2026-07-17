# Deconvolver2D1D-ALMA

2D-1D wavelet **deconvolution** for dirty ALMA spectral cubes, adapted from
the 2D-1D wavelet **denoising** framework in
[`Denoiser2D1D-improved`](../Denoiser2D1D-improved). Same dictionary
(spatial starlet scales x spectral wavelet scales), different inverse
problem: this repo goes from a dirty-beam-convolved cube straight to a
"clean" cube, instead of CLEAN's iterative Dirac-delta (point-source)
component fitting.

## Layout

```
src/
  wavelet2d1d.py    Self-contained 2D-1D a trous/starlet transform (pure numpy;
                     no pysparse/CosmoStat dependency -- see module docstring
                     for how/why it differs from the original wrapper)
  psf.py             Toy but structurally real ALMA-like dirty beam: random
                     antenna layout -> multi-hour-angle uv coverage -> gridded
                     sampling function -> inverse FFT
  toy_cube.py        Spatially-varying (non-stationary) rotating-ring toy
                     cube: a single source that splits into two spatially
                     separated blobs in intermediate velocity channels
  deconvolver.py     Deconvolver2D1D: ISTA/FISTA sparse deconvolution using
                     the 2D-1D wavelet dictionary as the sparsity prior
notebooks/
  deconvolution_demo.ipynb   End-to-end toy demo (see below)
```

## Denoising -> deconvolution: what actually changes

### The original problem (`Denoiser2D1D`)

```
Y = X + N            (H = identity)
```

`Denoiser2D1D._denoise_iterative_hard` re-estimates a per-sub-band
significance mask from the current model each iteration, then re-applies
that mask directly to the **data's own** wavelet coefficients and
reconstructs. There's no explicit gradient step in that code, because when
`H = I` the gradient of the data-fidelity term `||Y - X||^2` w.r.t. `X` is
just `X - Y` -- so "threshold the data's coefficients" already *is* the
gradient step, in disguise. That shortcut is what makes the original
denoiser so simple.

### The deconvolution problem (this repo)

```
Y = H(X) + N          H = per-channel 2D convolution with the dirty beam
```

`H` is no longer the identity, so the shortcut breaks and an explicit
forward-backward (ISTA/FISTA) step has to be added. Each iteration of
`Deconvolver2D1D.deconvolve` does:

1. **Residual** (this *is* new): `r = Y - H(x_aux)`
2. **Gradient step** (this *is* new): `z = x_aux + mu * H^T(r)`
   `H^T` is the adjoint of the dirty-beam convolution. Our mock beam is
   centro-symmetric by construction (baselines always occur as +/- pairs,
   see `psf.py`), so `H^T == H` and the same convolution routine is reused
   for both.
3. **Sparsify** (same idea as the original, applied to `z` instead of the raw data): forward 2D-1D transform, then
   per-sub-band soft/hard thresholding at `k_sigma` x (per-sub-band noise level).
4. **Reconstruct**: inverse 2D-1D transform.
5. **Positivity**: `max(0, .)`, same as the original.
6. **(FISTA) momentum**: extrapolate `x_aux` for the next iteration -- this
   is what turns plain ISTA (roughly the deconvolution analogue of the
   original's iterative hard-thresholding loop) into something that
   converges in a reasonable number of iterations.

This is proximal-gradient descent on

```
argmin_x  (1/2) * ||Y - H(x)||^2  +  lambda * ||W(x)||_1
```

i.e. exactly the objective solved by sparse radio-interferometric imagers
like MORESANE and PURIFY/SARA, just using this repo's own 2D-1D starlet
dictionary `W` instead of a single IUWT basis or a Dirac+wavelet
dictionary.

### The one genuinely new wrinkle: the missing zero-spacing flux

An interferometer never measures the true zero-length baseline, so the
dirty beam has **zero DC response** (`psf.sum() == 0`, verified in the
notebook). That means `H` has a null space at the sky's most diffuse mode,
and the data-fidelity term is completely blind to flux added along it. The
original denoiser always left its coarsest wavelet sub-band untouched
(harmless there, since `Y = X + N` bounds everything). Doing the same here
gives FISTA an unpenalized direction to grow in every iteration, and the
reconstructed flux diverges instead of converging (this was verified
empirically while building this repo -- see `_threshold_subbands` in
`deconvolver.py` for the fix: the coarsest sub-band is zeroed rather than
passed through). The practical upshot, also visible in the demo notebook,
is that the recovered cube's total flux is systematically *lower* than the
true cube's -- which is not a bug, it's the same fundamental "short-spacing
problem" every real ALMA continuum/line map has, and any deconvolution
method (CLEAN included) is subject to it.

### Noise-level calibration also needs the PSF in the loop

`Denoiser2D1D._decompose_and_estimate_noise` transforms an independent
white-noise realization and reads off its per-sub-band standard deviation.
Here, the noise entering the wavelet transform at each iteration has
already passed through `mu * H^T`, which reshapes (and generally amplifies,
through the beam's sidelobe structure) its variance -- so the calibration
noise realization has to be pushed through that same chain
(`_estimate_subband_noise_after_gradient` in `deconvolver.py`) rather than
transformed directly.

## The demo notebook

`notebooks/deconvolution_demo.ipynb`:

1. Builds a toy rotating-ring spectral cube (`toy_cube.rotating_ring_cube`)
   where the line-of-sight velocity field varies continuously across the
   source, so a single, spatially-connected ring shows up as **two**
   spatially separated blobs in most velocity channels and merges to one
   blob only at the rotation curve's extremes.
2. Builds a toy but structurally real ALMA-like dirty beam
   (`psf.mock_alma_dirty_beam`) from a random antenna layout and simulated
   Earth-rotation (multi-hour-angle) uv coverage.
3. Convolves the cube with the dirty beam channel-by-channel and adds
   noise, then shows the dirty beam and the dirty central channel side by
   side.
4. Runs `Deconvolver2D1D` and compares true / dirty / recovered channel
   maps, including the channel where the source visibly splits into two
   blobs -- showing the beam's sidelobe structure corrupting that
   morphology in the dirty map, and the deconvolution recovering it.
5. Sanity-checks convergence (residual std vs. iteration) and blob count
   (true vs. recovered) with a small dependency-free connected-components
   check in `toy_cube.count_blobs`.

## Requirements

Pure `numpy` + `matplotlib`. No `scipy`, no `pysparse`/CosmoStat, so it runs
anywhere the original `Denoiser2D1D-improved` repo's `mse`/`mse_flux`
U-Net paths would, without the `pysparse` C++ bindings needed for its
`mse_flux_hallucination` loss.
