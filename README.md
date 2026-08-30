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
  clean.py           Classic Hogbom CLEAN, from scratch, as the baseline
  io_fits.py          Loader for real ALMA FITS products (dirty cube,
                     per-channel dirty beam, CASA CLEAN benchmark), plus
                     PSF conditioning helpers (taper_psf, crop_psf_support)
  uv_operator.py      NonUniformFourierOperator: dense non-uniform Fourier
                     ("uv-plane") measurement operator, an alternative to
                     image-domain dirty-beam convolution (see below)
  uv_deconvolver.py   UVDeconvolver2D1D: FISTA + 2D-1D wavelet sparsity,
                     run directly against visibilities via uv_operator.py
  uv_real_data.py     Loader for exported real TW Hya visibilities
                     (data/export_visibilities.py's output) -- prepared,
                     not yet run against real data, see below
  alma_fourier.py     Measurement operators for real ALMA data, defined via
                     the normal operator N = A^H W A rather than a separate
                     forward/adjoint pair. CASA's own gridder/degridder does
                     the Fourier transforms. See "Real ALMA data" below.
  fista_2d1d.py       FISTA + 2D-1D starlet sparsity driven by an
                     alma_fourier operator
notebooks/
  deconvolution_demo.ipynb    End-to-end toy demo (see below)
  real_data_demo.ipynb        Same pipeline on a real TW Hya ALMA cube (see below)
  uv_deconvolution_demo.ipynb Toy demo of direct uv-plane deconvolution (see below)
scripts/
  split_line_ms.py             Cuts the CO(2-1) window out of the 55 GB
                               calibrated_final.ms.contsub (stage 0)
  deconvolve_alma.py           End-to-end real-data pipeline:
                               image -> validate -> deconvolve -> export
  build_demo_notebook.py       Builds/executes the toy notebook
  build_real_data_notebook.py  Builds/executes the real-data notebook
  build_uv_demo_notebook.py    Builds/executes the uv-plane toy notebook
  run_real_pipeline.py         Resumable/checkpointed CLEAN + wavelet runner
                                for the real cube (see below)
data/
  prep_wavelet_data.py       CASA script documenting exactly how the TW Hya
                             FITS products were produced (tclean calls,
                             niter=0 dirty + niter=5000 multiscale
                             benchmark). The FITS files themselves are
                             gitignored (large binary data).
  export_visibilities.py     CASA script to export real TW Hya visibilities
                             directly (for the uv-plane path) -- see below.
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

## Baseline: classic Hogbom CLEAN (`src/clean.py`)

For comparison, `src/clean.py` implements classic Hogbom CLEAN from scratch
-- the delta-function/point-source component fitting this whole repo is an
alternative to (see the very first line of this README). Run independently
per channel (CLEAN has no notion of the spectral axis): find the brightest
residual pixel, subtract `gain` x that peak's dirty-beam response recentered
there, repeat down to a 3-sigma stopping threshold, then restore (delta
model convolved with a Gaussian "clean beam" fit to the dirty beam's main
lobe, plus the leftover residual added back **unfiltered**).

That last point is the crux of the comparison in the demo notebook: CLEAN's
restored image keeps the data's full, unfiltered noise floor, while
`Deconvolver2D1D`'s thresholding denoises as an intrinsic part of every
FISTA iteration -- run both **soft** and **hard** (see `deconvolver.py`'s
`threshold_type`).

**With realistic thermal noise** (peak SNR 8, an earlier, noisier mock uv
coverage):

| method | recovered flux | RMSE vs. true | x better than dirty |
|---|---|---|---|
| dirty (no deconvolution) | 50% of true | 0.378 | 1.0x |
| Hogbom CLEAN (restored) | 67% of true | 0.352 | 1.1x |
| 2D-1D wavelet, soft threshold | 54% of true | 0.020 | 18.7x |
| 2D-1D wavelet, hard threshold | 96% of true | 0.021 | 18.0x |

CLEAN only modestly improves on the raw dirty map here (it removes some
sidelobe structure, but adds the noisy residual straight back); both
wavelet variants get more than an order of magnitude closer to the truth in
RMSE.

**With no noise** (the notebook's current default -- isolates the dirty
beam's effect from thermal-noise effects; see `psf.mock_alma_dirty_beam_rings`
below):

| method | recovered flux | RMSE vs. true | x better than dirty |
|---|---|---|---|
| dirty (no deconvolution) | -16% of true (!) | 0.045 | 1.0x |
| Hogbom CLEAN (restored) | 87% of true | 0.0092 | 4.9x |
| 2D-1D wavelet, soft threshold | 71% of true | 0.0111 | 4.0x |
| 2D-1D wavelet, hard threshold | 76% of true | 0.0114 | 3.9x |

With nothing to overfit, CLEAN can clean very deep (thousands of components)
and is actually the *best* RMSE here, edging out both wavelet variants --
this isolates CLEAN's real weakness to specifically not being able to tell
signal from noise the way the wavelet thresholding does, rather than to
fitting extended structure with delta functions per se. (The dirty map's
total flux going *negative*, despite the true cube being non-negative
everywhere, is another visible symptom of the missing zero-spacing flux
below.)

In both regimes, soft vs. hard threshold is its own bias/variance
trade-off: hard thresholding keeps surviving coefficients at full amplitude
(more flux recovered, less biased) at the cost of a speckled, higher-
variance reconstruction; soft thresholding shrinks every coefficient
(systematically undershoots peak brightness) in exchange for a cleaner
result. See the notebook for the full 5-way (true / dirty / CLEAN / soft /
hard) comparison on individual channel maps and the blob-count check.

## Toy ALMA dirty beam: concentric-fringe variant (`psf.ring_antenna_layout`)

`psf.mock_alma_dirty_beam` (fully random antenna radii) gives a dirty beam
with a speckled sidelobe field. Real ALMA dirty beam images more often show
a "target"/concentric-fringe pattern radiating out from the main lobe.
`psf.ring_antenna_layout` + `psf.mock_alma_dirty_beam_rings` reproduce that:
antennas are placed on a handful of discrete radii instead of continuously
random ones, so pairwise baseline lengths cluster onto a few dominant
"shells" -- each close to a fully-sampled annulus in the (u, v) plane once
rotated through many hour angles -- and the dirty beam becomes close to a
sum of a few Bessel-J0-like concentric rings at those baseline lengths, the
same basic mechanism that gives a filled circular aperture its Airy pattern.
`psf.azimuthal_ring_coherence` gives a quick numeric diagnostic (ratio of
a ring's azimuthal mean to its azimuthal scatter) for how ring-like vs.
speckle-like a given beam is.

## Doing it in the uv plane instead of image space

Everything above -- `Deconvolver2D1D`, `clean.py`, and the real TW Hya work
-- operates in **image (pixel/"Dirac") space**: the raw visibilities are
gridded onto a regular array and inverse-FFTed exactly once (what
`tclean(..., niter=0)` does), producing a single dirty image and a single,
fixed dirty beam; the forward model `H` is then a stationary 2D convolution
with that one beam. `uv_operator.py` and `uv_deconvolver.py` instead solve
the same sparse-recovery problem directly against the **scattered,
ungridded (u, v) sample points** an interferometer actually measures:

```
V(u, v) = integral  I(l, m) * exp(-2*pi*i*(u*l + v*m))  dl dm     (per baseline sample)
```

i.e. the forward operator `Phi` degrids straight from a sky model to
visibilities at their real, individual (u, v) locations, with no gridding
step and no dirty beam anywhere. This is what real compressed-sensing
radio-interferometric imagers (MORESANE, PURIFY/SARA -- see this repo's
early literature-search discussion) do, typically via an accelerated
non-uniform FFT (NUFFT).

### `NonUniformFourierOperator`: a dense, verified, non-gridding operator

`uv_operator.NonUniformFourierOperator` implements `Phi`/`Phi^H` as an
explicit dense matrix (`Fr`, `Fi`, the real/imaginary parts of the
non-uniform DFT matrix) rather than an accelerated NUFFT -- tractable at the
toy image sizes and uv-point counts used here (a few thousand of each), and
it makes the operator's correctness directly checkable rather than trusting
a gridding-kernel implementation. `uv_deconvolution_demo.ipynb` verifies two
things before trusting it inside FISTA:

1. **The adjoint identity** `Re(<Phi x, v>) == <x, Phi^H v>`, checked against
   random vectors (confirmed to machine precision, relative error ~1e-15).
2. **Consistency with the existing gridded/FFT dirty beam machinery**:
   `operator.adjoint` applied to unit-weight visibilities is the *direct*
   (ungridded) point-source response of those (u, v) points -- visually and
   structurally consistent with `psf.dirty_beam_from_uv`'s FFT-based dirty
   beam built from the same points, despite the two being computed by
   completely different code paths.

Building this exposed a real, previously-invisible bug: `psf.py`'s
`dirty_beam_from_uv` bins `np.histogram2d(u, v, ...)` with `u` onto the
array's row axis and `v` onto the column axis, whereas the standard imaging
convention (and this operator) has rows <-> v (m, y-like) and columns <-> u
(l, x-like) -- the two conventions are transposed relative to each other.
This was invisible in every earlier notebook because the mock ring beams are
close to circularly symmetric, so a transpose doesn't visibly change
anything; it became obvious immediately once a genuinely asymmetric (two
offset blobs) source was compared pixel-for-pixel against a physically
labeled uv-plane forward model. The demo notebook works around it locally
(swapping the argument order into `dirty_beam_from_uv` and into its own
gridding helper) rather than changing `psf.py`'s established convention,
which is internally self-consistent everywhere else it's used.

`operator.lipschitz_constant()` uses power iteration on `adjoint(forward(.))`
directly (there is no `max|FFT(psf)|^2` shortcut once the operator is a
non-uniform transform rather than a stationary convolution) -- the uv-plane
analogue of `deconvolver.lipschitz_constant`.

### `UVDeconvolver2D1D`: same FISTA loop, different `H`

`uv_deconvolver.UVDeconvolver2D1D` is algorithmically almost identical to
`Deconvolver2D1D` -- same 2D-1D starlet dictionary, same per-sub-band
soft/hard thresholding, same null-space handling (the coarsest sub-band is
still zeroed, reusing `deconvolver._threshold_subbands` directly: an
interferometer's uv sampling essentially never includes a true zero-length
baseline either way, gridded or not, so the sky's most diffuse mode stays
almost entirely unconstrained regardless of which domain the fit happens
in). The only thing that changes is what `H`/`H^T` *are*: `operator.
forward_cube`/`adjoint_cube` (visibility <-> image) instead of `convolve_cube`
(image <-> image). Noise calibration follows the same principle as
`_estimate_subband_noise_after_gradient` -- white *visibility*-domain noise
is propagated through `mu * adjoint_cube(...)` before its per-sub-band
wavelet-domain level is read off, rather than assuming it stays white.

### Toy demo results (`notebooks/uv_deconvolution_demo.ipynb`)

Both paths are run against the exact same noisy synthetic visibilities of a
small (40x40x17) rotating-ring toy cube: (a) grid them once into a dirty
image + dirty beam and run the existing image-domain `Deconvolver2D1D`, (b)
run `UVDeconvolver2D1D` directly against the ungridded visibilities.

| method | recovered flux | RMSE vs. true | x better than dirty |
|---|---|---|---|
| Gridded dirty image (no deconvolution) | 0% of true | 0.0222 | 1.0x |
| Image-domain wavelet (`Deconvolver2D1D`) | 40% of true | 0.0275 | 0.8x (worse than dirty) |
| uv-plane wavelet (`UVDeconvolver2D1D`) | 101% of true | 0.0105 | 2.1x |

This is an honest, unforced result, not tuned to make either path look
better: the uv-plane fit uses strictly more of the actual data (every
visibility individually, at its own exact (u, v) location) than the
image-domain path, which first collapses everything through one
histogram-binned dirty image/beam pair, and at this toy problem's scale that
shows up directly in both flux recovery and the kinematic-splitting
channel's blob count (uv-plane: 2/2 blobs recovered, matching truth;
image-domain: 1/2, the second blob's peak shrunk just under the detection
threshold -- consistent with soft-thresholding's known bias, not a bug).
Whether that gap holds up at a different noise level or source complexity
isn't established by one run; it's reported as-is.

The real cost, also visible in the notebook: `NonUniformFourierOperator`'s
dense matrix means every FISTA iteration costs `O(n_pixels * n_uv)`, versus
the image-domain path's `O(n_pixels * log(n_pixels))` FFT convolution --
exactly why this notebook uses a much smaller cube than
`deconvolution_demo.ipynb` (40x40x17 vs. 80x80x41).

### Real TW Hya visibilities: prepared, not yet run

`data/export_visibilities.py` is a CASA script (run the same way as
`prep_wavelet_data.py` -- it needs `casatools`, not available in the sandbox
this repo's other real-data tooling was built in) that exports the actual
calibrated TW Hya visibilities -- (u, v) in wavelength units per channel,
complex visibility values, weights -- for the same spw/40-channel selection
`prep_wavelet_data.py` images. Because `NonUniformFourierOperator` is a
dense matrix, real per-visibility counts (10^5-10^7 per channel before
averaging) are far too many to use directly, so the export script
time/baseline-averages down to ~3000 effective uv points per channel, an
explicit, documented simplification rather than a silent one.
`src/uv_real_data.py`'s `load_twhya_visibilities` reads the resulting
`twhya_visibilities.npz` with plain numpy and builds a single shared
`NonUniformFourierOperator` from the central channel's (u, v) (rather than
one operator per channel, which would need ~40x the memory/compute of the
toy demo for a dense operator) -- justified here specifically because the
40-channel window spans a narrow spectral line, so the channel-to-channel
(u, v) drift in wavelength units is small.

This code has been written and reviewed but **not run against real data**:
producing `twhya_visibilities.npz` requires a local CASA session (the same
constraint `prep_wavelet_data.py` already has), which hasn't happened yet.
Once that file exists, the natural next step is a `uv_real_data_demo.ipynb`
built and verified the same way `real_data_demo.ipynb` was for the
image-domain path -- not yet built, since building it against untested code
and no real data to check it with would just be speculation.

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
4. Runs `Deconvolver2D1D` **and** classic Hogbom CLEAN (`clean.py`) on the
   same dirty cube, and compares true / dirty / CLEAN / 2D-1D-wavelet
   channel maps, including the channel where the source visibly splits into
   two blobs.
5. Sanity-checks convergence (residual std vs. iteration) and blob count
   (true vs. recovered, per method) with a small dependency-free connected-
   components check in `toy_cube.count_blobs`.

## Real data: TW Hya (`notebooks/real_data_demo.ipynb`)

Everything above uses a toy cube and a toy dirty beam. `real_data_demo.ipynb`
runs the identical `Deconvolver2D1D` and `clean.py` code, unmodified, on a
genuine ALMA dataset: **TW Hya**, a well-studied nearby protoplanetary disk,
imaged in the CO(3-2) line at 345.796 GHz. `data/prep_wavelet_data.py`
documents the exact CASA commands used to produce the FITS products from the
calibrated measurement set:

```
tclean(..., specmode='cube', nchan=40, imsize=[256,256], cell='0.08arcsec',
       weighting='briggs', robust=0.5, niter=0)                # dirty cube + psf
tclean(..., deconvolver='multiscale', scales=[0,5,15],
       niter=5000, threshold='15mJy')                            # CASA benchmark, for comparison
```

`io_fits.load_twhya_data` loads and centrally crops these (256x256 -> 128x128,
still comfortably larger than the disk itself), replaces the primary-beam-
masked NaNs with 0, and estimates the noise level via a robust MAD statistic
over the whole (masked) cube.

### A real conditioning problem the toy data never exposed

The first real-data run completely failed to converge: `residual_std` stayed
pinned near its starting value regardless of how aggressively the threshold
schedule was tuned, and the recovered peak brightness stalled around 2% of
the true peak. The cause turned out to be specific to genuine ALMA beams and
invisible in the toy setup: `psf.mock_alma_dirty_beam*` is compact and decays
to ~0 well within its own array, but the real TW Hya dirty beam's sidelobes
do **not** decay within the imaged field of view -- even at the edge of the
full 256-pixel CASA image, the beam is still ~5-9% of its peak. `convolve_cube`
implements linear convolution via FFT (zero-pad, multiply, crop), and
convolving with a beam that's hard-truncated at a non-negligible amplitude
introduces an edge discontinuity that blows up the operator's Lipschitz
constant (measured: **L ~ 26755** for the untouched, full-support real beam,
vs. L ~ 17-235 for the toy beams) -- confirmed independently via power
iteration (L ~ 14790, same order of magnitude, ruling out an FFT-formula
artifact). With L that large, FISTA's step size `mu = 1/L` is small enough
that convergence is impractically slow.

The fix, in `io_fits.py` (applied only to the wavelet path -- Hogbom CLEAN
does purely local spatial-domain subtraction and has no such dependence, so
it keeps using the real, untouched beam):

- **`crop_psf_support`**: crop the beam to a centered 41-pixel window (still
  comfortably covers the main lobe and several sidelobe rings, ~1.6 arcsec
  radius at this data's 0.08 arcsec/pixel scale) before anything else.
- **`taper_psf`**: apply a radial raised-cosine window so the (now-cropped)
  beam goes smoothly to 0 at its own edge instead of being hard-truncated.

Combined, these bring L from 26755 down to **1653.5** -- a 16x larger usable
step size -- at the cost of discarding very-low-level far sidelobes that
mostly sit below the map's own noise floor anyway.

### Results (channel 17, the disk's peak-brightness channel; all methods agree on the peak's location, pixel (64, 64))

| method | peak (Jy/beam) | background std | x quieter than dirty |
|---|---|---|---|
| dirty (no deconvolution) | 0.904 | 0.0465 | 1.0x |
| CASA multiscale CLEAN (benchmark, niter=5000) | 1.735 | 0.0397 | 1.17x |
| Hogbom CLEAN (this repo, from scratch) | 0.138 | 0.0442 | 1.05x |
| 2D-1D wavelet, soft threshold (120 iterations) | 0.156 | 0.0022 | 21.6x |
| 2D-1D wavelet, hard threshold (120 iterations) | 0.166 | 0.0024 | 19.0x |

Two honest, non-cherry-picked observations the notebook calls out explicitly:

- **This repo's from-scratch Hogbom CLEAN badly underperforms CASA's
  production multiscale CLEAN** on real data (peak 0.138 vs. 1.735) --
  unsurprising, since it's a simple from-scratch single-scale implementation
  stopped at a plain 3-sigma threshold (~109 components/channel), not tuned
  or scale-aware like CASA's `deconvolver='multiscale'` run to 5000
  iterations. It's included as an apples-to-apples "simple point-source
  CLEAN" reference, not a claim of matching production imaging pipelines.
- **The wavelet deconvolution is still visibly under-converged after 120
  iterations** on this real, badly-conditioned problem: recovered peak
  brightness (~0.16) is well short of the dirty map's own peak (0.90) or
  CASA's (1.74), even though the convergence plot (`std(dirty - H(model))`
  vs. iteration) shows real, ongoing improvement rather than a stall. What
  it already gets right, even under-converged: the correct source location
  in every channel, and roughly a **20x lower background noise floor** than
  either CLEAN variant or the raw dirty map -- letting it run substantially
  more iterations (the toy notebook's much better-conditioned problem
  converges in ~70) would continue closing the peak-brightness gap.

`scripts/run_real_pipeline.py` is the resumable/checkpointed runner behind
these numbers -- real-data FISTA iterations are too slow to redo from
scratch in one shot every time the notebook is rebuilt, so progress is saved
to `scripts/.checkpoints/*.npz` and each invocation continues a shared
global threshold-decay schedule (`k_start=6.0 -> k_end=1.5`) from wherever
the last one left off.

## Requirements

Pure `numpy` + `matplotlib`, plus `astropy` for real-FITS I/O
(`io_fits.py`/`real_data_demo.ipynb` only -- everything else needs no extra
dependency). No `scipy`, no `pysparse`/CosmoStat, so it runs anywhere the
original `Denoiser2D1D-improved` repo's `mse`/`mse_flux` U-Net paths would,
without the `pysparse` C++ bindings needed for its `mse_flux_hallucination`
loss.

---

## Real ALMA data: `alma_fourier.py` + `deconvolve_alma.py`

Everything above works on a *given* dirty cube and dirty beam. This part goes
back to the visibilities of an actual ALMA observation --
`data/calibrated_final.ms.contsub`, a 55 GB 3-pointing 12 m-array mosaic of
IRAS F23007+0836 (NGC 7469): 1,248,030 rows, 4 spw x 1920 channels, baselines
14.3-3171 m, CO(2-1) redshifted to ~226.79 GHz.

### The formulation

Interferometric deconvolution never needs `A` and `A^H` separately. The
gradient of the weighted data-fidelity term only ever uses

    grad f(x) = A^H W A x - A^H W y = N x - d

so the operator to define is the **normal operator** `N`, not an adjoint
pair. This matters because CASA's gridder applies imaging weights that its
degridder does not: `im.ft` and `im.makeimage` are *not* adjoints of each
other, which is the unfixable flaw in `casa_uv_operator.py`. But one CASA
major cycle computes `d - N x` exactly, in one call, weights and all.

Two operators implement this, with the same interface:

* `CASAImager.normal(x)` -- a real degrid + grid through
  `casatools.synthesisimager`, carrying Briggs weighting, the w-term,
  per-channel chromatic uv scaling, flagging and the mosaic primary beams.
  Exact; one pass over the visibilities per call.
* `PSFNormalOperator.normal(x)` -- per-channel FFT convolution with the PSF
  on a zero-padded grid. Exact whenever `N` is shift-invariant, since then
  `N = B *` with `B = N delta` = the PSF. Two FFTs per channel. `L = max|OTF|`
  in closed form, so the FISTA step `mu = 1/L` is exact rather than estimated
  by power iteration.

### Verified, not assumed

The normalization (`dirty [Jy/beam] = psf [peak 1] (*) model [Jy/pixel]`) is
checked against the data rather than trusted:

| check | result |
|---|---|
| PSF peak per channel after normalization | exactly 1.000000, all 90 channels |
| 1 Jy delta through a real CASA degrid/grid vs. the PSF | peak ratio 0.999987, max error 0.35% |
| `PSFNormalOperator` delta round trip | 2.2e-16 |
| FFT operator vs. one real CASA major cycle | 0.55% rms, 4.7% max (inner quarter) |
| clean beam vs. PSF main lobe | 0.010 max abs (0.144 with the PA sign flipped) |

Three things this caught that would otherwise have been silent:

* **`specmode`, not `mode`.** The tool-level cube key is `specmode`; `mode` is
  accepted and ignored, collapsing the cube to one MFS channel.
* **Double normalization.** In cube mode `CubeMajorCycleAlgorithm` already
  divides by sumwt, so the normalizer's `divideresidualbyweight` divides a
  second time -- it scaled the dirty image down by 4.65e7 to a 1e-11 rms.
  Cube mode also needs `synthesisimager.normalizerinfo()` or gridding aborts
  with "imagename not specified".
* **The coarse wavelet band must NOT be zeroed here.** The image-domain
  deconvolver zeroes it because `beam.sum() == 0` -- no zero-spacing baseline.
  That is a property of the *image grid*, not the array: the uv cell is
  `1/field_of_view`, so a 12.8" field gives 16 kilo-lambda cells while the
  shortest baseline is 10.8 kilo-lambda, and the short baselines land inside
  the central uv cell. `sum(psf)` is 39-51, not 0, so the coarse band carries
  measured flux. `PSFNormalOperator.recommend_zero_coarse()` decides this from
  the data.

### Running it

```bash
python3 scripts/split_line_ms.py                  # 55 GB -> 12 GB line window
python3 scripts/deconvolve_alma.py --outdir results/ngc7469_co21
```

Stages are `image | validate | deconvolve | export` (`--stage`, repeatable),
each cached, so the solver can be re-run without re-gridding.

### Result

90 channels x 5 km/s (4610-5055 km/s), 320^2 at 0.04", mosaic gridder,
Briggs robust 0.5, beam 0.167" x 0.128". 50 FISTA iterations, ~8.5 min.

    dirty rms  1.591e-03 -> residual rms  7.542e-04 Jy/beam   (2.1x)
    dirty peak 2.728e-02 -> residual peak 5.287e-03 Jy/beam   (5.2x)

The recovered cube is physically sensible: 79% of the flux lies inside r < 3",
peaking in the 1-3" annuli -- NGC 7469's circumnuclear starburst ring -- and
the model is consistent with zero in the line-free edge channels, with
emission running 4665-5015 km/s and peaking at 2.04 Jy at 4945 km/s.

### Known limits

* **Two field/velocity mistakes are baked into the defaults as fixes.** With
  `phasecenter=""` CASA centres on field 0, but this mosaic *surrounds* the
  target, so the source sits 6.3" away and a small field images empty sky with
  the real emission aliasing in. And spw 0,2 alone clip the blue wing of the
  line (4600-4696 km/s exists only in spw 1,3). Both are now defaults in
  `ImagingConfig` with the measurement that established them.
* **Per-channel noise is not uniform** with all four spws: 4696-4955 km/s has
  all four, the wings only two. The MAD thresholding estimates one noise level
  per sub-band across the whole cube, so it under-thresholds the middle and
  over-thresholds the wings.
* **FISTA stalls after ~10 iterations** at a residual peak of 5.29e-3
  (7 sigma), located at r = 6.2" -- the field edge, where the mosaic makes `N`
  genuinely position-dependent and the shift-invariant PSF operator cannot
  represent it (the 4.7-11% max discrepancy in the CASA cross-check lives
  there too). Inside r < 3" the residual is 5.0 sigma. Fitting that emission
  properly needs `CASAImager.normal` as the solver's operator instead of the
  FFT one, at ~1 visibility pass per iteration.
* **Integrated flux: 404 Jy km/s** from the model (371 Jy km/s inside r < 4").
  The 676 Jy km/s that a plain sum over the PB-corrected restored cube gives
  is *not* the line flux -- it integrates residual noise over every pixel of
  the 12.8" field and is only an upper bound. Both are printed by the export
  stage. Neither has been compared against a CASA `tclean` run on the same
  data, which is the obvious missing benchmark.
