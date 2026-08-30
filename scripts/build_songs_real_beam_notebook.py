#!/usr/bin/env python
"""
Builds and executes notebooks/songs_real_beam_demo.ipynb: a synthetic-but-
real-beam test of `Deconvolver2D1D` -- takes a genuine ALMA dirty beam,
built from the real (u, v) sample points of the calibrated TW Hya
measurement set (`data/twhya/twhya_visibilities.npz`, see
`data/twhya/export_visibilities.py`), and applies it to a *different* cube
than it was ever observed with: the SONGS simulated galaxy cube
(`data/songs-cubes/raw_cube.h5`), which comes with a fully known ground
truth. Unlike `real_data_demo.ipynb` (real beam + real data, no ground
truth) or `deconvolution_demo.ipynb` (mock beam + toy cube, ground truth),
this notebook gets both a real dirty beam *and* a known ground truth.

Building the beam via `psf.dirty_beam_from_uv` directly from the real (u, v)
points at a grid size matched to the SONGS cube (rather than loading the
already-256-pixel CASA-gridded `twhya_psf_cube.fits` and rebinning it down)
avoids interpolation artifacts entirely -- an earlier version of this
notebook did the rebin and it visibly broke the beam's centro-symmetry;
gridding the raw (u, v) points fresh at the target size does not have that
problem. The raw MS only stores each baseline once (i < j), so the (u, v)
points are mirrored (u, v) and (-u, -v) before gridding, same as
`psf.uv_coverage`'s own convention for its mock antenna layouts, to get an
exactly centro-symmetric, real-valued beam.

Cell-execution harness is shared across all build_*_notebook.py scripts --
see notebook_builder.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_builder import NotebookBuilder  # noqa: E402

nb = NotebookBuilder("notebooks/songs_real_beam_demo.ipynb", chdir_to_notebooks=True)
md, code = nb.md, nb.code


# ---------------------------------------------------------------------------
md(r"""
# 2D-1D wavelet deconvolution with a real ALMA dirty beam, on a simulated galaxy cube (known ground truth)

Combines two things that don't otherwise appear together in this repo's
other notebooks:

- a **genuine ALMA dirty beam**, built directly from the real (u, v) sample
  points of the calibrated TW Hya measurement set
  (`data/twhya/twhya_visibilities.npz`, exported by
  `data/twhya/export_visibilities.py`) via `psf.dirty_beam_from_uv` --
  instead of the mock antenna-layout beams used in the toy notebook, or
  loading the already-gridded `twhya_psf_cube.fits` (which is fixed at
  CASA's own 256x256 pixel grid and would need rebinning/interpolation to
  match a different cube's grid).
- a cube with **known ground truth** (`data/songs-cubes/raw_cube.h5`, a
  simulated 3-galaxy cube), instead of TW Hya's real (truth-free) data.

The real (u, v) points have nothing physically to do with the SONGS cube's
grid (100x100 px, 2.75 kpc/px, 20 km/s/channel) -- this notebook does not
pretend otherwise. It grids those (u, v) points directly at a size matched
to the SONGS cube's spatial grid (rather than gridding at CASA's native
scale and then resampling), giving a real, sidelobe-bearing dirty beam with
an actual known answer to check the deconvolution against.

Pipeline:
1. load the real (u, v) points for one representative channel, mirror them
   for Hermitian symmetry, and grid a dirty beam at the SONGS cube's own
   spatial size,
2. convolve every channel of the SONGS cube with that one shared beam
   (+ noise) -> dirty cube,
3. fit a Gaussian clean beam to the dirty beam's main lobe, and convolve the
   *true* cube with it -> a resolution-matched ground truth ("clean image")
   to compare against, exactly as classic CLEAN's own restored-image
   convention does,
4. run `Deconvolver2D1D` on the dirty cube, convolve its output with the
   same clean beam -> restored model, and compare against the ground truth
   from step 3.
""")

code(r"""
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

import numpy as np
import h5py
import matplotlib.pyplot as plt

from psf import dirty_beam_from_uv, beam_fwhm_pixels
from deconvolver import Deconvolver2D1D, convolve_cube
from clean import make_gaussian_beam, hogbom_clean_cube, restore_clean_cube

np.random.seed(0)
plt.rcParams['figure.facecolor'] = 'white'
DATA_DIR = '../data'
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Load the SONGS simulated cube (ground truth) and real ALMA (u, v) points
""")

code(r"""
with h5py.File(os.path.join(DATA_DIR, 'songs-cubes', 'raw_cube.h5'), 'r') as f:
    true_cube = f['cube'][:].astype(np.float64)
    spatial_res_kpc = f.attrs['spatial_resolution_kpc_per_px']
    spectral_res_kms = f.attrs['spectral_resolution_km_s']
    n_gals = int(f.attrs['n_gals'])

nz, ny, nx = true_cube.shape
peak_chan = int(np.argmax(true_cube.sum(axis=(1, 2))))
print(f'SONGS cube shape (nz, ny, nx) = {true_cube.shape}, {n_gals} galaxies')
print(f'spatial resolution = {spatial_res_kpc} kpc/px, spectral resolution = {spectral_res_kms} km/s/channel')
print(f'true cube min/max/sum = {true_cube.min():.4g} / {true_cube.max():.4g} / {true_cube.sum():.4g}')
print(f'brightest channel (by total flux) = {peak_chan}')
""")

code(r"""
vis_data = np.load(os.path.join(DATA_DIR, 'twhya', 'twhya_visibilities.npz'))
u_lambda, v_lambda = vis_data['u_lambda'], vis_data['v_lambda']  # (n_chan, n_uv)
k0 = u_lambda.shape[0] // 2
u0, v0 = u_lambda[k0], v_lambda[k0]
print(f'real TW Hya (u, v): {u0.shape[0]} averaged uv points (channel {k0} of {u_lambda.shape[0]})')

# the MS stores each baseline once (i < j); mirror for Hermitian (u, v) <-> (-u, -v)
# symmetry, same convention psf.uv_coverage uses for the mock antenna layouts.
u_full = np.concatenate([u0, -u0])
v_full = np.concatenate([v0, -v0])
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Grid a real ALMA dirty beam directly at the SONGS cube's spatial size

`grid_size` is set one pixel larger than the SONGS cube's spatial size
(101 rather than 100): `dirty_beam_from_uv`'s FFT-shift bookkeeping is only
exactly centro-symmetric for an odd grid size (a true center pixel to
reflect about). `deconvolver.convolve_cube` doesn't require the beam and
cube to share a shape -- it convolves at `cube_size + beam_size - 1` and
crops back to the cube's own size -- so a 101x101 beam works fine against
the 100x100 SONGS cube.
""")

code(r"""
beam, sampling = dirty_beam_from_uv(u_full, v_full, grid_size=101, natural_weighting=False)
print(f'beam shape = {beam.shape}, peak = {beam.max():.3f}, sum = {beam.sum():.3e}')
print(f'main-lobe FWHM = {beam_fwhm_pixels(beam):.2f} px')
print(f'beam is centro-symmetric: {np.allclose(beam, beam[::-1, ::-1], atol=1e-6)}')
print(f'uv sampling fill fraction = {(sampling > 0).mean():.3f}')

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
im0 = axs[0].imshow(beam, origin='lower', cmap='afmhot', vmin=-0.15, vmax=0.5)
axs[0].set_title('Real ALMA dirty beam (from real u,v, gridded at SONGS scale)')
plt.colorbar(im0, ax=axs[0], fraction=0.046)
im1 = axs[1].imshow(np.log10(sampling + 1e-3), origin='lower', cmap='viridis')
axs[1].set_title('Gridded (u, v) sampling (log scale)')
plt.colorbar(im1, ax=axs[1], fraction=0.046)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Dirty cube (true cube (*) real beam + noise) and the clean-beam ground truth

Every channel of the true cube is convolved with the same shared 2D beam
(one shared beam across all channels, same convention the toy notebook
uses). Gaussian noise is added at a modest peak SNR, consistent with the
"realistic thermal noise" run described in `README.md`.

For the ground truth, a Gaussian **clean beam** is fit to the dirty beam's
own main lobe (`clean.make_gaussian_beam`, flux-conserving) and convolved
with the *true* cube -- this is the standard CLEAN convention for what
"ground truth at the achievable resolution" means: no deconvolution method
can recover detail finer than the beam actually constrains, so comparing
against the raw true cube would be an unfair, too-strict target.
""")

code(r"""
dirty_noiseless = convolve_cube(true_cube, beam)

target_peak_snr = 40.0
sigma_noise = dirty_noiseless.max() / target_peak_snr
rng = np.random.default_rng(0)
dirty_cube = dirty_noiseless + rng.normal(0.0, sigma_noise, size=dirty_noiseless.shape)

fwhm = beam_fwhm_pixels(beam)
clean_beam = make_gaussian_beam(beam.shape, fwhm, normalize='sum')
ground_truth = convolve_cube(true_cube, clean_beam)

print(f'clean beam FWHM = {fwhm:.2f} px, sum = {clean_beam.sum():.4f} (flux-conserving)')
print(f'dirty cube (noiseless) min/max = {dirty_noiseless.min():.4g} / {dirty_noiseless.max():.4g}')
print(f'injected noise sigma = {sigma_noise:.4g} (peak SNR = {target_peak_snr:.0f})')
print(f'true total flux = {true_cube.sum():.4g}, ground-truth (clean-beam-convolved) flux = {ground_truth.sum():.4g}')
""")

code(r"""
# --- Required figure: dirty beam, clean beam, dirty channel, ground-truth channel ---
fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))

im0 = axs[0].imshow(beam, origin='lower', cmap='afmhot', vmin=-0.15, vmax=0.5)
axs[0].set_title('Real ALMA dirty beam')
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(clean_beam, origin='lower', cmap='afmhot')
axs[1].set_title(f'Fitted clean beam (FWHM={fwhm:.1f}px)')
plt.colorbar(im1, ax=axs[1], fraction=0.046)

im2 = axs[2].imshow(dirty_cube[peak_chan], origin='lower', cmap='inferno')
axs[2].set_title(f'Dirty cube, channel {peak_chan}')
plt.colorbar(im2, ax=axs[2], fraction=0.046)

im3 = axs[3].imshow(ground_truth[peak_chan], origin='lower', cmap='inferno', vmin=0)
axs[3].set_title(f'Ground truth (clean beam (*) true), channel {peak_chan}')
plt.colorbar(im3, ax=axs[3], fraction=0.046)

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. 2D-1D wavelet deconvolution, then restore with the clean beam

Same `Deconvolver2D1D` FISTA loop used throughout this repo, then the
wavelet-sparse model is convolved with the *same* clean beam used for the
ground truth in step 3 -- putting both on the same footing ("what does this
method recover, at the resolution the beam actually supports") before
comparing.
""")

code(r"""
deconvolver = Deconvolver2D1D(
    num_scales_2d=4, num_scales_1d=3,
    threshold_type='soft', positivity=True, verbose=True,
)
model, history = deconvolver.deconvolve(
    dirty_cube, beam, sigma_noise,
    n_iter=100, k_start=6.0, k_end=2.5, fista=True,
)
""")

code(r"""
model_restored = convolve_cube(model, clean_beam)

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

rows = [
    ('Dirty (no deconvolution)', dirty_cube.sum(), rmse(dirty_cube, ground_truth)),
    ('2D-1D wavelet model (raw, unrestored)', model.sum(), rmse(model, true_cube)),
    ('2D-1D wavelet, restored with clean beam', model_restored.sum(), rmse(model_restored, ground_truth)),
]
print(f'ground-truth (clean-beam) total flux = {ground_truth.sum():.4g}')
print(f'true (unconvolved) total flux         = {true_cube.sum():.4g}\n')
print(f"{'method':40s} {'flux':>10s} {'RMSE':>10s} {'x better than dirty':>20s}")
rmse_dirty = rows[0][2]
for name, flux, e in rows:
    print(f'{name:40s} {flux:10.3f} {e:10.4f} {rmse_dirty/e:19.2f}x')
""")

code(r"""
# Moment 0 (channel-summed intensity) for the true (raw, unconvolved) cube,
# the raw dirty cube, and the raw (unrestored -- no clean-beam convolution)
# wavelet model, top row; one representative channel slice of each, bottom row.
true_mom0 = true_cube.sum(axis=0)
dirty_mom0 = dirty_cube.sum(axis=0)
model_mom0 = model.sum(axis=0)

fig, axs = plt.subplots(2, 3, figsize=(15, 9))

im00 = axs[0, 0].imshow(true_mom0, origin='lower', cmap='inferno', vmin=0)
axs[0, 0].set_title('True cube (raw, unconvolved), moment 0')
plt.colorbar(im00, ax=axs[0, 0], fraction=0.046)

im01 = axs[0, 1].imshow(dirty_mom0, origin='lower', cmap='inferno')
axs[0, 1].set_title('Dirty cube, moment 0')
plt.colorbar(im01, ax=axs[0, 1], fraction=0.046)

im02 = axs[0, 2].imshow(model_mom0, origin='lower', cmap='inferno', vmin=0)
axs[0, 2].set_title('2D-1D wavelet model, moment 0 (raw, no clean beam)')
plt.colorbar(im02, ax=axs[0, 2], fraction=0.046)

im10 = axs[1, 0].imshow(true_cube[peak_chan], origin='lower', cmap='inferno', vmin=0)
axs[1, 0].set_title(f'True cube, channel {peak_chan}')
plt.colorbar(im10, ax=axs[1, 0], fraction=0.046)

im11 = axs[1, 1].imshow(dirty_cube[peak_chan], origin='lower', cmap='inferno')
axs[1, 1].set_title(f'Dirty cube, channel {peak_chan}')
plt.colorbar(im11, ax=axs[1, 1], fraction=0.046)

im12 = axs[1, 2].imshow(model[peak_chan], origin='lower', cmap='inferno', vmin=0)
axs[1, 2].set_title(f'2D-1D wavelet model, channel {peak_chan} (raw, no clean beam)')
plt.colorbar(im12, ax=axs[1, 2], fraction=0.046)

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Classic Hogbom CLEAN, for comparison

Same from-scratch Hogbom CLEAN (`src/clean.py`) used as the baseline
throughout this repo: run independently per channel against the real ALMA
beam, then restored with the same Gaussian clean beam used for the ground
truth above (`restore_clean_cube` fits its own clean beam to `beam`'s main
lobe internally -- it comes out identical to `clean_beam` computed in step 3,
since both use the same `beam_fwhm_pixels(beam)` fit).
""")

code(r"""
model_clean, residual_clean, n_components = hogbom_clean_cube(
    dirty_cube, beam, sigma_noise, gain=0.15, threshold_sigma=3.0,
    n_iter_max=500, verbose=True,
)
restored_clean, _ = restore_clean_cube(model_clean, residual_clean, beam)
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Soft thresholding's amplitude bias, and fixing it with reweighting

Soft thresholding is biased: it shrinks *every* wavelet coefficient above
threshold by the same fixed amount, so even genuinely strong, well-detected
structure gets systematically pulled down (the flux numbers above already
show this: the raw wavelet model recovers only a fraction of the true
flux). `Deconvolver2D1D.deconvolve_reweighted` corrects this with
iteratively-reweighted L1 (Candes/Wakin/Boyd 2008): after an initial soft-
threshold solve, it re-solves a few more times, each round replacing the
fixed threshold with a per-coefficient one derived from the *previous*
round's coefficients -- coefficients that came out large keep getting
smaller thresholds (less shrinkage) each round, while coefficients that
never clear the detection bar keep being suppressed normally. See
`_reweighted_threshold_subbands` in `deconvolver.py` for the exact formula.
""")

code(r"""
model_rw, history_rw = deconvolver.deconvolve_reweighted(
    dirty_cube, beam, sigma_noise,
    n_reweight=4, n_iter_first=100, n_iter_reweight=40,
    k_start=6.0, lam=2.5, fista=True,
)
model_rw_restored = convolve_cube(model_rw, clean_beam)
""")

code(r"""
rows = [
    ('Dirty (no deconvolution)', dirty_cube.sum(), rmse(dirty_cube, ground_truth)),
    ('Hogbom CLEAN (restored)', restored_clean.sum(), rmse(restored_clean, ground_truth)),
    ('2D-1D wavelet, soft (restored)', model_restored.sum(), rmse(model_restored, ground_truth)),
    ('2D-1D wavelet, soft + reweighted (restored)', model_rw_restored.sum(), rmse(model_rw_restored, ground_truth)),
]
print(f'ground-truth (clean-beam) total flux = {ground_truth.sum():.4g}\n')
print(f"{'method':44s} {'flux':>10s} {'% of truth':>11s} {'RMSE':>10s} {'x better than dirty':>20s}")
rmse_dirty = rows[0][2]
for name, flux, e in rows:
    print(f'{name:44s} {flux:10.3f} {100*flux/ground_truth.sum():10.0f}% {e:10.5f} {rmse_dirty/e:19.2f}x')
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Recovery: ground truth vs. dirty vs. CLEAN vs. 2D-1D wavelet (soft vs. reweighted)
""")

code(r"""
def compare_panels(k, title):
    fig, axs = plt.subplots(1, 5, figsize=(23, 4.6))
    vmax = max(ground_truth[k].max(), model_rw_restored[k].max(), 1e-6)
    panels = [
        ('Ground truth (clean beam (*) true)', ground_truth[k], 0),
        ('Dirty', dirty_cube[k], None),
        ('Hogbom CLEAN', restored_clean[k], None),
        ('2D-1D wavelet, soft', model_restored[k], 0),
        ('2D-1D wavelet, soft + reweighted', model_rw_restored[k], 0),
    ]
    for ax, (name, img, vmin) in zip(axs, panels):
        im = ax.imshow(img, origin='lower', cmap='inferno',
                        vmin=vmin, vmax=vmax if vmin is not None else None)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

compare_panels(peak_chan, f'Channel {peak_chan} (brightest, by total flux)')
""")

code(r"""
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(np.arange(1, len(history['residual_std']) + 1), history['residual_std'])
axs[0].axhline(sigma_noise, color='k', linestyle='--', linewidth=1, label='injected noise sigma')
axs[0].set_xlabel('FISTA iteration')
axs[0].set_ylabel('std(dirty - H(model))')
axs[0].set_title('Soft threshold (single run)')
axs[0].legend()

axs[1].plot(np.arange(1, len(history_rw['residual_std']) + 1), history_rw['residual_std'])
for b in history_rw['round_boundaries'][:-1]:
    axs[1].axvline(b, color='gray', linestyle=':', linewidth=1)
axs[1].axhline(sigma_noise, color='k', linestyle='--', linewidth=1, label='injected noise sigma')
axs[1].set_xlabel('FISTA iteration (concatenated across rounds)')
axs[1].set_ylabel('std(dirty - H(model))')
axs[1].set_title('Soft + reweighted (dotted lines = new round)')
axs[1].legend()

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## Takeaways

- This is a **hybrid** test regime: a real, structurally messy ALMA dirty
  beam (genuine sidelobes from real (u, v) coverage, not a hand-drawn PSF)
  applied to a cube it never actually observed, purely to get a real beam's
  behavior *and* a known ground truth in the same notebook
  (`real_data_demo.ipynb` has the former without the latter;
  `deconvolution_demo.ipynb` has the latter with a mock beam instead).
- Building the beam directly from the real (u, v) points at the target grid
  size, instead of loading the CASA-gridded FITS PSF and rebinning it,
  avoids interpolation/resampling artifacts entirely and keeps the beam
  exactly centro-symmetric -- important since `Deconvolver2D1D`'s gradient
  step relies on the beam being self-adjoint (H^T == H).
- The real (u, v) points have no defined physical relationship to the SONGS
  cube's kpc/px grid -- gridding them at a matched pixel count is a
  pipeline/robustness check (does a genuinely messy, real sidelobe
  structure still deconvolve correctly against a known answer), not a
  claim about what ALMA could actually resolve on a galaxy at this
  physical scale.
- Comparing against a **clean-beam-convolved** ground truth (not the raw
  true cube) is the same convention CLEAN itself uses for its restored
  image -- it isolates how well the deconvolution recovers structure at the
  resolution the beam's own main lobe actually supports, rather than
  penalizing every method for not inventing detail the data never
  constrained.
- **Reweighting substantially closes the soft-threshold flux gap.** Plain
  soft-thresholding recovers only a small fraction of the true flux here --
  expected, since it shrinks every coefficient above threshold by the same
  fixed amount regardless of how strong it is. Iteratively reweighting
  (`deconvolve_reweighted`) nearly triples the recovered flux and further
  improves RMSE against the ground truth, by letting coefficients that
  proved themselves large in earlier rounds keep less of their amplitude
  shrunk in later ones -- see the printed flux/RMSE table and the
  side-by-side channel panels above for exactly how much.
- **Hogbom CLEAN, run independently per channel against this real beam, is
  a real baseline here too** -- unlike the toy notebook's mock beam, this
  beam's genuine sidelobe structure (from real, sparsely-sampled (u, v)
  coverage) gives CLEAN's delta-function component fitting a harder time,
  and CLEAN's restored image still adds its leftover residual back
  completely unfiltered (no denoising step), unlike the wavelet methods'
  built-in thresholding. Check the flux/RMSE table and panels above for how
  it stacks up against soft and reweighted-soft on this specific real beam.
""")


if __name__ == "__main__":
    nb.build_and_execute()
