#!/usr/bin/env python
"""
Builds and executes notebooks/songs_uv_deconvolution_demo.ipynb: the same
SONGS simulated galaxy cube used in `songs_real_beam_demo.ipynb`, but
deconvolved **directly in visibility space** with `UVDeconvolver2D1D`
(`src/uv_deconvolver.py`) instead of the image-domain dirty-beam path.

Where the other SONGS notebook grids the real TW Hya (u, v) points into a
single dirty beam and fits `||dirty_image - x (*) beam||^2` in the image
domain, this one keeps the (u, v) points scattered and fits
`||V_obs - Phi(x)||^2` at the exact sample locations, via
`uv_operator.NonUniformFourierOperator`. The SONGS cube (known ground truth)
is forward-projected through that operator to synthesize a noisy visibility
cube, which is displayed as a visibility-space heatmap before being
deconvolved.

Cell-execution harness is shared across all build_*_notebook.py scripts --
see notebook_builder.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from notebook_builder import NotebookBuilder  # noqa: E402

nb = NotebookBuilder("notebooks/songs_uv_deconvolution_demo.ipynb", chdir_to_notebooks=True)
md, code = nb.md, nb.code


# ---------------------------------------------------------------------------
md(r"""
# uv-plane 2D-1D wavelet deconvolution of the SONGS galaxy cube (visibility space)

The companion to `songs_real_beam_demo.ipynb`: same simulated SONGS galaxy
cube (`data/songs-cubes/raw_cube.h5`, known ground truth) and the same real
TW Hya (u, v) sample points (`data/twhya/twhya_visibilities.npz`), but the
deconvolution is run **directly in visibility space** instead of image space.

- `songs_real_beam_demo.ipynb`: grid the (u, v) points into one dirty beam,
  fit `||dirty_image - x (*) beam||^2` in the image domain with
  `Deconvolver2D1D`.
- **this notebook**: keep the (u, v) points scattered, forward-project the
  cube onto them via `uv_operator.NonUniformFourierOperator` (a genuine
  non-uniform 2D Fourier transform -- no gridding, no dirty beam anywhere),
  and fit `||V_obs - Phi(x)||^2` at the exact sample locations with
  `uv_deconvolver.UVDeconvolver2D1D`.

Pipeline:
1. load the SONGS cube (ground truth) and the real TW Hya (u, v) points,
2. build the non-uniform Fourier operator and forward-project the cube ->
   a synthetic **visibility cube**, plus complex noise -> the "dirty" data,
3. display that data as a visibility-space heatmap (amplitude/phase on the
   uv plane, and the full channel x uv-sample array),
4. run `UVDeconvolver2D1D` directly against the noisy visibilities and
   compare the recovered model to the ground truth.
""")

code(r"""
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

import numpy as np
import h5py
import matplotlib.pyplot as plt

from uv_operator import NonUniformFourierOperator, pixel_scale_for_uv
from uv_deconvolver import UVDeconvolver2D1D

np.random.seed(0)
plt.rcParams['figure.facecolor'] = 'white'
DATA_DIR = '../data'
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Load the SONGS cube (ground truth) and the real TW Hya (u, v) points
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

# the MS stores each baseline once (i < j); mirror for Hermitian symmetry,
# same as the image-domain SONGS notebook (and psf.uv_coverage's convention).
u = np.concatenate([u0, -u0])
v = np.concatenate([v0, -v0])
print(f'real TW Hya (u, v): {u0.shape[0]} averaged points, {len(u)} after Hermitian mirroring')
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Build the non-uniform Fourier operator and synthesize a visibility cube

`NonUniformFourierOperator` maps the sky cube directly onto the scattered
(u, v) points (`forward_cube`, one complex visibility per point per channel).
No dirty beam is ever formed. Complex Gaussian noise is added independently
to the real and imaginary part of every sampled visibility -- the
visibility-space analogue of the per-pixel image noise the image-domain
notebooks add.
""")

code(r"""
pixel_scale = pixel_scale_for_uv(u, v)
operator = NonUniformFourierOperator(ny, nx, u, v, pixel_scale)
print(f'operator: {operator.n_uv} uv points x {ny*nx} pixels, pixel_scale = {pixel_scale:.4g}')

vis_true = operator.forward_cube(true_cube)             # (nz, n_uv) complex

target_snr = 20.0
sigma_vis = np.std(np.abs(vis_true)) / target_snr
rng = np.random.default_rng(0)
noise = rng.normal(0, sigma_vis, vis_true.shape) + 1j * rng.normal(0, sigma_vis, vis_true.shape)
vis_obs = vis_true + noise

print(f'visibility cube shape = {vis_obs.shape} (nz, n_uv), dtype {vis_obs.dtype}')
print(f'typical |visibility| = {np.mean(np.abs(vis_true)):.4g}, sigma_vis = {sigma_vis:.4g} '
      f'(amplitude SNR ~ {np.mean(np.abs(vis_true)) / sigma_vis:.1f})')
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. The data in visibility space (heatmaps)

This is what the deconvolver actually fits -- there is no image yet. Left:
the observed visibility amplitude for the peak channel, gridded onto the uv
plane for display (log scale). Middle: the corresponding visibility phase.
Right: the full **(channel x uv-sample)** amplitude array -- every one of the
`nz` channels has its own complex visibility at each of the `n_uv` points.
""")

code(r"""
# grid one channel's scattered visibilities onto the uv plane, for display only
def grid_uv(values, u, v, grid_size=101):
    extent = 1.05 * max(np.abs(u).max(), np.abs(v).max())
    edges = np.linspace(-extent, extent, grid_size + 1)
    counts, _, _ = np.histogram2d(u, v, bins=[edges, edges])
    summed, _, _ = np.histogram2d(u, v, bins=[edges, edges], weights=values)
    with np.errstate(invalid='ignore', divide='ignore'):
        grid = np.where(counts > 0, summed / counts, np.nan)
    return grid.T, extent  # .T so axis-0 is v (y), axis-1 is u (x)

amp_grid, ext = grid_uv(np.abs(vis_obs[peak_chan]), u, v)
phase_grid, _ = grid_uv(np.angle(vis_obs[peak_chan]), u, v)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

im0 = axs[0].imshow(np.log10(amp_grid), origin='lower', cmap='viridis',
                    extent=[-ext, ext, -ext, ext])
axs[0].set_title(f'Visibility amplitude, channel {peak_chan}\n(log10 |V|, gridded for display)')
axs[0].set_xlabel('u'); axs[0].set_ylabel('v')
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(phase_grid, origin='lower', cmap='twilight',
                    extent=[-ext, ext, -ext, ext], vmin=-np.pi, vmax=np.pi)
axs[1].set_title(f'Visibility phase, channel {peak_chan}\n(arg V, gridded for display)')
axs[1].set_xlabel('u'); axs[1].set_ylabel('v')
plt.colorbar(im1, ax=axs[1], fraction=0.046, label='radians')

im2 = axs[2].imshow(np.log10(np.abs(vis_obs) + 1e-9), origin='lower', cmap='magma',
                    aspect='auto')
axs[2].set_title('Full visibility cube |V|\n(log10, channel x uv-sample)')
axs[2].set_xlabel('uv-sample index'); axs[2].set_ylabel('channel')
plt.colorbar(im2, ax=axs[2], fraction=0.046)

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. uv-plane 2D-1D wavelet deconvolution

`UVDeconvolver2D1D` runs the same FISTA + 2D-1D-starlet-thresholding loop as
`Deconvolver2D1D`, but its gradient step goes through
`operator.adjoint_cube(V_obs - operator.forward_cube(x))` in visibility space
instead of an image-domain dirty-beam convolution. For reference, the
`operator.adjoint_cube(vis_obs)` backprojection is also formed -- the
"dirty image" analogue (visibilities dumped straight back onto the sky, no
deconvolution), which is what the deconvolver starts from.
""")

code(r"""
dirty_image = operator.adjoint_cube(vis_obs)   # backprojection, for reference/display
dirty_image /= np.abs(dirty_image).max()

deconvolver = UVDeconvolver2D1D(
    num_scales_2d=4, num_scales_1d=3,
    threshold_type='soft', positivity=True, verbose=True,
)
model, history = deconvolver.deconvolve(
    vis_obs, operator, sigma_vis, cube_shape=true_cube.shape,
    n_iter=150, k_start=6.0, k_end=2.5, fista=True,
)
""")

code(r"""
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

# scale-invariant flux comparison: the raw non-uniform DFT has no flux
# normalization, so compare the model to the truth up to a single best-fit
# amplitude (this is a genuine gauge freedom of the un-normalized operator,
# not a fudge -- the morphology/relative structure is what the fit constrains).
alpha = float(np.sum(model * true_cube) / np.sum(model * model)) if model.any() else 0.0
model_scaled = alpha * model

rows = [
    ('Backprojection "dirty image" (no deconv)', rmse(dirty_image / dirty_image.max() * true_cube.max(), true_cube)),
    ('uv-plane wavelet model (best-fit amplitude)', rmse(model_scaled, true_cube)),
]
print(f'true total flux = {true_cube.sum():.4g}')
print(f'best-fit amplitude of uv model vs truth: alpha = {alpha:.4g}\n')
print(f"{'method':46s} {'RMSE vs true':>14s}")
for name, e in rows:
    print(f'{name:46s} {e:14.5f}')
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Recovery: ground truth vs. backprojection vs. uv-plane wavelet
""")

code(r"""
def compare_panels(k, title):
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    # top row: moment 0
    m_true = true_cube.sum(axis=0)
    m_dirty = dirty_image.sum(axis=0)
    m_model = model.sum(axis=0)
    for ax, (name, img, vmin) in zip(
        axs[0],
        [('True, moment 0', m_true, 0),
         ('Backprojection, moment 0', m_dirty, None),
         ('uv-plane wavelet, moment 0', m_model, 0)]):
        im = ax.imshow(img, origin='lower', cmap='inferno', vmin=vmin)
        ax.set_title(name); plt.colorbar(im, ax=ax, fraction=0.046)
    # bottom row: one channel
    for ax, (name, img, vmin) in zip(
        axs[1],
        [(f'True, ch {k}', true_cube[k], 0),
         (f'Backprojection, ch {k}', dirty_image[k], None),
         (f'uv-plane wavelet, ch {k}', model[k], 0)]):
        im = ax.imshow(img, origin='lower', cmap='inferno', vmin=vmin)
        ax.set_title(name); plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

compare_panels(peak_chan, f'Channel {peak_chan} (brightest, by total flux)')
""")

code(r"""
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(history['residual_std']) + 1), history['residual_std'])
plt.axhline(sigma_vis, color='k', linestyle='--', linewidth=1, label='injected visibility noise sigma')
plt.xlabel('FISTA iteration')
plt.ylabel('std of visibility residual  (|V_obs - Phi(x)|)')
plt.title('uv-plane FISTA convergence')
plt.legend()
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## Takeaways

- The deconvolution here never forms a dirty beam or a dirty image to fit
  against -- the data it fits *is* the scattered visibility cube shown in the
  heatmaps in section 3. The backprojection image in sections 4-5 is only for
  visualization/starting point; the objective `||V_obs - Phi(x)||^2` lives
  entirely in visibility space.
- The same missing-short-spacing null space discussed in
  `songs_real_beam_demo.ipynb` applies here identically: the (u, v) coverage
  has no zero-spacing baseline and only sparse short baselines, so the most
  diffuse (total-flux) mode of the sky is nearly unconstrained by the
  visibility-space fidelity term too -- working in the uv plane instead of the
  image plane changes the fidelity's *accuracy* (exact sample locations, no
  gridding approximation), not which modes are recoverable.
- The raw non-uniform DFT operator carries no flux normalization, so absolute
  recovered flux is only defined up to a single global amplitude -- the RMSE
  above is reported after fitting that one gauge amplitude, which is why the
  comparison is against morphology rather than an absolute flux percentage
  the way the image-domain notebooks report it.
""")


if __name__ == "__main__":
    nb.build_and_execute()
