#!/usr/bin/env python
"""
Builds and executes notebooks/deconvolution_demo.ipynb from the cell
sources defined below, without depending on jupyter/nbconvert/nbformat
(none of which are needed to just run this repo). Each code cell is
executed in a single shared namespace; stdout and any matplotlib figure
created by the cell are captured and embedded as cell outputs, producing a
standard nbformat-v4 .ipynb that opens normally in Jupyter/VS Code with all
outputs already populated.

Usage: python3 scripts/build_demo_notebook.py
"""

import base64
import io
import json
import os
import sys
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
OUT_PATH = os.path.join(REPO_ROOT, "notebooks", "deconvolution_demo.ipynb")

CELLS = []  # list of ("markdown"|"code", source_string)


def md(text):
    CELLS.append(("markdown", text))


def code(text):
    CELLS.append(("code", text))


# ---------------------------------------------------------------------------
md(r"""
# 2D-1D wavelet deconvolution of a dirty ALMA spectral cube

Goes from a dirty-beam-convolved, noisy ALMA-like cube to a "clean" cube
using the `Deconvolver2D1D` sparse-recovery algorithm in `src/deconvolver.py`
-- a deconvolution counterpart of the `Denoiser2D1D` denoiser from
[`Denoiser2D1D-improved`](../../Denoiser2D1D-improved), built on the same
kind of 2D (spatial) x 1D (spectral) wavelet dictionary, but solving
`Y = H(X) + N` (dirty beam convolution) instead of `Y = X + N` (pure noise).
See the repo `README.md` for the full derivation of what has to change
between the two, including two non-obvious pitfalls encountered while
building this: the interferometer's missing zero-spacing flux, and how
noise-level calibration has to route through the beam.

This notebook:
1. builds a toy, spatially-varying (non-stationary) spectral cube: a single
   rotating ring that splits into two spatially separated blobs in most
   velocity channels due to its kinematics,
2. builds a toy but structurally real ALMA-like dirty beam from a mock
   antenna layout and simulated Earth-rotation synthesis,
3. convolves + adds noise to get a dirty cube, and shows the dirty beam and
   dirty central channel,
4. runs the 2D-1D wavelet deconvolution and compares true / dirty /
   recovered channel maps.
""")

code(r"""
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

import numpy as np
import matplotlib.pyplot as plt

from toy_cube import rotating_ring_cube, count_blobs
from psf import mock_alma_dirty_beam, beam_fwhm_pixels
from deconvolver import Deconvolver2D1D, convolve_cube

np.random.seed(0)
plt.rcParams['figure.facecolor'] = 'white'
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Toy non-stationary rotating-ring cube

The line-of-sight velocity at each point on the ring is `v_los = v_max *
sin(theta)`, i.e. it depends on spatial position (`theta`, the azimuthal
angle around the ring) -- the spectral line profile is *not* the same shape
everywhere in the field of view. At a fixed channel velocity `v0`, only the
two points on the ring where `v_los(theta) = v0` are bright, so a single,
spatially connected source shows up as two separated blobs in most
channels, merging into one only at the rotation curve's extremes (top/bottom
of the ring).
""")

code(r"""
cube, velocities, v_los = rotating_ring_cube(
    nz=41, ny=80, nx=80, v_max=220.0, line_sigma=18.0,
    ring_radius=11.0, ring_sigma=1.6, peak_flux=1.0,
)
nz, ny, nx = cube.shape
center_channel = nz // 2
print(f'cube shape (nz, ny, nx) = {cube.shape}')
print(f'true total flux = {cube.sum():.4g}')
print(f'peak brightness  = {cube.max():.4g}')

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = axs[0].imshow(v_los, origin='lower', cmap='RdBu_r')
axs[0].set_title('Underlying kinematics: v_los(x, y)')
plt.colorbar(im0, ax=axs[0], label='km/s', fraction=0.046)

im1 = axs[1].imshow(cube[center_channel], origin='lower', cmap='inferno')
axs[1].set_title(f'True cube, central channel (v={velocities[center_channel]:.0f} km/s)')
plt.colorbar(im1, ax=axs[1], label='flux', fraction=0.046)
plt.tight_layout()
plt.show()
""")

code(r"""
# Quick scan: how many spatially separated blobs does the true source show
# in each channel? (0.3x that channel's own peak, dependency-free
# connected-components count from toy_cube.count_blobs)
print(f"{'chan':>4} {'v (km/s)':>9} {'peak':>7} {'n_blobs':>7}")
for k in range(0, nz, 4):
    peak = cube[k].max()
    n_blobs, _ = count_blobs(cube[k], 0.3 * peak) if peak > 1e-6 else (0, None)
    print(f'{k:4d} {velocities[k]:9.0f} {peak:7.3f} {n_blobs:7d}')
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Toy ALMA-like dirty beam

Built from an actual (mock) antenna layout and simulated Earth-rotation
aperture synthesis, not hand-drawn: random antenna positions -> baseline
vectors for every antenna pair at several array rotation angles -> grid the
resulting (u, v) points -> the dirty beam is the inverse FFT of that
sampling function. Because baselines always occur as (i, j)/(j, i) +/- pairs,
the sampling function -- and hence the beam -- is exactly centro-symmetric,
which is also why the beam's convolution operator can be reused as its own
adjoint in the deconvolution's gradient step (see `deconvolver.py`).
""")

code(r"""
beam, sampling, (ant_x, ant_y), (u, v) = mock_alma_dirty_beam(
    n_ant=14, max_radius=100.0,
    hour_angles_deg=np.linspace(-40, 40, 5),
    grid_size=81, natural_weighting=False, seed=2,
)
print(f'beam shape = {beam.shape}, peak = {beam.max():.3f}')
print(f'beam.sum() = {beam.sum():.3e}  <-- ~0: no zero-length baseline is ever measured,')
print('               so the dirty beam has essentially zero DC (total-flux) response.')
print(f'uv sampling fill fraction = {(sampling > 0).mean():.3f}')
print(f'main-lobe FWHM ~ {beam_fwhm_pixels(beam):.1f} pixels')
print(f'beam is centro-symmetric: {np.allclose(beam, beam[::-1, ::-1])}')
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Dirty cube = true cube (*) dirty beam + noise

Convolve every channel of the true cube with the same dirty beam (FFT-based,
`deconvolver.convolve_cube`), then add Gaussian noise at a fixed peak SNR.
""")

code(r"""
dirty_clean = convolve_cube(cube, beam)  # PSF-convolved, no noise yet

peak_snr = 8.0
sigma_noise = dirty_clean.max() / peak_snr
rng = np.random.default_rng(42)
dirty_cube = dirty_clean + rng.normal(0.0, sigma_noise, size=dirty_clean.shape)

print(f'peak SNR = {peak_snr}, noise sigma = {sigma_noise:.4g}')
print(f'dirty cube min/max = {dirty_cube.min():.3g} / {dirty_cube.max():.3g}')
print(f'(true cube min/max was {cube.min():.3g} / {cube.max():.3g} -- note the dirty')
print(' map goes negative and overshoots the true peak: classic dirty-beam sidelobe bowls.')
""")

code(r"""
# --- Required figure: dirty PSF and the dirty central channel, one figure, subplots ---
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))

im0 = axs[0].imshow(beam, origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=1.0)
axs[0].set_title('Dirty beam (PSF)')
plt.colorbar(im0, ax=axs[0], fraction=0.046)

im1 = axs[1].imshow(dirty_cube[center_channel], origin='lower', cmap='inferno')
axs[1].set_title(f'Dirty cube, central channel (v={velocities[center_channel]:.0f} km/s)')
plt.colorbar(im1, ax=axs[1], fraction=0.046, label='flux')

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. 2D-1D wavelet deconvolution

`Deconvolver2D1D` runs FISTA (accelerated ISTA): at each iteration, a
gradient step w.r.t. the data-fidelity term `||dirty - H(x)||^2` is taken,
then the result is projected onto the sparse set by soft-thresholding its
2D-1D wavelet coefficients, then positivity is enforced. The detection
threshold decays from `k_start` to `k_end` (in units of sigma) over the
iterations. See `README.md` / `deconvolver.py` docstrings for the full
derivation from the original denoiser.
""")

code(r"""
deconvolver = Deconvolver2D1D(
    num_scales_2d=4, num_scales_1d=3,
    threshold_type='soft', positivity=True, verbose=True,
)
model, history = deconvolver.deconvolve(
    dirty_cube, beam, sigma_noise,
    n_iter=70, k_start=6.0, k_end=2.5, fista=True,
)
""")

code(r"""
rmse_dirty = np.sqrt(np.mean((dirty_clean - cube) ** 2))
rmse_model = np.sqrt(np.mean((model - cube) ** 2))
print(f'true flux            = {cube.sum():.4g}')
print(f'dirty flux           = {dirty_cube.sum():.4g}')
print(f'recovered flux       = {model.sum():.4g}  '
      f'({100*model.sum()/cube.sum():.0f}% of true -- see the zero-spacing-flux note in README.md)')
print(f'RMSE(dirty, true)    = {rmse_dirty:.4g}')
print(f'RMSE(recovered, true)= {rmse_model:.4g}  ({rmse_dirty/rmse_model:.1f}x lower)')
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Recovery: true vs. dirty vs. recovered

First the central channel, then the channel that best shows the
kinematic splitting into two blobs.
""")

code(r"""
def compare_panels(k, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    vmax = max(cube[k].max(), model[k].max(), 1e-6)
    for ax, img, name, cmap, vmin in [
        (axs[0], cube[k], 'True', 'inferno', 0),
        (axs[1], dirty_cube[k], 'Dirty', 'inferno', None),
        (axs[2], model[k], 'Recovered (2D-1D)', 'inferno', 0),
    ]:
        im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax if vmin is not None else None)
        ax.set_title(name)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

compare_panels(center_channel, f'Central channel (v={velocities[center_channel]:.0f} km/s)')
""")

code(r"""
# Find the channel with the widest true blob separation for the clearest
# demonstration of recovering the kinematic splitting.
best_k, best_n = None, 0
for k in range(nz):
    peak = cube[k].max()
    if peak < 0.3 * cube.max():
        continue
    n_blobs, _ = count_blobs(cube[k], 0.3 * peak)
    if n_blobs >= 2:
        best_k = k
        break
split_k = best_k if best_k is not None else center_channel

n_true, _ = count_blobs(cube[split_k], 0.3 * cube[split_k].max())
n_dirty, _ = count_blobs(np.clip(dirty_cube[split_k], 0, None), 4 * sigma_noise)
n_model, _ = count_blobs(model[split_k], 0.3 * model[split_k].max() if model[split_k].max() > 0 else 1e9)
print(f'channel {split_k} (v={velocities[split_k]:.0f} km/s): '
      f'n_blobs true={n_true}, dirty(4-sigma)={n_dirty}, recovered={n_model}')

compare_panels(split_k, f'Kinematic-splitting channel (v={velocities[split_k]:.0f} km/s)')
""")

code(r"""
plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, len(history['residual_std']) + 1), history['residual_std'])
plt.xlabel('iteration')
plt.ylabel('std(dirty - H(model))')
plt.title('FISTA convergence')
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## Takeaways

- The dirty beam's sidelobes (not a clean Gaussian PSF -- real interferometric
  dirty beams never are) visibly corrupt the central channel with negative
  bowls and spurious ripples; the 2D-1D wavelet deconvolution removes most of
  that structure and brings the RMSE to the true cube down substantially.
- The kinematic splitting (one ring, two blobs in most channels) is preserved
  through deconvolution -- the 2D-1D dictionary models extended/structured
  emission natively, unlike CLEAN's delta-function components, which is the
  practical payoff of using it here.
- Recovered total flux undershoots the true flux. This is expected, not a
  bug: the dirty beam has ~zero response at the zero-spacing (DC) mode
  (`beam.sum() ~ 0`), so that flux is fundamentally unconstrained by the
  data for *any* deconvolution method, CLEAN included. See `README.md` for
  how `Deconvolver2D1D` handles that null space (zeroing the coarsest
  wavelet sub-band instead of leaving it free, which otherwise makes the
  FISTA iteration diverge).
""")


# ---------------------------------------------------------------------------
def build_and_execute():
    sys.path.insert(0, SRC_DIR)
    namespace = {}
    nb_cells = []

    for i, (ctype, source) in enumerate(CELLS):
        source = source.strip("\n")
        if ctype == "markdown":
            nb_cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": source.splitlines(keepends=True),
            })
            continue

        print(f'--- executing code cell {i} ---', file=sys.stderr)
        stdout_buf = io.StringIO()
        outputs = []
        error = None
        plt.close("all")
        try:
            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(source, f"<cell {i}>", "exec"), namespace)
        except Exception as exc:  # noqa: BLE001
            error = exc
        finally:
            text = stdout_buf.getvalue()
            if text:
                outputs.append({
                    "output_type": "stream",
                    "name": "stdout",
                    "text": text.splitlines(keepends=True),
                })
            fignums = plt.get_fignums()
            for fignum in fignums:
                fig = plt.figure(fignum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("ascii")
                outputs.append({
                    "output_type": "display_data",
                    "data": {"image/png": b64, "text/plain": ["<Figure>"]},
                    "metadata": {},
                })
            plt.close("all")

        if error is not None:
            outputs.append({
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": [f"{type(error).__name__}: {error}"],
            })

        nb_cells.append({
            "cell_type": "code",
            "execution_count": i,
            "metadata": {},
            "source": source.splitlines(keepends=True),
            "outputs": outputs,
        })

        if error is not None:
            print(f'!!! cell {i} raised {error!r}', file=sys.stderr)
            break

    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(notebook, f, indent=1)
    print(f"Wrote {OUT_PATH}")

    had_error = any(
        any(o.get("output_type") == "error" for o in c.get("outputs", []))
        for c in nb_cells if c["cell_type"] == "code"
    )
    if had_error:
        print("NOTEBOOK BUILD HAD AN ERROR -- see traceback above", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    build_and_execute()
