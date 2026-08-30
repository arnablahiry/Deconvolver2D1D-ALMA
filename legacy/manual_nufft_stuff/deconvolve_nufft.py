"""
The `notebooks/test.ipynb` approach, as a script: sparse 2D-1D starlet
deconvolution driven by a genuine NUFFT (torchkbnufft) operating directly on
the real, scattered (u, v) points -- no shift-invariant-PSF approximation
(`../simple`'s fast path) and no per-iteration CASA major cycle
(`../simple/exact_operator.py`). A NUFFT IS the true non-uniform Fourier
transform, computed via fast gridding-with-a-small-kernel + FFT, so this is
simultaneously closer to "exact" than the PSF-convolution shortcut and much
cheaper per iteration than a real CASA degrid/grid pass.

======================================================================
WHAT'S DIFFERENT FROM notebooks/test.ipynb, AND WHY
======================================================================
* `load_visibilities.py` filters to a single spectral window (needed here --
  the notebook's TW Hya MS had one spw; this split MS still carries 4, with
  DIFFERENT channel counts, so reading across all of them at once the way
  the notebook did would either error or silently mix frequency axes) and
  applies the mosaic phase-rotation derived in that file's docstring. The
  notebook never needed this: TW Hya is a single pointing.
* UNLIKE the notebook, this version applies density compensation
  (`tkbn.calc_density_compensation_function`, Pipe's method) before every
  gridding operation. The notebook's plain, unweighted `A^H V` turned out not
  to be a design choice worth keeping on real ALMA data: without correcting
  for ALMA's very uneven (u, v) sampling density, the "dirty cube" was
  dominated by oversampled regions rather than the sky, and the symptom was
  concrete, not aesthetic -- 90% of the FISTA output was nonzero and the
  data misfit barely moved in 60 iterations (0.2%). Density compensation is
  the NUFFT analogue of "uniform" weighting -- not identical to the Briggs
  robust=0.5 used elsewhere in this project, but a real correction. See the
  comment above `dcomp = tkbn.calc_density_compensation_function(...)` below.
* Still MISSING relative to a true CASA mosaic combination: per-pointing
  PRIMARY BEAM weighting. Real mosaicking multiplies each pointing's
  contribution by its own primary beam response before combining, so
  pointings agree on relative flux in their overlap region. This script
  phase-corrects (gets the pointings onto a common grid, coherently) but
  does not PB-weight them, so relative flux near the mosaic's edges (well
  outside all 3 pointings' good coverage) should not be trusted. Near the
  common phase center, where the 3 pointings overlap and their primary
  beams are all close to their peak, this is a small effect.
* Fixed threshold, computed once (not re-derived from the shrinking
  residual every iteration): the exact same fix `../simple/deconvolve.py`
  needed. The notebook already did this right (`DIRTY_SCALE = step`,
  documented in its own cell 3c) -- kept unchanged here.
* Sparsity dictionary: plain 2D-1D starlet (`../src/wavelet2d1d_torch.py`'s
  `prox_2d1d`), matching the notebook exactly. `../simple` uses CDF 9/7 for
  the spectral axis instead; that swap was specific to that folder's brief
  and is not repeated here, to keep this a faithful script version of the
  notebook rather than a third variant.
"""

import os
import sys
import time

import numpy as np
import torch
import torchkbnufft as tkbn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from wavelet2d1d_torch import prox_2d1d, subband_mad_noise

from load_visibilities import load_mosaic_visibilities

# ---------------------------------------------------------------- config --
MS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "data", "ngc7469_co21.ms")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SPW = 0                    # single spw: see module docstring
CHAN_START, CHAN_END = 0, 60   # subset for a first, fast run -- see main()
CELL_ARCSEC = 0.04          # matches ../src and ../simple, for comparability
IM_SIZE = (320, 320)

NUM_SCALES_2D = 4
NUM_SCALES_1D = 2          # nz=60 here; needs nz > 2**NUM_SCALES_1D
K_SIGMA = 4.0
N_ITER = 60
POSITIVITY = True


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    print(f"[setup] device: {device}")

    ktraj, v_batched, cell_arcsec, meta = load_mosaic_visibilities(
        MS_PATH, spw=SPW, chan_start=CHAN_START, chan_end=CHAN_END,
        cell_size_arcsec=CELL_ARCSEC)
    ktraj, v_batched = ktraj.to(device), v_batched.to(device)
    n_chan = v_batched.shape[0]
    print(f"[setup] cube (nz, ny, nx) = ({n_chan}, {IM_SIZE[0]}, {IM_SIZE[1]})  "
          f"cell={cell_arcsec}\"  fov={IM_SIZE[0] * cell_arcsec:.2f}\"")

    nufft_ob = tkbn.KbNufft(im_size=IM_SIZE).to(device)
    adjnufft_ob = tkbn.KbNufftAdjoint(im_size=IM_SIZE).to(device)

    # -- density compensation: this is the missing "imaging weights" ---------
    # A plain adjoint NUFFT, A^H V with no correction, treats every visibility
    # as equally informative regardless of how densely sampled its (u, v)
    # neighborhood is. ALMA's (u, v) coverage is very uneven (many more
    # baselines at some radii than others), so A^H V ends up dominated by
    # whichever regions happen to be oversampled -- not by the sky. In
    # practice this showed up as a dirty cube with no real "beam": nearly
    # every pixel carried some low-level signal, 90% of the FISTA output was
    # nonzero, and the data misfit barely moved in 60 iterations (5.0505e6 ->
    # 5.0412e6, 0.2%) because the model was mostly fitting broad sidelobe
    # structure, not the source.
    #
    # `calc_density_compensation_function` is torchkbnufft's implementation
    # of Pipe's method: an iterative estimate of, roughly, 1/(local sample
    # density) at each (u, v) point, computed once from `ktraj` alone (shared
    # by every channel here) and applied to every visibility before gridding.
    # This is the NUFFT analogue of "uniform" imaging weighting -- not
    # identical to the Briggs robust=0.5 weights used in ../src and
    # ../simple, but a real correction, not the near-total absence of one.
    print("[setup] computing density compensation (Pipe's method) ...")
    dcomp = tkbn.calc_density_compensation_function(ktraj, IM_SIZE)

    # -- dirty cube: A^H (dcomp * V) ------------------------------------------
    with torch.no_grad():
        dirty_complex = adjnufft_ob(v_batched * dcomp, ktraj)
        dirty_cube = dirty_complex.real.squeeze(1)    # (nz, ny, nx), real part
    print(f"[setup] dirty cube: peak={dirty_cube.abs().max().item():.4e}  "
          f"rms={dirty_cube.std().item():.4e}")

    # -- Lipschitz constant of A^H diag(dcomp) A, by power iteration (exact
    #    for this operator; every channel shares one k-trajectory here, so
    #    one plane is enough). MUST include dcomp here too -- the operator
    #    whose Lipschitz constant sets a stable FISTA step is whichever one
    #    actually appears in the gradient below, not the plain A^H A. --------
    def cnorm(t):
        # torch.linalg.vector_norm doesn't support complex on MPS yet.
        return torch.sqrt((t.real ** 2 + t.imag ** 2).sum())

    print("[setup] estimating Lipschitz constant (power iteration) ...")
    with torch.no_grad():
        x = torch.randn((1, 1, *IM_SIZE), device=device, dtype=torch.complex64)
        x = x / cnorm(x)
        L = 0.0
        for i in range(20):
            x_new = adjnufft_ob(nufft_ob(x, ktraj) * dcomp, ktraj)
            L = (cnorm(x_new) / cnorm(x)).item()
            x = x_new / cnorm(x_new)
    step = 1.0 / L
    print(f"[setup] L = {L:.4e}   step = {step:.4e}")

    # -- per-sub-band noise, estimated ONCE from the dirty cube AT THE SCALE
    #    the FISTA iterate actually uses (v = z - step * grad; from x=0 that
    #    is exactly step * dirty). Getting this scale wrong -- estimating
    #    sigma from the raw, unscaled dirty cube instead -- is precisely the
    #    bug `../simple/deconvolve.py` hit: every threshold ends up
    #    ~1/step too large, every coefficient is crushed on iteration 0, and
    #    the loop runs to completion doing nothing (residual_rms and
    #    active=0 identical every printout). -----------------------------
    with torch.no_grad():
        noise_levels = subband_mad_noise(step * dirty_cube, NUM_SCALES_2D,
                                         NUM_SCALES_1D)
    thresholds = K_SIGMA * noise_levels
    print(f"[setup] threshold range: {thresholds.min().item():.3e} - "
          f"{thresholds.max().item():.3e}")

    # -- FISTA -------------------------------------------------------------
    X = torch.zeros((n_chan, 1, *IM_SIZE), dtype=torch.float32, device=device)
    X_prev, Y, t = X.clone(), X.clone(), 1.0

    print(f"[fista] {N_ITER} iterations ...")
    t0 = time.time()
    with torch.no_grad():
        for it in range(N_ITER):
            residual_vis = nufft_ob(Y.to(torch.complex64), ktraj) - v_batched
            grad = adjnufft_ob(residual_vis * dcomp, ktraj).real
            Z = Y - step * grad

            X_new = prox_2d1d(Z.squeeze(1), thresholds, NUM_SCALES_2D,
                              NUM_SCALES_1D, keep_coarse=True).unsqueeze(1)
            if POSITIVITY:
                X_new = torch.relu(X_new)

            t_new = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            Y = X_new + ((t - 1.0) / t_new) * (X_new - X_prev)
            X_prev, X, t = X_new, X_new, t_new

            if it % 5 == 0 or it == N_ITER - 1:
                data_loss = 0.5 * (residual_vis.real ** 2
                                   + residual_vis.imag ** 2).sum().item()
                active = (X > 0).float().mean().item() * 100
                print(f"iter {it:3d}  0.5||V-AX||^2={data_loss:.4e}  "
                      f"flux={X.sum().item():8.2f}  active={active:5.2f}%  "
                      f"max={X.max().item():.4e}")
    print(f"[fista] done in {time.time() - t0:.1f}s")

    os.makedirs(OUT_DIR, exist_ok=True)
    model_cube = X.squeeze(1).cpu().numpy()
    dirty_out = dirty_cube.cpu().numpy()
    np.save(os.path.join(OUT_DIR, "model.npy"), model_cube)
    np.save(os.path.join(OUT_DIR, "dirty.npy"), dirty_out)
    print(f"[done] saved {OUT_DIR}/model.npy, dirty.npy  shape {model_cube.shape}")


if __name__ == "__main__":
    main()
