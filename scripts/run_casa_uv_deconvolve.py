#!/usr/bin/env python
"""
End-to-end uv-plane 2D-1D wavelet deconvolution of the real TW Hya
visibilities with **all Fourier (visibility <-> image) operations done by
CASA** -- via `src/casa_uv_operator.CASAFourierOperator` -- rather than this
repo's dense non-uniform-DFT matrix or its own gridder.

The point (and why nothing in `uv_deconvolver.py` changes)
----------------------------------------------------------
`UVDeconvolver2D1D.deconvolve` never performs a Fourier transform itself: its
whole data-fidelity gradient is `x + mu * operator.adjoint_cube(vis_obs -
operator.forward_cube(x))`, i.e. every visibility<->image step is delegated to
the `operator`. Swapping `NonUniformFourierOperator` for `CASAFourierOperator`
therefore makes the *entire* solver run on CASA's gridder/degridder with no
change to the deconvolver code -- that operator seam is exactly the place to
substitute the transform implementation. Putting raw CASA calls inside
`uv_deconvolver.py` would only couple the solver to CASA and lose that.

*** MUST BE RUN IN A CASA ENVIRONMENT (casatools/casatasks), e.g. casa_env. ***
It is NOT runnable in this repo's pure-numpy sandbox (same constraint as
`export_visibilities.py` / `run_ms_clean_uv_grid.py`). Run from the directory
containing `twhya_calibrated.ms`:

    /path/to/casa_env/bin/python scripts/run_casa_uv_deconvolve.py

It writes `scripts/.checkpoints/casa_uv_model.npz` (the recovered model cube +
metadata), which `notebooks/real_uv_deconvolution_demo.ipynb` can then load
and plot in the ordinary sandbox -- the same run-heavy-CASA-separately,
load-checkpoint-in-notebook pattern `run_real_pipeline.py` and
`run_ms_clean_uv_grid.py` already use.

NOTE: `CASAFourierOperator` is written to the documented casatools API but has
NOT been executed/verified in this repo (no casatools here). Expect to debug
it in casa_env -- most likely the forward/adjoint gridding normalization (so
they are a true adjoint pair and `mu = 1/L` is the right step size). Also note
`uv_deconvolver.py` currently sets `mu = 1.0` (the `/ L` is commented out on
line ~141); for a real run restore `mu = 1.0 / L`, or FISTA diverges to NaN
regardless of which operator is used.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from casa_uv_operator import CASAFourierOperator   # noqa: E402  (needs casatools)
from uv_deconvolver import UVDeconvolver2D1D        # noqa: E402

# --- selection: same spw/40-channel CO(3-2) window as the rest of the pipeline,
#     imaged onto the uv-notebook's 64x64 / 0.2099"/px grid ---------------------
MS = "twhya_calibrated.ms"
NY = NX = 64
CELL_ARCSEC = 0.2099
SPW = "0"
NCHAN = 40
START = 1400          # CASA internal regridded-frame channel index (see run_ms_clean_uv_grid.py)
WIDTH = 5
ROBUST = 0.5
N_ITER = 150
K = 3.0

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".checkpoints", "casa_uv_model.npz")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    print("Building CASAFourierOperator (all forward/adjoint transforms via CASA gridder)...")
    operator = CASAFourierOperator(
        msname=MS, nx=NX, ny=NY, cell_arcsec=CELL_ARCSEC, spw=SPW, field="",
        nchan=NCHAN, start=START, width=WIDTH, weighting="briggs", robust=ROBUST,
        workdir=".",
    )

    print("Reading observed visibilities (DATA column) as (nz, n_uv)...")
    vis_obs = operator.observed_visibilities("DATA")
    nz = vis_obs.shape[0]
    print(f"  vis_obs shape = {vis_obs.shape}")

    # sigma_vis is ignored by the current (dynamic-MAD) deconvolver, but the
    # positional argument is still required.
    deconvolver = UVDeconvolver2D1D(
        num_scales_2d=3, num_scales_1d=3,
        threshold_type="soft", positivity=True, verbose=True,
    )
    print("Running iterative soft-thresholding deconvolution (CASA transforms every iteration)...")
    model, history = deconvolver.deconvolve(
        vis_obs, operator, sigma_vis=None, cube_shape=(nz, NY, NX),
        n_iter=N_ITER, k=K, fista=True,
    )

    np.savez_compressed(
        OUT,
        model=model,
        residual_std=np.asarray(history["residual_std"]),
        cell_arcsec=CELL_ARCSEC, ny=NY, nx=NX, robust=ROBUST, k=K, n_iter=N_ITER,
    )
    print(f"Wrote {OUT}: model {model.shape}, final flux {model.sum():.4g}")
    operator.close()


if __name__ == "__main__":
    main()
