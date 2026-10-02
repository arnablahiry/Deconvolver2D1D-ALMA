# Deconvolver2D1D-ALMA

2D-1D wavelet **deconvolution** for dirty ALMA spectral cubes, adapted from
the 2D-1D wavelet **denoising** framework in
[`Denoiser2D1D-improved`](../Denoiser2D1D-improved). Same kind of dictionary
(spatial starlet scales x spectral wavelet scales), different inverse
problem: this repo goes from a dirty-beam-convolved cube straight to a
"clean" cube, instead of CLEAN's iterative point-source component fitting.

## Layout

```
simple/
  grid_with_casa.py   The only step that touches visibilities: CASA's
                      synthesisimager grids the MS once into a PSF cube and a
                      dirty cube (simple/data/psf.npy, dirty.npy)
  uv_to_image.py      ShiftInvariantOperator: the normal operator
                      N = A^H W A as PSF convolution via FFT (pure numpy/scipy)
  exact_operator.py   ExactOperator: same interface, but one real CASA
                      degrid+grid pass per call (mosaic-accurate, slow)
  wavelets.py         2D starlet (spatial) x 1D CDF 9/7 (spectral) transform,
                      soft thresholding, MAD noise estimate; self-test via
                      `python simple/wavelets.py`
  deconvolve.py       FISTA + reweighted-L1 sparsity in that dictionary --
                      the whole algorithm as one loop
  casa_logs.py        Redirects CASA's timestamped log files into casa_logs/
scripts/
  split_line_ms.py                 Cuts a line window out of a large
                                   calibrated MS into a small working MS
  inspect_ms.py                    Prints fields / array / baselines /
                                   spectral setup of one or more MSs
  build_alma_visibility_notebook.py  Builds + executes
                                   notebooks/alma_visibility_inspection.ipynb
                                   (uv coverage, dirty image, dirty beam)
  notebook_builder.py              Cell-execution harness for the above
  download_ngc3110_compact.sh      EU ARC CalMS downloads of the compact
  download_ngc2369_compact.sh      12m configurations (see Data below)
notebooks/
  alma_visibility_inspection.ipynb
legacy/notebooks/                  Executed notebooks from the removed src/
                                   pipeline, kept for reference only
```

Every module docstring in `simple/` explains its step in full; start with
`deconvolve.py` (the optimization problem and why the model's amplitude sits
below the dirty image's) and `uv_to_image.py` (the measurement operator).

## Running the deconvolution

From the repo root:

```
python simple/grid_with_casa.py   # CASA env: MS -> simple/data/{psf,dirty}.npy
python simple/deconvolve.py       # pure numpy: -> simple/data/model.npy
```

`grid_with_casa.py` holds the MS path and imaging configuration (image size,
cell, channel range, rest frequency, phase centre) at the top of the file;
`exact_operator.py` reuses the same settings. Set `USE_EXACT_OPERATOR = True`
in `deconvolve.py` to trade the FFT operator (~0.2 s/iteration) for the exact
CASA one (~1-2 min/iteration).

## Data

Calibrated MSs live outside the repo, under `/nas/datasets/ALMA_visibilities`
(see `LAYOUT.txt` there). Project 2017.1.00255.S, band 6:

| galaxy | config | MOUS | EB(s) |
|---|---|---|---|
| NGC 3110 | 12m extended | `uid://A001/X1284/X1371` | `uid___A002_Xc7cf4e_X7866` |
| NGC 3110 | 12m compact | `uid://A001/X1284/X1373` | `uid___A002_Xcf92df_X4e8e`, `uid___A002_Xd09670_X3f8d` |
| NGC 2369 | 12m extended | `uid://A001/X1284/X136b` | `uid___A002_Xc845c0_X1710` |
| NGC 2369 | 12m compact | `uid://A001/X1284/X136d` | `uid___A002_Xcd07af_X20e` |

All come from the [EU ARC CalMS service](https://almascience.org/tools/eu-arc-network/the-european-arc-calms-service)
(already-calibrated `*.ms.split.cal`, DATA column only). The two
`scripts/download_*_compact.sh` scripts fetch the compact configurations;
their `mydir/<hash>` links are personal and expire.

## Requirements

- `numpy`, `scipy`, `matplotlib` for `simple/` outside the CASA steps
- `casatools` + `casatasks` (6.7) for `grid_with_casa.py`, `exact_operator.py`
  and everything in `scripts/`, e.g. the env at
  `/scratch/alahiry/conda/envs/deconvolver2d1d`
