# deconvgui

A browser front end for the 2D-1D wavelet deconvolution in this repository.
The server runs where the data are (your laptop or a cluster node) and the
browser only receives small arrays, frames and statistics, so it works the
same over an SSH tunnel as locally.

```
Load MS ──> inspect (fields, spws, baselines) ──> imaging parameters ──> dirty cube + PSF (tclean, niter=0)
Import cubes (dirty.npy + psf.npy, or a tclean FITS pair) ─────────────┘
                                                                          │
                     deconvolution parameters ──> run (live moment-0 frames, progress, residual/flux curves)
                                                                          │
     results: dirty / model / restored / residual (/ true sky), moment 0 or per channel,
     drag a box for zoomed crops + integrated spectra, evolution GIF, snapshot scrubber, convergence plots
```

## Install

From the repository root, in the environment you use for the solver (add
CASA there if you want "Load MS"):

```bash
pip install -e gui            # editable: the app always runs this checkout
pip install casatools casatasks   # optional, only for Load MS
pip install torch                 # optional, for the PyTorch solver on CPU / CUDA / MPS
```

## Run

**Laptop**

```bash
deconvgui                     # opens a browser tab
```

**Cluster (login node or any host you SSH into)**

```bash
deconvgui --port 8765 --root /nas/datasets/ALMA_visibilities --workdir /scratch/$USER/deconvgui
```

It prints the URL (with an access token) and the tunnel command, e.g.

```bash
# on your laptop
ssh -N -L 8765:node042:8765 alahiry@cluster-login
# then open http://localhost:8765/?token=...
```

**On a compute node** run it inside a job so the deconvolution gets the
node's CPUs/memory, and keep it alive across dropped connections:

```bash
srun --mem=64G --cpus-per-task=16 --time=8:00:00 --pty \
     deconvgui --port 8765 --root /nas/datasets/ALMA_visibilities --workdir /scratch/$USER/deconvgui
```

Use `tmux` (or `sbatch` with the output going to a file) so the server keeps
running when your SSH session ends. Jobs keep running server-side; reopening
the page picks up the live progress again. VS Code Remote-SSH forwards the
port for you.

Options: `--host` (default 127.0.0.1, i.e. reachable only through the
tunnel), `--root` (repeatable; folders the file browser may open, default
home + current directory), `--workdir` (default `~/deconvgui_work`, also
`$DECONVGUI_WORKDIR`), `--token`, `--no-token` (single-user laptop only),
`--no-browser`.

## How it uses the repository

* Every job (MS inspection, dirty imaging, deconvolution) runs in a **fresh
  worker process** that imports `simple/` and `scripts/` from this checkout
  at that moment. Edit `simple/deconvolve_pd.py`, or pull new commits, and
  the next run uses the new code; no restart. Only changes inside `gui/`
  itself need a restart of `deconvgui`.
* The solvers are called through their existing functions, unchanged:
  `deconvolve_pd.deconvolve`, `deconvolve.deconvolve`,
  `grid_with_casa.tclean_dirty_cube`, `inspect_ms.describe`,
  `restore.convolve_with_beam`. Parameters a function does not accept are
  dropped and listed in the job log, so the GUI keeps working when you change
  a signature.
* Live frames and cancellation come from a thin wrapper around the operator
  (`worker._WatchedOperator`): both solvers call `operator.gradient()` once
  per iteration, which is where the wrapper records the iterate.
* The header shows the checked-out commit and any local modifications or
  unresolved conflicts. It only reads git state (`GIT_OPTIONAL_LOCKS=0`);
  the app never fetches, pulls or commits.

## Solvers and devices

| solver | code | devices |
|---|---|---|
| primal-dual, PyTorch | `simple/gpu_pd.py` (`solve_pd`, `noise_lambdas`) | CPU, CUDA GPUs, Apple GPU (MPS) |
| FISTA, PyTorch | loop of `scripts/make_fista_run.py` on `gpu_pd` blocks | CPU, CUDA GPUs, Apple GPU (MPS) |
| primal-dual, numpy | `simple/deconvolve_pd.py` | CPU |
| FISTA + reweighting, numpy | `simple/deconvolve.py` | CPU |

The device list is detected at start-up (all CUDA GPUs visible to the
process, so it follows `CUDA_VISIBLE_DEVICES` / your SLURM allocation, plus MPS
on a Mac). Tick one or several devices: **one run uses one device**, and each
device runs one job at a time. Several runs (a sweep, written as a comma list
of k values, or runs submitted one after another) go into a queue and each
starts on the next free ticked device, so N GPUs run N runs in parallel. A
single run is not split across GPUs: the 2D-1D transform couples all channels
and the PSF convolution couples the whole field, so splitting one cube would
need halo exchange every iteration.

`gpu_pd.py` itself picks `cuda` or `cpu` at import. The GUI sets
`gpu_pd.DEV` to the chosen device inside the worker. For MPS it also builds the
PSF transfer function in float64 on the CPU (Metal has no float64) and draws
the random noise realizations with a CPU generator; this changes only the
module in that worker process, not the file. MPS needs a recent PyTorch on
macOS 14 or later (complex FFTs on Metal). If you later make
`gpu_pd.py` device-aware yourself, delete `_apply_mps_compat` in
`gui/deconvgui/worker.py` so the GUI uses your version unchanged.

## Thresholds and the null test

**Null test** (PyTorch solvers): the cube is replaced by a noise realization
with the power spectrum measured in the line-free channels, and everything else
is unchanged. A good threshold setting returns a model with (nearly) zero flux.

**λ calibration** (primal-dual): `noise` uses `gpu_pd.noise_lambdas`
(λ_b = k × MAD of the noise's own coefficients W n). This fails the null test:
the primal-dual optimum keeps x = 0 only if the noise can be written as Wᵀu with
|u| ≤ λ, and for the redundant 2D-1D frame those u are larger and spread
differently over the sub-bands than W n. `dual` (default) sets
λ_b = k × max(MAD_b(u), MAD_b(W n)) with u = W(WᵀW)⁻¹n for a noise realization
(conjugate gradient, done once before iterating). On the synthetic disc, 1000
iterations, pure-noise input: noise-calibrated k = 5 gives 33 % of the true
sky's flux; dual-calibrated k = 5, 3 and 2 give 0.1 %, 0.1 % and 0.7 %. FISTA
thresholds the primal coefficients directly, so it uses the noise calibration.

## Where things are stored

```
<workdir>/datasets/<dataset>/
    meta.json  dirty.npy  psf.npy  [truth.npy]
    results/<rNNN-solver>/
        params.json  model.npy  restored.npy  residual.npy
        frames.npy  frames.json  history.json  evolution.gif
```

Everything shown in the browser is rebuilt from these files, so results
survive a restart. Tick **example** on a dataset to keep it at the top of the
list (your worked, debugged cases with their GIFs).

## Changing the look or adding things

* `gui/deconvgui/static/app.css`: all colours, fonts and spacing are CSS
  variables at the top (dark theme first, light theme below).
* `gui/deconvgui/static/app.js`: plain JavaScript, no build step; reload the
  page after editing. Image panels are `ImagePanel`, plots are `linePlot`.
* New solver: add a branch in `worker.deconvolve_job` and a default
  parameter set in `DEFAULTS` in `app.js`.
* New API endpoint: `gui/deconvgui/server.py`.

## Notes

* The fast operator assumes one pointing. For a mosaic the inspection step
  says so; use the exact operator offline for those.
* `Synthetic example` makes a toy rotating disc with a random (u, v)
  coverage and stores the true sky, so the result panels can show it next to
  the model.
