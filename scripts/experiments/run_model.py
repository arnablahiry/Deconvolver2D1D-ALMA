#!/usr/bin/env python
"""
Generic single solve for the experiments (see common.py for the recipe).

    run_model.py --method pd|pdtau|fista|debias --galaxy ngc3110 --cfg compact [--grid coarse|fine|large]
                 [--sim] [--k 2] [--n-iter 20000] [--x0 model.npy] [--mask mask.npy]
                 [--bootstrap SEED --bootstrap-model model.npy] --out model.npy [--gif DIR --gif-title TITLE]

--bootstrap: replace the data by N x_ref + noise (seed), x_ref = --bootstrap-model (parametric bootstrap).
--inject: add N x_inj to the data (x_inj a 0.02" model cube, e.g. point sources; experiment 10).
Saves the model (real data: Jy/pixel at 0.02"; simulations: on their own grid) and a JSON of diagnostics.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import G, problem, solve, to_native, to_saved, RW  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--galaxy", default="ngc3110")
ap.add_argument("--cfg", default="compact"); ap.add_argument("--grid", default=None)
ap.add_argument("--sim", action="store_true"); ap.add_argument("--k", type=float, default=2.0)
ap.add_argument("--n-iter", type=int, default=20000); ap.add_argument("--x0"); ap.add_argument("--mask")
ap.add_argument("--bootstrap", type=int); ap.add_argument("--bootstrap-model")
ap.add_argument("--inject", help="0.02\" model cube (Jy/pixel) put through the PSF and added to the data")
ap.add_argument("--out", required=True); ap.add_argument("--gif"); ap.add_argument("--gif-title", default="")
a = ap.parse_args()
tag = f"[{os.path.basename(a.out)}]"
log = lambda m: print(f"{tag} {m}", flush=True)
t0 = time.time()

data = None
if a.bootstrap is not None:
    P0 = problem(a.galaxy, a.cfg, a.grid, k=a.k, mask=a.mask)
    xr = G.to_t(to_native(P0, np.load(a.bootstrap_model)))
    npl = G.noise_planes(G.to_t(P0.op.d.cpu().numpy()), P0.noise)
    noise = G.noise_with_ps(G.measured_ps(npl), P0.op.d.shape, seed=1000 + a.bootstrap)
    data = (P0.op.apply(xr) + noise).cpu().numpy().astype(np.float64)
    del P0, xr, npl, noise; torch.cuda.empty_cache()
if a.inject:
    P0 = problem(a.galaxy, a.cfg, a.grid, sim=a.sim, k=a.k, mask=a.mask)
    data = (P0.op.d + P0.op.apply(G.to_t(to_native(P0, np.load(a.inject))))).cpu().numpy().astype(np.float64)
    del P0; torch.cuda.empty_cache()
P = problem(a.galaxy, a.cfg, a.grid, sim=a.sim, k=a.k, mask=a.mask, data=data)
x0 = G.to_t(to_native(P, np.load(a.x0)) if not a.sim else np.load(a.x0)) if a.x0 else None
log(f"{a.method} {a.galaxy} {a.cfg} grid={P.grid} sim={a.sim} k={a.k:g} n_iter={a.n_iter} "
    f"(setup {time.time() - t0:.0f} s; line {P.line[0]}-{P.line[-1]}, {len(P.noise)} noise channels)")

grab = None
if a.gif:
    from evolution_gifs import FrameGrabber
    grab = FrameGrabber(np.unique(np.geomspace(1, a.n_iter, 140).astype(int)), P.line, cell=P.cell, crop=P.crop)
x = solve(P, a.method, n_iter=a.n_iter, x0=x0, callback=grab, log=log)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, to_saved(P, x))
r = -P.op.grad(x)
diag = dict(method=a.method, galaxy=a.galaxy, cfg=a.cfg, grid=P.grid, sim=a.sim, k=a.k, n_iter=a.n_iter,
            bootstrap=a.bootstrap, inject=a.inject, x0=a.x0, flux=float(x.sum()), flux_line=float(x[P.line].sum()),
            flux_noise_channels=float(x[[c for c in range(32) if c not in P.line]].sum()),
            resid_rms_line=float(r[P.line].std()), seconds=round(time.time() - t0))
json.dump(diag, open(a.out.replace(".npy", ".json"), "w"), indent=1)
log(f"saved {a.out}: flux {diag['flux']:.2f} Jy, line-free-channel flux {diag['flux_noise_channels']:.3f} Jy, "
    f"{diag['seconds']} s")
if grab is not None:
    from evolution_gifs import render
    name = {"pd": "primal-dual", "pdtau": "primal-dual (tau-lambda reweighting)", "fista": "FISTA", "debias": "debiasing"}[a.method]
    render(grab.frames, grab.trace, a.gif, os.path.basename(a.out).replace(".npy", ""),
           a.gif_title or f"NGC 3110 {a.cfg}, {name}",
           lambda it: f"k = {a.k:g}" + (", reweighted" if it > RW["start"] and a.method != "debias" else ""),
           a.n_iter, reweight_iters=range(RW["start"], a.n_iter, RW["every"]) if a.method != "debias" else ())
log("DONE")
