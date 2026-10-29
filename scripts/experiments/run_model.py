#!/usr/bin/env python
"""
Generic single solve for the experiments (see common.py for the recipe).

    run_model.py --method pd|pdtau|fista|debias --galaxy ngc3110 --cfg compact [--grid coarse|fine|large]
                 [--sim] [--k 2] [--n-iter 20000] [--x0 model.npy] [--mask mask.npy]
                 [--bootstrap SEED --bootstrap-model model.npy] --out model.npy [--gif DIR --gif-title TITLE]
                 [--target-res ARCSEC]

--bootstrap: replace the data by N x_ref + noise (seed), x_ref = --bootstrap-model (parametric bootstrap).
--inject: add N x_inj to the data (x_inj a 0.02" model cube, e.g. point sources; experiment 10).
--null SEED: replace the data by pure noise with the measured power spectrum of the line-free channels (null test;
lambda still from the real cube, as for the real solve).
--lam-profile learnlet: on the scales the 2D dictionary's filter bank was trained for, lambda_b = the learnlet's own
learnt threshold of that scale (x --lam-scale), in units of plane b's noise (common.learnt_threshold_lam); k elsewhere.
--k-from FILE:KEY: k read from a JSON file (e.g. data/experiments/22_learnlet/calibration.json:learnlet16).
--learnt-weight w: the learnt filter planes' lambda x w, the starlet planes' unchanged (0 = starlet prior, 1 = learnlet).
The 2D dictionary is gpu_pd's default, i.e. env DICT_2D (starlet unless set).
--target-res: deconvolve to a target resolution (circular Gaussian FWHM, arcsec): structure sharper than
that comes out as that Gaussian, broader structure at its own size (gpu_pd.Op, Magain et al. 1998).
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
from common import G, problem, solve, to_native, to_saved, RW, learnt_threshold_lam, learnt_weight_lam  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--galaxy", default="ngc3110")
ap.add_argument("--cfg", default="compact"); ap.add_argument("--grid", default=None)
ap.add_argument("--sim", action="store_true"); ap.add_argument("--k", type=float, default=2.0)
ap.add_argument("--n-iter", type=int, default=20000); ap.add_argument("--x0"); ap.add_argument("--mask")
ap.add_argument("--bootstrap", type=int); ap.add_argument("--bootstrap-model")
ap.add_argument("--inject", help="0.02\" model cube (Jy/pixel) put through the PSF and added to the data")
ap.add_argument("--out", required=True); ap.add_argument("--gif"); ap.add_argument("--gif-title", default="")
ap.add_argument("--target-res", type=float, help="target resolution, arcsec (FWHM)")
ap.add_argument("--no-mask", action="store_true", help="no support mask (the extended configuration has one by default)")
ap.add_argument("--null", type=int, help="null test: the data replaced by pure noise (measured power spectrum), this seed")
ap.add_argument("--lam-profile", choices=["noise", "learnlet"], default="noise")
ap.add_argument("--lam-scale", type=float, default=1.0, help="--lam-profile learnlet: the learnt thresholds x this")
ap.add_argument("--k-from", help="FILE:KEY, k from a JSON file (overrides --k)")
ap.add_argument("--learnt-weight", type=float, default=1.0,
                help="lambda of a filter-bank dictionary's learnt filter planes x this (starlet planes unchanged; "
                     "0 = the starlet prior, 1 = the full learnlet prior); common.learnt_weight_lam")
a = ap.parse_args()
if a.k_from:
    f_, key_ = a.k_from.rsplit(":", 1)
    a.k = float(json.load(open(f_))[key_])
if a.no_mask:
    a.mask = "none"
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
if a.null is not None:
    P0 = problem(a.galaxy, a.cfg, a.grid, k=a.k, mask=a.mask)
    ps = G.measured_ps(G.noise_planes(P0.op.d, P0.noise))
    data = G.noise_with_ps(ps, tuple(P0.op.d.shape), seed=a.null).cpu().numpy().astype(np.float64)
    del P0, ps; torch.cuda.empty_cache()
if a.target_res and (a.x0 or a.method == "debias"):
    sys.exit("--target-res: --x0 / debias would need the model in the solver's x space; not supported")
P = problem(a.galaxy, a.cfg, a.grid, sim=a.sim, k=a.k, mask=a.mask, data=data, target_res=a.target_res)
x0 = G.to_t(to_native(P, np.load(a.x0)) if not a.sim else np.load(a.x0)) if a.x0 else None
if a.lam_profile == "learnlet":
    P.lam, fac = learnt_threshold_lam(P, scale=a.lam_scale)
    log("learnlet threshold profile: lambda_b / noise_b per spatial scale (median over planes) = "
        + str([round(float(np.median([f for f, s in zip(fac, P.D.plane_scale) if s == j])), 2) for j in range(P.D.J + 1)]))
if a.learnt_weight != 1.0:
    P.lam, isf = learnt_weight_lam(P, a.learnt_weight)
    log(f"learnt filter planes ({int(isf.sum())} of {len(isf)}): lambda x {a.learnt_weight:g}")
log(f"{a.method} {a.galaxy} {a.cfg} grid={P.grid} sim={a.sim} k={a.k:g} n_iter={a.n_iter} 2D={P.D.dict_2d} "
    f"lam={a.lam_profile} null={a.null} "
    f"(setup {time.time() - t0:.0f} s; line {P.line[0]}-{P.line[-1]}, {len(P.noise)} noise channels)"
    + (f"; target resolution {a.target_res:g}\" = {a.target_res / P.cell:.2f} px" if a.target_res else ""))

grab = None
if a.gif:
    from evolution_gifs import FrameGrabber
    grab = FrameGrabber(np.unique(np.geomspace(1, a.n_iter, 140).astype(int)), P.line, cell=P.cell, crop=P.crop)
cb = grab if grab is None or not a.target_res else (lambda it, x, g: grab(it, P.op.to_sky(x), g))
x = solve(P, a.method, n_iter=a.n_iter, x0=x0, callback=cb, log=log)

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.save(a.out, to_saved(P, x))
r = P.op.residual_sky(x)                      # d - N y on the original dirty cube
x = P.op.to_sky(x)
diag = dict(method=a.method, galaxy=a.galaxy, cfg=a.cfg, grid=P.grid, sim=a.sim, k=a.k, n_iter=a.n_iter,
            dict_2d=P.D.dict_2d, lam_profile=a.lam_profile, lam_scale=a.lam_scale, learnt_weight=a.learnt_weight, null=a.null,
            target_res=a.target_res,
            bootstrap=a.bootstrap, inject=a.inject, x0=a.x0, flux=float(x.sum()), flux_line=float(x[P.line].sum()),
            flux_noise_channels=float(x[[c for c in range(32) if c not in P.line]].sum()),
            resid_rms_line=float(r[P.line].std()), seconds=round(time.time() - t0))
json.dump(diag, open(a.out.replace(".npy", ".json"), "w"), indent=1)
log(f"saved {a.out}: flux {diag['flux']:.2f} Jy, line-free-channel flux {diag['flux_noise_channels']:.3f} Jy, "
    f"{diag['seconds']} s")
if grab is not None:
    from evolution_gifs import render
    name = {"pd": "primal-dual", "pdtau": "primal-dual (tau-lambda reweighting)", "pdtau-mrs": "primal-dual, residual on the multiresolution support", "pdtau-syn": "primal-dual, synthesis back-projection",
            "fista": "FISTA", "debias": "debiasing"}[a.method]
    render(grab.frames, grab.trace, a.gif, os.path.basename(a.out).replace(".npy", ""),
           a.gif_title or f"NGC 3110 {a.cfg}, {name}",
           lambda it: f"k = {a.k:g}" + (", reweighted" if it > RW["start"] and a.method != "debias" else ""),
           a.n_iter, reweight_iters=range(RW["start"], a.n_iter, RW["every"]) if a.method != "debias" else ())
log("DONE")
